/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.fpm

import java.{util => ju}
import java.lang.{Iterable => JavaIterable}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag
import scala.reflect.runtime.universe._

import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{compact, render}

import org.apache.spark.{HashPartitioner, Partitioner, SparkContext, SparkException}
import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext.fakeClassTag
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.fpm.FPGrowth._
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel

/**
 * Model trained by [[FPGrowth]], which holds frequent itemsets.
 * @param freqItemsets frequent itemset, which is an RDD of [[FreqItemset]]
 * @tparam Item item type
 */
@Since("1.3.0")
class FPGrowthModel[Item: ClassTag] @Since("1.3.0") (
    @Since("1.3.0") val freqItemsets: RDD[FreqItemset[Item]])
  extends Saveable with Serializable {
  /**
   * Generates association rules for the [[Item]]s in [[freqItemsets]].
   * @param confidence minimal confidence of the rules produced
   */
  @Since("1.5.0")
  def generateAssociationRules(confidence: Double): RDD[AssociationRules.Rule[Item]] = {
    val associationRules = new AssociationRules(confidence)
    associationRules.run(freqItemsets)
  }

  /**
   * Save this model to the given path.
   * It only works for Item datatypes supported by DataFrames.
   *
   * This saves:
   *  - human-readable (JSON) model metadata to path/metadata/
   *  - Parquet formatted data to path/data/
   *
   * The model may be loaded using [[FPGrowthModel.load]].
   *
   * @param sc  Spark context used to save model data.
   * @param path  Path specifying the directory in which to save this model.
   *              If the directory already exists, this method throws an exception.
   */
  @Since("2.0.0")
  override def save(sc: SparkContext, path: String): Unit = {
    FPGrowthModel.SaveLoadV1_0.save(this, path)
  }

  override protected val formatVersion: String = "1.0"
}

@Since("2.0.0")
object FPGrowthModel extends Loader[FPGrowthModel[_]] {

  @Since("2.0.0")
  override def load(sc: SparkContext, path: String): FPGrowthModel[_] = {
    FPGrowthModel.SaveLoadV1_0.load(sc, path)
  }

  private[fpm] object SaveLoadV1_0 {

    private val thisFormatVersion = "1.0"

    private val thisClassName = "org.apache.spark.mllib.fpm.FPGrowthModel"

    def save(model: FPGrowthModel[_], path: String): Unit = {
      val sc = model.freqItemsets.sparkContext
      val spark = SparkSession.builder().config(sc.getConf).getOrCreate()

      val metadata = compact(render(
        ("class" -> thisClassName) ~ ("version" -> thisFormatVersion)))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))

      // Get the type of item class
      val sample = model.freqItemsets.first().items(0)
      val className = sample.getClass.getCanonicalName
      val classSymbol = runtimeMirror(getClass.getClassLoader).staticClass(className)
      val tpe = classSymbol.selfType

      val itemType = ScalaReflection.schemaFor(tpe).dataType
      val fields = Array(StructField("items", ArrayType(itemType)),
        StructField("freq", LongType))
      val schema = StructType(fields)
      val rowDataRDD = model.freqItemsets.map { x =>
        Row(x.items.toSeq, x.freq)
      }
      spark.createDataFrame(rowDataRDD, schema).write.parquet(Loader.dataPath(path))
    }

    def load(sc: SparkContext, path: String): FPGrowthModel[_] = {
      implicit val formats = DefaultFormats
      val spark = SparkSession.builder().config(sc.getConf).getOrCreate()

      val (className, formatVersion, metadata) = Loader.loadMetadata(sc, path)
      assert(className == thisClassName)
      assert(formatVersion == thisFormatVersion)

      val freqItemsets = spark.read.parquet(Loader.dataPath(path))
      val sample = freqItemsets.select("items").head().get(0)
      loadImpl(freqItemsets, sample)
    }

    def loadImpl[Item: ClassTag](freqItemsets: DataFrame, sample: Item): FPGrowthModel[Item] = {
      val freqItemsetsRDD = freqItemsets.select("items", "freq").rdd.map { x =>
        val items = x.getAs[Seq[Item]](0).toArray
        val freq = x.getLong(1)
        new FreqItemset(items, freq)
      }
      new FPGrowthModel(freqItemsetsRDD)
    }
  }
}

/**
 * A parallel FP-growth algorithm to mine frequent itemsets. The algorithm is described in
 * [[http://dx.doi.org/10.1145/1454008.1454027 Li et al., PFP: Parallel FP-Growth for Query
 *  Recommendation]]. PFP distributes computation in such a way that each worker executes an
 * independent group of mining tasks. The FP-Growth algorithm is described in
 * [[http://dx.doi.org/10.1145/335191.335372 Han et al., Mining frequent patterns without candidate
 *  generation]].
 *
 * @param minSupport the minimal support level of the frequent pattern, any pattern that appears
 *                   more than (minSupport * size-of-the-dataset) times will be output
 * @param numPartitions number of partitions used by parallel FP-growth
 *
 * @see [[http://en.wikipedia.org/wiki/Association_rule_learning Association rule learning
 *       (Wikipedia)]]
 *
 */
@Since("1.3.0")
class FPGrowth private (
    private var minSupport: Double,
    private var numPartitions: Int) extends Logging with Serializable {

  /**
   * Constructs a default instance with default parameters {minSupport: `0.3`, numPartitions: same
   * as the input data}.
   *
   */
  @Since("1.3.0")
  def this() = this(0.3, -1)

  /**
   * Sets the minimal support level (default: `0.3`).
   *
   */
  @Since("1.3.0")
  def setMinSupport(minSupport: Double): this.type = {
    require(minSupport >= 0.0 && minSupport <= 1.0,
      s"Minimal support level must be in range [0, 1] but got ${minSupport}")
    this.minSupport = minSupport
    this
  }

  /**
   * Sets the number of partitions used by parallel FP-growth (default: same as input data).
   *
   */
  @Since("1.3.0")
  def setNumPartitions(numPartitions: Int): this.type = {
    require(numPartitions > 0,
      s"Number of partitions must be positive but got ${numPartitions}")
    this.numPartitions = numPartitions
    this
  }

  /**
   * Computes an FP-Growth model that contains frequent itemsets.
   * @param data input data set, each element contains a transaction
   * @return an [[FPGrowthModel]]
   *
   */
  @Since("1.3.0")
  def run[Item: ClassTag](data: RDD[Array[Item]]): FPGrowthModel[Item] = {
    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("Input data is not cached.")
    }
    val count = data.count()
    val minCount = math.ceil(minSupport * count).toLong
    val numParts = if (numPartitions > 0) numPartitions else data.partitions.length
    val partitioner = new HashPartitioner(numParts)
    val freqItems = genFreqItems(data, minCount, partitioner)
    val freqItemsets = genFreqItemsets(data, minCount, freqItems, partitioner)
    new FPGrowthModel(freqItemsets)
  }

  /** Java-friendly version of [[run]]. */
  @Since("1.3.0")
  def run[Item, Basket <: JavaIterable[Item]](data: JavaRDD[Basket]): FPGrowthModel[Item] = {
    implicit val tag = fakeClassTag[Item]
    run(data.rdd.map(_.asScala.toArray))
  }

  /**
   * Generates frequent items by filtering the input data using minimal support level.
   * @param minCount minimum count for frequent itemsets
   * @param partitioner partitioner used to distribute items
   * @return array of frequent pattern ordered by their frequencies
   */
  private def genFreqItems[Item: ClassTag](
      data: RDD[Array[Item]],       //data는 transaction item들의 array.
      minCount: Long,               //minCount는 support threshold값.
      partitioner: Partitioner      //partitioner는 transaction item에 대해 reducing하는데 사용되는 partitioner.
      ): Array[Item] = {data.flatMap { t => val uniq = t.toSet
      /* Item들의 array를 생성, 이때 flatMap을 통해 각 set에 들어가는 transaction의 형태를 단순 원소로 변경.
      *  이 안에있는 item들을 각각 중복되지 않는 단일 원소로 uniq라는 set에 삽입.
      */
      if (t.length != uniq.size){
        throw new SparkException(s"Items in a transaction must be unique but got ${t.toSeq}.")
      }     //set uniq의 크기와 Item들의 array인 t의 길이가 서로 다르면 예외로 처리하여 메시지를 출력.
      t     //이렇게 해서 모든 원소가 unique한 itemset들을 생성.
    }.map(v => (v, 1L))                 //생성된 set에 대해 mapping을 진행, 이때 각 transaction item array들의 subset(itemset)과, 해당 set의 출현빈도수를 값으로 가짐.
      .reduceByKey(partitioner, _ + _)  //partitioner를 이용한 reducing과정, 각 itemset들의 출현 빈도수를 더함.
      .filter(_._2 >= minCount)         //reducing되어 나온 값들 중 minCount값을 기준으로하여 minCount값보다 작은 itemset은 filtering
      .collect()                        //각각의 reducing과정을 거친 itemset들을 다시 모아줌.
      .sortBy(-_._2)                    //minCount이상의 빈도수를 가진 itemset들끼리 출현빈도수를 기준으로 sorting.
      .map(_._1)                        //이들을 다시 mapping하여 array로 반환.
  }

  /**
   * Generate frequent itemsets by building FP-Trees, the extraction is done on each partition.
   * @param data transactions
   * @param minCount minimum count for frequent itemsets
   * @param freqItems frequent items
   * @param partitioner partitioner used to distribute transactions
   * @return an RDD of (frequent itemset, count)
   */
  private def genFreqItemsets[Item: ClassTag](
      data: RDD[Array[Item]],           //data는 transaction item들의 array.
      minCount: Long,                   //minCount는 support threshold값.
      freqItems: Array[Item],           //transaction으로부터 만들어진 itemset의 array. frequency에 따라 정렬됨.
      partitioner: Partitioner): RDD[FreqItemset[Item]] = {         //FreqItemset을 만들고, 여기에 freqItems를
    val itemToRank = freqItems.zipWithIndex.toMap                   //Index로 압축하여 Mapping한 itemToRank 변수를 생성.
    data.flatMap { transaction =>       //genCondTransactions에 의해 발생한 transaction을 flatMapping.
      genCondTransactions(transaction, itemToRank, partitioner)     //어떤 조건에 대한 transaction을 발생시키는 함수.
    }.aggregateByKey(new FPTree[Int], partitioner.numPartitions)(   //FPTree를 만들고, 여기에 발생하는 transaction을 삽입.
      (tree, transaction) => tree.add(transaction, 1L),             //tree에는 transaction자체를 삽입할 수도 있고,
      (tree1, tree2) => tree1.merge(tree2))                         //이미 구성된 다른 tree를 추가할 수도 있음.
    .flatMap { case (part, tree) =>
      tree.extract(minCount, x => partitioner.getPartition(x) == part)
    }   //Key를 기준으로 만든 FPTree에서 minCount이상을 만족하는 부분을 추출하여 flatMapping.
    .map { case (ranks, count) =>
      new FreqItemset(ranks.map(i => freqItems(i)).toArray, count)
    }   //flatMapping된 값을 빈도수별로 랭크를 매기고, 이를 mapping하여 FreqItemset array를 생성하고 반환.
  }

  /**
   * Generates conditional transactions.
   * @param transaction a transaction
   * @param itemToRank map from item to their rank
   * @param partitioner partitioner used to distribute transactions
   * @return a map of (target partition, conditional transaction)
   */
  private def genCondTransactions[Item: ClassTag](
      transaction: Array[Item],
      itemToRank: Map[Item, Int],
      partitioner: Partitioner): mutable.Map[Int, Array[Int]] = {
    val output = mutable.Map.empty[Int, Array[Int]]
    // Filter the basket by frequent items pattern and sort their ranks.
    val filtered = transaction.flatMap(itemToRank.get)
    ju.Arrays.sort(filtered)
    val n = filtered.length
    var i = n - 1
    while (i >= 0) {
      val item = filtered(i)
      val part = partitioner.getPartition(item)
      if (!output.contains(part)) {
        output(part) = filtered.slice(0, i + 1)
      }
      i -= 1
    }
    output
  }
}

@Since("1.3.0")
object FPGrowth {

  /**
   * Frequent itemset.
   * @param items items in this itemset. Java users should call [[FreqItemset#javaItems]] instead.
   * @param freq frequency
   * @tparam Item item type
   *
   */
  @Since("1.3.0")
  class FreqItemset[Item] @Since("1.3.0") (
      @Since("1.3.0") val items: Array[Item],
      @Since("1.3.0") val freq: Long) extends Serializable {

    /**
     * Returns items in a Java List.
     *
     */
    @Since("1.3.0")
    def javaItems: java.util.List[Item] = {
      items.toList.asJava
    }

    override def toString: String = {
      s"${items.mkString("{", ",", "}")}: $freq"
    }
  }
}
