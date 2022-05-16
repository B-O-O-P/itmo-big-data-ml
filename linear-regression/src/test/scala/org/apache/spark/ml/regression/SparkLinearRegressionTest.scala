package org.apache.spark.ml.regression

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.scalatest.flatspec._
import org.scalatest.matchers._

import scala.util.Random

class SparkLinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  import sqlContext.implicits._

  val delta: Double = 1e-3

  lazy val data: DataFrame = StandardScalerTest._data
  lazy val vectors: Seq[Vector] = StandardScalerTest._vectors

  val predictUDF: UserDefinedFunction = StandardScalerTest._predictUDF

  val labelColName = "label"
  val featuresColName = "features"

  lazy val expectedWeights: Vector = Vectors.dense(1.5, 0.3, -0.7)

  "Model" should "predict input data" in {
    val model: SparkLinearRegressionModel = new SparkLinearRegressionModel(expectedWeights)

    validateModel(model.transform(data))
  }

  "Estimator" should "produce functional model" in {
    val estimator = new SparkLinearRegression()

    val dataset = data.withColumn(labelColName, predictUDF(col(featuresColName)))
    val model = estimator.fit(dataset)

    validateModel(model.transform(dataset))
  }

  "Estimator" should "predict correctly" in {
    val estimator = new SparkLinearRegression().setMaxIter(10000)

    val randomData = Matrices
      .rand(500000, 3, Random.self)
      .rowIter
      .toSeq
      .map(x => Tuple1(x))
      .toDF(featuresColName)

    val dataset = randomData.withColumn(labelColName, predictUDF(col(featuresColName)))
    val model = estimator.fit(dataset)

    model.weights(0) should be(expectedWeights(0) +- delta)
    model.weights(1) should be(expectedWeights(1) +- delta)
    model.weights(2) should be(expectedWeights(2) +- delta)
  }

  private def validateModel(data: DataFrame): Unit = {
    val expectedVector = Vectors.dense(21.05, -1.6)

    val vector = data.collect().map(_.getAs[Double](1))

    vector.length should be(2)

    vector(0) should be(expectedVector(0) +- delta)
    vector(1) should be(expectedVector(1) +- delta)
  }

}

object StandardScalerTest extends WithSpark {
  lazy val expectedWeights: Vector = Vectors.dense(1.5, 0.3, -0.7)

  val featuresColName = "features"

  val expectedVector1: Vector = Vectors.dense(13.5, 12, 4)
  val expectedVector2: Vector = Vectors.dense(-1, 2, 1)

  lazy val _vectors: Seq[Vector] = Seq(expectedVector1, expectedVector2)

  lazy val _data: DataFrame = {
    import sqlContext.implicits._
    _vectors.map(x => Tuple1(x)).toDF(featuresColName)
  }

  val _predictUDF: UserDefinedFunction = udf { features: Any =>
    val arr = features.asInstanceOf[Vector].toArray

    expectedWeights(0) * arr.apply(0) + expectedWeights(1) * arr.apply(1) - expectedWeights(2) * arr.apply(2)
  }

}