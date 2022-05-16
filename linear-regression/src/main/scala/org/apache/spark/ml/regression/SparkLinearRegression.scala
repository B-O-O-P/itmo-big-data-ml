package org.apache.spark.ml.regression

import breeze.linalg.DenseVector
import breeze.linalg.euclideanDistance

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.ml.Estimator

import org.apache.spark.mllib.linalg.{Vectors => MllibVectors}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer

import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, Encoder}

class SparkLinearRegression(override val uid: String)
  extends Estimator[SparkLinearRegressionModel]
    with SparkLinearRegressionParams
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): SparkLinearRegressionModel = {
    implicit val vectorEncoder: Encoder[Vector] = ExpressionEncoder()

    val epochs = getMaxIter
    val lr = getLearningRate
    val tol = getTol

    val assembler = new VectorAssembler().setInputCols(Array(getFeaturesCol, getLabelCol)).setOutputCol("result")
    val vectors = assembler.transform(dataset).select("result").as[Vector]

    val weightsSize = vectors.first().size - 1
    var previousWeights = DenseVector.fill(weightsSize, Double.PositiveInfinity)
    val currentWeights = DenseVector.fill(weightsSize, 0.0)

    var iter = 0
    while (iter < epochs && euclideanDistance(currentWeights.toDenseVector, previousWeights.toDenseVector) > tol) {
      val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()

        data.foreach(row => {
          val x = row.asBreeze(0 until weightsSize).toDenseVector
          val y = row.asBreeze(-1)

          val delta = x * (x.dot(currentWeights) - y)
          summarizer.add(MllibVectors.fromBreeze(delta))
        })

        Iterator(summarizer)
      }).reduce(_ merge _)

      val mean = summary.mean.asBreeze

      previousWeights = currentWeights.copy
      currentWeights -= lr * mean

      iter += 1
    }

    copyValues(new SparkLinearRegressionModel(Vectors.fromBreeze(currentWeights)).setParent(this))
  }

  override def copy(extra: ParamMap): Estimator[SparkLinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}