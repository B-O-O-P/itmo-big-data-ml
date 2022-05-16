package org.apache.spark.ml.regression

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.param.{DoubleParam, Param}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.ml.util.SchemaUtils

import org.apache.spark.sql.types.StructType

trait SparkLinearRegressionParams extends PredictorParams with HasMaxIter with HasTol {

  final val learningRate: Param[Double] = new DoubleParam(
    this,
    "learningRate",
    "learning rate"
  )

  def getLearningRate: Double = $(learningRate)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setLearningRate(value: Double): this.type = set(learningRate, value)

  def setTol(value: Double): this.type = set(tol, value)

  setDefault(learningRate -> 0.05, maxIter -> 1e5.toInt, tol -> 1e-7)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getPredictionCol))
    }
  }
}