package org.apache.spark.ml.regression

import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}

import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType

class SparkLinearRegressionModel(
                                   override val uid: String,
                                   val weights: Vector,
                                 ) extends Model[SparkLinearRegressionModel] with SparkLinearRegressionParams {

  private[regression] def this(weights: Vector) = this(
    Identifiable.randomUID("linearRegressionModel"),
    weights
  )

  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputSchema = transformSchema(dataset.schema, logging = true)
    val predictUDF = udf { features: Any =>
      predict(features.asInstanceOf[Vector])
    }

    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))), outputSchema($(predictionCol)).metadata)
  }

  override def transformSchema(schema: StructType): StructType = {
    var outputSchema = validateAndTransformSchema(schema)
    if ($(predictionCol).nonEmpty) {
      outputSchema = SchemaUtils.updateNumeric(outputSchema, $(predictionCol))
    }

    outputSchema
  }


  override def copy(extra: ParamMap): SparkLinearRegressionModel = copyValues(
    new SparkLinearRegressionModel(weights), extra
  )

  private def predict(features: Vector) = features.asBreeze.dot(weights.asBreeze)

}
