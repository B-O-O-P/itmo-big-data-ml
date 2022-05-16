package org.apache.spark.ml.regression

import org.apache.spark.sql.{SQLContext, SparkSession}

trait WithSpark {
  lazy val sparkSession: SparkSession = WithSpark.sparkSession
  lazy val sqlContext: SQLContext = WithSpark.sqlContext
}

object WithSpark {
  private lazy val sparkSession = SparkSession
    .builder
    .appName("Spark Linear Regression Testing")
    .master("local[4]")
    .getOrCreate()

  private lazy val sqlContext = sparkSession.sqlContext
}