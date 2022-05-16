package org.apache.spark.ml.regression

import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec._
import org.scalatest.matchers._


class StartSparkTest extends AnyFlatSpec with should.Matchers with WithSpark {

  "Spark" should "start context" in {
    SparkSession.builder
      .appName("Spark Linear Regression App")
      .master("local[4]")
      .getOrCreate()

    Thread.sleep(60000)
  }
}