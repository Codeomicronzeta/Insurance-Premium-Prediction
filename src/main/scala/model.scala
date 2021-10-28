import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import spark.implicits._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel, LinearRegressionSummary, GBTRegressionModel, GBTRegressor, RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.mllib.util.MLUtils

// Importing the data
val data = spark.read.option("header", true).option("inferSchema", true).csv("insurance.csv")
data.show(5)

println("Schema: ")
data.printSchema()

// Using StringIndexing to assign numerical labels to string categorical features
println("String Indexing:")
val stringIndexer = new StringIndexer().setInputCols(Array("sex", "smoker", "region")).setOutputCols(Array("sex_ct", "smoker_ct", "region_ct"))
val stringIndexed = stringIndexer.fit(data).transform(data)
stringIndexed.show(5)

// Using One Hot Encoder for One Hot Encoding the categorical features
println("One Hot Encoding:")
val oheIndexer = new OneHotEncoder().setInputCols(Array("sex_ct", "smoker_ct", "region_ct")).setOutputCols(Array("sex_ohe", "smoker_ohe", "region_ohe"))
val oheIndexed = oheIndexer.fit(stringIndexed).transform(stringIndexed)
oheIndexed.show(5)

oheIndexed.printSchema()

// Using Vector Assembler to create the feature columns
println("Assembling the features")
val assembler = new VectorAssembler().setInputCols(Array("age", "bmi", "children", "sex_ohe", "smoker_ohe", "region_ohe")).setOutputCol("features_0")
val dataVec = assembler.transform(oheIndexed)
dataVec.select("features_0").show(5)

// Using StandardScaler for Normalizing the features
println("Normalizing the data")

val scaler = new StandardScaler()
  .setInputCol("features_0")
  .setOutputCol("features")
  .setWithStd(true)
  .setWithMean(true)

// Compute summary statistics by fitting the StandardScaler
val scalerModel = scaler.fit(dataVec)

val scaledData = scalerModel.transform(dataVec)

// Creating Training and Testing Set
println("Creating training and testing data")
val train, test = scaledData.randomSplit(weights=Array(0.8,0.2), seed=200)

println("We get array of dataframes, so we take the required dataframe element")
val train_df = train(0)
val test_df = test(0)

//Initializing the linear regression model, evaluators and cross validation functions
val lr = new LinearRegression().setLabelCol("expenses")

val lrparamGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.01, 0.1, 0.5))
.addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
.addGrid(lr.maxIter, Array(1, 5, 10))
.build()

val lrevaluator = new RegressionEvaluator().setPredictionCol("prediction").setLabelCol("expenses").setMetricName("rmse")

val cv = new CrossValidator()
  .setEstimator(lr)
  .setEvaluator(lrevaluator)
  .setEstimatorParamMaps(lrparamGrid)
  .setNumFolds(5) 

// Fitting the model on train set
println("Fitting the model on train set")
val lrcvModel = cv.fit(train_df)

// Running the model on Test data
println("Running the model on test data")
val pred = lrcvModel.transform(test_df)

// Evaluating the model performance using RMSE
println("Evaluating the model perfromance on the test set using Root Mean Squared Error (RMSE)")
val eval = lrevaluator.evaluate(pred)
println(s"RMSE: $eval")

// Initializing the Gradient Boosted Tree model, evaluators and cross validation functions
val gbt = new GBTRegressor().setLabelCol("expenses")

val gbtparamGrid = new ParamGridBuilder().addGrid(gbt.maxIter, Array(10, 100))
.addGrid(gbt.maxDepth, Array(2, 5))
.build()

val gbtevaluator = new RegressionEvaluator().setPredictionCol("prediction").setLabelCol("expenses").setMetricName("rmse")

val gbtcv = new CrossValidator()
  .setEstimator(gbt)
  .setEvaluator(gbtevaluator)
  .setEstimatorParamMaps(gbtparamGrid)
  .setNumFolds(5) 

// Fitting the GBT model on train set
val gbtcvModel = gbtcv.fit(train_df)

// Running the GBT model on test data
val gbt_pred = gbtcvModel.transform(test_df)

//Evaluating the GBT model perfromance on the test set using Root Mean Squared Error (RMSE)
val gbt_eval = gbtevaluator.evaluate(gbt_pred)
println(s"RMSE: $gbt_eval")

// Initializing the Random Forests model, evaluators and cross validation functions
val rf = new RandomForestRegressor().setLabelCol("expenses")

val rfparamGrid = new ParamGridBuilder().addGrid(rf.numTrees, Array(10, 100))
.addGrid(rf.maxDepth, Array(2, 5))
.build()

val rfevaluator = new RegressionEvaluator().setPredictionCol("prediction").setLabelCol("expenses").setMetricName("rmse")

val rfcv = new CrossValidator()
  .setEstimator(rf)
  .setEstimatorParamMaps(rfparamGrid)
  .setEvaluator(rfevaluator)
  .setNumFolds(5)

//Fitting the Random Forest model on train set
val rfcvModel = rfcv.fit(train_df)

//Running the Random Forest model on test data
val rf_pred = rfcvModel.transform(test_df)

//Evaluating the Random Forest model perfromance on the test set using Root Mean Squared Error (RMSE)
val rf_eval = rfevaluator.evaluate(rf_pred)
println(s"RMSE: $rf_eval")

// Creating dataframe for tabulating the results
val columns = Seq("Model", "Results (RMSE)")
val data = Seq(("Gradient Boosted Trees", gbt_eval), ("Random Forest", rf_eval), ("Linear Regression", eval))
val results = spark.createDataFrame(data).toDF(columns:_*)

results.show()
