import pyspark as pyspark
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression,NaiveBayes
from pyspark.ml.feature import HashingTF,IDF, StringIndexer, CountVectorizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, CountVectorizer, VectorIndexer, HashingTF, IDF
from pyspark.ml.classification import RandomForestClassifier
#from pyspark.mllib.feature import HashingTF
#from pyspark.mllib.feature import IDF
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder,TrainValidationSplit

def lr_train_cv(data):
    #Logistic Regression using Count Vector Features
    label_stringIdx = StringIndexer(inputCol="_c0", outputCol="label")
    lsmodel=label_stringIdx.fit(data)
    data=lsmodel.transform(data)
    data.cache()
    #(trainingData, testData) = data.randomSplit([0.9, 0.1], seed=100)
    countVectors = CountVectorizer(inputCol="filtered", outputCol="cfeatures", vocabSize=10000, minDF=5)
    '''hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
    idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features",minDocFreq=5)'''
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
    lr = LogisticRegression(regParam=0.3, elasticNetParam=0,featuresCol=countVectors.getOutputCol(), labelCol="label")
    pipeline = Pipeline(stages=[countVectors,lr])
    grid = ParamGridBuilder().addGrid(lr.maxIter, [20]).build()
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=grid,
                              evaluator=evaluator,
                              numFolds=10)
    cvmodel=crossval.fit(data)
    return (evaluator.evaluate(cvmodel.transform(data)),lsmodel.labels,cvmodel)

def lr_train_tvs(data):
    #Logistic Regression using Count Vector Features
    label_stringIdx = StringIndexer(inputCol="_c0", outputCol="label")
    lsmodel=label_stringIdx.fit(data)
    data=lsmodel.transform(data)
    #(trainingData, testData) = data.randomSplit([0.9, 0.1], seed=100)
    countVectors = CountVectorizer(inputCol="filtered", outputCol="cfeatures", vocabSize=10000, minDF=5)
    '''hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
    idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features",minDocFreq=5)'''
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
    lr = LogisticRegression(regParam=0.3, elasticNetParam=0,featuresCol=countVectors.getOutputCol(), labelCol="label")
    pipeline = Pipeline(stages=[countVectors,lr])
    grid = ParamGridBuilder().addGrid(lr.maxIter, [10,15,20]).build()
    crossval = TrainValidationSplit(estimator=pipeline,
                              estimatorParamMaps=grid,
                              evaluator=evaluator,
                              trainRatio=0.9)
    cvmodel=crossval.fit(data)
    return (evaluator.evaluate(cvmodel.transform(data)),lsmodel.labels,cvmodel)

def lr_train(data):
    #Logistic Regression using Count Vector Features
    label_stringIdx = StringIndexer(inputCol="_c0", outputCol="label")
    lsmodel=label_stringIdx.fit(data)
    data=lsmodel.transform(data)
    (trainingData, testData) = data.randomSplit([0.9, 0.1], seed=100)
    countVectors = CountVectorizer(inputCol="filtered", outputCol="cfeatures", vocabSize=10000, minDF=5)
    '''hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
    idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features",minDocFreq=5)'''

    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0,featuresCol=countVectors.getOutputCol(), labelCol="label")
    pipeline = Pipeline(stages=[countVectors,lr])
    pipelineFit = pipeline.fit(trainingData)
    predictions = pipelineFit.transform(testData)
    #predictions.show(5)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
    #evaluator.evaluate(predictions)
    return (evaluator.evaluate(predictions),lsmodel.labels,pipelineFit)

def lr2_train(data):
    print data.columns
    countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
    label_stringIdx = StringIndexer(inputCol="_c0", outputCol="label")
    pipeline = Pipeline(stages=[countVectors,label_stringIdx])
    # Fit the pipeline to training documents.
    pipelineFit = pipeline.fit(data)
    dataset = pipelineFit.transform(data)
    dataset.show(5)
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)

    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
    lrModel = lr.fit(trainingData)
    predictions = lrModel.transform(testData)

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    evaluator.evaluate(predictions)
    return evaluator.evaluate(predictions), lrModel


def rf_train(data):
    #Random Forest Classifier 0.60
    countVectors = CountVectorizer(inputCol = "filtered", outputCol = "rfFeatures", vocabSize = 200, minDF =7)
    label_stringIdx = StringIndexer(inputCol = "_c0", outputCol = "label")
    lsmodel=label_stringIdx.fit(data)
    data=lsmodel.transform(data)
    (trainingData, testData) = data.randomSplit([0.9, 0.1], seed=100)
    rf = RandomForestClassifier(labelCol = "label", featuresCol = "rfFeatures", numTrees = 10)
    pipeline = Pipeline(stages = [ countVectors, rf])
    pipelineFit = pipeline.fit(trainingData)
    predictions = pipelineFit.transform(testData)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    return (evaluator.evaluate(predictions), lsmodel.labels, pipelineFit)

def rf_train_cv(data):
    #Random Forest Classifier 0.60
    countVectors = CountVectorizer(inputCol = "filtered", outputCol = "rfFeatures", vocabSize = 200, minDF =7)
    label_stringIdx = StringIndexer(inputCol = "_c0", outputCol = "label")
    lsmodel=label_stringIdx.fit(data)
    data=lsmodel.transform(data)
    data.cache()
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
    rf = RandomForestClassifier(labelCol = "label", featuresCol = "rfFeatures")
    pipeline = Pipeline(stages = [ countVectors, rf])
    grid = ParamGridBuilder().addGrid(rf.numTrees, [10]).build()
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=grid,
                              evaluator=evaluator,
                              numFolds=10)
    cvmodel = crossval.fit(data)
    return (evaluator.evaluate(cvmodel.transform(data)), lsmodel.labels, cvmodel)

def rf_train1(data):
    # Random Forest Classifier 0.53
    (trainingData, testData) = data.randomSplit([0.9, 0.1], seed=100)
    hashingTF = HashingTF(inputCol="filtered", outputCol= "hFeatures", numFeatures=10)
    #countVectors = CountVectorizer(inputCol="filtered", outputCol="hFeatures", vocabSize=200, minDF=7)
    idf = IDF(inputCol="hFeatures", outputCol="rfFeatures", minDocFreq = 7)
    label_stringIdx = StringIndexer(inputCol="_c0", outputCol="label")
    rf = RandomForestClassifier(labelCol="label", featuresCol="rfFeatures", numTrees=10)
    pipeline = Pipeline(stages=[label_stringIdx, hashingTF, idf, rf])
    pipelineFit = pipeline.fit(trainingData)
    predictions = pipelineFit.transform(testData)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

    return evaluator.evaluate(predictions), rf


def nb_train(data):
    #Naive Bayes Classifier
    label_stringIdx = StringIndexer(inputCol="_c0", outputCol="label")
    lsmodel=label_stringIdx.fit(data)
    data=lsmodel.transform(data)
    (trainingData, testData) = data.randomSplit([0.9, 0.1], seed=100)
    countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
    idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features", minDocFreq=5)
    nb = NaiveBayes(smoothing=1)
    pipeline = Pipeline(stages=[countVectors,nb])
    pipelineFit = pipeline.fit(trainingData)
    predictions = pipelineFit.transform(testData)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")

    return (evaluator.evaluate(predictions),lsmodel.labels,pipelineFit)

def nb_train_cv(data):
    #Naive Bayes Classifier
    label_stringIdx = StringIndexer(inputCol="_c0", outputCol="label")
    lsmodel=label_stringIdx.fit(data)
    data=lsmodel.transform(data)
    data.cache()
    #(trainingData, testData) = data.randomSplit([0.9, 0.1], seed=100)
    countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
    idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features", minDocFreq=5)
    nb = NaiveBayes()
    pipeline = Pipeline(stages=[countVectors,nb])
    grid = ParamGridBuilder().addGrid(nb.smoothing, [1]).build()
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=grid,
                              evaluator=evaluator,
                              numFolds=10)
    cvmodel = crossval.fit(data)
    return (evaluator.evaluate(cvmodel.transform(data)), lsmodel.labels, cvmodel)

'''hashingTF = HashingTF()
tf = hashingTF.transform(data.rdd)
# While applying HashingTF only needs a single pass to the data, applying IDF needs two passes:
# First to compute the IDF vector and second to scale the term frequencies by IDF.
tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)
print tfidf.columns
idfIgnore = IDF(minDocFreq=2).fit(tf)
tfidfIgnore = idfIgnore.transform(tf)
data=data.withColumn("tfidf",tfidf)

print predictions.filter(predictions['prediction'] == 0)\
    .select("filtered", "_c0", "probability", "label", "prediction") \
    .orderBy("probability", ascending=False) \
    .show(n=10, truncate=30)

rf = RandomForestClassifier(labelCol="_c0", \
                            featuresCol="filtered", \
                            numTrees=100, \
                            maxDepth=4, \
                            maxBins=32)
# Train model with Training Data
rfModel = rf.fit(trainingData)
predictions = rfModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("_c0", "filtered", "prediction") \
    .show(n=10, truncate=30)

hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(data)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
rescaledData.select("_c0", "features").show()

labelIndexer = StringIndexer(inputCol="_c0", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = \
    VectorIndexer(inputCol="filtered", outputCol="indexedFeatures", maxCategories=4).fit(data)
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only'''