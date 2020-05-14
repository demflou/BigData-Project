from __future__ import print_function

# $example on$
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import split
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
# $example off$
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer


def lower_clean_str(x):
  punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
  lowercased_str = x.lower()
  for ch in punc:
    lowercased_str = lowercased_str.replace(ch, '')
  return lowercased_str


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("TokenizerExample")\
        .getOrCreate()

    # $example on$
    '''sentenceDataFrame = spark.createDataFrame([
        (0, "Hi I heard about Spark"),
        (1, "I wish Java could use case classes"),
        (2, "Logistic,regression,models,are,neat")
    ], ["id", "sentence"])'''

    lcs = udf(lower_clean_str)
    sentenceDataFrame = spark.read.csv("../train.csv", header=False, sep="\t")
    sentenceDataFrame=sentenceDataFrame.withColumnRenamed('_c0','rating')
    sentenceDataFrame=sentenceDataFrame.withColumnRenamed('_c1','sentence')
    sentenceDataFrame= sentenceDataFrame.withColumn("sentence", lcs("sentence"))
    # df=df.withColumn("_c1",rews("_c1")) #3/3
    '''expres = [split(col("sentence"), " ").alias("sentence")]
    sentenceDataFrame = sentenceDataFrame.withColumn("sentence", *expres)
    remover = StopWordsRemover(inputCol="sentence", outputCol="filtered")
    swlist = remover.getStopWords()
    swlist.append("")
    remover.setStopWords(swlist)
    final = remover.transform(sentenceDataFrame.select("sentence"))'''

    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

    countTokens = udf(lambda words: len(words), IntegerType())

    tokenized = tokenizer.transform(sentenceDataFrame)
    print(tokenized.columns)
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    swlist = remover.getStopWords()
    swlist.append("")
    remover.setStopWords(swlist)
    tokenized = remover.transform(tokenized.select("words"))

    tokenized.select("sentence", "words")\
        .withColumn("tokens", countTokens(col("words"))).show()

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
    tf = hashingTF.transform(tokenized)
    tf.select('rawFeatures').take(2)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(tf)
    tfidf = idfModel.transform(tf)
    print (tfidf.select("features").first())
    spark.stop()