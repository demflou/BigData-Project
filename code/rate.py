import pyspark.sql.functions as func
from nltk.corpus import stopwords
from pyspark.sql.functions import udf
from pyspark.sql.functions import col, regexp_replace, split
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover, IndexToString
from rfclassifier import *
import nltk
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

def lower_clean_str(x):
  punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
  lowercased_str = x.lower()
  for ch in punc:
    lowercased_str = lowercased_str.replace(ch, '')
  return lowercased_str


'''def remove_extra_ws(x):  #alternative method to cut extra whitespaces 1/3
  return " ".join(x.split())'''

def rate_transform(x):
    if ((x=="1") or (x=="2") or (x=="3")):
        x="0"
    elif ((x=="4") or (x=="5")):
        x="1"
    return x

def parse_data(path):
    spark = SparkSession.builder.appName("BigDataProject").getOrCreate()
    if (path == "../train.csv"):
        lcs = udf(lower_clean_str)
        rt = udf(rate_transform)
        # rews=udf(remove_extra_ws) #2/3
        df = spark.read.csv(path, header=False, sep="\t");
        df = df.withColumn("_c1", lcs("_c1"))
        # df=df.withColumn("_c1",rews("_c1")) #3/3
        expres = [split(col("_c1"), " ").alias("_c1")]
        df = df.withColumn("_c1",*expres)
        remover = StopWordsRemover(inputCol="_c1", outputCol="filtered")
        swlist = remover.getStopWords()
        swlist= swlist + list(set(stopwords.words('english')))+ ['']
        remover.setStopWords(swlist)
        #remover.transform(df).select("filtered")

        final = remover.transform(df.select("_c1"))
        df = df.withColumn('row_index', func.monotonically_increasing_id())
        final = final.withColumn('row_index', func.monotonically_increasing_id())
        final = final.join(df["row_index", "_c0"], on=["row_index"]).drop("row_index").drop("_c1")
        final = final.withColumn("_c0", rt("_c0"))
        # fdf.show()
        return final
    elif (path == "../test.csv"):
        lcs = udf(lower_clean_str)
        # rews=udf(remove_extra_ws) #2/3
        df = spark.read.csv(path, header=False, sep="\t");
        initial=df
        df = df.withColumn("_c0", lcs("_c0"))
        # df=df.withColumn("_c1",rews("_c1")) #3/3
        expres = [split(col("_c0"), " ").alias("_c0")]
        df = df.select(*expres)
        remover = StopWordsRemover(inputCol="_c0", outputCol="filtered")
        swlist = remover.getStopWords()
        swlist= swlist + list(set(stopwords.words('english')))+ ['']
        remover.setStopWords(swlist)
        remover.transform(df).select("filtered")
        final = remover.transform(df.select("_c0"))
        return final,initial

    else:
        print "Wrong File or Path"
        return -1

def find_best(data):
    classifiers = []
    classifiers.append(lr_train_cv(data))
    print "lr trained"
    classifiers.append(rf_train_cv(data))
    print "rf trained"
    classifiers.append(nb_train_cv(data))
    print "nb trained"
    classifiers.sort(key=lambda tup: tup[0])
    print classifiers
    #print str(classifiers[-1][2].stages[-1])+ " is the best with accuracy: " +str(classifiers[-1][0])
    return classifiers[-1]

def main():
    spark = SparkSession.builder.appName("BigDataProject").getOrCreate()
    df_train = parse_data("../train.csv")
    df_test,df = parse_data("../test.csv")
    #df_train.cache()
    df_test.cache()
    best=find_best(df_train)
    predictions=best[-1].transform(df_test)
    #predictions.show()
    converter = IndexToString(inputCol="prediction", outputCol="originalCategory",labels=best[1])
    converted=converter.transform(predictions)
    df = df.withColumn('row_index', func.monotonically_increasing_id())
    converted = converted.withColumn('row_index', func.monotonically_increasing_id())
    df = df.join( converted["row_index", "originalCategory"], on=["row_index"]).drop("row_index")
    df.show()
    df.repartition(1).write.csv('../predictions',sep="\t")
    spark.stop

if __name__=="__main__":
    main()