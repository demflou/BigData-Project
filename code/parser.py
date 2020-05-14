from nltk.corpus import stopwords
from pyspark.sql.functions import udf
from pyspark.sql.functions import col, regexp_replace, split
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover
from rfclassifier import lr_train
import pyspark.sql.functions as func
import nltk

def lower_clean_str(x):
  punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
  lowercased_str = x.lower()
  for ch in punc:
    lowercased_str = lowercased_str.replace(ch, '')
  return lowercased_str

'''def remove_extra_ws(x):  #alternative method to cut extra whitespaces 1/3
  return " ".join(x.split())'''

lcs=udf(lower_clean_str)
#rews=udf(remove_extra_ws) #2/3
spark = SparkSession.builder.appName("Parsing and removing stopwords").getOrCreate()
df = spark.read.csv("../train.csv",header=False,sep="\t");
df=df.withColumn("_c1",lcs("_c1"))
#df=df.withColumn("_c1",rews("_c1")) #3/3
expres = [split(col("_c1")," ").alias("_c1")]
df=df.withColumn("_c1",*expres)
remover = StopWordsRemover(inputCol="_c1", outputCol="filtered")
swlist = remover.getStopWords()
swlist= swlist + list(set(stopwords.words('english')))+ ['']
remover.setStopWords(swlist)
final = remover.transform(df.select("_c1"))
df= df.withColumn('row_index', func.monotonically_increasing_id())
final = final.withColumn('row_index', func.monotonically_increasing_id())
#print final.columns
#print final.first()
final = final.join(df["row_index","_c0"], on=["row_index"]).drop("row_index").drop("_c1")
print lr_train(final)