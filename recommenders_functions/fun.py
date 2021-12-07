# set the environment path to find Recommenders
import sys
import os
import numpy as np
import surprise
import papermill as pm
import scrapbook as sb
import pandas as pd


import pyspark
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, FloatType, IntegerType, LongType, StructType, StructField
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover
from pyspark.ml.feature import HashingTF, CountVectorizer, VectorAssembler
from pyspark.sql.window import Window
import pyspark.sql.functions as F

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.utils.notebook_utils import is_jupyter
from recommenders.datasets.python_splitters import python_random_split
from recommenders.datasets.spark_splitters import spark_random_split
from recommenders.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k, 
                                                     recall_at_k, get_top_k_items)
from recommenders.models.surprise.surprise_utils import predict, compute_ranking_predictions
from recommenders.evaluation.spark_evaluation import SparkRatingEvaluation, SparkRankingEvaluation, SparkDiversityEvaluation
from recommenders.utils.spark_utils import start_or_get_spark


#constants
COL_USER= "UserId"
COL_ITEM= "MovieId"
COL_RATING= "Rating"
COL_TITLE = "title"
COL_GENRES ="genres"


#setup spark1
spark = start_or_get_spark("ALS PySpark", memory="16g")
spark.conf.set("spark.sql.crossJoin.enabled", "true")
spark

#setup spark
def spark_setup(size = "16g"):
    spark = start_or_get_spark("ALS PySpark", memory=size)
    spark.conf.set("spark.sql.crossJoin.enabled", "true")
    spark


#schemas
def get_movielens_schema(col_user = COL_USER, col_item = COL_ITEM, col_rating = COL_RATING, col_title = COL_TITLE, col_genres = COL_GENRES):
    schema = StructType(
        (
            StructField(col_user, IntegerType()),
            StructField(col_item, IntegerType()),
            StructField(col_rating, FloatType()),
            StructField("Timestamp", LongType()),
            StructField(col_title, StringType()),
            StructField(col_genres, StringType())
        )
    )
    return schema

def get_predictions_schema(col_user = COL_USER, col_item = COL_ITEM, col_pred = 'prediction'):
    schema = StructType(
        (
            StructField(col_user, IntegerType()),
            StructField(col_item, IntegerType()),
            StructField(col_pred, FloatType()),
        )
    )
    return schema


#convert Movielens pandas to spark and spark to pandas
def movielens_to_spark(df, schema):
    spark_df = spark.createDataFrame(df, schema=schema)
    print("Spark df created, info: ")
    print()
    spark_df.printSchema()
    spark_df.show(5)
    return spark_df

def movielens_split_data(spark_movielens):

    train_df_spark, test_df_spark = spark_random_split(spark_movielens.select(COL_USER, COL_ITEM, COL_RATING), ratio=0.75, seed=123)
    users = train_df_spark.select(COL_USER).distinct()
    items = train_df_spark.select(COL_ITEM).distinct()
    user_item = users.crossJoin(items)
    
    return user_item, train_df_spark, test_df_spark
    
    

def movielens_to_pandas(df):
    return df.toPandas()


#preprocess Movielens for content based
def create_feature_data(movielens_spark):
    
    movies = (
        movielens_spark.groupBy("MovieId", "title", "genres").count()
        .na.drop()  # remove rows with null values
        .withColumn("genres", F.split(F.col("genres"), "\|"))  # convert to array of genres
        .withColumn("title", F.regexp_replace(F.col("title"), "[\(),:^0-9]", ""))  # remove year from title
        .drop("count")  # remove unused columns
    )
    
    # tokenize "title" column
    title_tokenizer = Tokenizer(inputCol="title", outputCol="title_words")
    tokenized_data = title_tokenizer.transform(movies)


    # remove stop words
    remover = StopWordsRemover(inputCol="title_words", outputCol="text")
    clean_data = remover.transform(tokenized_data).drop("title", "title_words")
    
    # step 1: perform HashingTF on column "text"
    text_hasher = HashingTF(inputCol="text", outputCol="text_features", numFeatures=1024)
    hashed_data = text_hasher.transform(clean_data)


    # step 2: fit a CountVectorizerModel from column "genres".
    count_vectorizer = CountVectorizer(inputCol="genres", outputCol="genres_features")
    count_vectorizer_model = count_vectorizer.fit(hashed_data)
    vectorized_data = count_vectorizer_model.transform(hashed_data)


    # step 3: assemble features into a single vector
    assembler = VectorAssembler(
        inputCols=["text_features", "genres_features"],
        outputCol="features",
    )

    feature_data = assembler.transform(vectorized_data).select("MovieId", "features")
    
    return feature_data



#get top k dataframe
def create_topk_reco():
    pass


#get metrics
def get_metrics():
    pass