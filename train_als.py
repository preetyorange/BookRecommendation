import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "book_ratings.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "als_book_reco")
USER_RECS_PATH = os.path.join(BASE_DIR, "data", "user_recommendations.csv")


def create_spark(app_name: str = "BookRecoALS") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def load_ratings(spark: SparkSession):
    ratings = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(DATA_PATH)
    )
    ratings = (
        ratings
        .dropna(subset=["userId", "bookId", "rating"])
        .withColumn("user_id", col("userId").cast("int"))
        .withColumn("book_id", col("bookId").cast("int"))
        .withColumn("rating", col("rating").cast("float"))
    )
    return ratings


def evaluate_regression(predictions):
    rmse_eval = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction",
    )
    mae_eval = RegressionEvaluator(
        metricName="mae",
        labelCol="rating",
        predictionCol="prediction",
    )
    rmse = rmse_eval.evaluate(predictions)
    mae = mae_eval.evaluate(predictions)
    return rmse, mae


def evaluate_ranking(model, test_df, k: int = 10, relevant_rating_threshold: float = 4.0):
   
    spark = test_df.sparkSession

    # Users that appear in the test split
    test_users = test_df.select("user_id").distinct()

    # Predicted top-k items per user (as arrays)
    user_recs = model.recommendForUserSubset(test_users, k)
    pred = user_recs.select(
        col("user_id"),
        col("recommendations.book_id").alias("pred_items"),
    )

    # Ground truth relevant items per user from test
    truth = (
        test_df.filter(col("rating") >= float(relevant_rating_threshold))
        .groupBy("user_id")
        .agg({"book_id": "collect_set"})
        .withColumnRenamed("collect_set(book_id)", "true_items")
    )

    # Join and build (pred, true) pairs
    paired = pred.join(truth, on="user_id", how="inner").select("pred_items", "true_items")

    # If no users have relevant items in test for the chosen threshold, skip
    if paired.rdd.isEmpty():
        return None

    ranking_rdd = paired.rdd.map(lambda row: (row["pred_items"], row["true_items"]))
    metrics = RankingMetrics(ranking_rdd)

    return {
        "precisionAtK": metrics.precisionAt(k),
        "meanAveragePrecision": metrics.meanAveragePrecision,
        "ndcgAtK": metrics.ndcgAt(k),
    }


def train_and_generate_recs():
    spark = create_spark()

    ratings = load_ratings(spark)

    train, test = ratings.randomSplit([0.8, 0.2], seed=42)

    als = ALS(
        maxIter=10,
        regParam=0.1,
        rank=20,
        userCol="user_id",
        itemCol="book_id",
        ratingCol="rating",
        nonnegative=True,
        coldStartStrategy="drop",
    )

    model = als.fit(train)

    predictions = model.transform(test)

    rmse, mae = evaluate_regression(predictions)
    print(f"Test RMSE = {rmse:.4f}")
    print(f"Test MAE  = {mae:.4f}")

    ranking_metrics = evaluate_ranking(model, test, k=10, relevant_rating_threshold=4.0)
    if ranking_metrics is None:
        print("Ranking metrics skipped (no relevant items in test for threshold=4.0).")
    else:
       
        k = 10
        test_users = test.select("user_id").distinct()
        user_recs = model.recommendForUserSubset(test_users, k).select(
            col("user_id"),
            col("recommendations.book_id").alias("pred_items"),
        )
        truth = (
            test.filter(col("rating") >= 4.0)
            .groupBy("user_id")
            .agg({"book_id": "collect_set"})
            .withColumnRenamed("collect_set(book_id)", "true_items")
        )
        joined = user_recs.join(truth, on="user_id", how="inner")
        recall_sum = joined.rdd.map(
            lambda r: (
                len(set(r["pred_items"]).intersection(set(r["true_items"]))) / float(len(r["true_items"]))
                if r["true_items"] else 0.0
            )
        ).mean()

        print(f"Precision@{k} = {ranking_metrics['precisionAtK']:.4f}")
        print(f"Recall@{k}    = {recall_sum:.4f}")
        print(f"MAP           = {ranking_metrics['meanAveragePrecision']:.4f}")
        print(f"NDCG@{k}      = {ranking_metrics['ndcgAtK']:.4f}")

    # Generate top-N recommendations for all users in the dataset
    users = ratings.select("user_id").distinct()
    user_recs = model.recommendForUserSubset(users, 10)

    from pyspark.sql.functions import explode

    exploded = user_recs.select(
        col("user_id"),
        explode("recommendations").alias("rec"),
    ).select(
        col("user_id"),
        col("rec.book_id").alias("book_id"),
        col("rec.rating").alias("score"),
    )

    # Join back with titles from original CSV
    books = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(DATA_PATH)
        .select(col("bookId").alias("book_id"), col("title"))
    )

    exploded = exploded.join(books, on="book_id", how="left")

    # Save as a simple CSV for the frontend/API to consume
    exploded.coalesce(1).write.mode("overwrite").option("header", True).csv(
        USER_RECS_PATH + "_tmp"
    )

    # Move the single part file to a fixed name for easy loading
    import glob
    import shutil

    part_files = glob.glob(os.path.join(USER_RECS_PATH + "_tmp", "part-*.csv"))
    if part_files:
        os.makedirs(os.path.dirname(USER_RECS_PATH), exist_ok=True)
        shutil.move(part_files[0], USER_RECS_PATH)
    shutil.rmtree(USER_RECS_PATH + "_tmp", ignore_errors=True)

    print(f"User recommendations saved to {USER_RECS_PATH}")

    spark.stop()


if __name__ == "__main__":
    train_and_generate_recs()
