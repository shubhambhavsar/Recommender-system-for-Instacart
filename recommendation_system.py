from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, min, max
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.evaluation import RankingMetrics
import time

# ------------------------------Initial Setup-------------------------------
spark = SparkSession.builder \
    .appName("CollaborativeFiltering-RecommendAisles") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

products = spark.read.csv('gs://toolsforai/products.csv', header=True, inferSchema=True)
orders = spark.read.csv('gs://toolsforai/orders.csv', header=True, inferSchema=True)
order_products = spark.read.csv('gs://toolsforai/order_products.csv', header=True, inferSchema=True)
aisles = spark.read.csv('gs://toolsforai/aisles.csv', header=True, inferSchema=True)


# -------------------------------Data Preprocessing-------------------------
# Limiting Data for Banana Buyers Only
banana_product_id = products.filter(col("product_name") == "Banana").select("product_id").first()["product_id"]
banana_buyers = order_products.join(orders, "order_id") \
    .filter(col("product_id") == banana_product_id) \
    .select("user_id").distinct()

filtered_orders = orders.join(banana_buyers, "user_id")
filtered_order_products = order_products.join(filtered_orders, "order_id")

# Aggregate User-Aisle Interactions
filtered_order_products = filtered_order_products.join(products, "product_id").join(aisles, "aisle_id")
user_aisle_interactions = filtered_order_products.groupBy("user_id", "aisle_id") \
    .agg(count("*").alias("interaction_strength"))

# Normalize Interaction Strengths
interaction_stats = user_aisle_interactions.agg(
    min("interaction_strength").alias("min_interaction"),
    max("interaction_strength").alias("max_interaction")
).collect()

min_interaction = interaction_stats[0]["min_interaction"]
max_interaction = interaction_stats[0]["max_interaction"]

normalized_user_aisle_interactions = user_aisle_interactions.withColumn(
    "normalized_strength",
    (col("interaction_strength") - min_interaction) / (max_interaction - min_interaction)
)

# Preview Normalized Interactions
print("Normalized User-Aisle Interaction Data:")
normalized_user_aisle_interactions.show(5)

# Prepare data in RDD format for ALS modeling
als_data_raw = user_aisle_interactions.rdd.map(
    lambda row: Rating(row["user_id"], row["aisle_id"], row["interaction_strength"])
)

# Data Splitting
train_data_raw, test_data_raw = als_data_raw.randomSplit([0.8, 0.2], seed=42)

# ---------------------------------Model Training---------------------------

start_time = time.time()

als_model = ALS.train(
    train_data_raw, 
    rank=20, 
    iterations=20, 
    lambda_=0.01
)

end_time = time.time()

# Calculate runtime
training_runtime = end_time - start_time
print(f"Training runtime: {training_runtime:.2f} seconds")

# ----------Model Evaluation Strong and Weak Generalization-------------------------

unique_users = user_aisle_interactions.select("user_id").distinct()
_, test_users = unique_users.randomSplit([0.8, 0.2], seed=42)
test_users_df = test_users
test_data_strong = user_aisle_interactions.join(test_users_df, "user_id")

recommendations = als_model.recommendProductsForUsers(5)

# Weak Generalization
test_data_weak = test_data_raw
predicted_ranking_weak = recommendations.mapValues(lambda recs: [rec.product for rec in recs])
actual_ranking_weak = test_data_weak.map(lambda x: (x.user, int(x.product))).groupByKey().mapValues(list)
formatted_ranking_weak = predicted_ranking_weak.join(actual_ranking_weak).map(lambda x: (x[1][0], x[1][1]))

metrics_weak = RankingMetrics(formatted_ranking_weak)
map_at_5_weak = metrics_weak.meanAveragePrecisionAt(5)
recall_at_5_weak = metrics_weak.recallAt(5)

# Strong Generalization
predicted_ranking_strong = recommendations.mapValues(lambda recs: [rec.product for rec in recs])
actual_ranking_strong = test_data_strong.rdd.map(lambda x: (x["user_id"], int(x["aisle_id"]))).groupByKey().mapValues(list)
formatted_ranking_strong = predicted_ranking_strong.join(actual_ranking_strong).map(lambda x: (x[1][0], x[1][1]))

metrics_strong = RankingMetrics(formatted_ranking_strong)
map_at_5_strong = metrics_strong.meanAveragePrecisionAt(5)
recall_at_5_strong = metrics_strong.recallAt(5)

# Print Generalization Results
print("Weak Generalization - MAP@5:", map_at_5_weak)
print("Weak Generalization - Recall@5:", recall_at_5_weak)
print("Strong Generalization - MAP@5:", map_at_5_strong)
print("Strong Generalization - Recall@5:", recall_at_5_strong)

# ----------Model Evaluation for 1000 Random Users-------------------------

random_users_sample = test_data_raw.map(lambda x: x.user).distinct().takeSample(False, 1000, seed=42)
random_users_sample_broadcast = spark.sparkContext.broadcast(set(random_users_sample))

# Filter for Random Users
filtered_recommendations_sample = recommendations.filter(lambda x: x[0] in random_users_sample_broadcast.value)
filtered_test_data_sample = test_data_raw.filter(lambda x: x.user in random_users_sample_broadcast.value)

predicted_ranking_sample = filtered_recommendations_sample.mapValues(lambda recs: [rec.product for rec in recs])
actual_ranking_sample = filtered_test_data_sample.map(lambda x: (x.user, int(x.product))).groupByKey().mapValues(list)
formatted_ranking_sample = predicted_ranking_sample.join(actual_ranking_sample).map(lambda x: (x[1][0], x[1][1]))

metrics_sample = RankingMetrics(formatted_ranking_sample)
map_at_5_sample = metrics_sample.meanAveragePrecisionAt(5)
recall_at_5_sample = metrics_sample.recallAt(5)

# Print Results
print("1000 Random Users - MAP@5:", map_at_5_sample)
print("1000 Random Users - Recall@5:", recall_at_5_sample)

# -----------------------Aisle Recommendations for Top Bannana Buyers-------------------

top_banana_buyers = filtered_order_products.filter(col("product_id") == banana_product_id) \
    .groupBy("user_id") \
    .count() \
    .orderBy(col("count").desc()) \
    .limit(5)

top_banana_buyers_ids = [row["user_id"] for row in top_banana_buyers.collect()]
top_banana_buyers_broadcast = spark.sparkContext.broadcast(top_banana_buyers_ids)

recommendations_top_users = recommendations.filter(lambda x: x[0] in top_banana_buyers_broadcast.value).collect()

print("\nAisle Recommendations for Top Banana Buyers:")
if not recommendations_top_users:
    print("No Recommendations Found for Top Banana Buyers")
else:
    for user_id, recs in recommendations_top_users:
        recommendations_list = [(r.product, r.rating) for r in recs]
        print(f"User ID: {user_id}, Recommendations: {recommendations_list}")

recommendations_top_users_df = spark.createDataFrame([
    (user_id, r.product, r.rating)
    for user_id, recs in recommendations_top_users
    for r in recs
], schema=["user_id", "aisle_id", "score"])

# Join recommendations with aisle names
recommendations_with_names_df = recommendations_top_users_df.join(
    aisles.withColumnRenamed("aisle_id", "aisle_key"),  
    recommendations_top_users_df["aisle_id"] == col("aisle_key"),
    "left"
).select(
    "user_id",
    "aisle_id",
    "score",
    "aisle"
)

print("Aisle Recommendations for Top Banana Buyers:")
recommendations_with_names_df.show(25, truncate=False)

recommendations_with_names_df = recommendations_with_names_df.limit(10000)  # Limit to 10,000 rows

output_path = "gs://toolsforai//recommendations_with_aisle_names.csv"

# Save the recommendations
recommendations_with_names_df.write.csv(output_path, header=True, mode="overwrite")
print(f"Results saved to {output_path}")

#----------------------------------------------------------------------------------