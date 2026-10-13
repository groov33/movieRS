"""
Train and Cache Script for Hybrid Movie Recommendation System
Builds metadata vectors, decomposes collaborative ratings matrix,
runs Precision@K, Recall@K, and NDCG@K evaluation benchmark, and caches models.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.generate_dataset import generate_ratings, MOVIES_DATA
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeFilteringRecommender
from src.hybrid import HybridRecommender
from src.evaluation import evaluate_models

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Generating Movie Catalog and Ratings Dataset...")
    print("=" * 60)
    movies_df = pd.DataFrame(MOVIES_DATA)
    movies_df.to_csv(os.path.join(data_dir, "movies.csv"), index=False)

    ratings_df = generate_ratings(movies_df, n_users=60, seed=42)
    ratings_df.to_csv(os.path.join(data_dir, "ratings.csv"), index=False)
    print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} user ratings.")

    print("\n" + "=" * 60)
    print("STEP 2: Training Content-Based TF-IDF Recommender...")
    print("=" * 60)
    cb_engine = ContentBasedRecommender()
    cb_engine.fit(movies_df)
    print("TF-IDF matrix & pairwise cosine similarity computed.")

    print("\n" + "=" * 60)
    print("STEP 3: Splitting Ratings & Training SVD Collaborative Engine...")
    print("=" * 60)
    train_ratings, test_ratings = train_test_split(ratings_df, test_size=0.20, random_state=42)
    
    cf_engine = CollaborativeFilteringRecommender(n_factors=8, random_state=42)
    cf_engine.fit(train_ratings, all_movie_ids=movies_df["movie_id"].tolist())
    print("Matrix factorization (TruncatedSVD) fitted on user-item matrix.")

    print("\n" + "=" * 60)
    print("STEP 4: Initializing Hybrid Recommender Engine...")
    print("=" * 60)
    hybrid_engine = HybridRecommender(content_engine=cb_engine, cf_engine=cf_engine)
    hybrid_engine.set_ratings_data(ratings_df)

    print("\n" + "=" * 60)
    print("STEP 5: Benchmarking Quality: Precision@K, Recall@K, NDCG@K...")
    print("=" * 60)
    eval_df = evaluate_models(hybrid_engine, test_ratings, k_values=[5, 10])
    print("\nEvaluation Benchmark Comparison Table:")
    print(eval_df.to_string(index=False))

    eval_csv = os.path.join(models_dir, "evaluation_benchmark.csv")
    eval_df.to_csv(eval_csv, index=False)

    print("\n" + "=" * 60)
    print("STEP 6: Serializing and Caching Model Artifacts...")
    print("=" * 60)
    joblib.dump(cb_engine, os.path.join(models_dir, "content_engine.joblib"))
    joblib.dump(cf_engine, os.path.join(models_dir, "collaborative_engine.joblib"))
    joblib.dump(hybrid_engine, os.path.join(models_dir, "hybrid_engine.joblib"))

    # Summary json
    summary = {
        "movies_count": len(movies_df),
        "ratings_count": len(ratings_df),
        "users_count": ratings_df["user_id"].nunique(),
        "benchmark": eval_df.to_dict(orient="records")
    }
    with open(os.path.join(models_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Models and evaluation metrics successfully cached to models/ directory!")

if __name__ == "__main__":
    main()
