"""
Recommendation System Quality Evaluation Module
Evaluates model variants (Content-Based, Collaborative Filtering, and Hybrid)
using Precision@K, Recall@K, and NDCG@K metrics on holdout rating test splits.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def compute_dcg_at_k(relevance_scores: list, k: int) -> float:
    """Compute Discounted Cumulative Gain at rank K."""
    relevance_scores = np.asarray(relevance_scores, dtype=float)[:k]
    if relevance_scores.size == 0:
        return 0.0
    # Formula: sum((2^rel - 1) / log2(rank + 1))
    discounts = np.log2(np.arange(2, relevance_scores.size + 2))
    return float(np.sum((2 ** relevance_scores - 1) / discounts))

def compute_ndcg_at_k(recommended_ids: list, actual_ratings_dict: dict, k: int = 5) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K."""
    top_k = recommended_ids[:k]
    # Relevance: rating >= 4.0 gets relevance 1 (or normalized star value)
    actual_gains = [1.0 if actual_ratings_dict.get(mid, 0) >= 4.0 else 0.0 for mid in top_k]
    dcg = compute_dcg_at_k(actual_gains, k)

    # Ideal ranking gains
    ideal_gains = sorted([1.0 if r >= 4.0 else 0.0 for r in actual_ratings_dict.values()], reverse=True)
    idcg = compute_dcg_at_k(ideal_gains, k)

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)

def evaluate_models(hybrid_engine, test_ratings_df: pd.DataFrame, k_values=[5, 10]) -> dict:
    """
    Evaluate Content-Based, Collaborative Filtering, and Hybrid models
    across Precision@K, Recall@K, and NDCG@K on test users.
    """
    users = test_ratings_df["user_id"].unique()
    
    # Track metrics for each model variant
    variants = {
        "Content-Based (alpha=1.0)": 1.0,
        "Collaborative Filtering (alpha=0.0)": 0.0,
        "Hybrid Recommender (alpha=0.5)": 0.5
    }

    results = {}

    for var_name, alpha in variants.items():
        results[var_name] = {f"P@{k}": [] for k in k_values}
        results[var_name].update({f"R@{k}": [] for k in k_values})
        results[var_name].update({f"NDCG@{k}": [] for k in k_values})

        for uid in users:
            user_test = test_ratings_df[test_ratings_df["user_id"] == uid]
            relevant_movies = set(user_test[user_test["rating"] >= 4.0]["movie_id"].tolist())
            actual_dict = dict(zip(user_test["movie_id"], user_test["rating"]))

            if not relevant_movies:
                continue

            # Generate top recommendations for maximum K
            max_k = max(k_values)
            recs = hybrid_engine.recommend(user_id=uid, top_k=max_k, alpha=alpha, exclude_rated=False)
            rec_ids = [r["movie_id"] for r in recs]

            for k in k_values:
                top_k_ids = rec_ids[:k]
                hits = len(set(top_k_ids).intersection(relevant_movies))

                # Precision@K
                precision = hits / k
                # Recall@K
                recall = hits / len(relevant_movies)
                # NDCG@K
                ndcg = compute_ndcg_at_k(top_k_ids, actual_dict, k=k)

                results[var_name][f"P@{k}"].append(precision)
                results[var_name][f"R@{k}"].append(recall)
                results[var_name][f"NDCG@{k}"].append(ndcg)

    # Average metrics
    summary = []
    for var_name, metrics in results.items():
        row = {"Model Variant": var_name}
        for metric_name, val_list in metrics.items():
            row[metric_name] = round(float(np.mean(val_list)), 4) if val_list else 0.0
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    return summary_df
