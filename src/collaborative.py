"""
Collaborative Filtering Module
Implements Matrix Factorization via Truncated SVD and User-Item Interaction Matrix
to predict personalized ratings for unseen movies.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

class CollaborativeFilteringRecommender:
    def __init__(self, n_factors: int = 8, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
        self.user_item_matrix = None
        self.predicted_ratings_matrix = None
        self.users = []
        self.movies = []
        self.user_to_idx = {}
        self.movie_to_idx = {}
        self.global_mean = 3.5

    def fit(self, ratings_df: pd.DataFrame, all_movie_ids: list = None):
        """Fit Truncated SVD on the user-movie rating pivot table."""
        self.global_mean = float(ratings_df["rating"].mean())
        
        # Pivot ratings into User x Movie matrix
        pivot = ratings_df.pivot_table(index="user_id", columns="movie_id", values="rating")
        
        # Ensure all catalog movies are present in columns
        if all_movie_ids:
            for mid in all_movie_ids:
                if mid not in pivot.columns:
                    pivot[mid] = np.nan

        pivot = pivot.sort_index(axis=1)
        self.users = list(pivot.index)
        self.movies = list(pivot.columns)
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.users)}
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(self.movies)}

        # Fill missing ratings with item mean or global mean for SVD decomposition
        item_means = pivot.mean(axis=0).fillna(self.global_mean)
        pivot_filled = pivot.apply(lambda col: col.fillna(item_means[col.name]), axis=0)

        # Decompose User x Movie matrix
        self.user_item_matrix = pivot.values
        user_factors = self.svd.fit_transform(pivot_filled.values)
        item_factors = self.svd.components_

        # Reconstructed predictions
        self.predicted_ratings_matrix = np.dot(user_factors, item_factors)
        # Clip to realistic rating bounds [1.0, 5.0]
        self.predicted_ratings_matrix = np.clip(self.predicted_ratings_matrix, 1.0, 5.0)

        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict expected rating for a single (user, movie) pair."""
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return self.global_mean

        u_idx = self.user_to_idx[user_id]
        m_idx = self.movie_to_idx[movie_id]
        return float(self.predicted_ratings_matrix[u_idx, m_idx])

    def get_user_predicted_scores(self, user_id: int) -> dict:
        """
        Return dict mapping movie_id -> normalized collaborative score [0, 1].
        Normalized from the 1.0-5.0 scale: (rating - 1.0) / 4.0.
        """
        if user_id not in self.user_to_idx:
            # Cold-start fallback
            norm_global = (self.global_mean - 1.0) / 4.0
            return {mid: norm_global for mid in self.movies}

        u_idx = self.user_to_idx[user_id]
        preds = self.predicted_ratings_matrix[u_idx]
        
        # Normalize 1-5 scale into 0-1 range
        norm_preds = np.clip((preds - 1.0) / 4.0, 0.0, 1.0)
        return {mid: float(norm_preds[self.movie_to_idx[mid]]) for mid in self.movies}
