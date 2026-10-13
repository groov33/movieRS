"""
Hybrid Movie Recommendation Engine
Combines content-based metadata similarity and collaborative filtering matrix factorization
with configurable alpha balance and cold-start resilience.
"""

import numpy as np
import pandas as pd
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeFilteringRecommender

class HybridRecommender:
    def __init__(self, content_engine: ContentBasedRecommender, cf_engine: CollaborativeFilteringRecommender):
        self.content_engine = content_engine
        self.cf_engine = cf_engine
        self.movies_df = content_engine.movies_df
        self.ratings_df = None

    def set_ratings_data(self, ratings_df: pd.DataFrame):
        self.ratings_df = ratings_df

    def recommend(self, user_id: int, seed_movie_id: int = None, top_k: int = 6, 
                  alpha: float = 0.5, exclude_rated: bool = True) -> list:
        """
        Produce top-K hybrid recommendations.
        alpha = 1.0 -> 100% Content-Based
        alpha = 0.0 -> 100% Collaborative Filtering
        alpha = 0.5 -> Balanced Hybrid
        """
        all_movies = self.movies_df.copy()
        movie_ids = all_movies["movie_id"].tolist()

        # Determine movies already rated by the user
        rated_ids = set()
        if self.ratings_df is not None and user_id in self.cf_engine.users:
            user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
            rated_ids = set(user_ratings["movie_id"].tolist())

        # 1. Compute Collaborative Scores (in [0, 1])
        cf_scores = self.cf_engine.get_user_predicted_scores(user_id)

        # 2. Compute Content Scores (in [0, 1])
        if seed_movie_id is not None and seed_movie_id in self.content_engine.movie_id_to_idx:
            content_scores = self.content_engine.get_content_scores_for_movie(seed_movie_id)
        else:
            # Aggregate content profile from user's highest-rated movies
            if rated_ids:
                user_favs = self.ratings_df[(self.ratings_df["user_id"] == user_id) & (self.ratings_df["rating"] >= 4.0)]
                if not user_favs.empty:
                    top_rated_mids = user_favs.sort_values(by="rating", ascending=False)["movie_id"].head(3).tolist()
                    accum_scores = np.zeros(len(movie_ids))
                    for mid in top_rated_mids:
                        mid_scores = self.content_engine.get_content_scores_for_movie(mid)
                        for idx, m in enumerate(movie_ids):
                            accum_scores[idx] += mid_scores.get(m, 0.0)
                    accum_scores /= max(1, len(top_rated_mids))
                    content_scores = {mid: float(accum_scores[i]) for i, mid in enumerate(movie_ids)}
                else:
                    content_scores = {mid: 0.5 for mid in movie_ids}
            else:
                # Cold-start fallback
                content_scores = {mid: 0.5 for mid in movie_ids}

        # 3. Calculate Hybrid Score
        scored_movies = []
        for mid in movie_ids:
            if exclude_rated and mid in rated_ids and (seed_movie_id != mid):
                continue
            if seed_movie_id == mid:
                continue

            c_score = content_scores.get(mid, 0.0)
            cf_score = cf_scores.get(mid, 0.5)
            h_score = (alpha * c_score) + ((1.0 - alpha) * cf_score)

            movie_meta = self.movies_df[self.movies_df["movie_id"] == mid].iloc[0]

            # Generate recommendation rationale
            if alpha >= 0.7:
                reason = f"High content alignment with themes of '{movie_meta['title']}' ({movie_meta['genres']})"
            elif alpha <= 0.3:
                reason = f"Popular among viewers sharing your rating profile (Est. rating: {1.0 + cf_score*4.0:.1f}★)"
            else:
                reason = f"Personalized blend: strong thematic match + high affinity score"

            scored_movies.append({
                "movie_id": int(mid),
                "title": movie_meta["title"],
                "genres": movie_meta["genres"],
                "director": movie_meta["director"],
                "release_year": int(movie_meta["release_year"]),
                "vote_average": float(movie_meta["vote_average"]),
                "poster": movie_meta.get("poster", ""),
                "overview": movie_meta["overview"],
                "content_score": round(float(c_score), 4),
                "collaborative_score": round(float(cf_score), 4),
                "predicted_stars": round(1.0 + (cf_score * 4.0), 2),
                "hybrid_score": round(float(h_score), 4),
                "reason": reason
            })

        # Rank descending by hybrid score
        scored_movies = sorted(scored_movies, key=lambda x: x["hybrid_score"], reverse=True)
        return scored_movies[:top_k]
