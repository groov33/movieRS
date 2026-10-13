"""
Content-Based Recommendation Engine
Implements TF-IDF vectorization and Cosine Similarity over movie metadata.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import create_metadata_soup, build_tfidf_vectorizer

class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = build_tfidf_vectorizer()
        self.movies_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}

    def fit(self, movies_df: pd.DataFrame):
        """Fit TF-IDF matrix and compute pairwise cosine similarity matrix."""
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.movie_id_to_idx = {row["movie_id"]: idx for idx, row in self.movies_df.iterrows()}
        self.idx_to_movie_id = {idx: row["movie_id"] for idx, row in self.movies_df.iterrows()}

        soup_series = create_metadata_soup(self.movies_df)
        self.tfidf_matrix = self.vectorizer.fit_transform(soup_series)
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        return self

    def get_similar_movies(self, movie_id: int, top_k: int = 5) -> list:
        """Retrieve top-K most similar movies using cosine similarity."""
        if movie_id not in self.movie_id_to_idx:
            return []

        idx = self.movie_id_to_idx[movie_id]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort descending by similarity score, excluding the query movie itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_k]

        results = []
        source_movie = self.movies_df.iloc[idx]

        for match_idx, score in sim_scores:
            matched_movie = self.movies_df.iloc[match_idx]
            
            # Find common attributes for explanation
            src_genres = set(str(source_movie.get("genres", "")).split("|"))
            tgt_genres = set(str(matched_movie.get("genres", "")).split("|"))
            common_genres = list(src_genres.intersection(tgt_genres))

            same_director = source_movie.get("director") == matched_movie.get("director")
            
            explanation = []
            if same_director:
                explanation.append(f"Directed by {matched_movie.get('director')}")
            if common_genres:
                explanation.append(f"Shared genres: {', '.join(common_genres)}")

            results.append({
                "movie_id": int(matched_movie["movie_id"]),
                "title": matched_movie["title"],
                "genres": matched_movie["genres"],
                "director": matched_movie["director"],
                "release_year": int(matched_movie["release_year"]),
                "vote_average": float(matched_movie["vote_average"]),
                "overview": matched_movie["overview"],
                "similarity_score": round(float(score), 4),
                "explanation": " • ".join(explanation) if explanation else "Thematic story similarity"
            })

        return results

    def get_content_scores_for_movie(self, movie_id: int) -> dict:
        """Return dictionary mapping movie_id -> similarity score in [0, 1]."""
        if movie_id not in self.movie_id_to_idx:
            return {mid: 0.0 for mid in self.movie_id_to_idx}

        idx = self.movie_id_to_idx[movie_id]
        scores = self.cosine_sim[idx]
        return {self.idx_to_movie_id[i]: float(scores[i]) for i in range(len(scores))}
