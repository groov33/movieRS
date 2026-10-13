"""
FastAPI Inference Endpoint for Hybrid Movie Recommendation System
Provides RESTful APIs for content-based, collaborative, and hybrid recommendations,
model evaluation benchmarks, movie search, and user rating submissions.
"""

import os
import json
import joblib
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Define request/response schemas
class RatingSubmission(BaseModel):
    user_id: int = Field(..., description="ID of the user submitting the rating")
    movie_id: int = Field(..., description="ID of the movie being rated")
    rating: float = Field(..., ge=0.5, le=5.0, description="Star rating between 0.5 and 5.0")

class MovieResponse(BaseModel):
    movie_id: int
    title: str
    genres: str
    director: str
    release_year: int
    vote_average: float
    overview: str
    poster: Optional[str] = ""

class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    director: str
    release_year: int
    vote_average: float
    poster: Optional[str] = ""
    overview: str
    content_score: Optional[float] = 0.0
    collaborative_score: Optional[float] = 0.0
    predicted_stars: Optional[float] = 0.0
    hybrid_score: Optional[float] = 0.0
    reason: Optional[str] = ""

class HealthResponse(BaseModel):
    status: str
    movies_count: int
    users_count: int
    ratings_count: int

# Initialize FastAPI App
app = FastAPI(
    title="Hybrid Movie Recommendation API",
    description="Production-grade recommendation serving engine combining Content-Based TF-IDF and SVD Collaborative Filtering.",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Global instances
movies_df = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
cb_engine = joblib.load(os.path.join(MODELS_DIR, "content_engine.joblib"))
cf_engine = joblib.load(os.path.join(MODELS_DIR, "collaborative_engine.joblib"))
hybrid_engine = joblib.load(os.path.join(MODELS_DIR, "hybrid_engine.joblib"))
hybrid_engine.set_ratings_data(ratings_df)

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Verify API service health and loaded dataset statistics."""
    return {
        "status": "healthy",
        "movies_count": len(movies_df),
        "users_count": int(ratings_df["user_id"].nunique()),
        "ratings_count": len(ratings_df)
    }

@app.get("/movies", response_model=List[MovieResponse], tags=["Catalog"])
def list_movies(
    genre: Optional[str] = Query(None, description="Filter by genre substring, e.g. 'Sci-Fi'"),
    search: Optional[str] = Query(None, description="Search term in title or director"),
    limit: int = Query(30, ge=1, le=100)
):
    """List movies catalog with optional search and genre filtering."""
    df_filtered = movies_df.copy()
    if genre:
        df_filtered = df_filtered[df_filtered["genres"].str.contains(genre, case=False, na=False)]
    if search:
        mask = (df_filtered["title"].str.contains(search, case=False, na=False)) | \
               (df_filtered["director"].str.contains(search, case=False, na=False))
        df_filtered = df_filtered[mask]
        
    return df_filtered.head(limit).to_dict(orient="records")

@app.get("/movies/{movie_id}", response_model=MovieResponse, tags=["Catalog"])
def get_movie_by_id(movie_id: int):
    """Retrieve details for a single movie by its ID."""
    match = movies_df[movies_df["movie_id"] == movie_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found.")
    return match.iloc[0].to_dict()

@app.get("/recommend/content/{movie_id}", tags=["Recommendations"])
def get_content_recommendations(movie_id: int, top_k: int = Query(5, ge=1, le=20)):
    """
    Retrieve top-K content-based recommendations for a target movie
    based on TF-IDF cosine similarity of metadata tags and synopsis.
    """
    if movie_id not in cb_engine.movie_id_to_idx:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found in catalog.")
    recs = cb_engine.get_similar_movies(movie_id=movie_id, top_k=top_k)
    return {
        "target_movie_id": movie_id,
        "strategy": "Content-Based (TF-IDF + Cosine Similarity)",
        "recommendations": recs
    }

@app.get("/recommend/hybrid", response_model=List[RecommendationItem], tags=["Recommendations"])
def get_hybrid_recommendations(
    user_id: int = Query(..., description="Target User ID"),
    seed_movie_id: Optional[int] = Query(None, description="Optional seed movie ID for context"),
    top_k: int = Query(6, ge=1, le=20, description="Number of recommendations"),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Weight: 1.0 = Pure Content, 0.0 = Pure Collaborative"),
    exclude_rated: bool = Query(True, description="Filter out movies already rated by user")
):
    """
    Generate personalized hybrid recommendations combining content similarity and collaborative filtering.
    """
    recs = hybrid_engine.recommend(
        user_id=user_id,
        seed_movie_id=seed_movie_id,
        top_k=top_k,
        alpha=alpha,
        exclude_rated=exclude_rated
    )
    return recs

@app.get("/evaluation", tags=["Model Quality"])
def get_evaluation_benchmark():
    """Retrieve Precision@K, Recall@K, and NDCG@K comparison benchmarks across model variants."""
    summary_path = os.path.join(MODELS_DIR, "summary.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)
    return summary["benchmark"]

@app.post("/ratings", tags=["User Actions"])
def submit_rating(payload: RatingSubmission):
    """Submit a new user rating and dynamically update user profile."""
    global ratings_df
    new_entry = pd.DataFrame([{
        "user_id": payload.user_id,
        "movie_id": payload.movie_id,
        "rating": payload.rating
    }])
    ratings_df = pd.concat([ratings_df, new_entry], ignore_index=True)
    hybrid_engine.set_ratings_data(ratings_df)
    return {
        "status": "success",
        "message": f"Rating {payload.rating} recorded for User {payload.user_id} on Movie {payload.movie_id}."
    }
