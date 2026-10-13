"""
Metadata Preprocessing & Feature Extraction Module
Combines movie metadata (genres, director, cast, keywords, overview) into a unified soup
and applies TF-IDF vectorization with n-gram extraction.
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text: str) -> str:
    """Clean and standardize text string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s|]', ' ', text)
    return text.strip()

def create_metadata_soup(df: pd.DataFrame) -> pd.Series:
    """
    Construct rich metadata soup by weighting director, top cast, genres,
    keywords, and plot overview tokens.
    """
    soups = []
    for _, row in df.iterrows():
        # Clean tags
        genres = " ".join([g.lower().replace(" ", "") for g in str(row.get("genres", "")).split("|")])
        # Double weight for director to prioritize authorial style
        directors = " ".join([d.lower().replace(" ", "") for d in str(row.get("director", "")).split("|")] * 2)
        cast = " ".join([c.lower().replace(" ", "") for c in str(row.get("cast", "")).split("|")[:3]])
        keywords = clean_text(str(row.get("keywords", "")))
        overview = clean_text(str(row.get("overview", "")))

        soup = f"{genres} {directors} {cast} {keywords} {overview}"
        soups.append(soup)

    return pd.Series(soups, index=df.index)

def build_tfidf_vectorizer(max_features=5000):
    """Initialize TfidfVectorizer configured for recommendation tasks."""
    return TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2)
    )
