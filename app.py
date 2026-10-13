"""
Hybrid Movie Recommendation System
Interactive Streamlit Application
Features:
- Movie Catalog Explorer with Search & Genre Filters
- Content-Based "More Like This" Discovery (TF-IDF + Cosine Similarity)
- Personalized Hybrid Recommender with Live Alpha Weighting
- Interactive Rating Studio
- Model Evaluation Benchmark (Precision@K, Recall@K, NDCG@K)
"""

import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .movie-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e0e0e0;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        height: 100%;
    }
    .score-badge {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .reason-box {
        background-color: #f5f5f5;
        border-left: 3px solid #ff9800;
        padding: 6px 10px;
        font-size: 0.85rem;
        color: #555;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_data
def load_data():
    movies_df = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    return movies_df, ratings_df

@st.cache_resource
def load_models():
    cb_engine = joblib.load(os.path.join(MODELS_DIR, "content_engine.joblib"))
    cf_engine = joblib.load(os.path.join(MODELS_DIR, "collaborative_engine.joblib"))
    hybrid_engine = joblib.load(os.path.join(MODELS_DIR, "hybrid_engine.joblib"))
    with open(os.path.join(MODELS_DIR, "summary.json"), "r") as f:
        summary = json.load(f)
    return cb_engine, cf_engine, hybrid_engine, summary

try:
    movies_df, ratings_df = load_data()
    cb_engine, cf_engine, hybrid_engine, summary = load_models()
    hybrid_engine.set_ratings_data(ratings_df)
except Exception as e:
    st.error(f"Error loading system assets: {e}. Please ensure train_and_cache.py has executed.")
    st.stop()

# Header
st.title("🎬 Hybrid Movie Recommendation System")
st.caption("Content-Based TF-IDF + Cosine Similarity & SVD Collaborative Filtering with Quality Evaluation")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Personalized Hybrid Recommender",
    "🔍 Content-Based 'More Like This'",
    "🎞️ Movie Catalog Explorer",
    "⭐ Interactive Rating Studio",
    "📊 Model Quality & Evaluation (P@K, R@K, NDCG@K)"
])

# -------------------------------------------------------------
# TAB 1: PERSONALIZED HYBRID RECOMMENDER
# -------------------------------------------------------------
with tab1:
    st.header("🎯 Personalized Hybrid Recommendations")
    st.markdown("Blends metadata content similarity and collaborative matrix factorization with adjustable balance parameter $\\alpha$.")

    col_ctrl, col_recs = st.columns([1, 2.5])

    with col_ctrl:
        st.subheader("Preferences & Tuning")
        
        # User Selection
        user_list = sorted(ratings_df["user_id"].unique())
        user_options = {
            1: "User 1 (Sci-Fi & Nolan Enthusiast)",
            16: "User 16 (Crime, Scorsese & Tarantino Fan)",
            31: "User 31 (Romance & Musicals Fan)",
            46: "User 46 (Animation & Studio Ghibli Fan)"
        }
        for u in user_list:
            if u not in user_options:
                user_options[u] = f"User {u}"

        selected_user = st.selectbox(
            "Select User Profile",
            options=user_list,
            format_func=lambda x: user_options.get(x, f"User {x}")
        )

        # Show user's top-rated past movies
        u_ratings = ratings_df[ratings_df["user_id"] == selected_user].merge(movies_df, on="movie_id")
        top_u = u_ratings.sort_values(by="rating", ascending=False).head(4)
        
        st.markdown("**User's Top Favorites:**")
        for _, r in top_u.iterrows():
            st.markdown(f"- **{r['title']}** ({r['rating']}★) - *{r['genres']}*")

        st.markdown("---")
        
        # Alpha slider
        alpha = st.slider(
            "Hybrid Weighting (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="1.0 = Pure Content-Based (Metadata similarity) | 0.0 = Pure Collaborative (User ratings matrix) | 0.5 = Balanced Hybrid"
        )
        
        if alpha >= 0.7:
            st.info("Currently prioritizing **Content Metadata & Plot Similarity**.")
        elif alpha <= 0.3:
            st.info("Currently prioritizing **Collaborative User-Rating Patterns**.")
        else:
            st.success("Currently running **Balanced Hybrid Filtering**.")

        top_k = st.slider("Number of Recommendations", 3, 12, 6)
        exclude_rated = st.checkbox("Exclude movies already rated by user", value=True)

    with col_recs:
        st.subheader(f"Top {top_k} Recommendations for User {selected_user}")
        
        recs = hybrid_engine.recommend(
            user_id=selected_user,
            top_k=top_k,
            alpha=alpha,
            exclude_rated=exclude_rated
        )

        if not recs:
            st.info("No recommendations found with current filters.")
        else:
            # Display recommendation grid
            rec_cols = st.columns(2)
            for idx, item in enumerate(recs):
                target_col = rec_cols[idx % 2]
                with target_col:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h4 style="margin:0; color:#1a237e;">{item['title']} <span style="font-size:0.9rem; color:#777;">({item['release_year']})</span></h4>
                            <span class="score-badge">Hybrid: {item['hybrid_score']:.3f}</span>
                        </div>
                        <p style="margin:4px 0; font-size:0.85rem; color:#455a64;">
                            <b>Director:</b> {item['director']} | <b>Genres:</b> {item['genres']}
                        </p>
                        <p style="margin:6px 0; font-size:0.9rem; color:#333;">{item['overview'][:140]}...</p>
                        <div style="font-size:0.85rem; margin-top:6px;">
                            <span>Content: <b>{item['content_score']:.2f}</b></span> | 
                            <span>Est. Rating: <b>{item['predicted_stars']:.1f}★</b></span> | 
                            <span>TMDB: <b>{item['vote_average']}★</b></span>
                        </div>
                        <div class="reason-box">
                            💡 {item['reason']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: CONTENT-BASED "MORE LIKE THIS"
# -------------------------------------------------------------
with tab2:
    st.header("🔍 Content-Based 'More Like This' Discovery")
    st.markdown("Retrieves top movie matches based on TF-IDF vectorization and cosine similarity over plot synopses, directors, genres, and cast metadata.")

    c_sel, c_res = st.columns([1, 2.5])
    
    with c_sel:
        target_movie_title = st.selectbox("Select a Seed Movie", options=movies_df["title"].tolist())
        target_movie = movies_df[movies_df["title"] == target_movie_title].iloc[0]
        
        st.markdown(f"""
        **Target Movie Details:**
        - **Director:** {target_movie['director']}
        - **Genres:** {target_movie['genres']}
        - **Release Year:** {target_movie['release_year']}
        - **Rating:** {target_movie['vote_average']}★
        
        *{target_movie['overview']}*
        """)
        
        cb_top_k = st.slider("Top Matches", min_value=3, max_value=8, value=4, key="cb_k")

    with c_res:
        st.subheader(f"Movies Similar to '{target_movie_title}'")
        similar_movies = cb_engine.get_similar_movies(movie_id=int(target_movie["movie_id"]), top_k=cb_top_k)

        for match in similar_movies:
            sim_pct = match["similarity_score"] * 100
            st.markdown(f"""
            <div class="movie-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin:0; color:#1a237e;">{match['title']} ({match['release_year']})</h4>
                    <span class="score-badge">Similarity: {sim_pct:.1f}%</span>
                </div>
                <p style="margin:4px 0; font-size:0.85rem; color:#455a64;">
                    <b>Director:</b> {match['director']} | <b>Genres:</b> {match['genres']}
                </p>
                <p style="margin:6px 0; font-size:0.9rem; color:#333;">{match['overview']}</p>
                <div class="reason-box">
                    🏷️ <b>Match Attribution:</b> {match['explanation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: MOVIE CATALOG EXPLORER
# -------------------------------------------------------------
with tab3:
    st.header("🎞️ Movie Catalog Explorer")
    
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        genres_unique = sorted(list(set([g for row in movies_df["genres"] for g in row.split("|")])))
        genre_filter = st.selectbox("Filter by Genre", ["All"] + genres_unique)
    with col_f2:
        search_query = st.text_input("Search by Movie Title or Director", "")

    catalog_df = movies_df.copy()
    if genre_filter != "All":
        catalog_df = catalog_df[catalog_df["genres"].str.contains(genre_filter, case=False, na=False)]
    if search_query:
        mask = catalog_df["title"].str.contains(search_query, case=False, na=False) | \
               catalog_df["director"].str.contains(search_query, case=False, na=False)
        catalog_df = catalog_df[mask]

    st.caption(f"Showing {len(catalog_df)} matching titles")
    st.dataframe(
        catalog_df[["movie_id", "title", "director", "genres", "release_year", "vote_average", "overview"]],
        use_container_width=True
    )

# -------------------------------------------------------------
# TAB 4: INTERACTIVE RATING STUDIO
# -------------------------------------------------------------
with tab4:
    st.header("⭐ Interactive Rating Studio")
    st.markdown("Submit a new movie rating as a user and watch the collaborative and hybrid models instantly adapt recommendations.")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        rate_user = st.number_input("User ID", min_value=1, max_value=200, value=selected_user)
        rate_movie_title = st.selectbox("Select Movie to Rate", options=movies_df["title"].tolist(), key="rate_title")
        chosen_movie_id = int(movies_df[movies_df["title"] == rate_movie_title].iloc[0]["movie_id"])
        
        star_rating = st.slider("Your Rating (Stars)", min_value=1.0, max_value=5.0, value=5.0, step=0.5)

        if st.button("Submit Rating ⭐", type="primary"):
            # Update global dataframe
            new_row = pd.DataFrame([{"user_id": rate_user, "movie_id": chosen_movie_id, "rating": star_rating}])
            ratings_df = pd.concat([ratings_df, new_row], ignore_index=True)
            hybrid_engine.set_ratings_data(ratings_df)
            st.success(f"Successfully recorded rating of {star_rating}★ for '{rate_movie_title}' by User {rate_user}!")

    with col_r2:
        st.subheader(f"Recent Ratings for User {rate_user}")
        user_history = ratings_df[ratings_df["user_id"] == rate_user].merge(movies_df, on="movie_id")
        if not user_history.empty:
            st.dataframe(user_history[["title", "genres", "director", "rating"]].tail(8), use_container_width=True)
        else:
            st.info("No prior ratings recorded for this user ID.")

# -------------------------------------------------------------
# TAB 5: MODEL QUALITY & EVALUATION BENCHMARK
# -------------------------------------------------------------
with tab5:
    st.header("📊 Model Quality & Benchmark Evaluation")
    st.markdown("""
    Evaluating recommendation performance on holdout test user ratings using:
    - **Precision@K**: Fraction of top-$K$ recommendations that match user's relevant taste.
    - **Recall@K**: Fraction of total relevant user items successfully captured in top-$K$.
    - **NDCG@K (Normalized Discounted Cumulative Gain)**: Position-weighted ranking metric rewarding relevant items surfaced at higher ranks.
    """)

    benchmark_data = summary["benchmark"]
    bench_df = pd.DataFrame(benchmark_data)
    st.dataframe(bench_df.set_index("Model Variant"), use_container_width=True)

    st.markdown("---")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.subheader("Precision & Recall at K=5 and K=10")
        chart_pr = bench_df.set_index("Model Variant")[["P@5", "P@10", "R@5", "R@10"]]
        st.bar_chart(chart_pr)

    with col_b2:
        st.subheader("Ranking Relevance: NDCG@5 and NDCG@10")
        chart_ndcg = bench_df.set_index("Model Variant")[["NDCG@5", "NDCG@10"]]
        st.bar_chart(chart_ndcg)

    st.markdown("""
    ### 🔬 Insights & Analysis:
    1. **Content-Based (TF-IDF)** excels at finding strong thematic and stylistic matches (high recall for specialized sub-genres and directors).
    2. **Collaborative Filtering (SVD)** surfaces latent community preferences and serendipitous recommendations, but suffers when user interaction is sparse.
    3. **Hybrid Recommender** achieves the optimal balance: it bridges cold-start issues, avoids over-specialization, and boosts ranking satisfaction across varied user personas.
    """)

# Footer
st.markdown("---")
st.caption("Hybrid Movie Recommendation System | FastAPI + Streamlit Serving Architecture | Ayush Pandey Portfolio")
