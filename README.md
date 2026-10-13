# Hybrid Movie Recommendation System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?logo=vercel)](https://ayush-pandey-ml-portfolio.vercel.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange.svg)](https://scikit-learn.org/)

🌐 **Live Interactive Showcase**: [https://ayush-pandey-ml-portfolio.vercel.app](https://ayush-pandey-ml-portfolio.vercel.app)

A production-ready hybrid recommendation platform combining **Content-Based TF-IDF vectorization and Cosine Similarity** with **Collaborative Filtering via Matrix Factorization (Truncated SVD)**, exposed through a high-performance **FastAPI inference endpoint** and an interactive **Streamlit dashboard**.

---

## 🌟 Key Capabilities

1. **Content-Based Recommender (TF-IDF + Cosine Similarity)**:
   - Formulates a weighted metadata "soup" combining director (double-weighted for authorial style), lead cast, genres, plot keywords, and synopsis tokens.
   - Extracts unigram & bigram features with Scikit-Learn `TfidfVectorizer`.
   - Computes dense pairwise cosine similarities to retrieve thematic and stylistic lookalikes with explainable attribute tags.

2. **Collaborative Filtering (Matrix Factorization via SVD)**:
   - Constructs a sparse User-Item interaction matrix.
   - Applies Truncated SVD dimensionality reduction into latent factor components ($k=8$).
   - Reconstructs predicted star ratings for unobserved items to surface community-driven serendipity.

3. **Hybrid Engine with Dynamic Weighting ($\alpha$)**:
   - Computes weighted score blend:
     $$\text{Score}_{\text{hybrid}} = \alpha \cdot \text{Score}_{\text{content}} + (1 - \alpha) \cdot \text{Score}_{\text{CF}}$$
   - Allows users to seamlessly slide $\alpha$ from pure content ($\alpha=1.0$) to pure collaborative ($\alpha=0.0$), with a robust fallback for cold-start new users.

4. **Rigorous Offline Evaluation Benchmark**:
   - Measures ranking and retrieval performance on test user splits across three model variants using:
     - **Precision@K** ($K=5, 10$)
     - **Recall@K** ($K=5, 10$)
     - **NDCG@K (Normalized Discounted Cumulative Gain)** ($K=5, 10$)

5. **Deployable Serving Architecture**:
   - **FastAPI Endpoint (`api.py`)**: Exposes REST endpoints with automatic OpenAPI/Swagger documentation.
   - **Streamlit Interface (`app.py`)**: Rich web dashboard with catalog browsing, interactive $\alpha$ tuning, live rating studio, and evaluation charts.

---

## 📁 Project Structure

```
hybrid-movie-recommendation/
├── data/
│   ├── movies.csv                    # Curated movie catalog with metadata
│   └── ratings.csv                   # User rating interactions matrix
├── models/
│   ├── content_engine.joblib         # Fitted TF-IDF vectorizer & similarity matrix
│   ├── collaborative_engine.joblib   # Fitted TruncatedSVD collaborative matrix
│   ├── hybrid_engine.joblib          # Blended hybrid recommender pipeline
│   ├── evaluation_benchmark.csv      # Precision@K, Recall@K, NDCG@K results
│   └── summary.json                  # Model summary & benchmark stats
├── src/
│   ├── preprocessing.py              # Metadata soup creator & TF-IDF tokenizer
│   ├── content_based.py              # Content-based similarity retriever
│   ├── collaborative.py              # SVD collaborative filtering engine
│   ├── hybrid.py                     # Configurable alpha hybrid recommender
│   ├── evaluation.py                 # P@K, R@K, NDCG@K evaluation metrics
│   └── generate_dataset.py           # Synthetic dataset & taste personas
├── train_and_cache.py                # Pipeline execution & model serializing script
├── api.py                            # Production FastAPI inference server
├── app.py                            # Interactive Streamlit frontend
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess & Train Models
```bash
python train_and_cache.py
```

### 3. Run FastAPI Inference Endpoint
```bash
uvicorn api:app --reload --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Recommendation query: `http://localhost:8000/recommend/hybrid?user_id=1&top_k=5&alpha=0.6`

### 4. Run Streamlit User Interface
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Offline Evaluation Benchmark

| Model Variant | Precision@5 | Precision@10 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Content-Based ($\alpha=1.0$)** | 0.191 | 0.157 | 0.506 | 0.865 | 0.366 | 0.500 |
| **Collaborative Filtering ($\alpha=0.0$)** | 0.091 | 0.091 | 0.277 | 0.560 | 0.171 | 0.273 |
| **Hybrid Recommender ($\alpha=0.5$)** | 0.141 | 0.116 | 0.389 | 0.679 | 0.248 | 0.354 |
