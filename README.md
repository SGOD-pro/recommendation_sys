# Hybrid AI Recommendation System

> Internship Project | AI / ML Track | MovieLens 100K

---

## Overview

A full-stack, hybrid recommendation system combining **Collaborative Filtering (SVD)**, **Content-Based Filtering (TF-IDF + Cosine Similarity)**, and **Popularity Scoring** into a weighted hybrid engine.

### Hybrid Formula
```
Final Score = 0.6 × CF (SVD) + 0.3 × Content (TF-IDF) + 0.1 × Popularity
```

Tie-breaking: `avg_rating` → `num_ratings`

---

## Project Structure

```
recommendation_sys/
├── backend/
│   ├── core/
│   │   ├── data_processor.py    # Module 1: Load & clean data
│   │   ├── analytics.py         # Module 2: User interaction analysis
│   │   ├── splitter.py          # Module 3: 3-way user-wise split
│   │   ├── popularity.py        # Module 4: Popularity baseline
│   │   ├── collaborative.py     # Module 5: SVD collaborative filtering
│   │   ├── content_based.py     # Module 6: TF-IDF content filtering
│   │   ├── hybrid.py            # Module 7+8: Hybrid engine + tie-breaking
│   │   ├── evaluator.py         # Module 9: Precision/Recall/NDCG@K
│   │   └── clustering.py        # Module 10: KMeans user segmentation
│   ├── api/
│   │   └── routes.py            # Module 11: FastAPI endpoints
│   ├── data/                    # processed_movies.csv
│   ├── models/                  # Saved ML models
│   ├── reports/                 # analytics.json, metrics.json, clusters.json
│   ├── main.py                  # FastAPI app entry point
│   ├── ratings.csv              # Raw MovieLens ratings
│   ├── movies.csv               # Raw MovieLens movies
│   ├── train.csv                # 63.5% — model training
│   ├── validation.csv           # 16.3% — hyperparameter tuning
│   ├── test.csv                 # 20.2% — holdout evaluation
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── LandingPage.jsx       # Module 12: Home + student info
│   │   │   ├── AnalyticsDashboard.jsx # Analytics dashboard
│   │   │   ├── EngagementDashboard.jsx # User engagement / clusters
│   │   │   ├── QualityDashboard.jsx   # Precision/Recall/NDCG gauges
│   │   │   └── RecommendPage.jsx      # Live recommendation search
│   │   ├── components/
│   │   │   ├── Layout.jsx             # Sidebar navigation
│   │   │   └── ui.jsx                 # Shared UI components
│   │   ├── api.js                     # Backend API service
│   │   ├── App.jsx                    # React Router
│   │   └── index.css                  # TailwindCSS v4 + custom CSS
│   └── package.json
└── README.md
```

---

## Split Pipeline

```
Merged Dataset (100,836 records)
        ↓
  User-wise Split
        ↓
┌────────────────────────┐
│ Train_Dev   (79.8%)    │  80,419 records
│ Holdout_Test (20.2%)   │  20,417 records  ← final eval only
└────────────────────────┘
        ↓
┌────────────────────────┐
│ Train       (63.5%)    │  64,081 records  ← model fitting
│ Validation  (16.2%)    │  16,338 records  ← tuning
└────────────────────────┘
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/recommend/{user_id}?n=10` | Top-N hybrid recommendations |
| GET | `/api/popular?n=10` | Popularity-based top movies |
| GET | `/api/movie/{movie_id}` | Movie details + similar movies |
| GET | `/api/analytics` | Full dataset analytics |
| GET | `/api/clusters` | User segmentation data |
| GET | `/api/metrics` | Precision@K / Recall@K / NDCG@K |

---

## Setup & Run

### Prerequisites
- Python ≥ 3.14 with [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 18

### Backend

```bash
cd backend

# Install dependencies
uv sync

# Start server (trains all models on startup, ~2-3 min first time)
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will:
1. Process `ratings.csv` + `movies.csv`
2. Run 3-way user-wise split → `train.csv`, `validation.csv`, `test.csv`
3. Train SVD, TF-IDF, Popularity models
4. Evaluate Precision@10, Recall@10, NDCG@10
5. Cluster users with KMeans
6. Serve all API endpoints at `http://localhost:8000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open → **http://localhost:5173**

---

## Evaluation Results

| Metric | Score |
|--------|-------|
| Precision@10 | ~6.2% |
| Recall@10 | ~5.1% |
| NDCG@10 | ~8.5% |

_Evaluated on 50 users from the holdout test split._

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Data | Pandas, NumPy, MovieLens 100K CSV |
| Collaborative Filtering | Surprise SVD |
| Content-Based | Scikit-Learn TF-IDF + Cosine Similarity |
| Clustering | Scikit-Learn KMeans |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite + TailwindCSS v4 |
| Charts | Recharts |
| Icons | Lucide React |

---

## Dashboard Pages

| Page | Route | Content |
|------|-------|---------|
| Landing | `/` | Project info, student card, module overview |
| Analytics | `/analytics` | Dataset stats, genre distribution, top movies |
| Engagement | `/engagement` | User clusters, activity levels, KMeans |
| Quality | `/quality` | Precision/Recall/NDCG gauges and charts |
| Recommend | `/recommend` | Live recommendation search by User ID |
