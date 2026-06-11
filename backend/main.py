"""
Hybrid AI Recommendation System - FastAPI Backend
Main entry point that initializes all models and starts the API server.
"""
import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.data_processor import load_and_process_data, get_processed_data
from core.analytics import get_full_analytics
from core.splitter import create_three_way_split, get_train_val_test
from core.hybrid import HybridRecommender
from core.evaluator import RecommendationEvaluator
from core.clustering import UserSegmentation
from api.routes import router, init_routes

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all models on startup."""
    print("=" * 60)
    print("  Hybrid AI Recommendation System")
    print("  Weights: 70% SVD · 20% Content · 10% Popularity")
    print("  Split  : Chronological (timestamp-based)")
    print("="* 60)

    # Delete any stale random-split CSVs so the new chronological split runs fresh
    for fname in ["train.csv", "validation.csv", "test.csv"]:
        fpath = os.path.join(os.path.dirname(__file__), fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    # Module 1: Data Processing
    print("\n[1/6] Processing data...")
    df = load_and_process_data()

    # Module 2: Analytics
    print("\n[2/6] Generating analytics...")
    analytics_data = get_full_analytics()

    # Module 3: 3-Way Train/Val/Test Split
    print("\n[3/6] Creating train / validation / holdout-test split...")
    train_df, validation_df, test_df = create_three_way_split()
    print(f"  train={len(train_df):,}  val={len(validation_df):,}  test={len(test_df):,}")

    # Modules 4-8: Train Hybrid Recommender (on train_df only)
    print("\n[4/6] Training recommendation models (on train split)...")
    hybrid = HybridRecommender(
        weight_cf=0.4,
        weight_cb=0.1,
        weight_pop=0.5,
        content_profile_strategy="weighted",
        popularity_method="bayesian",
    )
    hybrid.fit(train_df)

    # Module 9: Evaluation (against holdout test split)
    print("\n[5/6] Evaluating recommendations (on holdout test split)...")
    evaluator = RecommendationEvaluator(hybrid, train_df, test_df)
    # n_users=9999 → evaluates every eligible user in the holdout test set
    metrics = evaluator.evaluate(k=10, n_users=9999)

    # Module 10: User Clustering
    print("\n[6/6] Clustering users...")
    segmentation = UserSegmentation(n_clusters=5)
    segmentation.fit(df)
    cluster_data = segmentation.get_clusters()

    # Save reports
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "analytics.json"), "w") as f:
        json.dump(analytics_data, f, indent=2)
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(REPORTS_DIR, "clusters.json"), "w") as f:
        json.dump(cluster_data, f, indent=2)

    # Initialize routes with trained models
    init_routes(hybrid, analytics_data, cluster_data, metrics)

    print("\n" + "=" * 60)
    print("  Server ready! All models trained.")
    print("=" * 60)
    yield
    print("Shutting down...")


app = FastAPI(
    title="Hybrid AI Recommendation System",
    description="Personalized movie recommendations using Collaborative Filtering, Content-Based Filtering, and Popularity",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"message": "Hybrid AI Recommendation System API", "version": "1.0.0",
            "endpoints": ["/api/recommend/{user_id}", "/api/popular", "/api/movie/{movie_id}",
                          "/api/analytics", "/api/clusters", "/api/metrics"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
