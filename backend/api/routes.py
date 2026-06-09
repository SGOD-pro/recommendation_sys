"""
Module 11 - FastAPI Routes
All API endpoints for the recommendation system.
"""
from fastapi import APIRouter, HTTPException

router = APIRouter()

# These will be set by main.py after models are trained
_hybrid = None
_analytics = None
_clusters = None
_metrics = None


def init_routes(hybrid, analytics_data, cluster_data, metrics_data):
    global _hybrid, _analytics, _clusters, _metrics
    _hybrid = hybrid
    _analytics = analytics_data
    _clusters = cluster_data
    _metrics = metrics_data


@router.get("/recommend/{user_id}")
def get_recommendations(user_id: int, n: int = 10):
    if _hybrid is None:
        raise HTTPException(503, "Models not loaded")
    try:
        recs = _hybrid.recommend(user_id, n)
        return {"user_id": user_id, "count": len(recs), "recommendations": recs}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/popular")
def get_popular(n: int = 10):
    if _hybrid is None:
        raise HTTPException(503, "Models not loaded")
    recs = _hybrid.pop.recommend(n)
    return {"count": len(recs), "movies": recs.to_dict("records")}


@router.get("/movie/{movie_id}")
def get_movie(movie_id: int):
    if _hybrid is None:
        raise HTTPException(503, "Models not loaded")
    details = _hybrid.get_movie_details(movie_id)
    if details is None:
        raise HTTPException(404, "Movie not found")
    return details


@router.get("/analytics")
def get_analytics():
    if _analytics is None:
        raise HTTPException(503, "Analytics not loaded")
    return _analytics


@router.get("/clusters")
def get_clusters():
    if _clusters is None:
        raise HTTPException(503, "Clusters not loaded")
    return _clusters


@router.get("/metrics")
def get_metrics():
    if _metrics is None:
        raise HTTPException(503, "Metrics not loaded")
    return _metrics
