"""
Module 2 - User Interaction Analysis
Generates comprehensive analytics from the processed dataset:
- Dataset statistics
- User statistics  
- Movie statistics
- Genre statistics
"""

import pandas as pd
import numpy as np
from .data_processor import get_processed_data


def get_dataset_statistics(df: pd.DataFrame) -> dict:
    """Get basic dataset statistics."""
    return {
        "total_users": int(df["userId"].nunique()),
        "total_movies": int(df["movieId"].nunique()),
        "total_ratings": int(len(df)),
        "rating_range": {"min": float(df["rating"].min()), "max": float(df["rating"].max())},
        "avg_rating": round(float(df["rating"].mean()), 2),
        "median_rating": float(df["rating"].median()),
    }


def get_user_statistics(df: pd.DataFrame) -> dict:
    """Get user-level statistics."""
    user_ratings = df.groupby("userId").agg(
        num_ratings=("rating", "count"),
        avg_rating=("rating", "mean"),
    ).reset_index()

    # Most active users (by number of ratings)
    most_active = (
        user_ratings.nlargest(10, "num_ratings")
        .rename(columns={"num_ratings": "total_ratings", "avg_rating": "average_rating"})
    )
    most_active["average_rating"] = most_active["average_rating"].round(2)

    # Ratings per user distribution
    ratings_per_user = user_ratings["num_ratings"].describe().to_dict()
    ratings_per_user = {k: round(float(v), 2) for k, v in ratings_per_user.items()}

    return {
        "most_active_users": most_active.to_dict("records"),
        "average_ratings_per_user": round(float(user_ratings["num_ratings"].mean()), 2),
        "ratings_per_user_distribution": ratings_per_user,
    }


def get_movie_statistics(df: pd.DataFrame) -> dict:
    """Get movie-level statistics."""
    movie_stats = df.groupby(["movieId", "title"]).agg(
        num_ratings=("rating", "count"),
        avg_rating=("rating", "mean"),
    ).reset_index()

    # Most rated movies
    most_rated = movie_stats.nlargest(10, "num_ratings").copy()
    most_rated["avg_rating"] = most_rated["avg_rating"].round(2)

    # Highest rated movies (minimum 50 ratings to avoid bias)
    qualified = movie_stats[movie_stats["num_ratings"] >= 50].copy()
    highest_rated = qualified.nlargest(10, "avg_rating")
    highest_rated["avg_rating"] = highest_rated["avg_rating"].round(2)

    return {
        "most_rated_movies": most_rated[["title", "num_ratings", "avg_rating"]].to_dict("records"),
        "highest_rated_movies": highest_rated[["title", "num_ratings", "avg_rating"]].to_dict("records"),
    }


def get_genre_statistics(df: pd.DataFrame) -> dict:
    """Get genre-level statistics."""
    # Explode genres (pipe-separated)
    genre_df = df.copy()
    genre_df["genres"] = genre_df["genres"].str.split("|")
    genre_exploded = genre_df.explode("genres")

    # Filter out '(no genres listed)'
    genre_exploded = genre_exploded[genre_exploded["genres"] != "(no genres listed)"]

    # Genre popularity (by count of ratings)
    genre_counts = genre_exploded.groupby("genres").agg(
        total_ratings=("rating", "count"),
        avg_rating=("rating", "mean"),
        num_movies=("movieId", "nunique"),
    ).reset_index()
    genre_counts["avg_rating"] = genre_counts["avg_rating"].round(2)

    # Sort by total ratings
    genre_counts = genre_counts.sort_values("total_ratings", ascending=False)

    return {
        "genre_distribution": genre_counts.to_dict("records"),
        "most_popular_genres": genre_counts.head(10).to_dict("records"),
        "total_genres": int(genre_counts.shape[0]),
    }


def get_rating_distribution(df: pd.DataFrame) -> dict:
    """Get distribution of rating values."""
    dist = df["rating"].value_counts().sort_index()
    return {
        "labels": [str(r) for r in dist.index.tolist()],
        "values": dist.values.tolist(),
    }


def get_full_analytics() -> dict:
    """Generate all analytics for the dashboard."""
    df = get_processed_data()

    return {
        "dataset_stats": get_dataset_statistics(df),
        "user_stats": get_user_statistics(df),
        "movie_stats": get_movie_statistics(df),
        "genre_stats": get_genre_statistics(df),
        "rating_distribution": get_rating_distribution(df),
    }


if __name__ == "__main__":
    import json
    analytics = get_full_analytics()
    print(json.dumps(analytics, indent=2))
