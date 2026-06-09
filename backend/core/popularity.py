"""
Module 4 - Popularity-Based Recommendation
Baseline recommender for cold-start users.
Score = average_rating × log(number_of_ratings)
"""

import pandas as pd
import numpy as np
import os
from .data_processor import get_processed_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class PopularityRecommender:
    """Popularity-based recommender for cold-start scenarios."""

    def __init__(self):
        self.popular_movies = None

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        """Compute popularity scores for all movies."""
        movie_stats = train_df.groupby(["movieId", "title", "genres"]).agg(
            avg_rating=("rating", "mean"),
            num_ratings=("rating", "count"),
        ).reset_index()

        # Score = avg_rating × log(num_ratings)
        # Using log to prevent movies with many low-quality ratings from dominating
        movie_stats["popularity_score"] = (
            movie_stats["avg_rating"] * np.log1p(movie_stats["num_ratings"])
        )

        # Sort by popularity score (descending), then by avg_rating, then num_ratings for tie-breaking
        movie_stats = movie_stats.sort_values(
            ["popularity_score", "avg_rating", "num_ratings"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        self.popular_movies = movie_stats
        print(f"[Popularity] Computed popularity scores for {len(movie_stats)} movies")
        return self

    def recommend(self, n: int = 10, exclude_movie_ids: list = None) -> pd.DataFrame:
        """Get top-N popular movies."""
        if self.popular_movies is None:
            raise ValueError("Model not fitted. Call fit() first.")

        recommendations = self.popular_movies.copy()
        if exclude_movie_ids:
            recommendations = recommendations[~recommendations["movieId"].isin(exclude_movie_ids)]

        return recommendations.head(n)[
            ["movieId", "title", "genres", "avg_rating", "num_ratings", "popularity_score"]
        ].reset_index(drop=True)

    def get_movie_popularity_score(self, movie_id: int) -> float:
        """Get popularity score for a specific movie."""
        if self.popular_movies is None:
            return 0.0
        match = self.popular_movies[self.popular_movies["movieId"] == movie_id]
        if match.empty:
            return 0.0
        return float(match.iloc[0]["popularity_score"])

    def get_all_scores(self) -> dict:
        """Get all popularity scores as a dictionary {movieId: score}."""
        if self.popular_movies is None:
            return {}
        # Normalize scores to [0, 1]
        max_score = self.popular_movies["popularity_score"].max()
        if max_score == 0:
            return {}
        scores = self.popular_movies.set_index("movieId")["popularity_score"] / max_score
        return scores.to_dict()


if __name__ == "__main__":
    from .splitter import get_train_test
    train_df, _ = get_train_test()
    recommender = PopularityRecommender()
    recommender.fit(train_df)
    print("\nTop 10 Popular Movies:")
    print(recommender.recommend(10).to_string(index=False))
