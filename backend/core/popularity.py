"""
Module 4 - Popularity-Based Recommendation.

Supports both the original popularity score and the Phase 4 Bayesian average
score so experiments can compare them honestly.
"""

import pandas as pd
import numpy as np
import os
from .data_processor import get_processed_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class PopularityRecommender:
    """Popularity-based recommender.

    scoring_method:
        "legacy"   -> avg_rating * log1p(num_ratings)
        "bayesian" -> (v/(v+m))*R + (m/(v+m))*C
    """

    def __init__(
        self,
        scoring_method: str = "bayesian",
        m_threshold: float | None = None,
        m_percentile: float = 95.0,
    ):
        """
        Args:
            scoring_method: "legacy" or "bayesian".
            m_threshold: explicit Bayesian minimum-votes threshold. When None,
                         it is derived from m_percentile.
            m_percentile: percentile of vote counts used as the minimum-votes
                          threshold m in the Bayesian Average formula.
        """
        self.popular_movies = None
        if scoring_method not in {"legacy", "bayesian"}:
            raise ValueError("scoring_method must be 'legacy' or 'bayesian'")
        self.scoring_method = scoring_method
        self.m_threshold_override = m_threshold
        self.m_percentile = m_percentile
        self.global_mean: float = 0.0
        self.m_threshold: float = 0.0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        """Compute popularity scores for all movies."""

        movie_stats = (
            train_df
            .groupby(["movieId", "title", "genres"])
            .agg(avg_rating=("rating", "mean"), num_ratings=("rating", "count"))
            .reset_index()
        )

        self.global_mean = float(train_df["rating"].mean())
        if self.m_threshold_override is None:
            self.m_threshold = float(
                np.percentile(movie_stats["num_ratings"].values, self.m_percentile)
            )
        else:
            self.m_threshold = float(self.m_threshold_override)

        C = self.global_mean
        m = self.m_threshold

        v = movie_stats["num_ratings"]
        R = movie_stats["avg_rating"]
        movie_stats["legacy_popularity_score"] = R * np.log1p(v)
        movie_stats["bayesian_popularity_score"] = (v / (v + m)) * R + (m / (v + m)) * C
        if self.scoring_method == "legacy":
            movie_stats["popularity_score"] = movie_stats["legacy_popularity_score"]
        else:
            movie_stats["popularity_score"] = movie_stats["bayesian_popularity_score"]

        movie_stats = movie_stats.sort_values(
            ["popularity_score", "avg_rating", "num_ratings"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        self.popular_movies = movie_stats
        print(
            f"[Popularity] {self.scoring_method}  |  C={C:.3f}  m={m:.0f}  "
            f"movies={len(movie_stats)}"
        )
        return self

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------

    def recommend(self, n: int = 10, exclude_movie_ids: list = None) -> pd.DataFrame:
        """Get top-N popular movies."""
        if self.popular_movies is None:
            raise ValueError("Model not fitted. Call fit() first.")

        recommendations = self.popular_movies.copy()
        if exclude_movie_ids:
            recommendations = recommendations[
                ~recommendations["movieId"].isin(exclude_movie_ids)
            ]

        return recommendations.head(n)[
            ["movieId", "title", "genres", "avg_rating", "num_ratings", "popularity_score"]
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Score lookups
    # ------------------------------------------------------------------

    def get_movie_popularity_score(self, movie_id: int) -> float:
        """Get raw Bayesian popularity score for a specific movie."""
        if self.popular_movies is None:
            return 0.0
        match = self.popular_movies[self.popular_movies["movieId"] == movie_id]
        if match.empty:
            return 0.0
        return float(match.iloc[0]["popularity_score"])

    def get_all_scores(self) -> dict:
        """Get all popularity scores normalised to [0, 1]."""
        if self.popular_movies is None:
            return {}
        max_score = self.popular_movies["popularity_score"].max()
        if max_score == 0:
            return {}
        scores = (
            self.popular_movies.set_index("movieId")["popularity_score"] / max_score
        )
        return scores.to_dict()


if __name__ == "__main__":
    from .splitter import get_train_test
    train_df, _ = get_train_test()
    recommender = PopularityRecommender()
    recommender.fit(train_df)
    print("\nTop 10 Popular Movies:")
    print(recommender.recommend(10).to_string(index=False))
