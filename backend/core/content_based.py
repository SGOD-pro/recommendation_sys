"""
Module 6 - Content-Based Recommendation
Uses TF-IDF on movie features (title + genres) and cosine similarity
to find similar movies.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class ContentBasedRecommender:
    """Content-based recommender using TF-IDF and cosine similarity."""

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = None
        self.movie_df = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}

    def fit(self, df: pd.DataFrame) -> "ContentBasedRecommender":
        """Build TF-IDF matrix from movie features."""
        # Get unique movies
        self.movie_df = df.drop_duplicates("movieId")[["movieId", "title", "genres"]].reset_index(drop=True)

        # Create feature text: title + genres (replace | with space)
        self.movie_df["feature_text"] = (
            self.movie_df["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
            + " "
            + self.movie_df["genres"].str.replace("|", " ", regex=False)
        )

        # Build TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movie_df["feature_text"])

        # Create index mappings
        for idx, movie_id in enumerate(self.movie_df["movieId"].values):
            self.movie_id_to_idx[int(movie_id)] = idx
            self.idx_to_movie_id[idx] = int(movie_id)

        print(f"[Content-Based] TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        return self

    def get_similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """Find N most similar movies to the given movie."""
        if movie_id not in self.movie_id_to_idx:
            return pd.DataFrame(columns=["movieId", "title", "genres", "similarity_score"])

        idx = self.movie_id_to_idx[movie_id]

        # Compute cosine similarity of this movie with all others
        sim_scores = cosine_similarity(
            self.tfidf_matrix[idx:idx+1], self.tfidf_matrix
        ).flatten()

        # Get top N+1 (excluding the movie itself)
        top_indices = sim_scores.argsort()[::-1][1:n+1]

        results = []
        for i in top_indices:
            results.append({
                "movieId": self.idx_to_movie_id[i],
                "title": self.movie_df.iloc[i]["title"],
                "genres": self.movie_df.iloc[i]["genres"],
                "similarity_score": round(float(sim_scores[i]), 4),
            })

        return pd.DataFrame(results)

    def get_content_scores_for_user(self, user_id: int, train_df: pd.DataFrame) -> dict:
        """
        Get content-based scores for unseen movies based on user's history.
        Uses the average similarity to user's top-rated movies.
        """
        # Get user's rated movies (top rated)
        user_movies = train_df[train_df["userId"] == user_id].sort_values("rating", ascending=False)

        if user_movies.empty:
            return {}

        # Take the user's top 20 rated movies for profile
        top_movies = user_movies.head(20)
        user_movie_ids = top_movies["movieId"].values

        # Get indices for user's movies
        user_indices = []
        for mid in user_movie_ids:
            if int(mid) in self.movie_id_to_idx:
                user_indices.append(self.movie_id_to_idx[int(mid)])

        if not user_indices:
            return {}

        # Compute average user profile vector
        user_profile = self.tfidf_matrix[user_indices].mean(axis=0)
        user_profile = np.asarray(user_profile)

        # Compute similarity with all movies
        sim_scores = cosine_similarity(user_profile, self.tfidf_matrix).flatten()

        # Create score dictionary (exclude user's already-rated movies)
        rated_movie_ids = set(int(m) for m in train_df[train_df["userId"] == user_id]["movieId"].values)
        scores = {}
        for idx, score in enumerate(sim_scores):
            movie_id = self.idx_to_movie_id[idx]
            if movie_id not in rated_movie_ids:
                scores[movie_id] = float(score)

        return scores

    def get_similarity_between(self, movie_id_a: int, movie_id_b: int) -> float:
        """Get cosine similarity between two movies."""
        if movie_id_a not in self.movie_id_to_idx or movie_id_b not in self.movie_id_to_idx:
            return 0.0
        idx_a = self.movie_id_to_idx[movie_id_a]
        idx_b = self.movie_id_to_idx[movie_id_b]
        sim = cosine_similarity(
            self.tfidf_matrix[idx_a:idx_a+1], self.tfidf_matrix[idx_b:idx_b+1]
        )
        return float(sim[0][0])

    def save_model(self, filepath: str = None):
        """Save the content-based model."""
        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, "content_model.pkl")
        with open(filepath, "wb") as f:
            pickle.dump({
                "vectorizer": self.tfidf_vectorizer,
                "matrix": self.tfidf_matrix,
                "movie_df": self.movie_df,
                "movie_id_to_idx": self.movie_id_to_idx,
                "idx_to_movie_id": self.idx_to_movie_id,
            }, f)
        print(f"[Content-Based] Model saved to {filepath}")

    def load_model(self, filepath: str = None):
        """Load a saved content-based model."""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "content_model.pkl")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.tfidf_vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["matrix"]
        self.movie_df = data["movie_df"]
        self.movie_id_to_idx = data["movie_id_to_idx"]
        self.idx_to_movie_id = data["idx_to_movie_id"]
        print(f"[Content-Based] Model loaded from {filepath}")


if __name__ == "__main__":
    from .data_processor import get_processed_data
    df = get_processed_data()
    cb = ContentBasedRecommender()
    cb.fit(df)
    cb.save_model()

    # Find similar movies to Toy Story (movieId=1)
    print("\nMovies similar to Toy Story:")
    print(cb.get_similar_movies(1, 10).to_string(index=False))
