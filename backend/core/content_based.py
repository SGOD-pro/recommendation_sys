"""
Module 6 - Content-Based Recommendation
Uses TF-IDF on movie features (title + genres) and cosine similarity.

Phase 2 upgrade: Rating-weighted user profile
    profile = Σ(r × v) / Σ(r)  for items rated >= 4.0

Phase 3 upgrade: Genre Affinity scoring
    Per-user genre distribution built from training interactions.
    genre_affinity_score = dot(user_genre_prefs, movie_genre_vector)
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

    def __init__(
        self,
        profile_size: int | None = 50,
        profile_strategy: str = "weighted",
        liked_threshold: float = 4.0,
    ):
        """
        Args:
            profile_size: maximum number of liked movies to include in user
                          profile. Options: 20, 50, or None (all liked).
            profile_strategy: "weighted" for rating-weighted profiles or
                              "mean" for the legacy unweighted average.
            liked_threshold: minimum rating included in the improved profile.
        """
        if profile_strategy not in {"weighted", "mean"}:
            raise ValueError("profile_strategy must be 'weighted' or 'mean'")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = None
        self.movie_df = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.profile_size = profile_size
        self.profile_strategy = profile_strategy
        self.liked_threshold = liked_threshold

        # Phase 3: Genre Affinity internals
        self._all_genres: list[str] = []
        self._movie_genre_matrix: np.ndarray | None = None   # shape (n_movies, n_genres)
        self._genre_to_idx: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "ContentBasedRecommender":
        """Build TF-IDF matrix and genre index from movie features."""

        # Unique movies
        self.movie_df = (
            df.drop_duplicates("movieId")[["movieId", "title", "genres"]]
            .reset_index(drop=True)
        )

        # Feature text: clean title + genres
        self.movie_df["feature_text"] = (
            self.movie_df["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
            + " "
            + self.movie_df["genres"].str.replace("|", " ", regex=False)
        )

        # TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.movie_df["feature_text"]
        )

        # Index mappings
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        for idx, movie_id in enumerate(self.movie_df["movieId"].values):
            self.movie_id_to_idx[int(movie_id)] = idx
            self.idx_to_movie_id[idx] = int(movie_id)

        # Phase 3: build genre vocabulary and binary movie-genre matrix
        self._build_genre_index()

        print(f"[Content-Based] TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        return self

    # ------------------------------------------------------------------
    # Genre Affinity (Phase 3)
    # ------------------------------------------------------------------

    def _build_genre_index(self):
        """Build a binary movie × genre matrix for affinity scoring."""
        genre_sets = self.movie_df["genres"].str.split("|")
        all_genres = sorted({g for gs in genre_sets for g in gs if g != "(no genres listed)"})
        self._all_genres = all_genres
        self._genre_to_idx = {g: i for i, g in enumerate(all_genres)}

        n_movies = len(self.movie_df)
        n_genres = len(all_genres)
        mat = np.zeros((n_movies, n_genres), dtype=np.float32)
        for row_idx, (_, row) in enumerate(self.movie_df.iterrows()):
            for g in row["genres"].split("|"):
                if g in self._genre_to_idx:
                    mat[row_idx, self._genre_to_idx[g]] = 1.0
        # L1-normalise each row so every movie's genre vector sums to 1
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self._movie_genre_matrix = mat / row_sums

    def get_user_genre_profile(self, user_id: int, train_df: pd.DataFrame) -> np.ndarray:
        """
        Build a genre preference vector for a user.
        Weighted by rating (higher-rated movies contribute more).

        Returns an L1-normalised array of length n_genres.
        """
        user_movies = train_df[train_df["userId"] == user_id].copy()
        if user_movies.empty:
            return np.zeros(len(self._all_genres), dtype=np.float32)

        liked_movies = user_movies[user_movies["rating"] >= self.liked_threshold]
        if liked_movies.empty:
            return np.zeros(len(self._all_genres), dtype=np.float32)

        profile = np.zeros(len(self._all_genres), dtype=np.float32)
        for _, row in liked_movies.iterrows():
            mid = int(row["movieId"])
            rating = float(row["rating"])
            if mid in self.movie_id_to_idx:
                m_idx = self.movie_id_to_idx[mid]
                profile += rating * self._movie_genre_matrix[m_idx]

        total = profile.sum()
        if total > 0:
            profile /= total
        return profile

    def get_genre_affinity_scores(self, user_id: int, train_df: pd.DataFrame) -> dict:
        """
        Compute a genre affinity score ∈ [0, 1] for every unseen movie.

        score = dot(user_genre_profile, movie_genre_vector)

        Returns {movieId: score}
        """
        if self._movie_genre_matrix is None:
            return {}

        user_genre_vec = self.get_user_genre_profile(user_id, train_df)  # (n_genres,)
        if user_genre_vec.sum() == 0:
            return {}

        # Matrix multiply: (n_movies, n_genres) @ (n_genres,) → (n_movies,)
        all_scores = self._movie_genre_matrix @ user_genre_vec  # shape (n_movies,)

        # Normalise to [0, 1]
        max_s = all_scores.max()
        if max_s > 0:
            all_scores = all_scores / max_s

        # Build exclusion set
        rated_ids = set(int(m) for m in train_df[train_df["userId"] == user_id]["movieId"].values)

        scores = {}
        for idx, score in enumerate(all_scores):
            mid = self.idx_to_movie_id[idx]
            if mid not in rated_ids:
                scores[mid] = float(score)

        return scores

    # ------------------------------------------------------------------
    # Weighted User Profile (Phase 2)
    # ------------------------------------------------------------------

    def get_content_scores_for_user(self, user_id: int, train_df: pd.DataFrame) -> dict:
        """
        Compute content-based similarity scores for all unseen movies.

        Phase 2: rating-weighted TF-IDF user profile.
            profile = Σ(r × v) / Σ(r)   for liked movies (rating >= 4.0)
        """
        user_movies = train_df[train_df["userId"] == user_id].copy()
        if user_movies.empty:
            return {}

        if self.profile_strategy == "mean":
            liked = user_movies.nlargest(self.profile_size or len(user_movies), "rating")
        else:
            liked = user_movies[user_movies["rating"] >= self.liked_threshold]
            if liked.empty:
                return {}

        if self.profile_size is not None:
            liked = liked.nlargest(self.profile_size, "rating")

        # Weighted TF-IDF profile
        indices = []
        weights = []
        for _, row in liked.iterrows():
            mid = int(row["movieId"])
            if mid in self.movie_id_to_idx:
                indices.append(self.movie_id_to_idx[mid])
                weights.append(float(row["rating"]))

        if not indices:
            return {}

        if self.profile_strategy == "mean":
            weights_arr = np.ones(len(weights), dtype=np.float64)
        else:
            weights_arr = np.array(weights, dtype=np.float64)
        weight_sum = weights_arr.sum()
        if weight_sum == 0:
            return {}

        # Sparse weighted sum → dense profile
        vecs = self.tfidf_matrix[indices]           # (k, n_features) sparse
        # Weighted mean: multiply each row by its weight then divide by total
        user_profile = np.asarray(
            vecs.T.dot(weights_arr) / weight_sum      # (n_features,)
        ).reshape(1, -1)

        # Cosine similarity against all movies
        sim_scores = cosine_similarity(user_profile, self.tfidf_matrix).flatten()

        # Build exclusion set
        rated_ids = set(int(m) for m in user_movies["movieId"].values)

        scores = {}
        for idx, score in enumerate(sim_scores):
            mid = self.idx_to_movie_id[idx]
            if mid not in rated_ids:
                scores[mid] = float(score)

        return scores

    # ------------------------------------------------------------------
    # Utilities (unchanged API)
    # ------------------------------------------------------------------

    def get_similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """Find N most similar movies to the given movie."""
        if movie_id not in self.movie_id_to_idx:
            return pd.DataFrame(columns=["movieId", "title", "genres", "similarity_score"])

        idx = self.movie_id_to_idx[movie_id]
        sim_scores = cosine_similarity(
            self.tfidf_matrix[idx:idx + 1], self.tfidf_matrix
        ).flatten()
        top_indices = sim_scores.argsort()[::-1][1:n + 1]

        results = [
            {
                "movieId": self.idx_to_movie_id[i],
                "title": self.movie_df.iloc[i]["title"],
                "genres": self.movie_df.iloc[i]["genres"],
                "similarity_score": round(float(sim_scores[i]), 4),
            }
            for i in top_indices
        ]
        return pd.DataFrame(results)

    def get_similarity_between(self, movie_id_a: int, movie_id_b: int) -> float:
        """Get cosine similarity between two movies."""
        if movie_id_a not in self.movie_id_to_idx or movie_id_b not in self.movie_id_to_idx:
            return 0.0
        idx_a = self.movie_id_to_idx[movie_id_a]
        idx_b = self.movie_id_to_idx[movie_id_b]
        sim = cosine_similarity(
            self.tfidf_matrix[idx_a:idx_a + 1], self.tfidf_matrix[idx_b:idx_b + 1]
        )
        return float(sim[0][0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, filepath: str = None):
        """Save the content-based model."""
        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, "content_model.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.tfidf_vectorizer,
                    "matrix": self.tfidf_matrix,
                    "movie_df": self.movie_df,
                    "movie_id_to_idx": self.movie_id_to_idx,
                    "idx_to_movie_id": self.idx_to_movie_id,
                    "all_genres": self._all_genres,
                    "genre_to_idx": self._genre_to_idx,
                    "movie_genre_matrix": self._movie_genre_matrix,
                    "profile_size": self.profile_size,
                    "profile_strategy": self.profile_strategy,
                    "liked_threshold": self.liked_threshold,
                },
                f,
            )
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
        self._all_genres = data.get("all_genres", [])
        self._genre_to_idx = data.get("genre_to_idx", {})
        self._movie_genre_matrix = data.get("movie_genre_matrix", None)
        self.profile_size = data.get("profile_size", 50)
        self.profile_strategy = data.get("profile_strategy", "weighted")
        self.liked_threshold = data.get("liked_threshold", 4.0)
        print(f"[Content-Based] Model loaded from {filepath}")


if __name__ == "__main__":
    from .data_processor import get_processed_data

    df = get_processed_data()
    cb = ContentBasedRecommender()
    cb.fit(df)
    cb.save_model()

    print("\nMovies similar to Toy Story:")
    print(cb.get_similar_movies(1, 10).to_string(index=False))
