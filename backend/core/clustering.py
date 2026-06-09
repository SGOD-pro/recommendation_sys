"""
Module 10 - User Segmentation Analytics
KMeans clustering on user profiles for analytics purposes.
Features: avg_rating, genre preferences, num_ratings, activity_level
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class UserSegmentation:
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.user_profiles = None
        self.cluster_summaries = None

    def _build_user_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature vectors for each user."""
        # Basic stats per user
        user_stats = df.groupby("userId").agg(
            avg_rating=("rating", "mean"),
            num_ratings=("rating", "count"),
            rating_std=("rating", "std"),
        ).reset_index()
        user_stats["rating_std"] = user_stats["rating_std"].fillna(0)

        # Genre preferences: proportion of each genre per user
        genre_df = df.copy()
        genre_df["genres"] = genre_df["genres"].str.split("|")
        genre_exploded = genre_df.explode("genres")
        genre_exploded = genre_exploded[genre_exploded["genres"] != "(no genres listed)"]

        # Get top genres
        top_genres = genre_exploded["genres"].value_counts().head(10).index.tolist()

        # Create genre proportion features
        for genre in top_genres:
            genre_counts = genre_exploded[genre_exploded["genres"] == genre].groupby("userId").size()
            total_counts = genre_exploded.groupby("userId").size()
            col_name = f"genre_{genre.lower().replace('-', '_')}"
            user_stats[col_name] = user_stats["userId"].map(
                (genre_counts / total_counts).to_dict()
            ).fillna(0)

        # Activity level (quantile-based)
        user_stats["activity_level"] = pd.qcut(
            user_stats["num_ratings"], q=4, labels=[1, 2, 3, 4], duplicates="drop"
        ).astype(float)

        return user_stats

    def fit(self, df: pd.DataFrame) -> "UserSegmentation":
        """Build user profiles and cluster them."""
        self.user_profiles = self._build_user_profiles(df)
        feature_cols = [c for c in self.user_profiles.columns if c != "userId"]
        X = self.user_profiles[feature_cols].values

        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.user_profiles["cluster"] = self.kmeans.fit_predict(X_scaled)

        self._generate_summaries(df)
        print(f"[Clustering] {self.n_clusters} clusters created")
        return self

    def _generate_summaries(self, df: pd.DataFrame):
        """Generate human-readable summaries for each cluster."""
        summaries = []
        genre_cols = [c for c in self.user_profiles.columns if c.startswith("genre_")]

        for cluster_id in range(self.n_clusters):
            members = self.user_profiles[self.user_profiles["cluster"] == cluster_id]
            n = len(members)
            avg_r = round(float(members["avg_rating"].mean()), 2)
            avg_n = round(float(members["num_ratings"].mean()), 1)

            # Top genre for this cluster
            if genre_cols:
                genre_avgs = members[genre_cols].mean()
                top_genre = genre_avgs.idxmax().replace("genre_", "").replace("_", " ").title()
            else:
                top_genre = "Mixed"

            # Name the cluster
            labels = {0: "Casual Viewers", 1: "Sci-Fi Enthusiasts", 2: "Drama Lovers",
                      3: "Comedy Fans", 4: "Action Buffs"}
            name = labels.get(cluster_id, f"Cluster {cluster_id}")

            summaries.append({
                "cluster_id": cluster_id, "name": name, "size": int(n),
                "avg_rating": avg_r, "avg_num_ratings": avg_n,
                "top_genre": top_genre,
                "percentage": round(n / len(self.user_profiles) * 100, 1),
            })
        self.cluster_summaries = summaries

    def get_clusters(self) -> dict:
        if self.cluster_summaries is None:
            return {"clusters": [], "total_users": 0}
        distribution = self.user_profiles["cluster"].value_counts().sort_index()
        return {
            "clusters": self.cluster_summaries,
            "total_users": int(len(self.user_profiles)),
            "cluster_distribution": {"labels": [s["name"] for s in self.cluster_summaries],
                                     "values": distribution.values.tolist()},
            "activity_distribution": {
                "labels": ["Low", "Medium", "High", "Very High"],
                "values": self.user_profiles["activity_level"].value_counts().sort_index().values.tolist(),
            },
        }
