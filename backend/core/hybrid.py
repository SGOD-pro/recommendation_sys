"""
Module 7 & 8 - Hybrid Recommendation Engine with Tie Breaking
Combines collaborative, content-based, and popularity scores.
Final Score = 0.6 × CF + 0.3 × CB + 0.1 × Popularity
Tie-break: avg_rating then num_ratings
"""
import pandas as pd
import numpy as np
from .collaborative import CollaborativeFilteringRecommender
from .content_based import ContentBasedRecommender
from .popularity import PopularityRecommender


class HybridRecommender:
    # Personalised-first weighting: 70% SVD · 20% Content · 10% Popularity
    WEIGHT_CF  = 0.7
    WEIGHT_CB  = 0.2
    WEIGHT_POP = 0.1

    def __init__(self):
        self.cf = CollaborativeFilteringRecommender()
        self.cb = ContentBasedRecommender()
        self.pop = PopularityRecommender()
        self.train_df = None
        self.movie_stats = None
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "HybridRecommender":
        self.train_df = train_df.copy()
        self.movie_stats = train_df.groupby("movieId").agg(
            avg_rating=("rating", "mean"), num_ratings=("rating", "count")
        ).to_dict("index")
        print("[Hybrid] Training CF (SVD)...")
        self.cf.fit(train_df)
        print("[Hybrid] Training CB (TF-IDF)...")
        self.cb.fit(train_df)
        print("[Hybrid] Training Popularity...")
        self.pop.fit(train_df)
        self.is_fitted = True
        print("[Hybrid] All models trained!")
        return self

    def recommend(self, user_id: int, n: int = 10) -> list[dict]:
        if not self.is_fitted:
            raise ValueError("Not fitted")
        user_movies = self.train_df[self.train_df["userId"] == user_id]
        if user_movies.empty:
            pop_recs = self.pop.recommend(n)
            return [{"movieId": int(r["movieId"]), "title": r["title"], "genres": r["genres"],
                      "score": round(float(r["popularity_score"]), 4), "method": "popularity"} for _, r in pop_recs.iterrows()]

        cf_preds = self.cf.get_all_predictions_for_user(user_id)
        cb_scores = self.cb.get_content_scores_for_user(user_id, self.train_df)
        pop_scores = self.pop.get_all_scores()
        all_ids = set(cf_preds.keys()) | set(cb_scores.keys())
        if not all_ids:
            pop_recs = self.pop.recommend(n, list(user_movies["movieId"]))
            return [{"movieId": int(r["movieId"]), "title": r["title"], "genres": r["genres"],
                      "score": round(float(r["popularity_score"]), 4), "method": "popularity"} for _, r in pop_recs.iterrows()]

        # Normalize CF to [0,1]
        if cf_preds:
            vals = list(cf_preds.values())
            mn, mx = min(vals), max(vals)
            rng = mx - mn if mx != mn else 1.0
            cf_norm = {k: (v - mn) / rng for k, v in cf_preds.items()}
        else:
            cf_norm = {}

        scores = []
        for mid in all_ids:
            cf_s = cf_norm.get(mid, 0.0)
            cb_s = cb_scores.get(mid, 0.0)
            p_s = pop_scores.get(mid, 0.0)
            final = self.WEIGHT_CF * cf_s + self.WEIGHT_CB * cb_s + self.WEIGHT_POP * p_s
            st = self.movie_stats.get(mid, {"avg_rating": 0, "num_ratings": 0})
            scores.append({"movieId": int(mid), "score": final, "cf_score": cf_s,
                           "cb_score": cb_s, "pop_score": p_s,
                           "avg_rating": st["avg_rating"], "num_ratings": st["num_ratings"]})

        scores.sort(key=lambda x: (x["score"], x["avg_rating"], x["num_ratings"]), reverse=True)
        top = scores[:n]
        mi = self.cb.movie_df.set_index("movieId")
        results = []
        for it in top:
            m = it["movieId"]
            info = mi.loc[m] if m in mi.index else None
            results.append({"movieId": m, "title": info["title"] if info is not None else f"Movie {m}",
                            "genres": info["genres"] if info is not None else "Unknown",
                            "score": round(it["score"], 4), "cf_score": round(it["cf_score"], 4),
                            "cb_score": round(it["cb_score"], 4), "pop_score": round(it["pop_score"], 4), "method": "hybrid"})
        return results

    def get_movie_details(self, movie_id: int) -> dict | None:
        if self.cb.movie_df is None:
            return None
        match = self.cb.movie_df[self.cb.movie_df["movieId"] == movie_id]
        if match.empty:
            return None
        movie = match.iloc[0]
        st = self.movie_stats.get(movie_id, {"avg_rating": 0, "num_ratings": 0})
        similar = self.cb.get_similar_movies(movie_id, 5)
        return {"movieId": int(movie_id), "title": movie["title"], "genres": movie["genres"],
                "avg_rating": round(float(st.get("avg_rating", 0)), 2),
                "num_ratings": int(st.get("num_ratings", 0)),
                "similar_movies": similar.to_dict("records") if not similar.empty else []}
