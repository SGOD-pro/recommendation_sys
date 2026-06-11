"""
Module 7b - XGBoost Learning-to-Rank Model (Redesigned)

Phase 5  – XGBRanker with objective="rank:ndcg".
           Label: 1 if rating >= 4, else 0.
           Ranking by predicted relevance score.

Phase 6  – Negative sampling: 5 unseen movies per positive interaction.
           Strict leakage guard: val/test movies per user are never in negatives.

Phase 7  – 11-feature advanced feature set:
    1.  SVD predicted rating
    2.  Content-Based cosine similarity (rating-weighted profile)
    3.  Bayesian popularity score (normalised)
    4.  Genre affinity score
    5.  User average rating
    6.  Movie average rating
    7.  User rating count
    8.  Movie rating count
    9.  SVD latent dot product  (u · i)
    10. User latent factor norm ‖u‖₂
    11. Item latent factor norm ‖i‖₂

Choice rationale (XGBClassifier vs XGBRanker):
    XGBRanker is the better fit for recommendation ranking because the loss is
    query-aware: each user is a query group and the model directly optimizes
    top-of-list ordering (NDCG). XGBClassifier is simpler and can work well,
    but it optimizes pointwise probability calibration rather than the relative
    ordering among candidates for the same user.
"""

import os
import random
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

FEATURE_NAMES = [
    "cf_pred",
    "cb_sim",
    "pop_score",
    "genre_affinity",
    "user_avg_rating",
    "movie_avg_rating",
    "user_rating_count",
    "movie_rating_count",
    "svd_dot_product",
    "user_factor_norm",
    "item_factor_norm",
]


class XGBoostRecommender:
    """XGBoost LambdaMART-style learning-to-rank recommender."""

    def __init__(
        self,
        n_estimators:  int   = 300,
        max_depth:     int   = 6,
        learning_rate: float = 0.05,
        neg_samples:   int   = 5,
        relevance_threshold: float = 4.0,
    ):
        """
        Args:
            n_estimators:          XGBoost trees.
            max_depth:             tree depth.
            learning_rate:         step size.
            neg_samples:           negative samples per positive interaction.
            relevance_threshold:   rating cutoff for positive label.
        """
        try:
            from .collaborative import CollaborativeFilteringRecommender
            from .content_based import ContentBasedRecommender
            from .popularity import PopularityRecommender
        except ImportError:
            from core.collaborative import CollaborativeFilteringRecommender
            from core.content_based import ContentBasedRecommender
            from core.popularity import PopularityRecommender

        self.cf  = CollaborativeFilteringRecommender()
        self.cb  = ContentBasedRecommender()
        self.pop = PopularityRecommender()

        self.neg_samples          = neg_samples
        self.relevance_threshold  = relevance_threshold

        self.model = xgb.XGBRanker(
            objective      = "rank:ndcg",
            n_estimators   = n_estimators,
            max_depth      = max_depth,
            learning_rate  = learning_rate,
            subsample      = 0.8,
            colsample_bytree = 0.8,
            eval_metric    = "ndcg@10",
            random_state   = 42,
        )

        self.train_df          = None
        self.user_profiles     = {}    # {userId: np.ndarray (1, n_features)}
        self.movie_stats       = {}    # {movieId: {movie_avg_rating, movie_rating_count}}
        self.user_stats        = {}    # {userId:  {user_avg_rating,  user_rating_count}}
        self.pop_scores        = {}    # {movieId: float [0,1]}
        self.movie_details_dict = {}  # {movieId: {title, genres}}
        self.all_movie_ids     = []   # sorted list for fast negative sampling
        self.is_fitted         = False

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _svd_factors(self, user_id: int, movie_id: int):
        """Return (u_vec, i_vec) from the SVD trainset; (None, None) on miss."""
        ts = self.cf.model.trainset
        try:
            iu = ts.to_inner_uid(user_id)
            u  = self.cf.model.pu[iu]
        except ValueError:
            u = None
        try:
            ii = ts.to_inner_iid(movie_id)
            i  = self.cf.model.qi[ii]
        except ValueError:
            i = None
        return u, i

    def _build_feature_row(
        self,
        user_id: int,
        movie_id: int,
        u_prof: np.ndarray | None,
        genre_vec: np.ndarray | None,
        u_stats: dict,
        global_mean: float,
    ) -> list[float]:
        """Build one feature vector for (user_id, movie_id)."""

        # 1. CF predicted rating
        cf_pred = self.cf.predict(user_id, movie_id)

        # 2. Content-Based cosine similarity
        cb_sim = 0.0
        if u_prof is not None and movie_id in self.cb.movie_id_to_idx:
            m_idx = self.cb.movie_id_to_idx[movie_id]
            m_vec = self.cb.tfidf_matrix[m_idx]
            cb_sim = float(cosine_similarity(u_prof, m_vec)[0][0])

        # 3. Bayesian popularity score
        pop_score = self.pop_scores.get(movie_id, 0.0)

        # 4. Genre affinity
        ga = 0.0
        if genre_vec is not None and movie_id in self.cb.movie_id_to_idx:
            m_idx = self.cb.movie_id_to_idx[movie_id]
            ga = float(np.dot(self.cb._movie_genre_matrix[m_idx], genre_vec))
            # Normalise by max possible (sum of genre_vec weights)
            denom = genre_vec.sum()
            if denom > 0:
                ga /= denom

        # 5 & 6. User and Movie stats
        m_stats = self.movie_stats.get(
            movie_id, {"movie_avg_rating": global_mean, "movie_rating_count": 0}
        )

        # 9-11. SVD latent factors
        u_vec, i_vec = self._svd_factors(user_id, movie_id)
        dot_product   = float(np.dot(u_vec, i_vec)) if (u_vec is not None and i_vec is not None) else 0.0
        u_norm        = float(np.linalg.norm(u_vec)) if u_vec is not None else 0.0
        i_norm        = float(np.linalg.norm(i_vec)) if i_vec is not None else 0.0

        return [
            cf_pred,
            cb_sim,
            pop_score,
            ga,
            u_stats["user_avg_rating"],
            m_stats["movie_avg_rating"],
            u_stats["user_rating_count"],
            m_stats["movie_rating_count"],
            dot_product,
            u_norm,
            i_norm,
        ]

    # ------------------------------------------------------------------
    # Phase 6: Negative Sampling
    # ------------------------------------------------------------------

    def _generate_negatives(
        self,
        train_df: pd.DataFrame,
        holdout_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        For every user, sample `neg_samples` unseen movies per positive interaction.

        Leakage guard:
            - Exclude movies the user rated in train_df (positives).
            - Exclude movies appearing in holdout_df for that user (val/test leak).

        Returns a DataFrame with [userId, movieId, label=0].
        """
        rng = random.Random(42)
        all_ids_set = set(self.all_movie_ids)
        neg_rows = []

        for user_id, group in train_df.groupby("userId"):
            observed_ids = set(int(m) for m in group["movieId"].values)
            positive_ids = set(
                int(m)
                for m in group[group["rating"] >= self.relevance_threshold]["movieId"].values
            )
            if not positive_ids:
                continue

            # Strict leakage guard
            forbidden = observed_ids.copy()
            if holdout_df is not None:
                user_holdout = holdout_df[holdout_df["userId"] == user_id]
                forbidden |= set(int(m) for m in user_holdout["movieId"].values)

            candidates = list(all_ids_set - forbidden)
            if not candidates:
                continue

            n_neg = min(len(positive_ids) * self.neg_samples, len(candidates))
            sampled = rng.sample(candidates, n_neg)
            for mid in sampled:
                neg_rows.append({"userId": user_id, "movieId": mid, "label": 0})

        return pd.DataFrame(neg_rows)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df:  pd.DataFrame,
        holdout_df: pd.DataFrame | None = None,
    ) -> "XGBoostRecommender":
        """
        Fit all sub-models and train the XGBoost ranker.

        Args:
            train_df:   training interactions (userId, movieId, rating).
            holdout_df: val + test DataFrame merged; used for leakage guard.
        """
        self.train_df       = train_df.copy()
        self.all_movie_ids  = sorted(int(m) for m in train_df["movieId"].unique())

        # ── Sub-models ──────────────────────────────────────────────────
        print("[XGBoost] Fitting Collaborative model (SVD)...")
        self.cf.fit(train_df)

        print("[XGBoost] Fitting Content-Based model (TF-IDF + Genre)...")
        self.cb.fit(train_df)
        self.movie_details_dict = (
            self.cb.movie_df.set_index("movieId")[["title", "genres"]].to_dict("index")
        )

        print("[XGBoost] Fitting Popularity model (Bayesian)...")
        self.pop.fit(train_df)
        self.pop_scores = self.pop.get_all_scores()

        # ── User profiles (rating-weighted TF-IDF) ─────────────────────
        print("[XGBoost] Computing weighted user TF-IDF profiles...")
        global_mean = float(train_df["rating"].mean())

        for user_id, group in train_df.groupby("userId"):
            liked = group[group["rating"] >= 4.0]
            if liked.empty:
                liked = group
            liked = liked.nlargest(50, "rating")

            indices = [self.cb.movie_id_to_idx[int(m)] for m in liked["movieId"].values
                       if int(m) in self.cb.movie_id_to_idx]
            weights  = [float(r) for m, r in zip(liked["movieId"].values, liked["rating"].values)
                        if int(m) in self.cb.movie_id_to_idx]

            if indices:
                w = np.array(weights)
                vecs = self.cb.tfidf_matrix[indices]
                self.user_profiles[user_id] = np.asarray(
                    vecs.T.dot(w) / w.sum()
                ).reshape(1, -1)
            else:
                self.user_profiles[user_id] = None

        # ── Rating statistics ────────────────────────────────────────────
        print("[XGBoost] Computing movie and user statistics...")
        self.movie_stats = train_df.groupby("movieId").agg(
            movie_avg_rating=("rating", "mean"),
            movie_rating_count=("rating", "count"),
        ).to_dict("index")

        self.user_stats = train_df.groupby("userId").agg(
            user_avg_rating=("rating", "mean"),
            user_rating_count=("rating", "count"),
        ).to_dict("index")

        # Observed samples: liked ratings are relevant, lower ratings are explicit negatives.
        observed_df = train_df[["userId", "movieId", "rating"]].copy()
        observed_df["label"] = (
            observed_df["rating"] >= self.relevance_threshold
        ).astype(int)

        # ── Negative samples (Phase 6) ────────────────────────────────────
        print(f"[XGBoost] Generating {self.neg_samples}× negative samples...")
        neg_df = self._generate_negatives(train_df, holdout_df)

        # ── Build feature matrix ──────────────────────────────────────────
        print("[XGBoost] Extracting features for all samples...")

        sample_df = pd.concat(
            [
                observed_df[["userId", "movieId", "label"]],
                neg_df[["userId", "movieId", "label"]] if not neg_df.empty else neg_df,
            ],
            ignore_index=True,
        ).sort_values(["userId", "label"], ascending=[True, False])

        rows_X = []
        rows_y = []
        groups = []

        for uid, group in sample_df.groupby("userId", sort=False):
            uid = int(uid)
            candidate_ids = [int(mid) for mid in group["movieId"].values]
            X_group = self._build_candidate_matrix(uid, candidate_ids, global_mean)
            groups.append(len(group))
            rows_X.extend(X_group.tolist())
            rows_y.extend(group["label"].astype(int).tolist())

        X_train = pd.DataFrame(rows_X, columns=FEATURE_NAMES, dtype=np.float32)
        y_train = np.array(rows_y, dtype=np.float32)

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos

        print(
            f"[XGBoost] Training XGBRanker on {len(X_train):,} samples "
            f"({n_pos:,} relevant / {n_neg:,} non-relevant)  "
            f"groups={len(groups):,}  features={X_train.shape[1]}"
        )
        self.model.fit(X_train, y_train, group=groups)

        self.is_fitted = True
        print("[XGBoost] Training complete!")
        return self

    # ------------------------------------------------------------------
    # Feature importance report (Phase 7)
    # ------------------------------------------------------------------

    def feature_importance_report(self) -> pd.DataFrame:
        """Return a sorted DataFrame of feature importances (gain)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        imp = self.model.get_booster().get_score(importance_type="gain")
        df  = (
            pd.DataFrame(
                [(FEATURE_NAMES[int(k[1:])] if k.startswith("f") else k, v)
                 for k, v in imp.items()],
                columns=["feature", "gain"],
            )
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
        return df

    # ------------------------------------------------------------------
    # Predict / Recommend  (vectorised)
    # ------------------------------------------------------------------

    def _build_candidate_matrix(
        self,
        user_id:      int,
        candidate_ids: list[int],
        global_mean:  float,
    ) -> np.ndarray:
        """Build the (n_candidates, 11) feature matrix in one vectorised pass."""

        u_stats    = self.user_stats.get(
            user_id, {"user_avg_rating": global_mean, "user_rating_count": 0}
        )
        u_prof     = self.user_profiles.get(user_id)
        genre_vec  = self.cb.get_user_genre_profile(user_id, self.train_df)

        n = len(candidate_ids)

        # Feature arrays
        cf_preds    = np.full(n, global_mean, dtype=np.float32)
        cb_sims     = np.zeros(n, dtype=np.float32)
        pop_arr     = np.zeros(n, dtype=np.float32)
        genre_arr   = np.zeros(n, dtype=np.float32)
        m_avg_arr   = np.full(n, global_mean, dtype=np.float32)
        m_cnt_arr   = np.zeros(n, dtype=np.float32)
        dot_arr     = np.zeros(n, dtype=np.float32)
        u_norm_arr  = np.zeros(n, dtype=np.float32)
        i_norm_arr  = np.zeros(n, dtype=np.float32)

        # ── CF batch (vectorised) ────────────────────────────────────────
        ts = self.cf.model.trainset
        try:
            inner_uid   = ts.to_inner_uid(user_id)
            u_bias      = self.cf.model.bu[inner_uid]
            u_factor    = self.cf.model.pu[inner_uid]
            cf_gm       = ts.global_mean
            u_norm_val  = float(np.linalg.norm(u_factor))

            base = cf_gm + u_bias
            cf_preds[:] = base

            inner_iids, cand_idx = [], []
            for idx, mid in enumerate(candidate_ids):
                try:
                    inner_iids.append(ts.to_inner_iid(mid))
                    cand_idx.append(idx)
                except ValueError:
                    pass

            if inner_iids:
                inner_iids = np.array(inner_iids)
                biases  = self.cf.model.bi[inner_iids]
                factors = self.cf.model.qi[inner_iids]          # (k, n_factors)
                cf_preds[cand_idx] = cf_gm + u_bias + biases + factors @ u_factor
                dot_arr[cand_idx]  = factors @ u_factor
                i_norm_arr[cand_idx] = np.linalg.norm(factors, axis=1)
            u_norm_arr[:] = u_norm_val
        except ValueError:
            pass   # unknown user → defaults stay as global mean

        # ── CB batch ────────────────────────────────────────────────────
        if u_prof is not None:
            sims = cosine_similarity(u_prof, self.cb.tfidf_matrix).flatten()
            for idx, mid in enumerate(candidate_ids):
                if mid in self.cb.movie_id_to_idx:
                    cb_sims[idx] = sims[self.cb.movie_id_to_idx[mid]]

        # ── Genre affinity batch ─────────────────────────────────────────
        if genre_vec is not None and genre_vec.sum() > 0:
            ga_all = self.cb._movie_genre_matrix @ genre_vec   # (n_movies,)
            denom  = genre_vec.sum()
            for idx, mid in enumerate(candidate_ids):
                if mid in self.cb.movie_id_to_idx:
                    genre_arr[idx] = ga_all[self.cb.movie_id_to_idx[mid]] / denom

        # ── Scalar lookups ───────────────────────────────────────────────
        for idx, mid in enumerate(candidate_ids):
            pop_arr[idx]   = self.pop_scores.get(mid, 0.0)
            ms = self.movie_stats.get(mid, {})
            m_avg_arr[idx] = ms.get("movie_avg_rating", global_mean)
            m_cnt_arr[idx] = ms.get("movie_rating_count", 0)

        X = np.column_stack([
            cf_preds,
            cb_sims,
            pop_arr,
            genre_arr,
            np.full(n, u_stats["user_avg_rating"],   dtype=np.float32),
            m_avg_arr,
            np.full(n, u_stats["user_rating_count"], dtype=np.float32),
            m_cnt_arr,
            dot_arr,
            u_norm_arr,
            i_norm_arr,
        ])
        return X.astype(np.float32)

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return the learned ranking score for a single (user, movie) pair."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        global_mean = float(self.train_df["rating"].mean())
        X = self._build_candidate_matrix(user_id, [movie_id], global_mean)
        X = pd.DataFrame(X, columns=FEATURE_NAMES, dtype=np.float32)
        return float(self.model.predict(X)[0])

    def recommend(self, user_id: int, n: int = 10) -> list[dict]:
        """Return top-N recommendations ranked by learned relevance score."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        user_rated    = set(int(m) for m in
                           self.train_df[self.train_df["userId"] == user_id]["movieId"].values)
        candidate_ids = [m for m in self.all_movie_ids if m not in user_rated]
        if not candidate_ids:
            return []

        global_mean = float(self.train_df["rating"].mean())
        X = self._build_candidate_matrix(user_id, candidate_ids, global_mean)
        X_df = pd.DataFrame(X, columns=FEATURE_NAMES, dtype=np.float32)
        scores = self.model.predict(X_df)

        top_idx = np.argsort(scores)[::-1][:n]

        results = []
        for idx in top_idx:
            mid  = candidate_ids[idx]
            info = self.movie_details_dict.get(mid, {"title": f"Movie {mid}", "genres": "Unknown"})
            results.append({
                "movieId":  mid,
                "title":    info["title"],
                "genres":   info["genres"],
                "score":    float(scores[idx]),
                "cf_score": float(X[idx, 0]),
                "cb_score": float(X[idx, 1]),
                "pop_score": float(X[idx, 2]),
                "genre_score": float(X[idx, 3]),
                "method":   "xgboost_ranker",
            })
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, filepath: str = None):
        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, "xgboost_recommender.pkl")
        with open(filepath, "wb") as f:
            pickle.dump({
                "model":              self.model,
                "user_profiles":      self.user_profiles,
                "movie_stats":        self.movie_stats,
                "user_stats":         self.user_stats,
                "pop_scores":         self.pop_scores,
                "movie_details_dict": self.movie_details_dict,
                "all_movie_ids":      self.all_movie_ids,
            }, f)
        print(f"[XGBoost] Model saved to {filepath}")

    def load_model(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "xgboost_recommender.pkl")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.model              = data["model"]
        self.user_profiles      = data["user_profiles"]
        self.movie_stats        = data["movie_stats"]
        self.user_stats         = data["user_stats"]
        self.pop_scores         = data["pop_scores"]
        self.movie_details_dict = data["movie_details_dict"]
        self.all_movie_ids      = data["all_movie_ids"]
        # Re-fit base models (CF trainset needed for SVD internals)
        try:
            from .splitter import get_train_val_test
        except ImportError:
            from core.splitter import get_train_val_test
        train_df, _, _ = get_train_val_test()
        self.train_df = train_df.copy()
        self.cf.fit(train_df)
        self.cb.fit(train_df)
        self.pop.fit(train_df)
        self.is_fitted = True
        print(f"[XGBoost] Model loaded from {filepath}")
