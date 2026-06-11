"""
Module 7 & 8 - Hybrid Recommendation Engine

Phase 1  – Automatic weight tuning via grid search on validation set.
Phase 3  – Genre affinity signal integrated as a 4th scoring dimension.
Phase 9  – MMR (Maximal Marginal Relevance) re-ranking for diversity.

Default weights (pre-tuning baseline):
    CF=0.7  CB=0.2  POP=0.1  GENRE=0.0

After Phase 1 tuning the best weights are stored on the instance and used
for all subsequent recommend() calls.

MMR re-ranking:
    After scoring all candidates, MMR re-ranks the top-50 to balance
    relevance vs genre diversity. Controlled by mmr_lambda:
        mmr_lambda=1.0  → pure relevance (MMR off, same as before)
        mmr_lambda=0.7  → default: 70% relevance, 30% diversity
        mmr_lambda=0.5  → equal relevance and diversity
"""
import itertools
import pandas as pd
import numpy as np
from .collaborative import CollaborativeFilteringRecommender
from .content_based import ContentBasedRecommender
from .popularity import PopularityRecommender
from .evaluator import RecommendationEvaluator


class HybridRecommender:
    """Hybrid recommender combining CF, CB, Popularity, Genre Affinity, and MMR."""

    # Baseline weights (overridden by tune_weights())
    WEIGHT_CF    = 0.7
    WEIGHT_CB    = 0.2
    WEIGHT_POP   = 0.1
    WEIGHT_GENRE = 0.0

    def __init__(
        self,
        weight_cf:    float = 0.7,
        weight_cb:    float = 0.2,
        weight_pop:   float = 0.1,
        weight_genre: float = 0.0,
        content_profile_size: int | None = 50,
        content_profile_strategy: str = "weighted",
        popularity_method: str = "bayesian",
        popularity_m_threshold: float | None = None,
        mmr_lambda: float = 0.7,
        mmr_candidates: int = 50,
    ):
        """
        Args:
            mmr_lambda:    MMR trade-off. 1.0 = pure relevance (disabled).
                           0.7 = default (70% relevance, 30% diversity).
            mmr_candidates: how many top candidates to re-rank with MMR.
                            50 is a good default — large enough pool, fast enough.
        """
        self.cf  = CollaborativeFilteringRecommender()
        self.cb  = ContentBasedRecommender(
            profile_size=content_profile_size,
            profile_strategy=content_profile_strategy,
        )
        self.pop = PopularityRecommender(
            scoring_method=popularity_method,
            m_threshold=popularity_m_threshold,
        )

        self.weight_cf    = weight_cf
        self.weight_cb    = weight_cb
        self.weight_pop   = weight_pop
        self.weight_genre = weight_genre
        self.content_profile_size = content_profile_size
        self.content_profile_strategy = content_profile_strategy
        self.popularity_method = popularity_method
        self.popularity_m_threshold = popularity_m_threshold
        self.mmr_lambda    = mmr_lambda
        self.mmr_candidates = mmr_candidates

        self.train_df    = None
        self.movie_stats = None
        self.is_fitted   = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "HybridRecommender":
        self.train_df = train_df.copy()
        self.movie_stats = train_df.groupby("movieId").agg(
            avg_rating=("rating", "mean"), num_ratings=("rating", "count")
        ).to_dict("index")

        print("[Hybrid] Training CF (SVD)...")
        self.cf.fit(train_df)
        print("[Hybrid] Training CB (TF-IDF + Genre)...")
        self.cb.fit(train_df)
        print("[Hybrid] Training Popularity (Bayesian)...")
        self.pop.fit(train_df)
        self.is_fitted = True
        print(
            f"[Hybrid] All models trained!  "
            f"weights → CF={self.weight_cf}  CB={self.weight_cb}  "
            f"POP={self.weight_pop}  GENRE={self.weight_genre}  "
            f"MMR_λ={self.mmr_lambda}"
        )
        return self

    # ------------------------------------------------------------------
    # Phase 1 – Automatic Weight Tuning
    # ------------------------------------------------------------------

    def tune_weights(
        self,
        val_df: pd.DataFrame,
        cf_values:    list[float] | None = None,
        cb_values:    list[float] | None = None,
        genre_values: list[float] | None = None,
        k: int = 10,
        n_eval_users: int = 200,
    ) -> dict:
        """
        Grid-search over (CF, CB, GENRE) weights; POP = 1 - CF - CB - GENRE.

        Ranking priority:  Recall@K  →  NDCG@K  →  Precision@K

        Args:
            val_df:        validation DataFrame to measure ranking quality.
            cf_values:     list of CF weight candidates.
            cb_values:     list of CB weight candidates.
            genre_values:  list of GENRE weight candidates.
            k:             cut-off for ranking metrics.
            n_eval_users:  max users to evaluate per combination (for speed).

        Returns:
            dict with keys: best_weights, best_metrics, comparison_table
        """
        if not self.is_fitted:
            raise ValueError("Call fit() before tune_weights().")

        if cf_values    is None: cf_values    = [0.4, 0.5, 0.6, 0.7, 0.8]
        if cb_values    is None: cb_values    = [0.1, 0.2, 0.3, 0.4]
        if genre_values is None: genre_values = [0.0, 0.05, 0.1]

        results = []
        combos  = list(itertools.product(cf_values, cb_values, genre_values))
        valid   = [(cf, cb, g) for cf, cb, g in combos if cf + cb + g <= 1.0]

        print(f"[Hybrid] Weight tuning: {len(valid)} valid combinations …")

        # Snapshot original weights
        orig_cf, orig_cb, orig_pop, orig_g = (
            self.weight_cf, self.weight_cb, self.weight_pop, self.weight_genre
        )

        for cf, cb, genre in valid:
            pop = round(1.0 - cf - cb - genre, 6)
            if pop < 0:
                continue

            # Apply candidate weights
            self.weight_cf    = cf
            self.weight_cb    = cb
            self.weight_pop   = pop
            self.weight_genre = genre

            evaluator = RecommendationEvaluator(self, self.train_df, val_df)
            m = evaluator.evaluate(k=k, n_users=n_eval_users)

            results.append({
                "cf": cf, "cb": cb, "pop": pop, "genre": genre,
                "recall":    m["recall_at_k"],
                "ndcg":      m["ndcg_at_k"],
                "precision": m["precision_at_k"],
            })

        # Sort by Recall then NDCG then Precision
        results.sort(key=lambda r: (r["recall"], r["ndcg"], r["precision"]), reverse=True)
        best = results[0]

        # Apply best weights permanently
        self.weight_cf    = best["cf"]
        self.weight_cb    = best["cb"]
        self.weight_pop   = best["pop"]
        self.weight_genre = best["genre"]

        print(
            f"\n[Hybrid] Best weights →  "
            f"CF={best['cf']}  CB={best['cb']}  POP={best['pop']}  GENRE={best['genre']}"
        )
        print(
            f"[Hybrid] Best val metrics →  "
            f"Recall@{k}={best['recall']}  NDCG@{k}={best['ndcg']}  "
            f"P@{k}={best['precision']}"
        )

        comparison_df = pd.DataFrame(results)
        return {
            "best_weights": {
                "cf": best["cf"], "cb": best["cb"],
                "pop": best["pop"], "genre": best["genre"],
            },
            "best_metrics": {
                "recall_at_k":    best["recall"],
                "ndcg_at_k":      best["ndcg"],
                "precision_at_k": best["precision"],
            },
            "comparison_table": comparison_df,
        }

    # ------------------------------------------------------------------
    # Phase 9 – MMR Re-ranking
    # ------------------------------------------------------------------

    def _mmr_rerank(
        self,
        candidates: list[dict],
        n: int,
        mi: pd.DataFrame,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance re-ranking.

        Picks items one at a time, each time choosing the candidate that
        maximises:
            MMR = λ * relevance_score  -  (1 - λ) * max_genre_overlap

        where max_genre_overlap is the highest Jaccard similarity between
        the candidate's genres and any already-selected item's genres.

        Args:
            candidates: pre-sorted list of scored items (top mmr_candidates).
            n:          final number of items to return.
            mi:         movie_df indexed by movieId for genre lookups.

        Returns:
            Re-ranked list of n items.
        """
        # If MMR is effectively disabled, just return top-n directly
        if self.mmr_lambda >= 1.0 or len(candidates) <= n:
            return candidates[:n]

        # Pre-build genre sets for all candidates (avoids repeated splits)
        genre_sets = {}
        for item in candidates:
            mid = item["movieId"]
            if mid in mi.index:
                genre_sets[mid] = set(mi.loc[mid, "genres"].split("|"))
            else:
                genre_sets[mid] = set()

        selected: list[dict] = []
        remaining = list(candidates)  # shallow copy

        while len(selected) < n and remaining:
            # First item: always pick the highest-scored candidate
            if not selected:
                selected.append(remaining.pop(0))
                continue

            selected_genres = [genre_sets[s["movieId"]] for s in selected]

            best_mmr   = -float("inf")
            best_idx   = 0

            for i, cand in enumerate(remaining):
                relevance = cand["score"]

                # Jaccard similarity against each already-selected item
                cand_genres = genre_sets[cand["movieId"]]
                if cand_genres:
                    overlaps = [
                        len(cand_genres & sg) / max(len(cand_genres | sg), 1)
                        for sg in selected_genres
                        if sg  # skip empty genre sets
                    ]
                    max_overlap = max(overlaps) if overlaps else 0.0
                else:
                    max_overlap = 0.0

                mmr_score = (
                    self.mmr_lambda * relevance
                    - (1.0 - self.mmr_lambda) * max_overlap
                )

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------

    def recommend(self, user_id: int, n: int = 10) -> list[dict]:
        if not self.is_fitted:
            raise ValueError("Not fitted")

        user_movies = self.train_df[self.train_df["userId"] == user_id]
        if user_movies.empty:
            pop_recs = self.pop.recommend(n)
            return [
                {
                    "movieId": int(r["movieId"]), "title": r["title"],
                    "genres": r["genres"],
                    "score": round(float(r["popularity_score"]), 4),
                    "method": "popularity",
                }
                for _, r in pop_recs.iterrows()
            ]

        cf_preds   = self.cf.get_all_predictions_for_user(user_id)
        cb_scores  = self.cb.get_content_scores_for_user(user_id, self.train_df)
        pop_scores = self.pop.get_all_scores()

        # Phase 3: genre affinity scores
        genre_scores: dict = {}
        if self.weight_genre > 0:
            genre_scores = self.cb.get_genre_affinity_scores(user_id, self.train_df)

        all_ids = set(cf_preds.keys()) | set(cb_scores.keys()) | set(pop_scores.keys())
        if not all_ids:
            pop_recs = self.pop.recommend(n, list(user_movies["movieId"]))
            return [
                {
                    "movieId": int(r["movieId"]), "title": r["title"],
                    "genres": r["genres"],
                    "score": round(float(r["popularity_score"]), 4),
                    "method": "popularity",
                }
                for _, r in pop_recs.iterrows()
            ]

        # Normalise CF to [0, 1]
        if cf_preds:
            vals = list(cf_preds.values())
            mn, mx = min(vals), max(vals)
            rng = mx - mn if mx != mn else 1.0
            cf_norm = {k: (v - mn) / rng for k, v in cf_preds.items()}
        else:
            cf_norm = {}

        scores = []
        for mid in all_ids:
            cf_s  = cf_norm.get(mid, 0.0)
            cb_s  = cb_scores.get(mid, 0.0)
            p_s   = pop_scores.get(mid, 0.0)
            g_s   = genre_scores.get(mid, 0.0)
            final = (
                self.weight_cf    * cf_s
                + self.weight_cb    * cb_s
                + self.weight_pop   * p_s
                + self.weight_genre * g_s
            )
            st = self.movie_stats.get(mid, {"avg_rating": 0, "num_ratings": 0})
            scores.append({
                "movieId":     int(mid),
                "score":       final,
                "cf_score":    cf_s,
                "cb_score":    cb_s,
                "pop_score":   p_s,
                "genre_score": g_s,
                "avg_rating":  st["avg_rating"],
                "num_ratings": st["num_ratings"],
            })

        # Sort by hybrid score, then avg_rating, then num_ratings as tie-breakers
        scores.sort(
            key=lambda x: (x["score"], x["avg_rating"], x["num_ratings"]),
            reverse=True,
        )

        # Phase 9: MMR re-ranking on top candidates for diversity
        mi = self.cb.movie_df.set_index("movieId")
        candidates = scores[:self.mmr_candidates]
        top = self._mmr_rerank(candidates, n, mi)

        results = []
        for it in top:
            m    = it["movieId"]
            info = mi.loc[m] if m in mi.index else None
            results.append({
                "movieId":     m,
                "title":       info["title"] if info is not None else f"Movie {m}",
                "genres":      info["genres"] if info is not None else "Unknown",
                "score":       round(it["score"], 4),
                "cf_score":    round(it["cf_score"], 4),
                "cb_score":    round(it["cb_score"], 4),
                "pop_score":   round(it["pop_score"], 4),
                "genre_score": round(it["genre_score"], 4),
                "method":      "hybrid",
            })
        return results

    # ------------------------------------------------------------------
    # Movie details helper (unchanged)
    # ------------------------------------------------------------------

    def get_movie_details(self, movie_id: int) -> dict | None:
        if self.cb.movie_df is None:
            return None
        match = self.cb.movie_df[self.cb.movie_df["movieId"] == movie_id]
        if match.empty:
            return None
        movie = match.iloc[0]
        st = self.movie_stats.get(movie_id, {"avg_rating": 0, "num_ratings": 0})
        similar = self.cb.get_similar_movies(movie_id, 5)
        return {
            "movieId":     int(movie_id),
            "title":       movie["title"],
            "genres":      movie["genres"],
            "avg_rating":  round(float(st.get("avg_rating", 0)), 2),
            "num_ratings": int(st.get("num_ratings", 0)),
            "similar_movies": similar.to_dict("records") if not similar.empty else [],
        }