"""
Module 9 - Recommendation Evaluation
Evaluates using Precision@K, Recall@K, and NDCG@K on the test dataset.
"""
import pandas as pd
import numpy as np


class RecommendationEvaluator:
    def __init__(self, hybrid_recommender, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.recommender = hybrid_recommender
        self.train_df = train_df
        self.test_df = test_df
        self.metrics = {}

    def _get_relevant_items(self, user_id: int, threshold: float = 3.5) -> set:
        """Items in test set with rating >= threshold are considered relevant."""
        user_test = self.test_df[self.test_df["userId"] == user_id]
        return set(user_test[user_test["rating"] >= threshold]["movieId"].values)

    def precision_at_k(self, user_id: int, k: int = 10) -> float:
        relevant = self._get_relevant_items(user_id)
        if not relevant:
            return 0.0
        recs = self.recommender.recommend(user_id, k)
        rec_ids = set(r["movieId"] for r in recs)
        if not rec_ids:
            return 0.0
        return len(relevant & rec_ids) / len(rec_ids)

    def recall_at_k(self, user_id: int, k: int = 10) -> float:
        relevant = self._get_relevant_items(user_id)
        if not relevant:
            return 0.0
        recs = self.recommender.recommend(user_id, k)
        rec_ids = set(r["movieId"] for r in recs)
        return len(relevant & rec_ids) / len(relevant)

    def ndcg_at_k(self, user_id: int, k: int = 10) -> float:
        relevant = self._get_relevant_items(user_id)
        if not relevant:
            return 0.0
        recs = self.recommender.recommend(user_id, k)
        rec_ids = [r["movieId"] for r in recs]
        dcg = sum(1.0 / np.log2(i + 2) for i, mid in enumerate(rec_ids) if mid in relevant)
        ideal = sorted([1.0 / np.log2(i + 2) for i in range(min(len(relevant), k))], reverse=True)
        idcg = sum(ideal)
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate(self, k: int = 10, n_users: int = 100) -> dict:
        """Evaluate on a sample of users from the test set."""
        test_users = self.test_df["userId"].unique()
        # Only evaluate users who have ratings in BOTH train and test
        valid_users = [u for u in test_users if u in self.train_df["userId"].values]
        # n_users=9999 (or any value > pool) → evaluate every eligible user
        sample_users = valid_users[:n_users]

        precisions, recalls, ndcgs = [], [], []
        print(f"[Evaluator] Evaluating on {len(sample_users)} users with K={k}...")

        for i, uid in enumerate(sample_users):
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(sample_users)}")
            precisions.append(self.precision_at_k(uid, k))
            recalls.append(self.recall_at_k(uid, k))
            ndcgs.append(self.ndcg_at_k(uid, k))

        self.metrics = {
            "precision_at_k": round(float(np.mean(precisions)), 4),
            "recall_at_k": round(float(np.mean(recalls)), 4),
            "ndcg_at_k": round(float(np.mean(ndcgs)), 4),
            "k": k,
            "n_users_evaluated": len(sample_users),
        }
        print(f"[Evaluator] Precision@{k}: {self.metrics['precision_at_k']}")
        print(f"[Evaluator] Recall@{k}: {self.metrics['recall_at_k']}")
        print(f"[Evaluator] NDCG@{k}: {self.metrics['ndcg_at_k']}")
        return self.metrics

    def get_metrics(self) -> dict:
        return self.metrics
