"""
Phase 8 - Full ranking-quality experiment runner.

Run from backend/:
    UV_CACHE_DIR=/tmp/uv-cache uv run python -m core.experiment_runner

The runner keeps the existing architecture and evaluates targeted changes:
baseline hybrid, weighted content profiles, Bayesian popularity tuning,
hybrid weight tuning, genre affinity, and XGBRanker.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.collaborative import CollaborativeFilteringRecommender
from core.evaluator import RecommendationEvaluator
from core.hybrid import HybridRecommender
from core.splitter import get_train_val_test


REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _eval_ranking(recommender, train_df, eval_df, *, k=10, n_users=200, label="") -> dict:
    t0 = time.time()
    evaluator = RecommendationEvaluator(recommender, train_df, eval_df, relevance_threshold=4.0)
    metrics = evaluator.evaluate(k=k, n_users=n_users)
    metrics["elapsed_sec"] = round(time.time() - t0, 2)
    if label:
        print(f"[{label}] ranking eval finished in {metrics['elapsed_sec']}s")
    return metrics


def _svd_rmse(cf: CollaborativeFilteringRecommender, test_df: pd.DataFrame) -> float:
    errors = [
        (float(row.rating) - cf.predict(int(row.userId), int(row.movieId))) ** 2
        for row in test_df.itertuples(index=False)
    ]
    return round(math.sqrt(float(np.mean(errors))), 5)


def _hybrid_cf_rmse(hybrid: HybridRecommender, test_df: pd.DataFrame) -> float:
    return _svd_rmse(hybrid.cf, test_df)


def _row(phase: str, model: str, metrics: dict, rmse, notes: str, extra: dict | None = None) -> dict:
    row = {
        "phase": phase,
        "model": model,
        "precision_at_10": metrics["precision_at_k"],
        "recall_at_10": metrics["recall_at_k"],
        "ndcg_at_10": metrics["ndcg_at_k"],
        "rmse": rmse,
        "n_users": metrics["n_users_evaluated"],
        "notes": notes,
    }
    if extra:
        row.update(extra)
    return row


def _print_rows(rows: list[dict], title: str) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    display_cols = [
        "phase",
        "model",
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
        "rmse",
        "notes",
    ]
    print(pd.DataFrame(rows)[display_cols].to_string(index=False))


def _save_report(rows: list[dict], artifacts: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORTS_DIR / "ranking_experiment_report.json"
    with json_path.open("w") as f:
        json.dump({"results": rows, "artifacts": artifacts}, f, indent=2, default=str)

    md_path = REPORTS_DIR / "ranking_experiment_report.md"
    df = pd.DataFrame(rows)
    best_hybrid = df[df["phase"].str.contains("P1|P3", regex=True)].sort_values(
        ["recall_at_10", "ndcg_at_10"], ascending=False
    ).head(1)
    best_overall = df.sort_values(["recall_at_10", "ndcg_at_10"], ascending=False).head(1)

    with md_path.open("w") as f:
        f.write("# Ranking Experiment Report\n\n")
        f.write("Primary objective: maximize Recall@10, with NDCG@10 as secondary tie-breaker.\n\n")
        f.write("```text\n")
        f.write(df.to_string(index=False))
        f.write("\n```\n")
        f.write("\n\n## Best Models\n\n")
        if not best_hybrid.empty:
            f.write(f"- Best hybrid: {best_hybrid.iloc[0]['model']}\n")
        if not best_overall.empty:
            f.write(f"- Best overall: {best_overall.iloc[0]['model']}\n")
        f.write("\n## Interpretation\n\n")
        f.write(
            "- RMSE is reported for SVD-backed rating prediction. XGBRanker emits relevance "
            "scores, not calibrated ratings, so RMSE is marked N/A for that model.\n"
            "- Hybrid metrics move when candidate ordering changes; SVD RMSE can stay fixed "
            "because the CF submodel is unchanged.\n"
            "- XGBRanker is production-relevant when serving can afford feature generation "
            "for a candidate set and when online ranking metrics matter more than rating "
            "prediction error.\n"
        )
        if "feature_importance" in artifacts:
            f.write("\n## XGBRanker Feature Importance\n\n")
            f.write("```text\n")
            f.write(pd.DataFrame(artifacts["feature_importance"]).to_string(index=False))
            f.write("\n```\n")

    print(f"\nSaved reports:\n  {json_path}\n  {md_path}")


def run_experiments() -> list[dict]:
    k = _env_int("EXPERIMENT_K", 10)
    n_users = _env_int("EXPERIMENT_USERS", 200)
    tune_users = _env_int("EXPERIMENT_TUNE_USERS", min(200, n_users))
    xgb_estimators = _env_int("XGB_ESTIMATORS", 300)

    print("=" * 80)
    print("Ranking-quality experiment report")
    print(f"K={k}  eval_users={n_users}  tune_users={tune_users}")
    print("=" * 80)

    train_df, val_df, test_df = get_train_val_test()
    holdout_df = pd.concat([val_df, test_df], ignore_index=True)
    rows: list[dict] = []
    artifacts: dict = {}

    print("\n[Baseline] Legacy hybrid: CF=0.7 CB=0.2 POP=0.1")
    baseline = HybridRecommender(
        weight_cf=0.7,
        weight_cb=0.2,
        weight_pop=0.1,
        weight_genre=0.0,
        content_profile_size=20,
        content_profile_strategy="mean",
        popularity_method="legacy",
    )
    baseline.fit(train_df)
    baseline_metrics = _eval_ranking(baseline, train_df, test_df, k=k, n_users=n_users, label="baseline")
    rows.append(
        _row(
            "Baseline",
            "Legacy Hybrid",
            baseline_metrics,
            _hybrid_cf_rmse(baseline, test_df),
            "Mean top-20 content profile + avg_rating*log1p(count)",
        )
    )

    print("\n[P2] Weighted content profile sweep")
    content_results = []
    best_content = None
    for profile_size, label in [(20, "top20"), (50, "top50"), (None, "all_liked")]:
        rec = HybridRecommender(
            weight_cf=0.7,
            weight_cb=0.2,
            weight_pop=0.1,
            weight_genre=0.0,
            content_profile_size=profile_size,
            content_profile_strategy="weighted",
            popularity_method="legacy",
        )
        rec.fit(train_df)
        val_metrics = _eval_ranking(rec, train_df, val_df, k=k, n_users=tune_users, label=f"P2-{label}-val")
        test_metrics = _eval_ranking(rec, train_df, test_df, k=k, n_users=n_users, label=f"P2-{label}-test")
        result = {
            "label": label,
            "profile_size": "all" if profile_size is None else profile_size,
            "validation": val_metrics,
            "test": test_metrics,
        }
        content_results.append(result)
        rows.append(
            _row(
                "P2",
                f"Weighted Content Profile ({label})",
                test_metrics,
                _hybrid_cf_rmse(rec, test_df),
                "profile=sum(rating*tfidf)/sum(rating), ratings>=4",
                {"validation_recall_at_10": val_metrics["recall_at_k"]},
            )
        )
        if best_content is None or (
            val_metrics["recall_at_k"],
            val_metrics["ndcg_at_k"],
        ) > (
            best_content["validation"]["recall_at_k"],
            best_content["validation"]["ndcg_at_k"],
        ):
            best_content = result
    artifacts["content_profile_sweep"] = content_results
    best_profile_size = None if best_content["profile_size"] == "all" else int(best_content["profile_size"])

    print("\n[P4] Bayesian popularity m sweep")
    m_values = [5, 10, 25, 50, 100, 200]
    popularity_results = []
    best_pop = None
    for m in m_values:
        rec = HybridRecommender(
            weight_cf=0.7,
            weight_cb=0.2,
            weight_pop=0.1,
            weight_genre=0.0,
            content_profile_size=best_profile_size,
            content_profile_strategy="weighted",
            popularity_method="bayesian",
            popularity_m_threshold=m,
        )
        rec.fit(train_df)
        val_metrics = _eval_ranking(rec, train_df, val_df, k=k, n_users=tune_users, label=f"P4-m{m}-val")
        result = {"m": m, "validation": val_metrics}
        popularity_results.append(result)
        if best_pop is None or (
            val_metrics["recall_at_k"],
            val_metrics["ndcg_at_k"],
        ) > (
            best_pop["validation"]["recall_at_k"],
            best_pop["validation"]["ndcg_at_k"],
        ):
            best_pop = result
    artifacts["popularity_m_sweep"] = popularity_results
    best_m = int(best_pop["m"])

    p4_rec = HybridRecommender(
        weight_cf=0.7,
        weight_cb=0.2,
        weight_pop=0.1,
        weight_genre=0.0,
        content_profile_size=best_profile_size,
        content_profile_strategy="weighted",
        popularity_method="bayesian",
        popularity_m_threshold=best_m,
    )
    p4_rec.fit(train_df)
    p4_metrics = _eval_ranking(p4_rec, train_df, test_df, k=k, n_users=n_users, label="P4-test")
    rows.append(
        _row(
            "P4",
            f"Bayesian Popularity (m={best_m})",
            p4_metrics,
            _hybrid_cf_rmse(p4_rec, test_df),
            "Best m selected by validation Recall@10 then NDCG@10",
        )
    )

    print("\n[P1] Hybrid weight tuning without genre")
    tuned = HybridRecommender(
        content_profile_size=best_profile_size,
        content_profile_strategy="weighted",
        popularity_method="bayesian",
        popularity_m_threshold=best_m,
    )
    tuned.fit(train_df)
    tune_result = tuned.tune_weights(
        val_df,
        cf_values=[0.6, 0.65, 0.7, 0.75, 0.8],
        cb_values=[0.1, 0.15, 0.2, 0.25],
        genre_values=[0.0],
        k=k,
        n_eval_users=tune_users,
    )
    artifacts["hybrid_weight_tuning"] = tune_result["comparison_table"].to_dict("records")
    p1_metrics = _eval_ranking(tuned, train_df, test_df, k=k, n_users=n_users, label="P1-test")
    rows.append(
        _row(
            "P1",
            f"Tuned Hybrid {tune_result['best_weights']}",
            p1_metrics,
            _hybrid_cf_rmse(tuned, test_df),
            "Weights selected on validation Recall@10, then NDCG@10",
        )
    )

    print("\n[P3] Hybrid weight tuning with genre affinity")
    genre_tuned = HybridRecommender(
        content_profile_size=best_profile_size,
        content_profile_strategy="weighted",
        popularity_method="bayesian",
        popularity_m_threshold=best_m,
    )
    genre_tuned.fit(train_df)
    genre_result = genre_tuned.tune_weights(
        val_df,
        cf_values=[0.4, 0.5, 0.6, 0.7, 0.8],
        cb_values=[0.1, 0.2, 0.3, 0.4],
        genre_values=[0.0, 0.05, 0.1],
        k=k,
        n_eval_users=tune_users,
    )
    artifacts["genre_weight_tuning"] = genre_result["comparison_table"].to_dict("records")
    p3_metrics = _eval_ranking(genre_tuned, train_df, test_df, k=k, n_users=n_users, label="P3-test")
    rows.append(
        _row(
            "P3",
            f"Tuned Hybrid + Genre {genre_result['best_weights']}",
            p3_metrics,
            _hybrid_cf_rmse(genre_tuned, test_df),
            "Genre affinity added as a fourth hybrid signal",
        )
    )

    print("\n[P5-P7] XGBRanker with negative sampling and advanced features")
    from core.xgboost_recommender import XGBoostRecommender

    xgb_rec = XGBoostRecommender(
        n_estimators=xgb_estimators,
        max_depth=6,
        learning_rate=0.05,
        neg_samples=5,
    )
    xgb_rec.fit(train_df, holdout_df=holdout_df)
    fi = xgb_rec.feature_importance_report()
    artifacts["feature_importance"] = fi.to_dict("records")
    xgb_metrics = _eval_ranking(xgb_rec, train_df, test_df, k=k, n_users=n_users, label="XGBRanker-test")
    rows.append(
        _row(
            "P5-P7",
            "XGBRanker rank:ndcg",
            xgb_metrics,
            "N/A",
            "Binary relevance, 5x unseen negatives per positive, 11 ranking features",
        )
    )

    _print_rows(rows, "Final holdout comparison")
    _save_report(rows, artifacts)
    return rows


if __name__ == "__main__":
    run_experiments()
