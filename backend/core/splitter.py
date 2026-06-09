"""
Module 3 - Train/Test Split (Chronological 3-Way User-Wise Split)

Strategy: CHRONOLOGICAL (not random)
  - Each user's interactions are sorted by timestamp (ascending).
  - The first 80% of interactions → Train_Dev
  - The last 20% of interactions  → Holdout_Test

This correctly simulates a real deployment:
  the model trains on past behaviour and predicts future preferences.

Pipeline:
    Merged Dataset (sorted by userId, timestamp)
          ↓
    Chronological User-Wise Split
          ↓
    ┌─────────────────────────┐
    │  Train_Dev  (first 80%) │  ← ALL model development
    │  Holdout_Test (last 20%)│  ← final evaluation ONLY
    └─────────────────────────┘
          ↓ (further split Train_Dev chronologically)
    ┌─────────────────────────┐
    │  Train      (first 80%) │  ← SVD / TF-IDF fitting
    │  Validation (last 20%)  │  ← hyperparameter tuning
    └─────────────────────────┘

Saved files:
    train.csv       → model fitting
    validation.csv  → tuning
    test.csv        → holdout evaluation
"""

import pandas as pd
import os
from .data_processor import load_and_process_data

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────────────
#  Core: chronological per-user split
# ─────────────────────────────────────────────────────────────────────────────

def chronological_user_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split each user's interactions chronologically.

    For every user:
      - Sort by timestamp (ascending → oldest first)
      - First (1 - test_size) fraction  → train
      - Last  test_size fraction         → test

    Users with fewer than 5 ratings go entirely to train (not enough
    history to create a meaningful holdout).

    Args:
        df        : DataFrame with columns [userId, movieId, rating, timestamp, ...]
        test_size : fraction of each user's MOST RECENT interactions for test

    Returns:
        (train_df, test_df)
    """
    if "timestamp" not in df.columns:
        raise ValueError(
            "timestamp column is required for chronological splitting. "
            "Set keep_timestamp=True in data_processor.py."
        )

    train_parts: list[pd.DataFrame] = []
    test_parts:  list[pd.DataFrame] = []

    for user_id, user_data in df.groupby("userId"):
        # Sort chronologically (oldest → newest)
        user_data = user_data.sort_values("timestamp")
        n = len(user_data)

        if n < 5:
            train_parts.append(user_data)
            continue

        n_test = max(1, int(n * test_size))
        train_parts.append(user_data.iloc[: n - n_test])   # first 80% oldest
        test_parts.append(user_data.iloc[n - n_test :])    # last 20%  newest

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)
    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
#  Full 3-way chronological split
# ─────────────────────────────────────────────────────────────────────────────

def create_three_way_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute the full chronological 3-way split on the processed dataset.

    Step 1 – Holdout split (per user, chronological)
        Merged dataset  →  Train_Dev (80% oldest)  +  Holdout_Test (20% newest)

    Step 2 – Dev split (applied only to Train_Dev, chronological)
        Train_Dev       →  Train (80% oldest)      +  Validation (20% newest)

    Returns:
        (train_df, validation_df, test_df)
    """
    # Always re-process so timestamp column is present
    df = load_and_process_data()
    total = len(df)
    print(f"[Splitter] Merged dataset  : {total:,} records  |  "
          f"{df['userId'].nunique()} users  |  {df['movieId'].nunique()} movies")
    print(f"[Splitter] Split strategy  : CHRONOLOGICAL (timestamp-based)")

    # Step 1 ── holdout split
    train_dev_df, holdout_test_df = chronological_user_split(df, test_size=0.20)
    print(f"[Splitter] Train_Dev       : {len(train_dev_df):,} records  "
          f"({len(train_dev_df)/total*100:.1f}%)  ← oldest 80% per user")
    print(f"[Splitter] Holdout_Test    : {len(holdout_test_df):,} records  "
          f"({len(holdout_test_df)/total*100:.1f}%)  ← newest 20% per user")

    # Step 2 ── dev split
    train_df, validation_df = chronological_user_split(train_dev_df, test_size=0.20)
    print(f"[Splitter]   ↳ Train       : {len(train_df):,} records  "
          f"({len(train_df)/total*100:.1f}%)")
    print(f"[Splitter]   ↳ Validation  : {len(validation_df):,} records  "
          f"({len(validation_df)/total*100:.1f}%)")

    # Save CSVs (drop timestamp — downstream models don't need it)
    def _save(frame: pd.DataFrame, path: str):
        frame.drop(columns=["timestamp"], errors="ignore").to_csv(path, index=False)

    _save(train_df,       os.path.join(ROOT_DIR, "train.csv"))
    _save(validation_df,  os.path.join(ROOT_DIR, "validation.csv"))
    _save(holdout_test_df, os.path.join(ROOT_DIR, "test.csv"))
    print("[Splitter] Saved → train.csv | validation.csv | test.csv  (timestamp dropped)")

    # Drop timestamp from returned DataFrames as well so callers see clean data
    train_df       = train_df.drop(columns=["timestamp"], errors="ignore")
    validation_df  = validation_df.drop(columns=["timestamp"], errors="ignore")
    holdout_test_df = holdout_test_df.drop(columns=["timestamp"], errors="ignore")

    return train_df, validation_df, holdout_test_df


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience loaders
# ─────────────────────────────────────────────────────────────────────────────

def get_train_val_test() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load splits from disk; (re-)create them if any file is missing."""
    train_path = os.path.join(ROOT_DIR, "train.csv")
    val_path   = os.path.join(ROOT_DIR, "validation.csv")
    test_path  = os.path.join(ROOT_DIR, "test.csv")

    if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        return (
            pd.read_csv(train_path),
            pd.read_csv(val_path),
            pd.read_csv(test_path),
        )

    return create_three_way_split()


def get_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy helper — returns only (train, test)."""
    train_df, _, test_df = get_train_val_test()
    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tr, va, te = create_three_way_split()
    print(f"\nFinal shapes → Train: {tr.shape}  |  Val: {va.shape}  |  Test: {te.shape}")
