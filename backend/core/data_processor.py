"""
Module 1 - Data Collection & Processing
Loads ratings.csv and movies.csv, merges them, cleans data,
and outputs processed_movies.csv
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.dirname(os.path.dirname(__file__))


def load_and_process_data() -> pd.DataFrame:
    """Load raw CSVs, merge, clean, and save processed dataset."""
    ratings_path = os.path.join(RAW_DIR, "ratings.csv")
    movies_path = os.path.join(RAW_DIR, "movies.csv")

    # Load raw data
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    print(f"[Data Processor] Loaded {len(ratings)} ratings and {len(movies)} movies")

    # Merge on movieId
    df = ratings.merge(movies, on="movieId", how="inner")

    # --- Data Cleaning ---
    # Check for missing values
    missing_before = df.isnull().sum().sum()
    print(f"[Data Processor] Missing values before cleaning: {missing_before}")

    # Drop rows with missing values
    df = df.dropna()

    # Remove duplicates (same user rating same movie multiple times)
    dupes_before = df.duplicated(subset=["userId", "movieId"]).sum()
    print(f"[Data Processor] Duplicate entries found: {dupes_before}")
    df = df.drop_duplicates(subset=["userId", "movieId"], keep="last")

    # Keep timestamp — used for chronological train/test splitting
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(int)

    # Ensure correct data types
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)

    # Sort by userId then timestamp (chronological order per user)
    df = df.sort_values(["userId", "timestamp"]).reset_index(drop=True)

    print(f"[Data Processor] Final dataset: {len(df)} records")
    print(f"[Data Processor] Users: {df['userId'].nunique()}, Movies: {df['movieId'].nunique()}")

    # Save processed dataset
    os.makedirs(DATA_DIR, exist_ok=True)
    processed_path = os.path.join(DATA_DIR, "processed_movies.csv")
    df.to_csv(processed_path, index=False)
    print(f"[Data Processor] Saved processed data to {processed_path}")

    return df


def get_processed_data() -> pd.DataFrame:
    """Load the processed dataset. If it doesn't exist, create it."""
    processed_path = os.path.join(DATA_DIR, "processed_movies.csv")
    if os.path.exists(processed_path):
        return pd.read_csv(processed_path)
    return load_and_process_data()


if __name__ == "__main__":
    df = load_and_process_data()
    print(df.head(10))
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
