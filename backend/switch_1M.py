"""
Switch to MovieLens 1M Dataset
==============================
Run this script ONCE from your project's backend/ folder:

    python switch_to_1m.py --ratings path/to/ratings.dat \
                            --movies  path/to/movies.dat

What it does:
  1. Converts ratings.dat → ratings.csv  (project format, with headers)
  2. Converts movies.dat  → movies.csv   (project format, with headers)
  3. Deletes stale cached files so everything rebuilds fresh
  4. Patches collaborative.py  → fixes rating_scale (0.5,5.0) → (1.0,5.0)
  5. Patches data_processor.py → reads 1M format (no merge needed, auto-detect)

After running this script:
    - Just start your server normally: uv run uvicorn main:app
    - Everything rebuilds automatically from the new data
"""

import argparse
import os
import re
import shutil
import pandas as pd


# ── Genre normalisation ───────────────────────────────────────────────────────
def fix_genres(g: str) -> str:
    """Normalise 1M genre quirks to match 100K conventions."""
    return (
        g.replace("Children's", "Children")
         .replace("Film-Noir", "Film-Noir")   # keep as-is
         .strip()
    )


# ── Step 1 & 2: Convert raw .dat files ───────────────────────────────────────
def convert_data(ratings_path: str, movies_path: str, out_dir: str) -> None:
    print("\n[1/5] Converting ratings.dat...")
    ratings_raw = pd.read_csv(
        ratings_path, sep="::", header=None,
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python", encoding="latin-1",
    )
    # 1M has integer ratings (1-5) → convert to float to match project schema
    ratings_raw["rating"] = ratings_raw["rating"].astype(float)
    ratings_raw["userId"]    = ratings_raw["userId"].astype(int)
    ratings_raw["movieId"]   = ratings_raw["movieId"].astype(int)
    ratings_raw["timestamp"] = ratings_raw["timestamp"].astype(int)

    ratings_out = os.path.join(out_dir, "ratings.csv")
    ratings_raw.to_csv(ratings_out, index=False)
    print(f"      Saved → {ratings_out}  ({len(ratings_raw):,} rows)")
    print(f"      Rating range: {ratings_raw['rating'].min()} – {ratings_raw['rating'].max()}")
    print(f"      Users: {ratings_raw['userId'].nunique():,}  |  Movies: {ratings_raw['movieId'].nunique():,}")

    print("\n[2/5] Converting movies.dat...")
    movies_raw = pd.read_csv(
        movies_path, sep="::", header=None,
        names=["movieId", "title", "genres"],
        engine="python", encoding="latin-1",
    )
    movies_raw["movieId"] = movies_raw["movieId"].astype(int)
    movies_raw["title"]   = movies_raw["title"].str.strip()
    movies_raw["genres"]  = movies_raw["genres"].apply(fix_genres)

    # Drop movies with no genres
    before = len(movies_raw)
    movies_raw = movies_raw[movies_raw["genres"] != "(no genres listed)"].copy()
    print(f"      Kept {len(movies_raw):,} / {before:,} movies")

    movies_out = os.path.join(out_dir, "movies.csv")
    movies_raw.to_csv(movies_out, index=False)
    print(f"      Saved → {movies_out}  ({len(movies_raw):,} rows)")


# ── Step 3: Delete stale cached files ─────────────────────────────────────────
def delete_stale_files(project_root: str) -> None:
    print("\n[3/5] Deleting stale cached files...")
    stale = [
        os.path.join(project_root, "train.csv"),
        os.path.join(project_root, "validation.csv"),
        os.path.join(project_root, "test.csv"),
        os.path.join(project_root, "data", "processed_movies.csv"),
    ]
    for f in stale:
        if os.path.exists(f):
            os.remove(f)
            print(f"      Deleted → {f}")
        else:
            print(f"      Not found (ok) → {f}")


# ── Step 4: Patch collaborative.py ───────────────────────────────────────────
def patch_collaborative(core_dir: str) -> None:
    print("\n[4/5] Patching collaborative.py (rating_scale)...")
    path = os.path.join(core_dir, "collaborative.py")
    if not os.path.exists(path):
        print(f"      SKIP — file not found: {path}")
        return

    with open(path, "r") as f:
        content = f.read()

    # Fix rating scale
    new_content = content.replace(
        "rating_scale=(0.5, 5.0)",
        "rating_scale=(1.0, 5.0)",
    )
    # Fix clip range
    new_content = new_content.replace(
        "np.clip(estimates, 0.5, 5.0)",
        "np.clip(estimates, 1.0, 5.0)",
    )

    if new_content == content:
        print("      Already patched or pattern not found — check manually")
    else:
        # Backup original
        shutil.copy(path, path + ".bak")
        with open(path, "w") as f:
            f.write(new_content)
        print(f"      Patched!  (backup saved as collaborative.py.bak)")
        print("      Changed: rating_scale=(0.5,5.0) → (1.0,5.0)")
        print("      Changed: np.clip(estimates, 0.5, 5.0) → (1.0, 5.0)")


# ── Step 5: Patch data_processor.py ──────────────────────────────────────────
def patch_data_processor(core_dir: str) -> None:
    print("\n[5/5] Patching data_processor.py (1M format support)...")
    path = os.path.join(core_dir, "data_processor.py")
    if not os.path.exists(path):
        print(f"      SKIP — file not found: {path}")
        return

    shutil.copy(path, path + ".bak")

    new_load_fn = '''def load_and_process_data() -> pd.DataFrame:
    """Load raw CSVs, merge if needed, clean, and save processed dataset.

    Supports both formats automatically:
      - 100K: ratings.csv has [userId,movieId,rating,timestamp]
              movies.csv has  [movieId,title,genres]  → merge needed
      - 1M:   ratings.csv has [userId,movieId,rating,timestamp]
              movies.csv has  [movieId,title,genres]  → merge needed
    Both use comma-separated CSV with headers after conversion.
    """
    ratings_path = os.path.join(RAW_DIR, "ratings.csv")
    movies_path  = os.path.join(RAW_DIR, "movies.csv")

    ratings = pd.read_csv(ratings_path)
    movies  = pd.read_csv(movies_path)

    print(f"[Data Processor] Loaded {len(ratings)} ratings and {len(movies)} movies")

    # Detect if ratings already has title/genres merged in (100K processed format)
    if "title" in ratings.columns and "genres" in ratings.columns:
        print("[Data Processor] Detected pre-merged format (100K style)")
        df = ratings.copy()
    else:
        print("[Data Processor] Merging ratings + movies...")
        df = ratings.merge(movies[["movieId", "title", "genres"]], on="movieId", how="inner")

    # --- Data Cleaning ---
    missing_before = df.isnull().sum().sum()
    print(f"[Data Processor] Missing values before cleaning: {missing_before}")
    df = df.dropna()

    dupes_before = df.duplicated(subset=["userId", "movieId"]).sum()
    print(f"[Data Processor] Duplicate entries found: {dupes_before}")
    df = df.drop_duplicates(subset=["userId", "movieId"], keep="last")

    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(int)

    df["userId"]  = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"]  = df["rating"].astype(float)

    df = df.sort_values(["userId", "timestamp"]).reset_index(drop=True)

    print(f"[Data Processor] Final dataset: {len(df)} records")
    print(f"[Data Processor] Users: {df[\'userId\'].nunique()}, Movies: {df[\'movieId\'].nunique()}")

    os.makedirs(DATA_DIR, exist_ok=True)
    processed_path = os.path.join(DATA_DIR, "processed_movies.csv")
    df.to_csv(processed_path, index=False)
    print(f"[Data Processor] Saved processed data to {processed_path}")

    return df
'''

    with open(path, "r") as f:
        content = f.read()

    # Replace the entire load_and_process_data function
    pattern = r'def load_and_process_data\(\).*?(?=\ndef |\Z)'
    new_content = re.sub(pattern, new_load_fn.strip(), content, flags=re.DOTALL)

    if new_content == content:
        print("      Could not auto-patch — pattern mismatch. Patch manually (see below).")
        print("      The only change needed: add merge detection at the top of load_and_process_data()")
    else:
        with open(path, "w") as f:
            f.write(new_content)
        print("      Patched!  (backup saved as data_processor.py.bak)")
        print("      Added: auto-detect 1M vs 100K format")


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(out_dir: str) -> None:
    ratings_path = os.path.join(out_dir, "ratings.csv")
    movies_path  = os.path.join(out_dir, "movies.csv")

    if os.path.exists(ratings_path):
        r = pd.read_csv(ratings_path, nrows=3)
        print("\n── ratings.csv sample (first 3 rows) ──")
        print(r.to_string(index=False))

    if os.path.exists(movies_path):
        m = pd.read_csv(movies_path, nrows=3)
        print("\n── movies.csv sample (first 3 rows) ──")
        print(m.to_string(index=False))

    print("\n" + "=" * 60)
    print("  Migration complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. cd into your backend/ folder")
    print("  2. uv run uvicorn main:app --host 0.0.0.0 --port 8000")
    print("  3. Everything rebuilds automatically from 1M data")
    print("\nTo run the experiment benchmark:")
    print("  uv run python -m core.experiment_runner")
    print("\nExpected improvement:")
    print("  Recall@10: ~2.5% → 5-12%  (dataset density: 1.7% → 4.5%)")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate project to MovieLens 1M")
    parser.add_argument("--ratings", required=True, help="Path to ratings.dat (1M raw file)")
    parser.add_argument("--movies",  required=True, help="Path to movies.dat  (1M raw file)")
    parser.add_argument(
        "--project",
        default=".",
        help="Path to your backend/ folder (default: current directory)",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(args.project)
    core_dir     = os.path.join(project_root, "core")

    print("=" * 60)
    print("  MovieLens 1M Migration Script")
    print(f"  Project root : {project_root}")
    print("=" * 60)

    # Validate inputs
    for p, label in [(args.ratings, "ratings.dat"), (args.movies, "movies.dat")]:
        if not os.path.exists(p):
            print(f"\nERROR: {label} not found at: {p}")
            exit(1)

    convert_data(args.ratings, args.movies, project_root)
    delete_stale_files(project_root)
    patch_collaborative(core_dir)
    patch_data_processor(core_dir)
    print_summary(project_root)