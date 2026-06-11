"""
Module 5b - Collaborative Filtering Hyperparameter Tuning
Performs grid search to find the best SVD hyperparameters.
Evaluates model on the chronological validation split using RMSE.
"""

import os
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.accuracy import rmse
import itertools
from concurrent.futures import ProcessPoolExecutor
from core.splitter import get_train_val_test

# Worker function for parallel hyperparameter evaluation
def evaluate_params(params):
    n_factors, n_epochs, lr_all, reg_all = params
    
    # Reloading/building datasets inside worker to avoid serializing complex Surprise structures
    # We load them from CSV directly or pass them. Reading from disk is fast enough, but let's load
    # train/val inside the worker or read from disk. Reading from disk is very clean.
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    train_path = os.path.join(ROOT_DIR, "train.csv")
    val_path = os.path.join(ROOT_DIR, "validation.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    reader = Reader(rating_scale=(0.5, 5.0))
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    trainset = train_data.build_full_trainset()
    
    val_testset = list(val_df[['userId', 'movieId', 'rating']].itertuples(index=False, name=None))
    
    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=42
    )
    model.fit(trainset)
    predictions = model.test(val_testset)
    val_rmse = rmse(predictions, verbose=False)
    
    return {
        'n_factors': n_factors,
        'n_epochs': n_epochs,
        'lr_all': lr_all,
        'reg_all': reg_all,
        'val_rmse': val_rmse
    }

def main():
    # Define hyperparameter grid
    n_factors_list = [20, 50, 100, 150]
    n_epochs_list = [20, 50, 100]
    lr_all_list = [0.002, 0.005, 0.01, 0.02]
    reg_all_list = [0.02, 0.05, 0.1, 0.2]
    
    combinations = list(itertools.product(n_factors_list, n_epochs_list, lr_all_list, reg_all_list))
    total_runs = len(combinations)
    
    print(f"[Tuning] Starting parallel Grid Search on {total_runs} combinations...")
    
    results = []
    # Use ProcessPoolExecutor to speed up grid search
    # Limit worker processes to avoid overloading the CPU, but still utilize parallel power
    max_workers = min(os.cpu_count() or 4, 8)
    print(f"[Tuning] Using {max_workers} parallel workers.")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = [executor.submit(evaluate_params, comb) for comb in combinations]
        
        # Gather results and print progress
        for i, future in enumerate(futures):
            res = future.result()
            results.append(res)
            if (i + 1) % 20 == 0 or (i + 1) == total_runs:
                print(f"  Progress: {i+1}/{total_runs} completed (Last RMSE: {res['val_rmse']:.5f})")
                
    # Create DataFrame from results
    df = pd.DataFrame(results)
    df = df.sort_values("val_rmse").reset_index(drop=True)
    
    # Print the top results
    print("\n[Tuning] --- Top 10 Parameter Combinations ---")
    print(df.head(10).to_string(index=False))
    
    # Save the full results table
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    results_path = os.path.join(reports_dir, "svd_tuning_results.csv")
    df.to_csv(results_path, index=False)
    print(f"\n[Tuning] Full results saved to: {results_path}")
    
    best = df.iloc[0]
    print("\n[Tuning] --- Best Parameters Found ---")
    print(f"  n_factors : {int(best['n_factors'])}")
    print(f"  n_epochs  : {int(best['n_epochs'])}")
    print(f"  lr_all    : {best['lr_all']}")
    print(f"  reg_all   : {best['reg_all']}")
    print(f"  Best RMSE : {best['val_rmse']:.5f}")

if __name__ == "__main__":
    main()
