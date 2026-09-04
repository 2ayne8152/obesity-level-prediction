"""
Random Forest Classifier Hyperparameter Tuning — Estimation of Obesity Levels
Based on Eating Habits and Physical Condition (UCI ML Repository, dataset id 544)
DOI: https://doi.org/10.24432/C5H31Z

Pipeline:
  1. Load data & Preprocess (imported from Preprocessing.py)
  2. Set up Pipeline and run RandomizedSearchCV over hyperparameter space
  3. Save best parameters and individual parameter evaluation metrics to text file
  4. Generate and save diagnostic tuning plots
"""

import contextlib
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

# Import centralized preprocessing function
from Preprocessing import get_preprocessed_data

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# PROGRESS BAR HELPER FOR JOBLIB
# --------------------------------------------------------------------------
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# --------------------------------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "tuning" / "Random_Forest"
RESULT_FILE = RESULTS_DIR / "rf_tuning_results.txt"

# --------------------------------------------------------------------------
# 1. LOAD AND PREPROCESS DATA
# --------------------------------------------------------------------------
print("Loading and preprocessing data...")
X_train, X_test, y_train, y_test, preprocessor, target_encoder = get_preprocessed_data()

print("\nClasses (in ordinal order):", list(target_encoder.classes_))

# --------------------------------------------------------------------------
# 2. SETUP PIPELINE & TUNING PARAMETERS
# --------------------------------------------------------------------------
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(random_state=42, n_jobs=-1),
        ),
    ]
)

param_distributions = {
    "classifier__n_estimators": [100, 200, 300, 500, 700, 1000],
    "classifier__max_depth": [10, 20, 30, 50, 75, 100, None],
    "classifier__max_features": ["sqrt", "log2", None],
    "classifier__criterion": ["gini", "entropy", "log_loss"],
    "classifier__min_samples_split": [2, 5, 10, 15],
    "classifier__min_samples_leaf": [1, 2, 4, 8],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_distributions,
    n_iter=100,
    scoring="accuracy",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=0,
    return_train_score=True,
)

# --------------------------------------------------------------------------
# 3. RUN TUNING WITH PROGRESS BAR
# --------------------------------------------------------------------------
print("\n============================================================")
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("============================================================")

total_fits = random_search.n_iter * cv.n_splits
start_time = time.time()

with tqdm_joblib(
    tqdm(desc="Tuning", total=total_fits, file=sys.stdout, mininterval=0.5)
):
    random_search.fit(X_train, y_train)

tuning_time = time.time() - start_time

print("\n============================================================")
print("TUNING RESULTS")
print("============================================================")

print("Best Parameters Found:")
for param, value in random_search.best_params_.items():
    clean_key = param.replace("classifier__", "")
    print(f"  - {clean_key}: {value}")

print(f"\nBest CV Accuracy: {random_search.best_score_ * 100:.2f}%")
print(f"Time Taken:       {tuning_time:.2f} seconds")

# --------------------------------------------------------------------------
# 4. SAVE BEST RESULTS AND ALL PARAMETER RESULTS
# --------------------------------------------------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

cv_results = pd.DataFrame(random_search.cv_results_)

parameters_to_plot = {
    "classifier__n_estimators": "Number of Trees",
    "classifier__max_depth": "Max Depth",
    "classifier__max_features": "Max Features",
    "classifier__min_samples_split": "Min Samples Split",
    "classifier__min_samples_leaf": "Min Samples Leaf",
    "classifier__criterion": "Splitting Criterion",
}

print(f"\nSaving results summary text file...")
with open(RESULT_FILE, "w") as file:
    file.write("RANDOM FOREST TUNING RESULTS\n")
    file.write("=" * 50 + "\n\n")

    # --- PART 1: FINAL TUNING RESULT (BEST) ---
    file.write("1. FINAL TUNING RESULT (BEST COMBINATION)\n")
    file.write("-" * 50 + "\n")
    file.write(f"Best CV Accuracy: {random_search.best_score_:.4f}\n")
    file.write(f"Tuning Execution Time: {tuning_time:.2f} seconds\n\n")
    file.write("Best Parameters:\n")
    for param, value in random_search.best_params_.items():
        file.write(f"  {param}: {value}\n")
    file.write("\n\n")

    # --- PART 2: ALL PARAMETER RESULTS ---
    file.write("2. ALL PARAMETER RESULTS (AVERAGE CV ACCURACY)\n")
    file.write("-" * 50 + "\n")

    for param_col, param_title in parameters_to_plot.items():
        search_col = f"param_{param_col}"

        temp_df = cv_results[[search_col, "mean_test_score"]].copy()
        temp_df[search_col] = temp_df[search_col].fillna("None").astype(str)
        grouped_df = (
            temp_df.groupby(search_col)["mean_test_score"].mean().reset_index()
        )

        def sort_key(x):
            try:
                return float(x)
            except ValueError:
                return float("inf")

        grouped_df["sort_val"] = grouped_df[search_col].apply(sort_key)
        grouped_df = grouped_df.sort_values("sort_val").drop(
            "sort_val", axis=1
        )

        file.write(f"{param_title} ({param_col})\n")
        for _, row in grouped_df.iterrows():
            file.write(
                f"  {row[search_col]}: {row['mean_test_score']:.4f}\n"
            )
        file.write("\n")

print(f"Saved {RESULT_FILE.relative_to(PROJECT_ROOT)}")

# --------------------------------------------------------------------------
# 5. GENERATE TUNING LINE GRAPHS
# --------------------------------------------------------------------------
print("\nGenerating hyperparameter tuning plots...")

for param_col, param_title in parameters_to_plot.items():
    search_col = f"param_{param_col}"
    temp_df = cv_results[[search_col, "mean_test_score"]].copy()
    temp_df[search_col] = temp_df[search_col].fillna("None").astype(str)

    grouped_df = (
        temp_df.groupby(search_col)["mean_test_score"].mean().reset_index()
    )

    def sort_key(x):
        try:
            return float(x)
        except ValueError:
            return float("inf")

    grouped_df["sort_val"] = grouped_df[search_col].apply(sort_key)
    grouped_df = grouped_df.sort_values("sort_val").drop("sort_val", axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(
        grouped_df[search_col],
        grouped_df["mean_test_score"] * 100,
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    plt.title(f"Accuracy vs {param_title}")
    plt.xlabel(param_title)
    plt.ylabel("Mean CV Accuracy (%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    safe_name = param_col.replace("classifier__", "")
    save_path = RESULTS_DIR / f"rf_{safe_name}_tuning.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path.relative_to(PROJECT_ROOT)}")

print(f"\nProcess Complete! Results saved to {RESULTS_DIR.relative_to(PROJECT_ROOT)}")