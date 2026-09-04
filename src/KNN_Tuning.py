"""
KNN Classifier Hyperparameter Tuning — Estimation of Obesity Levels Based on
Eating Habits and Physical Condition (UCI ML Repository, dataset id 544)
DOI: https://doi.org/10.24432/C5H31Z

Pipeline:
  1. Load data & Preprocess (imported from Preprocessing.py)
  2. Set up Pipeline and run 5-fold CV via GridSearchCV
  3. Output best hyperparameters and CV metrics
  4. Generate & save tuning plots and CSV results
"""

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Import centralized preprocessing function
from Preprocessing import get_preprocessed_data

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------------------------------
# Assuming this script is inside 'src/', .parent.parent gets you to the root 'Obesity' folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "tuning" / "KNN"

# --------------------------------------------------------------------------
# PARAMETER GRID CONFIGURATION
# --------------------------------------------------------------------------
PARAM_GRID = {
    "classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "classifier__weights": ["uniform", "distance"],
    "classifier__metric": ["euclidean", "manhattan", "minkowski"],
    "classifier__p": [1, 2],
}

# --------------------------------------------------------------------------
# 1. LOAD AND PREPROCESS DATA
# --------------------------------------------------------------------------
print("Loading and preprocessing data...")
X_train, X_test, y_train, y_test, preprocessor, target_encoder = get_preprocessed_data()

print("\nClasses (in ordinal order):", list(target_encoder.classes_))

# --------------------------------------------------------------------------
# 2. SET UP PIPELINE AND GRID SEARCH
# --------------------------------------------------------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            KNeighborsClassifier(n_jobs=-1),
        ),
    ]
)

print("\nInitializing GridSearchCV (5-fold CV)...")
grid = GridSearchCV(
    pipeline,
    PARAM_GRID,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

print("Fitting Grid Search across parameter space...")
start_time = time.time()
grid.fit(X_train, y_train)
tuning_time = time.time() - start_time

# --------------------------------------------------------------------------
# 3. PRINT TUNING RESULTS
# --------------------------------------------------------------------------
print("\n============================================================")
print("TUNING RESULTS")
print("============================================================")

print("Best Parameters:")
for key, value in grid.best_params_.items():
    clean_key = key.replace("classifier__", "")
    print(f"  - {clean_key}: {value}")

print(f"\nBest CV Accuracy: {grid.best_score_ * 100:.2f}%")
print(f"Time Taken:       {tuning_time:.2f} seconds")

# --------------------------------------------------------------------------
# 4. GENERATE AND SAVE TUNING PLOTS & RESULTS
# --------------------------------------------------------------------------
# Ensure output directory exists dynamically
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

cv_results = pd.DataFrame(grid.cv_results_)

print("\nGenerating tuning diagnostic plots...")
raw_params = ["n_neighbors", "weights", "metric", "p"]

for param in raw_params:
    col_name = f"param_classifier__{param}"
    grouped = cv_results.groupby(col_name)["mean_test_score"].mean().sort_index()

    plt.figure(figsize=(8, 5))
    plt.plot(
        [str(v) for v in grouped.index],
        grouped.values * 100,
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    plt.title(f"Mean CV Accuracy vs {param.capitalize()}")
    plt.ylabel("Mean CV Accuracy (%)")
    plt.xlabel(param.capitalize())
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = RESULTS_DIR / f"knn_{param}_tuning.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved {plot_path.relative_to(PROJECT_ROOT)}")

# Save full results table to CSV
csv_path = RESULTS_DIR / "knn_cv_results.csv"
cv_results.to_csv(csv_path, index=False)
print(f"Saved {csv_path.relative_to(PROJECT_ROOT)}")