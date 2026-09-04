"""
KNN Classifier — Estimation of Obesity Levels Based on Eating Habits
and Physical Condition (UCI ML Repository, dataset id 544)
DOI: https://doi.org/10.24432/C5H31Z

This version runs KNN directly with the best hyperparameters found
previously via tuning (no re-tuning, no baseline run).

Pipeline:
  1. Load data & Preprocess (imported from Preprocessing.py)
  2. Fit KNN with best-known hyperparameters
  3. Evaluation (accuracy, precision, recall, F1, ROC-AUC, classification
     report, confusion matrix, timing)
  4. Save the final model
  5. Generate learning curve
"""

import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
# ADDED MISSING IMPORTS HERE
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

# Import centralized preprocessing function
from Preprocessing import get_preprocessed_data

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "models" / "KNN"
MODELS_DIR = PROJECT_ROOT / "models"

# --------------------------------------------------------------------------
# BEST HYPERPARAMETERS (found previously via tuning)
# --------------------------------------------------------------------------
BEST_PARAMS = {
    "n_neighbors": 5,
    "weights": "distance",
    "metric": "manhattan",
    "p": 1,
}

# --------------------------------------------------------------------------
# 1. LOAD AND PREPROCESS DATA
# --------------------------------------------------------------------------
print("Loading and preprocessing data...")
X_train, X_test, y_train, y_test, preprocessor, target_encoder = get_preprocessed_data()

print("\nClasses (in ordinal order):", list(target_encoder.classes_))

# --------------------------------------------------------------------------
# 2. FIT KNN WITH BEST-KNOWN HYPERPARAMETERS
# --------------------------------------------------------------------------
best_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            KNeighborsClassifier(
                n_jobs=-1,
                **BEST_PARAMS,
            ),
        ),
    ]
)

print("\nFitting KNN with best-known hyperparameters...")
train_start = time.time()
best_model.fit(X_train, y_train)
training_time = time.time() - train_start

# --------------------------------------------------------------------------
# 3. EVALUATION ON TEST SET
# --------------------------------------------------------------------------
run_start = time.time()
final_preds = best_model.predict(X_test)
final_probs = best_model.predict_proba(X_test)
run_time = time.time() - run_start

final_acc = accuracy_score(y_test, final_preds)
print(f"\nTest accuracy: {final_acc:.4f}")
print(f"Model training time:  {training_time:.5f} seconds")
print(f"Model inference time: {run_time:.5f} seconds")

print("\nClassification report:\n")
print(
    classification_report(
        y_test, final_preds, target_names=target_encoder.classes_, digits=4
    )
)

# --------------------------------------------------------------------------
# ROC-AUC (multiclass, one-vs-rest)
# --------------------------------------------------------------------------
y_test_bin = label_binarize(y_test, classes=np.unique(y_train))

macro_roc_auc = roc_auc_score(y_test_bin, final_probs, multi_class="ovr", average="macro")
weighted_roc_auc = roc_auc_score(y_test_bin, final_probs, multi_class="ovr", average="weighted")
per_class_roc_auc = roc_auc_score(y_test_bin, final_probs, multi_class="ovr", average=None)

print(f"\nMacro-average ROC-AUC (OvR): {macro_roc_auc:.4f}")
print(f"Weighted-average ROC-AUC (OvR): {weighted_roc_auc:.4f}")

roc_auc_df = pd.DataFrame(
    {
        "Class": target_encoder.classes_,
        "ROC-AUC": per_class_roc_auc,
    }
).sort_values("ROC-AUC", ascending=False)

print("\nPer-class ROC-AUC (OvR):")
print(roc_auc_df.to_string(index=False))

# Weighted precision/recall/F1, for a report-table summary line
precision = precision_score(y_test, final_preds, average="weighted")
recall = recall_score(y_test, final_preds, average="weighted")
f1 = f1_score(y_test, final_preds, average="weighted")

print("\nSummary metrics:")
print(f"Accuracy:  {final_acc * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1 * 100:.2f}%")
print(f"ROC-AUC:   {weighted_roc_auc * 100:.2f}%")

# Ensure output directories exist dynamically
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Confusion matrix
cm = confusion_matrix(y_test, final_preds)
fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — KNN (Best Params)")
plt.tight_layout()

cm_path = RESULTS_DIR / "knn_confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\nSaved {cm_path.relative_to(PROJECT_ROOT)}")

# --------------------------------------------------------------------------
# 4. SAVE THE FINAL MODEL
# --------------------------------------------------------------------------
model_path = MODELS_DIR / "knn_model.pkl"
encoder_path = MODELS_DIR / "knn_target_encoder.pkl"

joblib.dump(best_model, model_path)
joblib.dump(target_encoder, encoder_path)

print(f"\nSaved trained pipeline to {model_path.relative_to(PROJECT_ROOT)}")
print(f"Saved target label encoder to {encoder_path.relative_to(PROJECT_ROOT)}")

# --------------------------------------------------------------------------
# 5. LEARNING CURVE
# --------------------------------------------------------------------------
print("\nGenerating Learning Curve...")
train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_model,
    X=X_train,
    y=y_train,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
    shuffle=True,
    random_state=RANDOM_STATE,
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(8, 6))

plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
plt.plot(train_sizes, val_mean, marker="s", label="Validation Accuracy")

plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.2,
)

plt.fill_between(
    train_sizes,
    val_mean - val_std,
    val_mean + val_std,
    alpha=0.2,
)

plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve - KNN (Best Params)")
plt.grid(True)
plt.legend()

plt.tight_layout()

lc_path = RESULTS_DIR / "knn_learning_curve.png"
plt.savefig(lc_path, dpi=150)
plt.close()
print(f"Saved {lc_path.relative_to(PROJECT_ROOT)}")