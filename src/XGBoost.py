"""
XGBoost Classifier — Estimation of Obesity Levels Based on Eating Habits
and Physical Condition (UCI ML Repository, dataset id 544)
DOI: https://doi.org/10.24432/C5H31Z

This version runs XGBoost directly with the best hyperparameters found
previously via RandomizedSearchCV (no re-tuning, no baseline run).

Pipeline:
  1. Load data & Preprocess (imported from Preprocessing.py)
  2. Fit XGBoost with best-known hyperparameters
  3. Evaluation (accuracy, classification report, ROC-AUC, confusion matrix, feature importance)
  4. Learning curve
  5. Save the final model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

from xgboost import XGBClassifier

# Import centralized preprocessing function
from Preprocessing import get_preprocessed_data

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------------------------------
# Assuming this script is inside 'src/', .parent.parent gets you to the root 'Obesity' folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "models" / "XGBoost"
MODELS_DIR = PROJECT_ROOT / "models"

# --------------------------------------------------------------------------
# BEST HYPERPARAMETERS (found previously via RandomizedSearchCV)
# --------------------------------------------------------------------------
BEST_PARAMS = {
    "subsample": 1.0,
    "reg_lambda": 1,
    "reg_alpha": 0,
    "n_estimators": 600,
    "min_child_weight": 1,
    "max_depth": 5,
    "learning_rate": 0.02,
    "gamma": 0.3,
    "colsample_bytree": 0.7,
}

# --------------------------------------------------------------------------
# 1. LOAD AND PREPROCESS DATA
# --------------------------------------------------------------------------
print("Loading and preprocessing data...")
X_train, X_test, y_train, y_test, preprocessor, target_encoder = get_preprocessed_data()

print("\nClasses (in ordinal order):", list(target_encoder.classes_))

# --------------------------------------------------------------------------
# 2. FIT XGBOOST WITH BEST-KNOWN HYPERPARAMETERS
# --------------------------------------------------------------------------
n_classes = len(target_encoder.classes_)

best_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **BEST_PARAMS,
            ),
        ),
    ]
)

print("\nFitting XGBoost with best-known hyperparameters...")
best_model.fit(X_train, y_train)

# --------------------------------------------------------------------------
# 3. EVALUATION ON TEST SET
# --------------------------------------------------------------------------
final_preds = best_model.predict(X_test)
final_acc = accuracy_score(y_test, final_preds)
print(f"\nTest accuracy: {final_acc:.4f}")

print("\nClassification report:\n")
print(
    classification_report(
        y_test, final_preds, target_names=target_encoder.classes_, digits=4
    )
)

# --------------------------------------------------------------------------
# ROC-AUC (multiclass, one-vs-rest)
# --------------------------------------------------------------------------
final_probs = best_model.predict_proba(X_test)

macro_roc_auc = roc_auc_score(
    y_test, final_probs, multi_class="ovr", average="macro"
)
weighted_roc_auc = roc_auc_score(
    y_test, final_probs, multi_class="ovr", average="weighted"
)
per_class_roc_auc = roc_auc_score(
    y_test, final_probs, multi_class="ovr", average=None
)

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

# Ensure output directories exist dynamically
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Confusion matrix
cm = confusion_matrix(y_test, final_preds)
fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — XGBoost (Best Params)")
plt.tight_layout()

cm_path = RESULTS_DIR / "xgboost_confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\nSaved {cm_path.relative_to(PROJECT_ROOT)}")

# --------------------------------------------------------------------------
# 4. FEATURE IMPORTANCE
# --------------------------------------------------------------------------
# Recover feature names dynamically from the preprocessor and clean prefixes
raw_feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
clean_feature_names = [name.split("__")[-1] for name in raw_feature_names]

importances = best_model.named_steps["classifier"].feature_importances_
feat_imp = (
    pd.Series(importances, index=clean_feature_names)
    .sort_values(ascending=False)
    .head(20)
)

plt.figure(figsize=(8, 8))
sns.barplot(x=feat_imp.values, y=feat_imp.index, color="steelblue")
plt.title("Top 20 Feature Importances — XGBoost (Best Params)")
plt.xlabel("Importance")
plt.tight_layout()

feat_imp_path = RESULTS_DIR / "xgboost_feature_importance.png"
plt.savefig(feat_imp_path, dpi=150)
plt.close()
print(f"Saved {feat_imp_path.relative_to(PROJECT_ROOT)}")

# --------------------------------------------------------------------------
# 5. SAVE THE FINAL MODEL
# --------------------------------------------------------------------------
model_path = MODELS_DIR / "xgboost_obesity_model.pkl"
encoder_path = MODELS_DIR / "xgboost_target_encoder.pkl"

joblib.dump(best_model, model_path)
joblib.dump(target_encoder, encoder_path)

print(f"\nSaved trained pipeline to {model_path.relative_to(PROJECT_ROOT)}")
print(f"Saved target label encoder to {encoder_path.relative_to(PROJECT_ROOT)}")

# --------------------------------------------------------------------------
# 6. LEARNING CURVE
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

plt.plot(train_sizes, train_mean, marker='o', label="Training Accuracy")
plt.plot(train_sizes, val_mean, marker='s', label="Validation Accuracy")

plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.2
)

plt.fill_between(
    train_sizes,
    val_mean - val_std,
    val_mean + val_std,
    alpha=0.2
)

plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve - XGBoost (Best Params)")
plt.grid(True)
plt.legend()

plt.tight_layout()

lc_path = RESULTS_DIR / "xgboost_learning_curve.png"
plt.savefig(lc_path, dpi=150)
plt.close()
print(f"Saved {lc_path.relative_to(PROJECT_ROOT)}")