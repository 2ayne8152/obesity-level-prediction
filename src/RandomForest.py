"""
RandomForest.py
Purpose: Runs Random Forest directly with the best hyperparameters found previously.
Evaluates the model and exports the .pkl files for GUI integration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.tree import plot_tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier

# Import centralized preprocessing function
from Preprocessing import get_preprocessed_data

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------------------------------
# Assuming this script is inside 'src/', .parent.parent gets you to the root 'Obesity' folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "models" / "Random_Forest"
MODELS_DIR = PROJECT_ROOT / "models"

# --------------------------------------------------------------------------
# BEST HYPERPARAMETERS (Copy these from your tuning results text file)
# --------------------------------------------------------------------------
BEST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "max_features": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "criterion": "entropy"
}

# --------------------------------------------------------------------------
# 1. LOAD AND PREPROCESS DATA
# --------------------------------------------------------------------------
print("Loading and preprocessing data...")
# Preprocessing.py handles data loading, ColumnTransformer scaling/encoding, 
# and the clinical-order label encoding automatically.
X_train, X_test, y_train, y_test, preprocessor, target_encoder = get_preprocessed_data()

print("\nClasses (in ordinal order):", list(target_encoder.classes_))

# --------------------------------------------------------------------------
# 2. FIT RANDOM FOREST WITH BEST HYPERPARAMETERS
# --------------------------------------------------------------------------
best_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **BEST_PARAMS
            ),
        ),
    ]
)

print("\nFitting final Random Forest model...")
best_model.fit(X_train, y_train)

# --------------------------------------------------------------------------
# 3. EVALUATION
# --------------------------------------------------------------------------
final_preds = best_model.predict(X_test)
final_probs = best_model.predict_proba(X_test)

final_acc = accuracy_score(y_test, final_preds)
weighted_roc_auc = roc_auc_score(y_test, final_probs, multi_class="ovr", average="weighted")

print(f"\nTest accuracy: {final_acc:.4f}")
print(f"Weighted-average ROC-AUC (OvR): {weighted_roc_auc:.4f}")
print("\nClassification report:\n")
print(classification_report(y_test, final_preds, target_names=target_encoder.classes_, digits=4))

# Ensure output directories exist dynamically
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Generate Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — Random Forest (Best Params)")
plt.tight_layout()

cm_path = RESULTS_DIR / "rf_confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Saved {cm_path.relative_to(PROJECT_ROOT)}")

# ==========================================================================
# 4. SAMPLE DECISION TREE GRAPH
# ==========================================================================
print("\nGenerating sample decision tree graph...")

# 1. Recover feature names dynamically from the preprocessor and clean prefixes
raw_feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
clean_feature_names = [name.split("__")[-1] for name in raw_feature_names]

# 2. Extract the very first decision tree (index 0) from the 300 tuned trees
sample_tree = best_model.named_steps["classifier"].estimators_[0]

# 3. Plot the tree
plt.figure(figsize=(25, 12))
plot_tree(
    sample_tree,
    feature_names=clean_feature_names,
    class_names=list(target_encoder.classes_),
    filled=True,
    rounded=True,
    max_depth=3,  # Capped at 3 so the image is readable in a document
    fontsize=9
)

plt.title("Sample Decision Tree from Tuned Random Forest (Max Depth = 3)")
plt.tight_layout()

tree_graph_path = RESULTS_DIR / "rf_sample_tree.png"
plt.savefig(tree_graph_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved sample decision tree to {tree_graph_path.relative_to(PROJECT_ROOT)}")

# --------------------------------------------------------------------------
# 5. SAVE THE FINAL MODEL FOR GUI
# --------------------------------------------------------------------------
model_path = MODELS_DIR / "random_forest_model.pkl"
encoder_path = MODELS_DIR / "rf_target_encoder.pkl"

joblib.dump(best_model, model_path)
joblib.dump(target_encoder, encoder_path)

print(f"\nSaved trained pipeline to {model_path.relative_to(PROJECT_ROOT)}")
print(f"Saved target label encoder to {encoder_path.relative_to(PROJECT_ROOT)}")