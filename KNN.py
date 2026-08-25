"""
KNN Classifier — Estimation of Obesity Levels Based on Eating Habits
and Physical Condition (UCI ML Repository, dataset id 544)
DOI: https://doi.org/10.24432/C5H31Z

This version runs KNN directly with the best hyperparameters found
previously via tuning (no re-tuning, no baseline run).

Preprocessing is shared with XGBoost.py / LogisticRegression.py:
  - binary_cols:  2-category columns -> single 0/1 column (OrdinalEncoder)
  - ordinal_cols: genuinely ORDERED categories (CAEC/CALC) -> single integer
                  column, in their real-world order (OrdinalEncoder)
  - nominal_cols: unordered multi-category column (MTRANS) -> one-hot
                  encoded, one category dropped
  - numeric_cols: continuous features -> standardized

Pipeline:
  1. Load data (via ucimlrepo, with a CSV fallback)
  2. Preprocessing (encode categoricals, encode target, train/test split)
  3. Fit KNN with best-known hyperparameters
  4. Evaluation (accuracy, precision, recall, F1, ROC-AUC, classification
     report, confusion matrix, timing)
  5. Save the final model

Install requirements:
    pip install ucimlrepo scikit-learn pandas numpy matplotlib seaborn
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
import joblib

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

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
# 1. LOAD DATA
# --------------------------------------------------------------------------
def load_data():
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=544)
        X = dataset.data.features
        y = dataset.data.targets.squeeze()  # Series
        return X, y
    except Exception as e:
        print(f"ucimlrepo fetch failed ({e}); falling back to local CSV.")
        df = pd.read_csv("csv/ObesityDataSet_raw_and_data_sinthetic.csv")
        y = df["NObeyesdad"]
        X = df.drop(columns=["NObeyesdad"])
        return X, y

X, y = load_data()
print("Feature matrix shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# --------------------------------------------------------------------------
# 2. PREPROCESSING
# --------------------------------------------------------------------------
# Identify column types.
#   - binary_cols:  2-category columns -> single 0/1 column
#   - ordinal_cols: genuinely ORDERED categories -> single integer column,
#                   in their real-world order (no/Sometimes/Frequently/Always)
#   - nominal_cols: unordered multi-category columns -> one-hot encoded,
#                   with one category dropped to avoid the dummy trap
#   - numeric_cols: continuous features -> standardized
binary_cols = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
ordinal_cols = ["CAEC", "CALC"]                    # truly ordered categories
nominal_cols = ["MTRANS"]                          # unordered, multi-category
numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Keep only columns that actually exist (robust to minor naming differences)
binary_cols = [c for c in binary_cols if c in X.columns]
ordinal_cols = [c for c in ordinal_cols if c in X.columns]
nominal_cols = [c for c in nominal_cols if c in X.columns]
numeric_cols = [c for c in numeric_cols if c in X.columns]

# Explicit category orders. For binary columns the order just fixes which
# label maps to 0 vs 1; for CAEC/CALC the order is the real severity order.
binary_categories = [["Female", "Male"], ["no", "yes"], ["no", "yes"], ["no", "yes"], ["no", "yes"]]
binary_categories = binary_categories[: len(binary_cols)]

ordinal_categories = [["no", "Sometimes", "Frequently", "Always"]] * len(ordinal_cols)

# Encode target labels in their natural CLINICAL order (kept consistent with
# XGBoost.py / LogisticRegression.py, so class_names/report ordering is
# identical across all scripts for easy side-by-side comparison).
class_order = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

target_encoder = LabelEncoder()
target_encoder.classes_ = np.array(class_order)
y_encoded = target_encoder.transform(y)
print("\nClasses (in ordinal order):", list(target_encoder.classes_))

# ColumnTransformer: ordinal-encode binary/ordinal columns, one-hot encode
# only the genuinely nominal column (MTRANS, drop="first"), and scale
# numeric columns. Same preprocessing as XGBoost.py / LogisticRegression.py.
preprocessor = ColumnTransformer(
    transformers=[
        ("bin", OrdinalEncoder(categories=binary_categories), binary_cols),
        ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
        (
            "nom",
            OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            nominal_cols,
        ),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Train/test split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

# --------------------------------------------------------------------------
# 3. FIT KNN WITH BEST-KNOWN HYPERPARAMETERS
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
# 4. EVALUATION ON TEST SET
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

# Ensure output directories exist
os.makedirs("results/graphs", exist_ok=True)
os.makedirs("pkl", exist_ok=True)

# Confusion matrix
cm = confusion_matrix(y_test, final_preds)
fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — KNN (Best Params)")
plt.tight_layout()
plt.savefig("results/graphs/knn_confusion_matrix.png", dpi=150)
plt.close()
print("\nSaved results/graphs/knn_confusion_matrix.png")

# --------------------------------------------------------------------------
# 5. SAVE THE FINAL MODEL
# --------------------------------------------------------------------------
joblib.dump(best_model, "pkl/knn_model.pkl")
joblib.dump(target_encoder, "pkl/knn_target_encoder.pkl")
print("\nSaved trained pipeline to pkl/knn_model.pkl")
print("Saved target label encoder to pkl/knn_target_encoder.pkl")

# --------------------------------------------------------------------------
# Example: how to load and use the saved model later
# --------------------------------------------------------------------------
# best_model = joblib.load("pkl/knn_model.pkl")
# target_encoder = joblib.load("pkl/knn_target_encoder.pkl")
# preds = best_model.predict(new_data_df)
# predicted_labels = target_encoder.inverse_transform(preds)