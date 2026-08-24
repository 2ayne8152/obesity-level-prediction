"""
RandomForest.py
Purpose: Runs Random Forest directly with the best hyperparameters found previously.
Evaluates the model and exports the .pkl files for GUI integration.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

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
# 1. LOAD DATA
# --------------------------------------------------------------------------
def load_data():
    data_path = "csv/ObesityDataSet_raw_and_data_sinthetic.csv" 
    df = pd.read_csv(data_path)
    y = df["NObeyesdad"]
    X = df.drop(columns=["NObeyesdad"])
    return X, y

X, y = load_data()

# --------------------------------------------------------------------------
# 2. STRICT PREPROCESSING (As defined in your documentation)
# --------------------------------------------------------------------------
binary_cols = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
nominal_cols = ["CAEC", "CALC", "MTRANS"]          
numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
categorical_cols = binary_cols + nominal_cols

# Encode target labels (7 obesity classes -> integers)
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# ColumnTransformer: one-hot encode categoricals, standardscaler for numeric
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

# --------------------------------------------------------------------------
# 3. FIT RANDOM FOREST WITH BEST HYPERPARAMETERS
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
# 4. EVALUATION
# --------------------------------------------------------------------------
final_preds = best_model.predict(X_test)
final_probs = best_model.predict_proba(X_test)

final_acc = accuracy_score(y_test, final_preds)
weighted_roc_auc = roc_auc_score(y_test, final_probs, multi_class="ovr", average="weighted")

print(f"\nTest accuracy: {final_acc:.4f}")
print(f"Weighted-average ROC-AUC (OvR): {weighted_roc_auc:.4f}")
print("\nClassification report:\n")
print(classification_report(y_test, final_preds, target_names=target_encoder.classes_, digits=4))

# Ensure output directories exist
os.makedirs("results/graphs", exist_ok=True)
os.makedirs("saved_model", exist_ok=True)

# Generate Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — Random Forest (Best Params)")
plt.tight_layout()
plt.savefig("results/graphs/rf_confusion_matrix.png", dpi=150)
plt.close()

# ==========================================================================
# 4.5. SAMPLE DECISION TREE GRAPH (ADD THIS NEW SECTION HERE)
# ==========================================================================
print("\nGenerating sample decision tree graph...")

# 1. Recover feature names after one-hot encoding
ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"]
ohe_feature_names = list(ohe.get_feature_names_out(categorical_cols))
all_feature_names = ohe_feature_names + numeric_cols

# 2. Extract the very first decision tree (index 0) from the 300 tuned trees
sample_tree = best_model.named_steps["classifier"].estimators_[0]

# 3. Plot the tree
plt.figure(figsize=(25, 12))
plot_tree(
    sample_tree,
    feature_names=all_feature_names,
    class_names=list(target_encoder.classes_),
    filled=True,
    rounded=True,
    max_depth=3,  # Capped at 3 so the image is readable in a Word document
    fontsize=9
)

plt.title("Sample Decision Tree from Tuned Random Forest (Max Depth = 3)")
plt.tight_layout()

tree_graph_path = "results/graphs/rf_sample_tree.png"
plt.savefig(tree_graph_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved sample decision tree to {tree_graph_path}")

# --------------------------------------------------------------------------
# 5. SAVE THE FINAL MODEL FOR GUI
# --------------------------------------------------------------------------
joblib.dump(best_model, "pkl/random_forest_model.pkl")
joblib.dump(target_encoder, "pkl/rf_target_encoder.pkl")
print("\nSaved trained pipeline to pkl/random_forest_model.pkl")
print("Saved target label encoder to pkl/rf_target_encoder.pkl")