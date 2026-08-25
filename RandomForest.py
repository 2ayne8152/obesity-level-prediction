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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
# SEVERITY-ORDERED TARGET ENCODER
# --------------------------------------------------------------------------
# LabelEncoder sorts classes alphabetically, which puts Obesity types before
# Overweight levels — medically wrong ordering. SeverityOrderedTargetEncoder
# forces the correct low-to-high severity sequence, matching LogisticRegression.py
# and XGBoost.py (which already uses severity order via its manual mapping).
# Tree-based models don't use label order for splitting, so accuracy is
# unaffected — only the class labels in the report/confusion matrix change
# from alphabetical to severity order, consistent across all three scripts.
SEVERITY_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

class SeverityOrderedTargetEncoder:
    """Encodes the target in explicit obesity-severity order rather than
    alphabetically. Exposes the same .classes_ / .fit_transform() /
    .inverse_transform() interface as LabelEncoder so nothing else in
    the script needs to change."""

    def __init__(self, categories):
        self.classes_ = np.array(categories)
        self._to_code = {c: i for i, c in enumerate(categories)}
        self._to_label = {i: c for i, c in enumerate(categories)}

    def fit_transform(self, y):
        return np.array([self._to_code[v] for v in y])

    def transform(self, y):
        return np.array([self._to_code[v] for v in y])

    def inverse_transform(self, y):
        return np.array([self._to_label[int(i)] for i in y])


import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler


RANDOM_STATE = 42

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
# LogisticRegression.py, even though XGBoost itself doesn't require an
# ordered target — this keeps class_names/report ordering identical across
# both scripts for easy side-by-side comparison).
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
# numeric columns. Same preprocessing as LogisticRegression.py.
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
os.makedirs("pkl", exist_ok=True)

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
# 4.5. SAMPLE DECISION TREE GRAPH
# ==========================================================================
print("\nGenerating sample decision tree graph...")

# 1. Recover feature names from the new ColumnTransformer structure
# Binary and Ordinal columns keep their original names
bin_feature_names = binary_cols
ord_feature_names = ordinal_cols

# Nominal columns get expanded by OneHotEncoder
nom_ohe = best_model.named_steps["preprocessor"].named_transformers_["nom"]
nom_feature_names = list(nom_ohe.get_feature_names_out(nominal_cols))

# Combine all names in the EXACT order they appear in the ColumnTransformer
all_feature_names = bin_feature_names + ord_feature_names + nom_feature_names + numeric_cols

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