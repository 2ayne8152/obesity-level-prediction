"""
XGBoost Classifier — Estimation of Obesity Levels Based on Eating Habits
and Physical Condition (UCI ML Repository, dataset id 544)
DOI: https://doi.org/10.24432/C5H31Z

This version runs XGBoost directly with the best hyperparameters found
previously via RandomizedSearchCV (no re-tuning, no baseline run).

Pipeline:
  1. Load data (via ucimlrepo, with a CSV fallback)
  2. Preprocessing (encode categoricals, encode target, train/test split)
  3. Fit XGBoost with best-known hyperparameters
  4. Evaluation (accuracy, classification report, ROC-AUC, confusion matrix, feature importance)
  5. Learning curve

Install requirements:
    pip install ucimlrepo xgboost scikit-learn pandas numpy matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import learning_curve, train_test_split
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

from xgboost import XGBClassifier

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# BEST HYPERPARAMETERS (found previously via RandomizedSearchCV)
# --------------------------------------------------------------------------
BEST_PARAMS = {
    "subsample": 1.0,
    "reg_lambda": 1.5,
    "reg_alpha": 0,
    "n_estimators": 400,
    "min_child_weight": 2,
    "max_depth": 8,
    "learning_rate": 0.05,
    "gamma": 0,
    "colsample_bytree": 1.0,
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
# Identify column types
binary_cols = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
nominal_cols = ["CAEC", "CALC", "MTRANS"]          # multi-category, unordered
numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Keep only columns that actually exist (robust to minor naming differences)
binary_cols = [c for c in binary_cols if c in X.columns]
nominal_cols = [c for c in nominal_cols if c in X.columns]
numeric_cols = [c for c in numeric_cols if c in X.columns]
categorical_cols = binary_cols + nominal_cols

# Encode target labels (7 obesity classes -> integers)
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print("\nClasses:", list(target_encoder.classes_))

# ColumnTransformer: one-hot encode categoricals, standardscaler for numeric.
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Train/test split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

# --------------------------------------------------------------------------
# 3. FIT XGBOOST WITH BEST-KNOWN HYPERPARAMETERS
# --------------------------------------------------------------------------
n_classes = len(np.unique(y_encoded))

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
# 4. EVALUATION ON TEST SET
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

# Confusion matrix
cm = confusion_matrix(y_test, final_preds)
fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — XGBoost (Best Params)")
plt.tight_layout()
plt.savefig("image/confusion_matrix.png", dpi=150)
plt.close()
print("\nSaved image/confusion_matrix.png")

# --------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE
# --------------------------------------------------------------------------
# Recover feature names after one-hot encoding
ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"]
ohe_feature_names = list(ohe.get_feature_names_out(categorical_cols))
all_feature_names = ohe_feature_names + numeric_cols

importances = best_model.named_steps["classifier"].feature_importances_
feat_imp = (
    pd.Series(importances, index=all_feature_names)
    .sort_values(ascending=False)
    .head(20)
)

plt.figure(figsize=(8, 8))
sns.barplot(x=feat_imp.values, y=feat_imp.index, color="steelblue")
plt.title("Top 20 Feature Importances — XGBoost (Best Params)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("image/feature_importance.png", dpi=150)
plt.close()
print("Saved image/feature_importance.png")

# --------------------------------------------------------------------------
# 6. SAVE THE FINAL MODEL
# --------------------------------------------------------------------------
import joblib

joblib.dump(best_model, "csv/xgboost_obesity_model.pkl")
joblib.dump(target_encoder, "csv/target_label_encoder.pkl")
print("\nSaved trained pipeline to csv/xgboost_obesity_model.pkl")
print("Saved target label encoder to csv/target_label_encoder.pkl")

# --------------------------------------------------------------------------
# LEARNING CURVE
# --------------------------------------------------------------------------
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
plt.savefig("tuning result/learning_curve.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# Example: how to load and use the saved model later
# --------------------------------------------------------------------------
# best_model = joblib.load("xgboost_obesity_model.pkl")
# target_encoder = joblib.load("target_label_encoder.pkl")
# preds = best_model.predict(new_data_df)
# predicted_labels = target_encoder.inverse_transform(preds)