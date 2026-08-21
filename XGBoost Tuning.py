"""
XGBoost Classifier — Estimation of Obesity Levels Based on Eating Habits
and Physical Condition (UCI ML Repository, dataset id 544)
DOI: https://doi.org/10.24432/C5H31Z

Pipeline:
  1. Load data (via ucimlrepo, with a CSV fallback)
  2. Preprocessing (encode categoricals, encode target, train/test split)
  3. Baseline XGBoost model
  4. Hyperparameter tuning (RandomizedSearchCV)
  5. Evaluation (accuracy, classification report, ROC-AUC, confusion matrix, feature importance)

Install requirements:
    pip install ucimlrepo xgboost scikit-learn pandas numpy matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
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
# 3. BASELINE MODEL
# --------------------------------------------------------------------------
n_classes = len(np.unique(y_encoded))

baseline_pipeline = Pipeline(
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
            ),
        ),
    ]
)

baseline_pipeline.fit(X_train, y_train)
baseline_preds = baseline_pipeline.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_preds)
print(f"\nBaseline XGBoost accuracy: {baseline_acc:.4f}")

# --------------------------------------------------------------------------
# 4. HYPERPARAMETER TUNING (RandomizedSearchCV)
# --------------------------------------------------------------------------
param_distributions = {
    "classifier__n_estimators": [100, 200, 300, 400, 500, 600],
    "classifier__max_depth": [3, 4, 5, 6, 7, 8, 9],
    "classifier__learning_rate": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
    "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "classifier__min_child_weight": [1, 2, 3, 5, 7],
    "classifier__gamma": [0, 0.1, 0.2, 0.3, 0.5],
    "classifier__reg_alpha": [0, 0.01, 0.1, 1, 5],
    "classifier__reg_lambda": [0.5, 1, 1.5, 2, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search_pipeline = Pipeline(
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
            ),
        ),
    ]
)

random_search = RandomizedSearchCV(
    estimator=search_pipeline,
    param_distributions=param_distributions,
    n_iter=100,             # number of parameter combinations to try
    scoring="accuracy",
    cv=cv,
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("\nStarting hyperparameter search (this can take a few minutes)...")
random_search.fit(X_train, y_train)
results = pd.DataFrame(random_search.cv_results_)

# ==========================================================
# Mean CV Accuracy for Each Hyperparameter Value
# ==========================================================

parameters = [
    "param_classifier__n_estimators",
    "param_classifier__max_depth",
    "param_classifier__learning_rate",
    "param_classifier__subsample",
    "param_classifier__colsample_bytree",
    "param_classifier__min_child_weight",
    "param_classifier__gamma",
    "param_classifier__reg_alpha",
    "param_classifier__reg_lambda"
]

for param in parameters:
    
    summary = (
        results.groupby(param)["mean_test_score"]
        .mean()
        .reset_index()
        .rename(columns={
            "mean_test_score": "Mean_CV_Accuracy"
        })
        .sort_values("Mean_CV_Accuracy", ascending=False)
    )

    filename = param.replace("param_classifier__", "")

    summary.to_csv(
        f"csv/{filename}_mean_cv_accuracy.csv",
        index=False
    )

print("\nBest parameters found:")
for k, v in random_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Best CV accuracy: {random_search.best_score_:.4f}")

best_model = random_search.best_estimator_

# --------------------------------------------------------------------------
# 5. FINAL EVALUATION ON TEST SET
# --------------------------------------------------------------------------
final_preds = best_model.predict(X_test)
final_acc = accuracy_score(y_test, final_preds)
print(f"\nTuned model test accuracy: {final_acc:.4f}")
print(f"(Baseline test accuracy was: {baseline_acc:.4f})")

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
plt.title("Confusion Matrix — Tuned XGBoost")
plt.tight_layout()
plt.savefig("image/confusion_matrix.png", dpi=150)
plt.close()
print("\nSaved image/confusion_matrix.png")

# --------------------------------------------------------------------------
# 6. FEATURE IMPORTANCE
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
plt.title("Top 20 Feature Importances — Tuned XGBoost")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("image/feature_importance.png", dpi=150)
plt.close()
print("Saved image/feature_importance.png")

# --------------------------------------------------------------------------
# 7. SAVE THE FINAL MODEL
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

plt.figure(figsize=(8,6))

plt.plot(train_sizes, train_mean, marker='o', label="Training Accuracy")
plt.plot(train_sizes, val_mean, marker='s', label="Validation Accuracy")

plt.fill_between(
    train_sizes,
    train_mean-train_std,
    train_mean+train_std,
    alpha=0.2
)

plt.fill_between(
    train_sizes,
    val_mean-val_std,
    val_mean+val_std,
    alpha=0.2
)

plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve - Tuned XGBoost")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("tuning result/learning_curve.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# NUMBER OF TREES
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__n_estimators")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Number of Trees")
plt.xlabel("n_estimators")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_n_estimators.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# MAX DEPTH
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__max_depth")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Max Depth")
plt.xlabel("Max Depth")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_max_depth.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# LEARNING RATE
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__learning_rate")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_learning_rate.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# SUBSAMPLE
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__subsample")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Subsample")
plt.xlabel("Subsample")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_subsample.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# COLUMN SAMPLE
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__colsample_bytree")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Colsample By Tree")
plt.xlabel("colsample_bytree")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_colsample.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# GAMMA
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__gamma")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Gamma")
plt.xlabel("Gamma")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_gamma.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# MINIMUM CHILD WEIGHT
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__min_child_weight")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Min Child Weight")
plt.xlabel("Min Child Weight")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_child_weight.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# ALPHA (L1 REGULARIZATION)
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__reg_alpha")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Alpha (L1)")
plt.xlabel("reg_alpha")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_alpha.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# LAMBDA (L2 REGULARIZATION)
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__reg_lambda")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Lambda (L2)")
plt.xlabel("reg_lambda")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_lambda.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# Example: how to load and use the saved model later
# --------------------------------------------------------------------------
# best_model = joblib.load("xgboost_obesity_model.pkl")
# target_encoder = joblib.load("target_label_encoder.pkl")
# preds = best_model.predict(new_data_df)
# predicted_labels = target_encoder.inverse_transform(preds)