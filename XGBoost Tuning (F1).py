"""
XGBoost Classifier — Estimation of Obesity Levels Based on Eating Habits
and Physical Condition (UCI ML Repository, dataset id 544)
DOI: https://doi.org/10.24432/C5H31Z

Pipeline:
  1. Load data (via ucimlrepo, with a CSV fallback)
  2. Preprocessing (encode categoricals, encode target, train/test split)
  3. Baseline XGBoost model
  4. Hyperparameter tuning (RandomizedSearchCV, optimized for macro-F1 so
     minority/boundary classes matter as much as majority classes)
  5. Evaluation (accuracy, classification report, confusion matrix +
     misclassification-pair breakdown, feature importance)
  6. Save the final (full-feature) model
  7. Behavioral-only model: retrain with Weight & Height removed, since
     NObeyesdad is derived from BMI and those two columns let the model
     essentially read the label off the inputs
  8. Side-by-side comparison of the full model vs. the behavioral-only model

Install requirements:
    pip install ucimlrepo xgboost scikit-learn pandas numpy matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from xgboost import XGBClassifier

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
def load_data():
    """
    Try loading via the official ucimlrepo package first. If that fails
    (e.g. no internet access), fall back to reading a local CSV named
    'ObesityDataSet_raw_and_data_sinthetic.csv' that you can download from:
    https://archive.ics.uci.edu/dataset/544
    """
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=544)
        X = dataset.data.features
        y = dataset.data.targets.squeeze()  # Series
        return X, y
    except Exception as e:
        print(f"ucimlrepo fetch failed ({e}); falling back to local CSV.")
        df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
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

# Train/test split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

# ColumnTransformer: one-hot encode categoricals, pass numeric columns through.
# Tree-based models like XGBoost don't need scaling, so numeric features
# are left as-is.
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
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
    n_iter=150,             # number of parameter combinations to try
    # Macro-F1 weighs every class equally (Insufficient_Weight, Normal_Weight,
    # Overweight_Level_I, etc. all count as much as the larger classes),
    # instead of accuracy, which lets the majority classes dominate the score.
    # This pushes the search toward params that actually fix the boundary
    # classes rather than just preserving overall accuracy.
    scoring="f1_macro",
    cv=cv,
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("\nStarting hyperparameter search (this can take a few minutes)...")
random_search.fit(X_train, y_train)

print("\nBest parameters found:")
for k, v in random_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Best CV macro-F1: {random_search.best_score_:.4f}")

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
        y_test, final_preds, target_names=target_encoder.classes_
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, final_preds)
fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — Tuned XGBoost")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("\nSaved confusion_matrix.png")

# --- Textual breakdown: which classes get confused with which, and how often ---
class_names = target_encoder.classes_
misclass_pairs = []
for true_idx in range(len(class_names)):
    for pred_idx in range(len(class_names)):
        if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
            misclass_pairs.append(
                {
                    "true_class": class_names[true_idx],
                    "predicted_as": class_names[pred_idx],
                    "count": cm[true_idx, pred_idx],
                    "pct_of_true_class": cm[true_idx, pred_idx] / cm[true_idx].sum() * 100,
                }
            )

misclass_df = pd.DataFrame(misclass_pairs).sort_values("count", ascending=False)
print("\nTop misclassification pairs (true class -> predicted class):")
print(misclass_df.head(15).to_string(index=False))
misclass_df.to_csv("misclassification_breakdown.csv", index=False)
print("\nSaved full misclassification breakdown to misclassification_breakdown.csv")

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
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Saved feature_importance.png")

# --------------------------------------------------------------------------
# 7. SAVE THE FINAL MODEL
# --------------------------------------------------------------------------
import joblib

joblib.dump(best_model, "xgboost_obesity_model.pkl")
joblib.dump(target_encoder, "target_label_encoder.pkl")
print("\nSaved trained pipeline to xgboost_obesity_model.pkl")
print("Saved target label encoder to target_label_encoder.pkl")

# --------------------------------------------------------------------------
# 8. BEHAVIORAL-ONLY MODEL (drop Weight & Height)
# --------------------------------------------------------------------------
# NObeyesdad is derived directly from BMI = Weight / Height^2, so a model
# that sees Weight and Height is close to just reading the label off the
# inputs. This second model removes both, so the only remaining predictors
# are eating habits, physical activity, and demographics — this tells us
# how much the *lifestyle* features alone actually explain about obesity
# level, independent of already knowing someone's weight.
print("\n" + "=" * 70)
print("BEHAVIORAL-ONLY MODEL (Weight & Height removed)")
print("=" * 70)

leak_cols = [c for c in ["Weight", "Height"] if c in numeric_cols]
numeric_cols_behavioral = [c for c in numeric_cols if c not in leak_cols]
print(f"Dropped features: {leak_cols}")
print(f"Remaining numeric features: {numeric_cols_behavioral}")
print(f"Categorical features (unchanged): {categorical_cols}")

X_train_beh = X_train.drop(columns=leak_cols)
X_test_beh = X_test.drop(columns=leak_cols)

preprocessor_beh = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
        ("num", "passthrough", numeric_cols_behavioral),
    ]
)

search_pipeline_beh = Pipeline(
    steps=[
        ("preprocessor", preprocessor_beh),
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

# Slightly fewer iterations than the main search to keep runtime reasonable —
# this is an exploratory comparison, not the final production model.
random_search_beh = RandomizedSearchCV(
    estimator=search_pipeline_beh,
    param_distributions=param_distributions,
    n_iter=60,
    scoring="f1_macro",
    cv=cv,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("\nTuning behavioral-only model...")
random_search_beh.fit(X_train_beh, y_train)

print("\nBest parameters (behavioral-only model):")
for k, v in random_search_beh.best_params_.items():
    print(f"  {k}: {v}")
print(f"Best CV macro-F1 (behavioral-only): {random_search_beh.best_score_:.4f}")

best_model_beh = random_search_beh.best_estimator_
preds_beh = best_model_beh.predict(X_test_beh)
acc_beh = accuracy_score(y_test, preds_beh)

print(f"\nBehavioral-only test accuracy: {acc_beh:.4f}")
print("\nClassification report (behavioral-only model):\n")
print(classification_report(y_test, preds_beh, target_names=target_encoder.classes_))

# Confusion matrix for the behavioral-only model
cm_beh = confusion_matrix(y_test, preds_beh)
fig, ax = plt.subplots(figsize=(9, 8))
disp_beh = ConfusionMatrixDisplay(confusion_matrix=cm_beh, display_labels=target_encoder.classes_)
disp_beh.plot(ax=ax, xticks_rotation=45, cmap="Oranges", colorbar=False)
plt.title("Confusion Matrix — Behavioral-Only XGBoost (no Weight/Height)")
plt.tight_layout()
plt.savefig("confusion_matrix_behavioral_only.png", dpi=150)
plt.close()
print("Saved confusion_matrix_behavioral_only.png")

# --------------------------------------------------------------------------
# 9. SIDE-BY-SIDE COMPARISON: FULL MODEL vs BEHAVIORAL-ONLY MODEL
# --------------------------------------------------------------------------
comparison = pd.DataFrame(
    {
        "Model": ["Full features (incl. Weight & Height)", "Behavioral-only (no Weight/Height)"],
        "Test Accuracy": [final_acc, acc_beh],
        "Best CV Macro-F1": [random_search.best_score_, random_search_beh.best_score_],
    }
)
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(comparison.to_string(index=False))
comparison.to_csv("model_comparison_full_vs_behavioral.csv", index=False)
print("\nSaved model_comparison_full_vs_behavioral.csv")

print(
    "\nInterpretation: the drop in accuracy from the full model to the "
    "behavioral-only model is a rough estimate of how much of the "
    "prediction task is 'solved' just by knowing someone's Weight/Height "
    "(i.e. BMI) directly, versus how much genuine signal comes from diet, "
    "activity, and lifestyle habits alone."
)

# Save the behavioral-only model too
joblib.dump(best_model_beh, "xgboost_obesity_model_behavioral_only.pkl")
print("Saved behavioral-only pipeline to xgboost_obesity_model_behavioral_only.pkl")

# --------------------------------------------------------------------------
# Example: how to load and use the saved models later
# --------------------------------------------------------------------------
# best_model = joblib.load("xgboost_obesity_model.pkl")
# best_model_beh = joblib.load("xgboost_obesity_model_behavioral_only.pkl")
# target_encoder = joblib.load("target_label_encoder.pkl")
# preds = best_model.predict(new_data_df)
# predicted_labels = target_encoder.inverse_transform(preds)