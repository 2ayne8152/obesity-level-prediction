"""
Ordinal Logistic Regression — Estimation of Obesity Levels Based on Eating
Habits and Physical Condition (UCI ML Repository, dataset id 544)
DOI: https://doi.org/10.24432/C5H31Z

This script mirrors the structure of XGBoost.py: same data loading, same
preprocessing (ColumnTransformer with one-hot encoding + scaling), same
train/test split, then fits an Ordinal (proportional-odds) Logistic
Regression instead of XGBoost, and evaluates it the same way (accuracy,
classification report, ROC-AUC, confusion matrix, feature importance,
learning curve).

Because obesity level is an ORDINAL target (Insufficient Weight < Normal
Weight < ... < Obesity Type III), the target is encoded in that clinical
order rather than alphabetically, and statsmodels' OrderedModel is wrapped
in a small scikit-learn-compatible class so it can sit inside the same
Pipeline / learning_curve machinery XGBoost.py uses.

Install requirements:
    pip install ucimlrepo statsmodels scikit-learn pandas numpy matplotlib seaborn joblib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

from statsmodels.miscmodels.ordinal_model import OrderedModel

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# MODEL SETTINGS (analogous to XGBoost.py's BEST_PARAMS)
# --------------------------------------------------------------------------
ORDINAL_PARAMS = {
    "distr": "logit",   # proportional-odds logistic model
    "method": "bfgs",
    "maxiter": 500,
    "disp": False,
}


# --------------------------------------------------------------------------
# 0. SCIKIT-LEARN WRAPPER FOR statsmodels' OrderedModel
# --------------------------------------------------------------------------
# OrderedModel doesn't implement the sklearn estimator interface, so it
# can't be dropped into a Pipeline or sklearn's learning_curve() as-is.
# This thin wrapper gives it fit / predict / predict_proba so the rest of
# this script can look exactly like XGBoost.py.
class OrderedLogisticClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, distr="logit", method="bfgs", maxiter=500, disp=False):
        self.distr = distr
        self.method = method
        self.maxiter = maxiter
        self.disp = disp

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.model_ = OrderedModel(y, X, distr=self.distr)
        self.result_ = self.model_.fit(
            method=self.method, maxiter=self.maxiter, disp=self.disp
        )
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        return self.result_.model.predict(self.result_.params, exog=X)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    @property
    def coef_(self):
        # params = [feature coefficients..., threshold cutoffs...]
        n_features = self.model_.exog.shape[1]
        return self.result_.params[:n_features]


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
# label maps to 0 vs 1; for CAEC/CALC the order is the real severity order,
# which is the whole point of encoding them as ordinals rather than dummies.
binary_categories = [["Female", "Male"], ["no", "yes"], ["no", "yes"], ["no", "yes"], ["no", "yes"]]
binary_categories = binary_categories[: len(binary_cols)]

ordinal_categories = [["no", "Sometimes", "Frequently", "Always"]] * len(ordinal_cols)

# Encode target labels in their natural CLINICAL order (not alphabetically —
# this matters for an ordinal model). A plain LabelEncoder().fit(y) would
# sort classes alphabetically and destroy the ordering the model relies on,
# so we set classes_ explicitly instead.
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
# only the genuinely nominal column (MTRANS, drop="first" to keep full
# column rank since OrderedModel has no intercept of its own), and scale
# numeric columns. sparse_output=False because OrderedModel needs a dense
# numeric matrix.
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
# 3. FIT ORDINAL LOGISTIC REGRESSION
# --------------------------------------------------------------------------
best_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", OrderedLogisticClassifier(**ORDINAL_PARAMS)),
    ]
)

print("\nFitting Ordinal Logistic Regression...")
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
plt.title("Confusion Matrix — Ordinal Logistic Regression")
plt.tight_layout()
plt.savefig("image/confusion_matrix_ordinal_logreg.png", dpi=150)
plt.close()
print("\nSaved image/confusion_matrix_ordinal_logreg.png")

# --------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE
# --------------------------------------------------------------------------
# Recover feature names in the same order ColumnTransformer concatenates them
ohe = best_model.named_steps["preprocessor"].named_transformers_["nom"]
nom_feature_names = list(ohe.get_feature_names_out(nominal_cols))
all_feature_names = binary_cols + ordinal_cols + nom_feature_names + numeric_cols

# A proportional-odds ordinal logit has ONE coefficient per feature (shared
# across all thresholds), so |coefficient| is a direct analogue of
# XGBoost's feature_importances_.
coefficients = best_model.named_steps["classifier"].coef_
feat_imp = (
    pd.Series(np.abs(coefficients), index=all_feature_names)
    .sort_values(ascending=False)
    .head(20)
)

plt.figure(figsize=(8, 8))
sns.barplot(x=feat_imp.values, y=feat_imp.index, color="steelblue")
plt.title("Top 20 Feature Importances (|coef|) — Ordinal Logistic Regression")
plt.xlabel("Absolute Coefficient")
plt.tight_layout()
plt.savefig("image/feature_importance_ordinal_logreg.png", dpi=150)
plt.close()
print("Saved image/feature_importance_ordinal_logreg.png")

# --------------------------------------------------------------------------
# 6. SAVE THE FINAL MODEL
# --------------------------------------------------------------------------
joblib.dump(best_model, "pkl/ordinal_logistic_regression_model.pkl")
joblib.dump(target_encoder, "pkl/ordinal_logistic_target_encoder.pkl")
print("\nSaved trained pipeline to pkl/ordinal_logistic_regression_model.pkl")
print("Saved target label encoder to pkl/ordinal_logistic_target_encoder.pkl")

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
plt.title("Learning Curve - Ordinal Logistic Regression")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("tuning result/learning_curve_ordinal_logreg.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# Example: how to load and use the saved model later
# --------------------------------------------------------------------------
# best_model = joblib.load("pkl/ordinal_logistic_regression_model.pkl")
# target_encoder = joblib.load("pkl/ordinal_logistic_target_encoder.pkl")
# preds = best_model.predict(new_data_df)
# predicted_labels = target_encoder.inverse_transform(preds)