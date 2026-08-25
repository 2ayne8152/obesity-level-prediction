"""
Ordinal Logistic Regression (with Optimisation-Method Tuning) — Estimation
of Obesity Levels Based on Eating Habits and Physical Condition
(UCI ML Repository, dataset id 544)
DOI: https://doi.org/10.24432/C5H31Z

This script merges two scripts from the project:

  1. LogisticRegression.py  — self-contained pipeline: loads data, builds a
     sklearn-compatible wrapper around statsmodels' OrderedModel, fits one
     fixed optimisation method ("bfgs"), and evaluates with accuracy,
     classification report, ROC-AUC, confusion matrix, feature importance
     and a learning curve.

  2. LRModelTuning.py       — hyperparameter tuning: 5-fold stratified CV
     that compares several statsmodels optimisation methods ("bfgs",
     "lbfgs", "newton", "nm", "powell") on mean CV accuracy (tie-broken by
     mean training time), then refits the winning method on the full
     training set.

Here, LogisticRegression.py supplies the overall structure (self-contained
data loading, the sklearn-compatible OrderedModel wrapper, Pipeline usage,
ROC-AUC evaluation, feature importance, learning curve) and the
model-tuning step from LRModelTuning.py (CV comparison across optimisation
methods, per-fold failure handling, method-comparison bar charts) is
inserted in place of the fixed ORDINAL_PARAMS["method"] = "bfgs" choice.

NOTE ON OrderedLogisticClassifier:
This script does NOT redefine OrderedLogisticClassifier locally — it is
imported from LogisticRegression.py instead. This matters for pickling:
joblib/pickle stores a reference to the class's *module of origin*, so if
the class were defined here (in __main__ when this script is run
directly), any other script (like GUI.py) that tries to joblib.load(...)
the saved model would fail with something like:
    ModuleNotFoundError: No module named '...'
    AttributeError: Can't get attribute 'OrderedLogisticClassifier' on <module '__main__' ...>
By importing the class from LogisticRegression.py here, and importing it
the same way in GUI.py, every script that loads this pickle resolves the
class the same way. IMPORTANT: LogisticRegression.py's own top-level
script code (data loading, training, plotting, etc.) must be wrapped in
`if __name__ == "__main__":` so that merely importing the class from it
does not re-run its entire training pipeline as a side effect.

See the "DIFFERENCES FROM THE ORIGINAL SCRIPTS" note at the bottom of this
file for a full list of what changed and why.

Install requirements:
    pip install ucimlrepo statsmodels scikit-learn pandas numpy matplotlib seaborn joblib
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    learning_curve,
    train_test_split,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    precision_recall_fscore_support,
)

from statsmodels.miscmodels.ordinal_model import OrderedModel

# OrderedLogisticClassifier now lives in LogisticRegression.py — imported
# here rather than redefined, so every script that loads a pickled model
# built with this class resolves it from the same place.
from LogisticRegression import OrderedLogisticClassifier

RANDOM_STATE = 42
N_SPLITS = 5

# Optimisation methods to compare during tuning (from LRModelTuning.py)
CANDIDATE_METHODS = ["bfgs", "lbfgs", "newton", "nm", "powell"]

# Output directories (created up front so both scripts' save paths work)
for directory in ["image", "pkl", "tuning result"]:
    os.makedirs(directory, exist_ok=True)


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
# 2. FEATURE / TARGET DEFINITIONS
# --------------------------------------------------------------------------
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

# Explicit category orders, keyed by column name (so create_preprocessor()
# can be rebuilt safely for every CV fold without relying on list position).
binary_categories = {
    "Gender": ["Female", "Male"],
    "family_history_with_overweight": ["no", "yes"],
    "FAVC": ["no", "yes"],
    "SMOKE": ["no", "yes"],
    "SCC": ["no", "yes"],
}

ordinal_categories = {
    "CAEC": ["no", "Sometimes", "Frequently", "Always"],
    "CALC": ["no", "Sometimes", "Frequently", "Always"],
}

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

class_names_dict = {i: name.replace("_", " ") for i, name in enumerate(class_order)}


# --------------------------------------------------------------------------
# 3. PREPROCESSOR FACTORY
# --------------------------------------------------------------------------
# A *factory function* (rather than one shared ColumnTransformer instance)
# so a brand-new, unfitted preprocessor can be created for every CV fold —
# this mirrors LRModelTuning.py's leak-safe fold handling, where each fold
# gets its own preprocessor fit only on that fold's training rows.
def create_preprocessor():
    current_binary_categories = [binary_categories[c] for c in binary_cols]
    current_ordinal_categories = [ordinal_categories[c] for c in ordinal_cols]

    return ColumnTransformer(
        transformers=[
            ("bin", OrdinalEncoder(categories=current_binary_categories), binary_cols),
            ("ord", OrdinalEncoder(categories=current_ordinal_categories), ordinal_cols),
            (
                "nom",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                nominal_cols,
            ),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )


# Train/test split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)


# --------------------------------------------------------------------------
# 4. MODEL TUNING: COMPARE OPTIMISATION METHODS VIA 5-FOLD STRATIFIED CV
# --------------------------------------------------------------------------
# This whole section is the piece brought in from LRModelTuning.py. Instead
# of hardcoding ORDINAL_PARAMS["method"] = "bfgs" (as the original
# LogisticRegression.py did), every candidate method is cross-validated and
# the best one (by mean accuracy, tie-broken by mean training time) is
# selected before the final model is fit.
def cross_validate_method(X_raw, y_arr, method, n_splits=N_SPLITS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_accuracies = []
    fold_training_times = []
    failed = False
    failed_fold = None
    error_message = ""

    for fold, (train_index, valid_index) in enumerate(skf.split(X_raw, y_arr), start=1):
        print(f"    Fold {fold}: ", end="")
        try:
            X_fold_train = X_raw.iloc[train_index].copy()
            X_fold_valid = X_raw.iloc[valid_index].copy()
            y_fold_train = y_arr[train_index]
            y_fold_valid = y_arr[valid_index]

            # Fresh preprocessor per fold, fit ONLY on the training rows
            fold_preprocessor = create_preprocessor()
            X_fold_train_processed = fold_preprocessor.fit_transform(X_fold_train)
            X_fold_valid_processed = fold_preprocessor.transform(X_fold_valid)

            training_start_time = time.perf_counter()
            model = OrderedModel(y_fold_train, X_fold_train_processed, distr="logit")
            result = model.fit(method=method, maxiter=500, disp=False)
            fold_training_time = time.perf_counter() - training_start_time
            fold_training_times.append(fold_training_time)

            probs = result.model.predict(result.params, exog=X_fold_valid_processed)
            y_pred = np.argmax(probs, axis=1).astype(int)

            fold_accuracy = accuracy_score(y_fold_valid, y_pred)
            fold_accuracies.append(fold_accuracy)

            print(f"Accuracy = {fold_accuracy:.4f}, Training Time = {fold_training_time:.4f}s")

        except Exception as e:
            failed = True
            failed_fold = fold
            error_message = f"{type(e).__name__}: {str(e)}"
            print("FAILED")
            print(f"        Error: {error_message}")
            if method.lower() == "newton":
                print("        Newton's method commonly fails when the Hessian is singular.")
            print(f"        Optimisation method '{method}' is marked as FAILED.")
            print("        Remaining folds will not be tested for this method.")
            break

    if failed or len(fold_accuracies) != n_splits:
        return np.nan, np.nan, fold_accuracies, fold_training_times, "Failed", failed_fold, error_message

    return (
        float(np.mean(fold_accuracies)),
        float(np.mean(fold_training_times)),
        fold_accuracies,
        fold_training_times,
        "Successful",
        None,
        "",
    )


print("\n" + "=" * 70)
print("PARAMETER TUNING: OPTIMISATION METHOD")
print("=" * 70)
print("\nUsing 5-fold cross-validation.")
print("Primary metric: Mean Cross-Validation Accuracy")
print("Secondary metric: Mean Training Time\n")

method_results = []
for method in CANDIDATE_METHODS:
    print("-" * 70)
    print(f"Testing Optimisation method = {method}")

    (mean_cv_accuracy, mean_training_time, fold_accuracies,
     fold_training_times, status, failed_fold, error_message) = cross_validate_method(
        X_train, y_train, method=method, n_splits=N_SPLITS
    )

    if status == "Successful":
        print("\n    Fold Accuracies:", [f"{a:.4f}" for a in fold_accuracies])
        print("    Fold Training Times:", [f"{t:.4f}s" for t in fold_training_times])
        print(f"    Mean CV Accuracy: {mean_cv_accuracy:.4f}")
        print(f"    Mean Training Time: {mean_training_time:.4f}s")
    else:
        print("\n    Mean CV Accuracy: FAILED")
        if failed_fold is not None:
            print(f"    Failed Fold: {failed_fold}")
        print(f"    Reason: {error_message}")

    method_results.append({
        "method": method,
        "mean_cv_accuracy": mean_cv_accuracy,
        "mean_training_time": mean_training_time,
        "status": status,
    })

method_results_df = pd.DataFrame(method_results)
print("\n" + "=" * 70)
print("OPTIMISATION METHOD TUNING RESULTS")
print("=" * 70)
print(method_results_df.to_string(index=False))

successful_results = method_results_df[method_results_df["status"] == "Successful"].copy()
if successful_results.empty:
    raise RuntimeError(
        "All optimisation methods failed during 5-fold cross-validation. "
        "No final model can be selected."
    )

# Method-comparison bar charts (from LRModelTuning.py), saved into the same
# "tuning result" folder LogisticRegression.py already uses for the
# learning curve plot.
plt.figure(figsize=(8, 5))
plt.bar(successful_results["method"], successful_results["mean_cv_accuracy"])
plt.xlabel("Optimisation Method")
plt.ylabel("Mean Cross-Validation Accuracy")
plt.title("Ordinal Logistic Regression: Mean CV Accuracy by Method")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("tuning result/method_tuning_accuracy.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(successful_results["method"], successful_results["mean_training_time"])
plt.xlabel("Optimisation Method")
plt.ylabel("Mean Training Time (seconds)")
plt.title("Ordinal Logistic Regression: Mean Training Time by Method")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("tuning result/method_tuning_training_time.png", dpi=150)
plt.close()
print("\nSaved tuning result/method_tuning_accuracy.png")
print("Saved tuning result/method_tuning_training_time.png")

# Select best method: highest mean CV accuracy, ties broken by lowest
# mean training time.
highest_accuracy = successful_results["mean_cv_accuracy"].max()
tied = successful_results[
    np.isclose(successful_results["mean_cv_accuracy"], highest_accuracy, rtol=1e-9, atol=1e-9)
]
best_result = tied.loc[tied["mean_training_time"].idxmin()] if len(tied) > 1 else tied.iloc[0]
best_method = best_result["method"]

print("\n" + "=" * 70)
print("BEST OPTIMISATION METHOD")
print("=" * 70)
print(f"Best Method: {best_method}")
print(f"Mean CV Accuracy: {best_result['mean_cv_accuracy']:.4f}")
print(f"Mean Training Time: {best_result['mean_training_time']:.4f} seconds")

# ORDINAL_PARAMS now takes its "method" from the tuning step above, instead
# of the fixed "bfgs" the original LogisticRegression.py used.
ORDINAL_PARAMS = {
    "distr": "logit",
    "method": best_method,
    "maxiter": 500,
    "disp": False,
}


# --------------------------------------------------------------------------
# 5. FIT FINAL ORDINAL LOGISTIC REGRESSION (best method, full training set)
# --------------------------------------------------------------------------
best_model = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
        ("classifier", OrderedLogisticClassifier(**ORDINAL_PARAMS)),
    ]
)

print(f"\nFitting final Ordinal Logistic Regression with method='{best_method}'...")
best_model.fit(X_train, y_train)

# --------------------------------------------------------------------------
# 6. EVALUATION ON TEST SET
# --------------------------------------------------------------------------
final_preds = best_model.predict(X_test)
final_acc = accuracy_score(y_test, final_preds)
print(f"\nTest accuracy: {final_acc:.4f}")

print("\nClassification report:\n")
print(classification_report(y_test, final_preds, target_names=target_encoder.classes_, digits=4))

# ROC-AUC (multiclass, one-vs-rest)
final_probs = best_model.predict_proba(X_test)
macro_roc_auc = roc_auc_score(y_test, final_probs, multi_class="ovr", average="macro")
weighted_roc_auc = roc_auc_score(y_test, final_probs, multi_class="ovr", average="weighted")
per_class_roc_auc = roc_auc_score(y_test, final_probs, multi_class="ovr", average=None)

print(f"\nMacro-average ROC-AUC (OvR): {macro_roc_auc:.4f}")
print(f"Weighted-average ROC-AUC (OvR): {weighted_roc_auc:.4f}")

roc_auc_df = pd.DataFrame(
    {"Class": target_encoder.classes_, "ROC-AUC": per_class_roc_auc}
).sort_values("ROC-AUC", ascending=False)
print("\nPer-class ROC-AUC (OvR):")
print(roc_auc_df.to_string(index=False))

# Confusion matrix
cm = confusion_matrix(y_test, final_preds)
fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix — Ordinal Logistic Regression ({best_method})")
plt.tight_layout()
plt.savefig("image/confusion_matrix_ordinal_logreg.png", dpi=150)
plt.close()
print("\nSaved image/confusion_matrix_ordinal_logreg.png")

# --------------------------------------------------------------------------
# 6b. SUMMARY METRICS TABLE (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
# --------------------------------------------------------------------------
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test, final_preds, average="macro", zero_division=0
)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_test, final_preds, average="weighted", zero_division=0
)

summary_metrics_df = pd.DataFrame(
    {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Macro":    [final_acc, precision_macro,    recall_macro,    f1_macro,    macro_roc_auc],
        "Weighted": [final_acc, precision_weighted, recall_weighted, f1_weighted, weighted_roc_auc],
    }
)

print("\n" + "=" * 70)
print("SUMMARY METRICS")
print("=" * 70)
print(summary_metrics_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

summary_metrics_df.to_csv("tuning result/summary_metrics_ordinal_logreg.csv", index=False)
print("\nSaved tuning result/summary_metrics_ordinal_logreg.csv")

# --------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE
# --------------------------------------------------------------------------
ohe = best_model.named_steps["preprocessor"].named_transformers_["nom"]
nom_feature_names = list(ohe.get_feature_names_out(nominal_cols))
all_feature_names = binary_cols + ordinal_cols + nom_feature_names + numeric_cols

coefficients = best_model.named_steps["classifier"].coef_
feat_imp = (
    pd.Series(np.abs(coefficients), index=all_feature_names)
    .sort_values(ascending=False)
    .head(20)
)

plt.figure(figsize=(8, 8))
sns.barplot(x=feat_imp.values, y=feat_imp.index, color="steelblue")
plt.title(f"Top 20 Feature Importances (|coef|) — Ordinal Logistic Regression ({best_method})")
plt.xlabel("Absolute Coefficient")
plt.tight_layout()
plt.savefig("image/feature_importance_ordinal_logreg.png", dpi=150)
plt.close()
print("Saved image/feature_importance_ordinal_logreg.png")

# --------------------------------------------------------------------------
# 8. SAVE THE FINAL MODEL
# --------------------------------------------------------------------------
joblib.dump(best_model, "pkl/ordinal_logistic_regression_model.pkl")
joblib.dump(target_encoder, "pkl/ordinal_logistic_target_encoder.pkl")
print("\nSaved trained pipeline to pkl/ordinal_logistic_regression_model.pkl")
print("Saved target label encoder to pkl/ordinal_logistic_target_encoder.pkl")

# --------------------------------------------------------------------------
# 9. LEARNING CURVE (uses the tuned best_method via ORDINAL_PARAMS)
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

train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
val_mean, val_std = np.mean(val_scores, axis=1), np.std(val_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
plt.plot(train_sizes, val_mean, marker="s", label="Validation Accuracy")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.title(f"Learning Curve - Ordinal Logistic Regression ({best_method})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("tuning result/learning_curve_ordinal_logreg.png", dpi=150)
plt.close()
print("Saved tuning result/learning_curve_ordinal_logreg.png")

# --------------------------------------------------------------------------
# 10. EXAMPLE PREDICTIONS (brought in from LRModelTuning.py)
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXAMPLE PREDICTIONS")
print("=" * 70)

for i in range(min(10, len(y_test))):
    actual_class = int(y_test[i]) if not hasattr(y_test, "iloc") else int(y_test.iloc[i])
    predicted_class = int(final_preds[i])
    print(f"\nActual:    {class_names_dict[actual_class]}")
    print(f"Predicted: {class_names_dict[predicted_class]}")
    print("Probabilities:")
    for class_index, probability in enumerate(final_probs[i]):
        print(f"  {class_names_dict[class_index]:22s}: {probability:.4f}")

# --------------------------------------------------------------------------
# Example: how to load and use the saved model later
# --------------------------------------------------------------------------
# from LogisticRegression import OrderedLogisticClassifier  # noqa: F401
# best_model = joblib.load("pkl/ordinal_logistic_regression_model.pkl")
# target_encoder = joblib.load("pkl/ordinal_logistic_target_encoder.pkl")
# preds = best_model.predict(new_data_df)
# predicted_labels = target_encoder.inverse_transform(preds)


# ============================================================================
# DIFFERENCES FROM THE ORIGINAL SCRIPTS
# ============================================================================
#
# vs. LogisticRegression.py (structure this script follows):
#   - The optimisation method is no longer a fixed value ("bfgs") inside
#     ORDINAL_PARAMS. It is now selected by the 5-fold CV tuning step
#     (section 4) before ORDINAL_PARAMS is even built.
#   - The single shared `preprocessor` ColumnTransformer became a
#     `create_preprocessor()` factory, so a fresh, independently-fit
#     instance can be produced per CV fold (needed for leak-safe tuning)
#     as well as for the final Pipeline.
#   - Category lists (binary_categories / ordinal_categories) became dicts
#     keyed by column name rather than plain lists, so create_preprocessor()
#     can rebuild the right category order regardless of column filtering.
#   - Added an "EXAMPLE PREDICTIONS" section, method-comparison bar charts,
#     and a consolidated Accuracy/Precision/Recall/F1/ROC-AUC summary table
#     (section 6b) — none present in the original.
#   - Confusion-matrix / feature-importance / learning-curve plot titles
#     now report which optimisation method was actually used.
#   - OrderedLogisticClassifier is now IMPORTED from LogisticRegression.py
#     rather than defined locally, so pickled models resolve the class
#     consistently regardless of which script loads them (see the module
#     docstring note above). LogisticRegression.py's own script body must
#     be wrapped in `if __name__ == "__main__":` for this import to be
#     side-effect-free.
#
# vs. LRModelTuning.py (source of the tuning logic):
#   - Data loading is self-contained again (ucimlrepo fetch with local-CSV
#     fallback, as in LogisticRegression.py) instead of importing an
#     already-split X_train/X_test/y_train/y_test from a separate
#     Preprocessing.py module.
#   - The final model is fit through the OrderedLogisticClassifier
#     sklearn-compatible wrapper inside a sklearn Pipeline, rather than
#     calling OrderedModel/`result.fit()` directly — this is what lets the
#     same final model be reused for `learning_curve()` in section 9,
#     which LRModelTuning.py did not compute at all.
#   - ROC-AUC (macro, weighted, and per-class) and a full summary metrics
#     table (Accuracy/Precision/Recall/F1/ROC-AUC) were added to the
#     test-set evaluation; LRModelTuning.py only reported accuracy, a
#     classification report and a confusion matrix.
#   - Artifact saving is simplified: one joblib dump of the whole fitted
#     Pipeline (preprocessor + classifier together) plus the target
#     encoder, instead of three separate artifacts (statsmodels
#     `result.save()`, a standalone preprocessor, and a feature-name list),
#     since the Pipeline already bundles preprocessing with the model.
#   - The per-fold try/except failure handling (including the Newton/
#     singular-Hessian note) and the "tie-break by mean training time"
#     method-selection rule were both kept as-is from LRModelTuning.py.
# ============================================================================