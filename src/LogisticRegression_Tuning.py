"""
Ordinal Logistic Regression (with Optimization-Method Tuning) — Estimation
of Obesity Levels Based on Eating Habits and Physical Condition
(UCI ML Repository, dataset id 544)
DOI: https://doi.org/10.24432/C5H31Z

This script standardises the Ordinal Logistic Regression pipeline to match the
project layout: loading preprocessed splits via `get_preprocessed_data()`,
tuning statsmodels' OrderedModel optimisation methods using Stratified K-Fold CV,
evaluating on test data, and exporting model artifacts & visualization outputs.
"""

import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline

from statsmodels.miscmodels.ordinal_model import OrderedModel

# Import centralized preprocessing function
from Preprocessing import get_preprocessed_data

# --------------------------------------------------------------------------
# PROJECT PATHS & CONFIGURATION
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "tuning" / "Ordinal_Logistic_Regression"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
N_SPLITS = 5
CANDIDATE_METHODS = ["bfgs", "lbfgs", "newton", "nm", "powell"]


# --------------------------------------------------------------------------
# 0. SCIKIT-LEARN WRAPPER FOR statsmodels' OrderedModel
# --------------------------------------------------------------------------
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
# HELPER FUNCTIONS
# --------------------------------------------------------------------------
def cross_validate_method(
    X_train, y_train, preprocessor, method, n_splits=N_SPLITS
):
    """Evaluates an optimisation method using leak-safe Stratified K-Fold CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_accuracies = []
    fold_training_times = []
    failed = False
    failed_fold = None
    error_message = ""

    for fold, (train_index, valid_index) in enumerate(
        skf.split(X_train, y_train), start=1
    ):
        print(f"    Fold {fold}: ", end="")
        try:
            X_fold_train = (
                X_train.iloc[train_index]
                if hasattr(X_train, "iloc")
                else X_train[train_index]
            )
            X_fold_valid = (
                X_train.iloc[valid_index]
                if hasattr(X_train, "iloc")
                else X_train[valid_index]
            )
            y_fold_train = y_train[train_index]
            y_fold_valid = y_train[valid_index]

            fold_pipeline = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    (
                        "classifier",
                        OrderedLogisticClassifier(
                            distr="logit", method=method, maxiter=500, disp=False
                        ),
                    ),
                ]
            )

            start_time = time.perf_counter()
            fold_pipeline.fit(X_fold_train, y_fold_train)
            fold_time = time.perf_counter() - start_time

            y_pred = fold_pipeline.predict(X_fold_valid)
            acc = accuracy_score(y_fold_valid, y_pred)

            fold_accuracies.append(acc)
            fold_training_times.append(fold_time)
            print(f"Accuracy = {acc:.4f}, Time = {fold_time:.4f}s")

        except Exception as e:
            failed = True
            failed_fold = fold
            error_message = f"{type(e).__name__}: {str(e)}"
            print("FAILED")
            print(f"        Error: {error_message}")
            if method.lower() == "newton":
                print(
                    "        Newton's method commonly fails when the Hessian is singular."
                )
            print(f"        Optimisation method '{method}' is marked as FAILED.")
            print("        Remaining folds skipped for this method.")
            break

    if failed or len(fold_accuracies) != n_splits:
        return (
            np.nan,
            np.nan,
            fold_accuracies,
            fold_training_times,
            "Failed",
            failed_fold,
            error_message,
        )

    return (
        float(np.mean(fold_accuracies)),
        float(np.mean(fold_training_times)),
        fold_accuracies,
        fold_training_times,
        "Successful",
        None,
        "",
    )


# --------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------
def main():
    # Ensure output directories exist dynamically
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. LOAD AND PREPROCESS DATA
    print("Loading and preprocessing data...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        target_encoder,
    ) = get_preprocessed_data()

    print("\nClasses (in ordinal order):", list(target_encoder.classes_))

    # 2. OPTIMISATION METHOD TUNING
    print("\n" + "=" * 70)
    print("PARAMETER TUNING: OPTIMISATION METHOD")
    print("=" * 70)
    print(f"\nUsing {N_SPLITS}-fold cross-validation.")
    print("Primary metric: Mean Cross-Validation Accuracy")
    print("Secondary metric: Mean Training Time\n")

    method_results = []
    for method in CANDIDATE_METHODS:
        print("-" * 70)
        print(f"Testing Optimisation method = {method}")

        (
            mean_cv_accuracy,
            mean_training_time,
            fold_accuracies,
            fold_training_times,
            status,
            failed_fold,
            error_message,
        ) = cross_validate_method(X_train, y_train, preprocessor, method)

        if status == "Successful":
            print(
                "\n    Fold Accuracies:", [f"{a:.4f}" for a in fold_accuracies]
            )
            print(
                "    Fold Training Times:",
                [f"{t:.4f}s" for t in fold_training_times],
            )
            print(f"    Mean CV Accuracy: {mean_cv_accuracy:.4f}")
            print(f"    Mean Training Time: {mean_training_time:.4f}s")
        else:
            print("\n    Mean CV Accuracy: FAILED")
            if failed_fold is not None:
                print(f"    Failed Fold: {failed_fold}")
            print(f"    Reason: {error_message}")

        method_results.append(
            {
                "method": method,
                "mean_cv_accuracy": mean_cv_accuracy,
                "mean_training_time": mean_training_time,
                "status": status,
            }
        )

    method_results_df = pd.DataFrame(method_results)
    print("\n" + "=" * 70)
    print("OPTIMISATION METHOD TUNING RESULTS")
    print("=" * 70)
    print(method_results_df.to_string(index=False))

    successful_results = method_results_df[
        method_results_df["status"] == "Successful"
    ].copy()
    if successful_results.empty:
        raise RuntimeError(
            "All optimisation methods failed during cross-validation."
        )

    # Plot Tuning Results
    plt.figure(figsize=(8, 5))
    plt.bar(successful_results["method"], successful_results["mean_cv_accuracy"])
    plt.xlabel("Optimisation Method")
    plt.ylabel("Mean Cross-Validation Accuracy")
    plt.title("Ordinal Logistic Regression: Mean CV Accuracy by Method")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "method_tuning_accuracy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(
        successful_results["method"], successful_results["mean_training_time"]
    )
    plt.xlabel("Optimisation Method")
    plt.ylabel("Mean Training Time (seconds)")
    plt.title("Ordinal Logistic Regression: Mean Training Time by Method")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "method_tuning_training_time.png", dpi=150)
    plt.close()

    # Select Best Method
    highest_accuracy = successful_results["mean_cv_accuracy"].max()
    tied = successful_results[
        np.isclose(
            successful_results["mean_cv_accuracy"],
            highest_accuracy,
            rtol=1e-9,
            atol=1e-9,
        )
    ]
    best_result = (
        tied.loc[tied["mean_training_time"].idxmin()]
        if len(tied) > 1
        else tied.iloc[0]
    )
    best_method = best_result["method"]

    print("\n" + "=" * 70)
    print("BEST OPTIMISATION METHOD")
    print("=" * 70)
    print(f"Best Method: {best_method}")
    print(f"Mean CV Accuracy: {best_result['mean_cv_accuracy']:.4f}")
    print(
        f"Mean Training Time: {best_result['mean_training_time']:.4f} seconds"
    )

    # 3. FIT FINAL ORDINAL LOGISTIC REGRESSION
    ORDINAL_PARAMS = {
        "distr": "logit",
        "method": best_method,
        "maxiter": 500,
        "disp": False,
    }

    best_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", OrderedLogisticClassifier(**ORDINAL_PARAMS)),
        ]
    )

    print(f"\nFitting final Ordinal Logistic Regression (method='{best_method}')...")
    best_model.fit(X_train, y_train)

    # 4. EVALUATION ON TEST SET
    final_preds = best_model.predict(X_test)
    final_acc = accuracy_score(y_test, final_preds)
    print(f"\nTest accuracy: {final_acc:.4f}")

    print("\nClassification report:\n")
    print(
        classification_report(
            y_test,
            final_preds,
            target_names=target_encoder.classes_,
            digits=4,
        )
    )

    # ROC-AUC (multiclass, one-vs-rest)
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
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=target_encoder.classes_
    )
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix — Ordinal Logistic Regression ({best_method})")
    plt.tight_layout()

    cm_path = RESULTS_DIR / "confusion_matrix_ordinal_logreg.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nSaved {cm_path.relative_to(PROJECT_ROOT)}")

    # Save summary metrics table
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, final_preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_test, final_preds, average="weighted", zero_division=0
        )
    )

    summary_metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            "Macro": [
                final_acc,
                precision_macro,
                recall_macro,
                f1_macro,
                macro_roc_auc,
            ],
            "Weighted": [
                final_acc,
                precision_weighted,
                recall_weighted,
                f1_weighted,
                weighted_roc_auc,
            ],
        }
    )
    summary_metrics_path = RESULTS_DIR / "summary_metrics_ordinal_logreg.csv"
    summary_metrics_df.to_csv(summary_metrics_path, index=False)
    print(f"Saved {summary_metrics_path.relative_to(PROJECT_ROOT)}")

    # 5. FEATURE IMPORTANCE
    raw_feature_names = best_model.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    clean_feature_names = [name.split("__")[-1] for name in raw_feature_names]

    coefficients = best_model.named_steps["classifier"].coef_
    feat_imp = (
        pd.Series(np.abs(coefficients), index=clean_feature_names)
        .sort_values(ascending=False)
        .head(20)
    )

    plt.figure(figsize=(8, 8))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, color="steelblue")
    plt.title(
        f"Top 20 Feature Importances (|coef|) — Ordinal Logistic Regression ({best_method})"
    )
    plt.xlabel("Absolute Coefficient")
    plt.tight_layout()

    feat_imp_path = RESULTS_DIR / "feature_importance_ordinal_logreg.png"
    plt.savefig(feat_imp_path, dpi=150)
    plt.close()
    print(f"Saved {feat_imp_path.relative_to(PROJECT_ROOT)}")

    # 6. SAVE THE FINAL MODEL
    model_path = MODELS_DIR / "ordinal_logistic_regression_model.pkl"
    encoder_path = MODELS_DIR / "ordinal_logistic_target_encoder.pkl"

    joblib.dump(best_model, model_path)
    joblib.dump(target_encoder, encoder_path)
    print(f"\nSaved trained pipeline to {model_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved target label encoder to {encoder_path.relative_to(PROJECT_ROOT)}")

    # 7. LEARNING CURVE
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
    plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
    plt.plot(train_sizes, val_mean, marker="s", label="Validation Accuracy")

    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
    )
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2,
    )

    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve - Ordinal Logistic Regression ({best_method})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    lc_path = RESULTS_DIR / "learning_curve_ordinal_logreg.png"
    plt.savefig(lc_path, dpi=150)
    plt.close()
    print(f"Saved {lc_path.relative_to(PROJECT_ROOT)}")

    # 8. EXAMPLE PREDICTIONS
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)

    class_names = list(target_encoder.classes_)
    for i in range(min(10, len(y_test))):
        actual_idx = (
            int(y_test[i]) if not hasattr(y_test, "iloc") else int(y_test.iloc[i])
        )
        pred_idx = int(final_preds[i])
        print(f"\nActual:    {class_names[actual_idx].replace('_', ' ')}")
        print(f"Predicted: {class_names[pred_idx].replace('_', ' ')}")
        print("Probabilities:")
        for idx, prob in enumerate(final_probs[i]):
            print(f"  {class_names[idx].replace('_', ' '):22s}: {prob:.4f}")


if __name__ == "__main__":
    main()