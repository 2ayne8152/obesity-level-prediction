import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from statsmodels.miscmodels.ordinal_model import OrderedResults


# ============================================================
# Import Test Data
# ============================================================

from Preprocessing import (
    X_test,
    y_test
)


# ============================================================
# File Paths
# ============================================================

model_path = (
    "saved_models/"
    "ordinal_logistic_regression.pkl"
)

preprocessor_path = (
    "saved_models/"
    "ordinal_logistic_preprocessor.pkl"
)


# ============================================================
# Load Saved Model
# ============================================================

print("\n" + "=" * 70)
print("LOADING ORDINAL LOGISTIC REGRESSION MODEL")
print("=" * 70)

try:

    result = OrderedResults.load(
        model_path
    )

    print(
        f"Model loaded successfully from:\n"
        f"{model_path}"
    )

except Exception as e:

    raise RuntimeError(
        f"Failed to load the Ordinal Logistic Regression model.\n"
        f"Path: {model_path}\n"
        f"Error: {type(e).__name__}: {str(e)}"
    )


# ============================================================
# Load Saved Preprocessor
# ============================================================

print("\n" + "-" * 70)
print("LOADING SAVED PREPROCESSOR")
print("-" * 70)

try:

    preprocessor = joblib.load(
        preprocessor_path
    )

    print(
        f"Preprocessor loaded successfully from:\n"
        f"{preprocessor_path}"
    )

except Exception as e:

    raise RuntimeError(
        f"Failed to load the saved preprocessing pipeline.\n"
        f"Path: {preprocessor_path}\n"
        f"Error: {type(e).__name__}: {str(e)}"
    )


# ============================================================
# Prepare Test Features
# ============================================================

print("\n" + "-" * 70)
print("PREPARING TEST DATA")
print("-" * 70)

try:

    X_test = X_test.copy()

    # --------------------------------------------------------
    # Apply the SAME fitted preprocessing used during training
    # --------------------------------------------------------
    #
    # This performs:
    #
    # Binary variables
    #     -> OrdinalEncoder
    #
    # CAEC / CALC
    #     -> OrdinalEncoder
    #
    # MTRANS
    #     -> OneHotEncoder
    #
    # Numerical variables
    #     -> StandardScaler
    #
    # The preprocessor was fitted ONLY on the training data.
    #

    X_test_processed = preprocessor.transform(
        X_test
    )

    print(
        "Test data preprocessing completed successfully."
    )

except Exception as e:

    raise RuntimeError(
        f"Failed to preprocess the test data.\n"
        f"Error: {type(e).__name__}: {str(e)}"
    )


# ============================================================
# Convert Processed Test Data to DataFrame
# ============================================================

try:

    # Obtain transformed feature names if supported
    feature_names = (
        preprocessor.get_feature_names_out()
    )

except Exception:

    # Fallback if feature names cannot be obtained
    feature_names = [
        f"Feature_{i}"
        for i in range(
            X_test_processed.shape[1]
        )
    ]


X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)


# ============================================================
# Prepare Target
# ============================================================

y_test = np.asarray(
    y_test
).astype(int)


# ============================================================
# Generate Predicted Probabilities
# ============================================================

print("\n" + "-" * 70)
print("GENERATING PREDICTIONS")
print("-" * 70)

try:

    predicted_probabilities = (
        result.model.predict(
            result.params,
            exog=X_test_processed
        )
    )

    print(
        "Predicted probabilities generated successfully."
    )

except Exception as e:

    raise RuntimeError(
        f"Failed to generate model predictions.\n"
        f"Error: {type(e).__name__}: {str(e)}"
    )


# ============================================================
# Generate Predicted Classes
# ============================================================

# Select the class with the highest predicted probability.

y_pred = np.argmax(
    predicted_probabilities,
    axis=1
)

y_pred = y_pred.astype(int)


# ============================================================
# Class Names
# ============================================================

class_names = [
    "Insufficient Weight",
    "Normal Weight",
    "Overweight Level I",
    "Overweight Level II",
    "Obesity Type I",
    "Obesity Type II",
    "Obesity Type III"
]


# ============================================================
# Overall Metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# ROC-AUC
# ============================================================

try:

    roc_auc = roc_auc_score(
        y_test,
        predicted_probabilities,
        multi_class="ovr",
        average="weighted"
    )

    roc_auc_status = "Successful"

except Exception as e:

    roc_auc = np.nan

    roc_auc_status = (
        f"Failed: "
        f"{type(e).__name__}: {str(e)}"
    )


# ============================================================
# Display Overall Performance
# ============================================================

print("\n" + "=" * 70)
print("ORDINAL LOGISTIC REGRESSION EVALUATION")
print("=" * 70)

print("\nSaved Model:")
print(
    model_path
)

print("\nSaved Preprocessor:")
print(
    preprocessor_path
)

print("\n" + "-" * 70)
print("OVERALL PERFORMANCE")
print("-" * 70)

print(
    f"Accuracy:  {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1-Score:  {f1:.4f}"
)

if not np.isnan(roc_auc):

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

else:

    print(
        f"ROC-AUC:   FAILED"
    )

    print(
        f"Reason:    {roc_auc_status}"
    )


# ============================================================
# 7x7 Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(
        range(
            len(class_names)
        )
    )
)


# ============================================================
# Display 7x7 Confusion Matrix in Terminal
# ============================================================

print("\n" + "-" * 70)
print("7x7 CONFUSION MATRIX")
print("-" * 70)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

print(
    cm_df.to_string()
)


# ============================================================
# Plot 7x7 Confusion Matrix
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix — Ordinal Logistic Regression"
)

plt.colorbar()


# ============================================================
# Axis Labels
# ============================================================

plt.xticks(
    np.arange(
        len(class_names)
    ),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    np.arange(
        len(class_names)
    ),
    class_names
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)


# ============================================================
# Display Values Inside Matrix
# ============================================================

threshold = cm.max() / 2

for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )


plt.tight_layout()

plt.show()


# ============================================================
# TP / FP / TN / FN By Class
# ============================================================

total_samples = cm.sum()

confusion_results = []


for class_index, class_name in enumerate(
    class_names
):

    # --------------------------------------------------------
    # True Positive
    # --------------------------------------------------------

    TP = cm[
        class_index,
        class_index
    ]


    # --------------------------------------------------------
    # False Positive
    # --------------------------------------------------------

    FP = (
        cm[:, class_index].sum()
        - TP
    )


    # --------------------------------------------------------
    # False Negative
    # --------------------------------------------------------

    FN = (
        cm[class_index, :].sum()
        - TP
    )


    # --------------------------------------------------------
    # True Negative
    # --------------------------------------------------------

    TN = (
        total_samples
        - TP
        - FP
        - FN
    )


    confusion_results.append({
        "Class": class_name,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN
    })


confusion_df = pd.DataFrame(
    confusion_results
)


# ============================================================
# Display TP / FP / TN / FN
# ============================================================

print("\n" + "-" * 70)
print("TP / FP / TN / FN BY CLASS (ONE-VS-REST)")
print("-" * 70)

print(
    confusion_df.to_string(
        index=False
    )
)


# ============================================================
# Classification Report
# ============================================================

print("\n" + "-" * 70)
print("CLASSIFICATION REPORT (7-CLASS DETAIL)")
print("-" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        labels=list(
            range(
                len(class_names)
            )
        ),
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# Evaluation Complete
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)