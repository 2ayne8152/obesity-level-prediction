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
# Load Saved Model
# ============================================================

model_path = (
    "saved_models/"
    "ordinal_logistic_regression.pkl"
)

result = OrderedResults.load(
    model_path
)


# ============================================================
# Load Saved Scaler
# ============================================================

scaler_path = (
    "saved_models/"
    "ordinal_logistic_scaler.pkl"
)

scaler = joblib.load(
    scaler_path
)


# ============================================================
# Load Saved Feature Columns
# ============================================================

features_path = (
    "saved_models/"
    "ordinal_logistic_features.pkl"
)

feature_columns = joblib.load(
    features_path
)


# ============================================================
# Prepare Test Features
# ============================================================

X_test = X_test.copy()

X_test = pd.get_dummies(
    X_test,
    columns=["MTRANS"],
    prefix="MTRANS",
    dtype=int
)


# ============================================================
# Ensure Same Feature Columns as Training Data
# ============================================================

X_test = X_test.reindex(
    columns=feature_columns,
    fill_value=0
)


# ============================================================
# Standardise Test Data
# ============================================================

X_test_scaled = scaler.transform(
    X_test
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=feature_columns,
    index=X_test.index
)


# ============================================================
# Prepare Target
# ============================================================

y_test = y_test.astype(int)


# ============================================================
# Generate Predicted Probabilities
# ============================================================

predicted_probabilities = (
    result.model.predict(
        result.params,
        exog=X_test_scaled
    )
)


# ============================================================
# Generate Predicted Classes
# ============================================================

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

roc_auc = roc_auc_score(
    y_test,
    predicted_probabilities,
    multi_class="ovr",
    average="weighted"
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

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)


# ============================================================
# 7x7 Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(class_names)))
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
    np.arange(len(class_names)),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    np.arange(len(class_names)),
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
        target_names=class_names,
        zero_division=0
    )
)