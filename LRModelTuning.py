import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold

from statsmodels.miscmodels.ordinal_model import OrderedModel


# ============================================================
# Import preprocessed train/test data
# ============================================================

from Preprocessing import (
    X_train,
    X_test,
    y_train,
    y_test
)


# ============================================================
# Keep RAW Training/Test Data
# ============================================================

# Keep the original data before converting MTRANS into
# dummy variables.
#
# Each cross-validation fold will perform its own
# preprocessing.

X_train_raw = X_train.copy()
X_test_raw = X_test.copy()

y_train = y_train.astype(int)
y_test = y_test.astype(int)


# ============================================================
# Prepare Features Function
# ============================================================

def prepare_features(X):

    X = X.copy()

    # MTRANS is a nominal variable.
    # Convert it into dummy variables so that the Ordinal
    # Logistic Regression model does not treat transportation
    # modes as ordered numerical values.

    X = pd.get_dummies(
        X,
        columns=["MTRANS"],
        prefix="MTRANS",
        dtype=int
    )

    return X


# ============================================================
# Cross-Validation Function
# ============================================================

def cross_validate_method(
    X,
    y,
    method,
    n_splits=5
):

    # --------------------------------------------------------
    # Create Stratified K-Fold Cross-Validation
    # --------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    fold_accuracies = []
    fold_training_times = []

    # Track whether any fold fails
    failed = False

    # Store the error message if a fold fails
    error_message = ""


    # ========================================================
    # Run Each Fold
    # ========================================================

    for fold, (train_index, validation_index) in enumerate(
        skf.split(X, y),
        start=1
    ):

        print(
            f"    Fold {fold}: ",
            end=""
        )

        try:

            # ------------------------------------------------
            # Split Fold
            # ------------------------------------------------

            X_fold_train = X.iloc[train_index].copy()
            X_fold_valid = X.iloc[validation_index].copy()

            y_fold_train = y.iloc[train_index].copy()
            y_fold_valid = y.iloc[validation_index].copy()


            # ------------------------------------------------
            # Prepare Features
            # ------------------------------------------------

            X_fold_train = prepare_features(
                X_fold_train
            )

            X_fold_valid = prepare_features(
                X_fold_valid
            )


            # ------------------------------------------------
            # Ensure Same Columns
            # ------------------------------------------------

            X_fold_valid = X_fold_valid.reindex(
                columns=X_fold_train.columns,
                fill_value=0
            )


            # ------------------------------------------------
            # Standardise Features
            # ------------------------------------------------

            # Fit the scaler ONLY on the fold-training data.
            # This prevents validation-data leakage.

            fold_scaler = StandardScaler()

            X_fold_train_scaled = fold_scaler.fit_transform(
                X_fold_train
            )

            X_fold_valid_scaled = fold_scaler.transform(
                X_fold_valid
            )


            # ------------------------------------------------
            # Convert Back to DataFrame
            # ------------------------------------------------

            X_fold_train_scaled = pd.DataFrame(
                X_fold_train_scaled,
                columns=X_fold_train.columns,
                index=X_fold_train.index
            )

            X_fold_valid_scaled = pd.DataFrame(
                X_fold_valid_scaled,
                columns=X_fold_train.columns,
                index=X_fold_valid.index
            )


            # ------------------------------------------------
            # Create Ordinal Logistic Regression Model
            # ------------------------------------------------

            model = OrderedModel(
                endog=y_fold_train,
                exog=X_fold_train_scaled,
                distr="logit"
            )


            # =================================================
            # Start Training Timer
            # =================================================

            training_start_time = time.perf_counter()


            # ------------------------------------------------
            # Train Model
            # ------------------------------------------------

            result = model.fit(
                method=method,
                disp=False
            )
            # ============================================================
            # Save Final Model
            # ============================================================

            os.makedirs(
                "saved_models",
                exist_ok=True
            )

            model_path = os.path.join(
                "saved_models",
                "ordinal_logistic_regression.pkl"
            )

            result.save(
                model_path
            )

            print(
                f"Model saved to: "
                f"{model_path}"
            )
        

            # =================================================
            # End Training Timer
            # =================================================

            training_end_time = time.perf_counter()

            fold_training_time = (
                training_end_time
                - training_start_time
            )

            fold_training_times.append(
                fold_training_time
            )


            # ------------------------------------------------
            # Predict Validation Probabilities
            # ------------------------------------------------

            predicted_probabilities = (
                result.model.predict(
                    result.params,
                    exog=X_fold_valid_scaled
                )
            )


            # ------------------------------------------------
            # Predict Final Class
            # ------------------------------------------------

            y_pred = np.argmax(
                predicted_probabilities,
                axis=1
            )

            y_pred = y_pred.astype(int)


            # ------------------------------------------------
            # Calculate Fold Accuracy
            # ------------------------------------------------

            fold_accuracy = accuracy_score(
                y_fold_valid,
                y_pred
            )

            fold_accuracies.append(
                fold_accuracy
            )


            print(
                f"Accuracy = {fold_accuracy:.4f}, "
                f"Training Time = "
                f"{fold_training_time:.4f}s"
            )


        # ====================================================
        # Handle Failed Optimisation
        # ====================================================

        except Exception as e:

            failed = True

            error_message = (
                f"{type(e).__name__}: {str(e)}"
            )

            print("FAILED")

            print(
                f"        Error: {error_message}"
            )

            # Stop testing remaining folds for this method.
            # A complete 5-fold CV result is required.

            break


    # ========================================================
    # Return Results if Method Failed
    # ========================================================

    if failed:

        return (
            np.nan,
            np.nan,
            fold_accuracies,
            fold_training_times,
            "Failed",
            error_message
        )


    # ========================================================
    # Mean Cross-Validation Accuracy
    # ========================================================

    mean_cv_accuracy = np.mean(
        fold_accuracies
    )


    # ========================================================
    # Mean Training Time Across 5 Folds
    # ========================================================

    mean_training_time = np.mean(
        fold_training_times
    )


    return (
        mean_cv_accuracy,
        mean_training_time,
        fold_accuracies,
        fold_training_times,
        "Successful",
        ""
    )


# ============================================================
# Parameter Tuning: Optimisation Method
# ============================================================

methods = [
    "bfgs",
    "lbfgs",
    "newton",
    "nm",
    "powell"
]

method_results = []


print("\n" + "=" * 70)
print("PARAMETER TUNING: OPTIMISATION METHOD")
print("=" * 70)

print("\nUsing 5-fold cross-validation.")
print("Primary metric: Mean Cross-Validation Accuracy")
print("Secondary metric: Mean Training Time")
print()


# ============================================================
# Test Each Optimisation Method
# ============================================================

for method in methods:

    print("-" * 70)

    print(
        f"Testing Optimisation method = {method}"
    )


    (
        mean_cv_accuracy,
        mean_training_time,
        fold_accuracies,
        fold_training_times,
        status,
        error_message
    ) = cross_validate_method(
        X_train_raw,
        y_train,
        method=method,
        n_splits=5
    )


    # --------------------------------------------------------
    # Successful Method
    # --------------------------------------------------------

    if status == "Successful":

        print(
            "\n    Fold Accuracies:",
            [
                f"{accuracy:.4f}"
                for accuracy in fold_accuracies
            ]
        )

        print(
            "    Fold Training Times:",
            [
                f"{training_time:.4f}s"
                for training_time in fold_training_times
            ]
        )

        print(
            f"    Mean CV Accuracy: "
            f"{mean_cv_accuracy:.4f}"
        )

        print(
            f"    Mean Training Time: "
            f"{mean_training_time:.4f}s"
        )


    # --------------------------------------------------------
    # Failed Method
    # --------------------------------------------------------

    else:

        print(
            "\n    Mean CV Accuracy: FAILED"
        )

        print(
            f"    Reason: {error_message}"
        )


    # --------------------------------------------------------
    # Store Results
    # --------------------------------------------------------

    method_results.append({
        "method": method,
        "mean_cv_accuracy": mean_cv_accuracy,
        "mean_training_time": mean_training_time,
        "status": status,
        "error": error_message
    })


# ============================================================
# Display Method Tuning Results
# ============================================================

method_results_df = pd.DataFrame(
    method_results
)


print("\n" + "=" * 70)
print("OPTIMISATION METHOD TUNING RESULTS")
print("=" * 70)

print(
    method_results_df[
        [
            "method",
            "mean_cv_accuracy",
            "mean_training_time",
            "status"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Select Successful Methods
# ============================================================

successful_results = method_results_df[
    method_results_df["status"] == "Successful"
].copy()


if successful_results.empty:

    raise RuntimeError(
        "All Optimisation methods failed during "
        "cross-validation. No final model can be selected."
    )


# ============================================================
# Graph 1:
# Mean Cross-Validation Accuracy
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    successful_results["method"],
    successful_results["mean_cv_accuracy"]
)

plt.xlabel(
    "Optimisation Method"
)

plt.ylabel(
    "Mean Cross-Validation Accuracy"
)

plt.title(
    "Ordinal Logistic Regression: "
    "Mean Cross-Validation Accuracy"
)

plt.ylim(
    0,
    1
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()


# ============================================================
# Graph 2:
# Mean Training Time
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    successful_results["method"],
    successful_results["mean_training_time"]
)

plt.xlabel(
    "Optimisation Method"
)

plt.ylabel(
    "Mean Training Time (seconds)"
)

plt.title(
    "Ordinal Logistic Regression: "
    "Mean Training Time Across 5 Folds"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()


# ============================================================
# Select Best Optimisation Method
# ============================================================

# ------------------------------------------------------------
# Primary Criterion:
# Highest Mean CV Accuracy
# ------------------------------------------------------------

highest_accuracy = (
    successful_results["mean_cv_accuracy"].max()
)


accuracy_tied_methods = successful_results[
    np.isclose(
        successful_results["mean_cv_accuracy"],
        highest_accuracy,
        rtol=1e-9,
        atol=1e-9
    )
].copy()


# ------------------------------------------------------------
# Secondary Criterion:
# Lowest Mean Training Time
# ------------------------------------------------------------

if len(accuracy_tied_methods) > 1:

    print("\n" + "=" * 70)
    print("ACCURACY TIE DETECTED")
    print("=" * 70)

    print(
        f"{len(accuracy_tied_methods)} methods achieved "
        f"the same mean CV accuracy."
    )

    print(
        "Mean training time will be used as the "
        "secondary selection criterion."
    )


    best_result = accuracy_tied_methods.loc[
        accuracy_tied_methods[
            "mean_training_time"
        ].idxmin()
    ]


else:

    best_result = accuracy_tied_methods.iloc[0]


# ============================================================
# Store Best Method
# ============================================================

best_method = best_result["method"]

best_mean_cv_accuracy = float(
    best_result["mean_cv_accuracy"]
)

best_mean_training_time = float(
    best_result["mean_training_time"]
)


# ============================================================
# Display Best Method
# ============================================================

print("\n" + "=" * 70)
print("BEST OPTIMISATION METHOD")
print("=" * 70)

print(
    f"Best Method: "
    f"{best_method}"
)

print(
    f"Mean CV Accuracy: "
    f"{best_mean_cv_accuracy:.4f}"
)

print(
    f"Mean Training Time: "
    f"{best_mean_training_time:.4f} seconds"
)


# ============================================================
# Prepare Full Training/Test Data
# ============================================================

X_train = prepare_features(
    X_train_raw
)

X_test = prepare_features(
    X_test_raw
)


# ============================================================
# Ensure Same Columns
# ============================================================

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


# ============================================================
# Standardise Full Training/Test Data
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

joblib.dump(
    scaler,
    "saved_models/ordinal_logistic_scaler.pkl"
)
joblib.dump(
    X_train.columns.tolist(),
    "saved_models/ordinal_logistic_features.pkl"
)


# ============================================================
# Convert Back to DataFrame
# ============================================================

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_train.columns,
    index=X_test.index
)


# ============================================================
# Create Final Ordinal Logistic Regression Model
# ============================================================

model = OrderedModel(
    endog=y_train,
    exog=X_train_scaled,
    distr="logit"
)


# ============================================================
# Train Final Model Using Best Method
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL ORDINAL LOGISTIC REGRESSION MODEL")
print("=" * 70)

print(
    f"Selected optimization method: "
    f"{best_method}"
)


# ------------------------------------------------------------
# Start Training Timer
# ------------------------------------------------------------

training_start_time = time.perf_counter()


# ------------------------------------------------------------
# Train Final Model
# ------------------------------------------------------------

result = model.fit(
    method=best_method,
    disp=False
)


# ------------------------------------------------------------
# End Training Timer
# ------------------------------------------------------------

training_end_time = time.perf_counter()


# ------------------------------------------------------------
# Calculate Final Training Time
# ------------------------------------------------------------

training_time = (
    training_end_time
    - training_start_time
)


# ============================================================
# Display Model Summary
# ============================================================

print("\n" + "=" * 70)
print("ORDINAL LOGISTIC REGRESSION MODEL")
print("=" * 70)

# Uncomment if model summary is required.
# print(result.summary())


# ============================================================
# Predict Probabilities
# ============================================================

predicted_probabilities = result.model.predict(
    result.params,
    exog=X_test_scaled
)


# ============================================================
# Predict Final Class
# ============================================================

# Select the class with the highest predicted probability.

y_pred = np.argmax(
    predicted_probabilities,
    axis=1
)

y_pred = y_pred.astype(int)


# ============================================================
# Model Evaluation
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Selected Method: "
    f"{best_method}"
)

print(
    f"Mean CV Accuracy: "
    f"{best_mean_cv_accuracy:.4f}"
)

print(
    f"Test Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Final Model Training Time: "
    f"{training_time:.4f} seconds"
)


# ============================================================
# Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Insufficient Weight",
            "Normal Weight",
            "Overweight Level I",
            "Overweight Level II",
            "Obesity Type I",
            "Obesity Type II",
            "Obesity Type III"
        ],
        zero_division=0
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# Display Example Predictions
# ============================================================

class_names = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}


print("\n" + "=" * 70)
print("EXAMPLE PREDICTIONS")
print("=" * 70)


for i in range(
    min(10, len(y_test))
):

    actual_class = int(
        y_test.iloc[i]
    )

    predicted_class = int(
        y_pred[i]
    )


    print(
        f"\nActual:    "
        f"{class_names[actual_class]}"
        f"\nPredicted: "
        f"{class_names[predicted_class]}"
    )


    print("Probabilities:")


    for class_index, probability in enumerate(
        predicted_probabilities[i]
    ):

        print(
            f"  "
            f"{class_names[class_index]:25s}: "
            f"{probability:.4f}"
        )