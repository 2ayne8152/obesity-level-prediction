import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold

from statsmodels.miscmodels.ordinal_model import OrderedModel


# ============================================================
# Import Training/Test Data
# ============================================================

from Preprocessing import (
    X_train,
    X_test,
    y_train,
    y_test
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

MODEL_DIRECTORY = "saved_models"

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIRECTORY,
    "ordinal_logistic_regression.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIRECTORY,
    "ordinal_logistic_preprocessor.pkl"
)

FEATURES_PATH = os.path.join(
    MODEL_DIRECTORY,
    "ordinal_logistic_features.pkl"
)


# ============================================================
# Create Saved Model Directory
# ============================================================

os.makedirs(
    MODEL_DIRECTORY,
    exist_ok=True
)


# ============================================================
# Keep RAW Training/Test Data
# ============================================================

# X_train and X_test from the new Preprocessing.py are still
# raw feature DataFrames.
#
# Preprocessing is performed separately inside each
# cross-validation fold to prevent data leakage.

X_train_raw = X_train.copy()
X_test_raw = X_test.copy()


# The new Preprocessing.py returns y_train and y_test as
# NumPy arrays.

y_train = np.asarray(
    y_train
).astype(int)

y_test = np.asarray(
    y_test
).astype(int)


# ============================================================
# Feature Definitions
# ============================================================

binary_cols = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "SMOKE",
    "SCC"
]

ordinal_cols = [
    "CAEC",
    "CALC"
]

nominal_cols = [
    "MTRANS"
]

numeric_cols = [
    "Age",
    "Height",
    "Weight",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE"
]


# ============================================================
# Keep Only Existing Columns
# ============================================================

binary_cols = [
    column
    for column in binary_cols
    if column in X_train_raw.columns
]

ordinal_cols = [
    column
    for column in ordinal_cols
    if column in X_train_raw.columns
]

nominal_cols = [
    column
    for column in nominal_cols
    if column in X_train_raw.columns
]

numeric_cols = [
    column
    for column in numeric_cols
    if column in X_train_raw.columns
]


# ============================================================
# Explicit Category Definitions
# ============================================================

binary_categories = {
    "Gender": [
        "Female",
        "Male"
    ],

    "family_history_with_overweight": [
        "no",
        "yes"
    ],

    "FAVC": [
        "no",
        "yes"
    ],

    "SMOKE": [
        "no",
        "yes"
    ],

    "SCC": [
        "no",
        "yes"
    ]
}


ordinal_categories = {
    "CAEC": [
        "no",
        "Sometimes",
        "Frequently",
        "Always"
    ],

    "CALC": [
        "no",
        "Sometimes",
        "Frequently",
        "Always"
    ]
}


# ============================================================
# Create Preprocessor
# ============================================================

def create_preprocessor():

    current_binary_categories = [
        binary_categories[column]
        for column in binary_cols
    ]

    current_ordinal_categories = [
        ordinal_categories[column]
        for column in ordinal_cols
    ]


    preprocessor = ColumnTransformer(
        transformers=[

            # ------------------------------------------------
            # Binary Variables
            # ------------------------------------------------

            (
                "bin",

                OrdinalEncoder(
                    categories=current_binary_categories
                ),

                binary_cols
            ),


            # ------------------------------------------------
            # Ordinal Variables
            # ------------------------------------------------

            (
                "ord",

                OrdinalEncoder(
                    categories=current_ordinal_categories
                ),

                ordinal_cols
            ),


            # ------------------------------------------------
            # Nominal Variables
            # ------------------------------------------------

            (
                "nom",

                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False
                ),

                nominal_cols
            ),


            # ------------------------------------------------
            # Numerical Variables
            # ------------------------------------------------

            (
                "num",

                StandardScaler(),

                numeric_cols
            )
        ],

        remainder="drop"
    )


    return preprocessor


# ============================================================
# Cross-Validation Function
# ============================================================

def cross_validate_method(
    X,
    y,
    method,
    n_splits=N_SPLITS
):

    # --------------------------------------------------------
    # Create Stratified K-Fold Cross-Validation
    # --------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )


    # --------------------------------------------------------
    # Store Fold Results
    # --------------------------------------------------------

    fold_accuracies = []
    fold_training_times = []


    # --------------------------------------------------------
    # Failure Tracking
    # --------------------------------------------------------

    failed = False

    failed_fold = None

    error_message = ""


    # ========================================================
    # Run Each Fold
    # ========================================================

    for fold, (
        train_index,
        validation_index
    ) in enumerate(
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

            X_fold_train = X.iloc[
                train_index
            ].copy()

            X_fold_valid = X.iloc[
                validation_index
            ].copy()


            y_fold_train = y[
                train_index
            ]

            y_fold_valid = y[
                validation_index
            ]


            # ------------------------------------------------
            # Create Fold-Specific Preprocessor
            # ------------------------------------------------

            fold_preprocessor = (
                create_preprocessor()
            )


            # ------------------------------------------------
            # Fit Preprocessor ONLY on Training Fold
            # ------------------------------------------------

            X_fold_train_processed = (
                fold_preprocessor.fit_transform(
                    X_fold_train
                )
            )


            # ------------------------------------------------
            # Transform Validation Fold
            # ------------------------------------------------

            X_fold_valid_processed = (
                fold_preprocessor.transform(
                    X_fold_valid
                )
            )


            # ------------------------------------------------
            # Get Feature Names
            # ------------------------------------------------

            feature_names = (
                fold_preprocessor
                .get_feature_names_out()
            )


            # ------------------------------------------------
            # Convert Processed Data to DataFrames
            # ------------------------------------------------

            X_fold_train_processed = pd.DataFrame(
                X_fold_train_processed,
                columns=feature_names,
                index=X_fold_train.index
            )

            X_fold_valid_processed = pd.DataFrame(
                X_fold_valid_processed,
                columns=feature_names,
                index=X_fold_valid.index
            )


            # ------------------------------------------------
            # Convert Targets to Series
            # ------------------------------------------------

            y_fold_train_series = pd.Series(
                y_fold_train,
                index=X_fold_train.index
            )

            y_fold_valid_series = pd.Series(
                y_fold_valid,
                index=X_fold_valid.index
            )


            # =================================================
            # Create Ordinal Logistic Regression Model
            # =================================================

            model = OrderedModel(
                endog=y_fold_train_series,
                exog=X_fold_train_processed,
                distr="logit"
            )


            # =================================================
            # Start Training Timer
            # =================================================

            training_start_time = (
                time.perf_counter()
            )


            # =================================================
            # Train Model
            # =================================================

            result = model.fit(
                method=method,
                disp=False
            )


            # =================================================
            # End Training Timer
            # =================================================

            training_end_time = (
                time.perf_counter()
            )


            fold_training_time = (
                training_end_time
                - training_start_time
            )


            # ------------------------------------------------
            # Store Training Time
            # ------------------------------------------------

            fold_training_times.append(
                fold_training_time
            )


            # =================================================
            # Predict Validation Probabilities
            # =================================================

            predicted_probabilities = (
                result.model.predict(
                    result.params,
                    exog=X_fold_valid_processed
                )
            )


            # =================================================
            # Predict Final Class
            # =================================================

            y_pred = np.argmax(
                predicted_probabilities,
                axis=1
            )

            y_pred = y_pred.astype(int)


            # =================================================
            # Calculate Fold Accuracy
            # =================================================

            fold_accuracy = accuracy_score(
                y_fold_valid_series,
                y_pred
            )


            fold_accuracies.append(
                fold_accuracy
            )


            # =================================================
            # Display Successful Fold
            # =================================================

            print(
                f"Accuracy = {fold_accuracy:.4f}, "
                f"Training Time = "
                f"{fold_training_time:.4f}s"
            )


        # ====================================================
        # Handle Failed Fold
        # ====================================================

        except Exception as e:

            failed = True

            failed_fold = fold

            error_message = (
                f"{type(e).__name__}: {str(e)}"
            )


            print(
                "FAILED"
            )


            print(
                f"        Error: {error_message}"
            )


            # ------------------------------------------------
            # Newton-Specific Explanation
            # ------------------------------------------------

            if method.lower() == "newton":

                print(
                    "        Newton's method failed during "
                    "this fold."
                )

                print(
                    "        This commonly occurs when the "
                    "Hessian matrix is singular and cannot "
                    "be inverted."
                )


            # ------------------------------------------------
            # Stop Remaining Folds
            # ------------------------------------------------

            print(
                f"        Optimisation method '{method}' "
                f"is marked as FAILED."
            )

            print(
                "        Remaining folds will not be "
                "tested for this method."
            )


            break


    # ========================================================
    # Failed Method
    # ========================================================

    if failed:

        return (
            np.nan,
            np.nan,
            fold_accuracies,
            fold_training_times,
            "Failed",
            failed_fold,
            error_message
        )


    # ========================================================
    # Verify Complete 5-Fold CV
    # ========================================================

    if len(fold_accuracies) != n_splits:

        return (
            np.nan,
            np.nan,
            fold_accuracies,
            fold_training_times,
            "Failed",
            None,
            "Incomplete cross-validation."
        )


    # ========================================================
    # Calculate Mean CV Accuracy
    # ========================================================

    mean_cv_accuracy = np.mean(
        fold_accuracies
    )


    # ========================================================
    # Calculate Mean Training Time
    # ========================================================

    mean_training_time = np.mean(
        fold_training_times
    )


    # ========================================================
    # Return Successful Results
    # ========================================================

    return (
        mean_cv_accuracy,
        mean_training_time,
        fold_accuracies,
        fold_training_times,
        "Successful",
        None,
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
print(
    "PARAMETER TUNING: OPTIMISATION METHOD"
)
print("=" * 70)


print(
    "\nUsing 5-fold cross-validation."
)

print(
    "Primary metric: Mean Cross-Validation Accuracy"
)

print(
    "Secondary metric: Mean Training Time"
)

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
        failed_fold,
        error_message
    ) = cross_validate_method(
        X_train_raw,
        y_train,
        method=method,
        n_splits=N_SPLITS
    )


    # ========================================================
    # Successful Method
    # ========================================================

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


    # ========================================================
    # Failed Method
    # ========================================================

    else:

        print(
            "\n    Mean CV Accuracy: FAILED"
        )


        if failed_fold is not None:

            print(
                f"    Failed Fold: "
                f"{failed_fold}"
            )


        print(
            f"    Reason: "
            f"{error_message}"
        )


    # ========================================================
    # Store Results
    # ========================================================

    method_results.append({

        "method": method,

        "mean_cv_accuracy":
            mean_cv_accuracy,

        "mean_training_time":
            mean_training_time,

        "status":
            status,

        "failed_fold":
            failed_fold,

        "error":
            error_message
    })


# ============================================================
# Display Method Tuning Results
# ============================================================

method_results_df = pd.DataFrame(
    method_results
)


print("\n" + "=" * 70)
print(
    "OPTIMISATION METHOD TUNING RESULTS"
)
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


# ------------------------------------------------------------
# Check That At Least One Method Succeeded
# ------------------------------------------------------------

if successful_results.empty:

    raise RuntimeError(
        "All optimisation methods failed during "
        "5-fold cross-validation. "
        "No final model can be selected."
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
    successful_results[
        "mean_cv_accuracy"
    ].max()
)


accuracy_tied_methods = successful_results[
    np.isclose(
        successful_results[
            "mean_cv_accuracy"
        ],
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
    print(
        "ACCURACY TIE DETECTED"
    )
    print("=" * 70)


    print(
        f"{len(accuracy_tied_methods)} methods achieved "
        f"the same mean CV accuracy."
    )


    print(
        "Mean training time will be used as the "
        "secondary selection criterion."
    )


    best_result = (
        accuracy_tied_methods.loc[
            accuracy_tied_methods[
                "mean_training_time"
            ].idxmin()
        ]
    )


else:

    best_result = (
        accuracy_tied_methods.iloc[0]
    )


# ============================================================
# Store Best Method
# ============================================================

best_method = best_result[
    "method"
]


best_mean_cv_accuracy = float(
    best_result[
        "mean_cv_accuracy"
    ]
)


best_mean_training_time = float(
    best_result[
        "mean_training_time"
    ]
)


# ============================================================
# Display Best Method
# ============================================================

print("\n" + "=" * 70)
print(
    "BEST OPTIMISATION METHOD"
)
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

print("\n" + "=" * 70)
print(
    "PREPARING FULL TRAINING AND TEST DATA"
)
print("=" * 70)


# ------------------------------------------------------------
# Create Final Preprocessor
# ------------------------------------------------------------

final_preprocessor = (
    create_preprocessor()
)


# ------------------------------------------------------------
# Fit ONLY on Full Training Data
# ------------------------------------------------------------

X_train_processed = (
    final_preprocessor.fit_transform(
        X_train_raw
    )
)


# ------------------------------------------------------------
# Transform Test Data
# ------------------------------------------------------------

X_test_processed = (
    final_preprocessor.transform(
        X_test_raw
    )
)


# ============================================================
# Get Feature Names
# ============================================================

final_feature_names = (
    final_preprocessor
    .get_feature_names_out()
)


# ============================================================
# Convert Processed Data to DataFrames
# ============================================================

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=final_feature_names,
    index=X_train_raw.index
)


X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=final_feature_names,
    index=X_test_raw.index
)


# ============================================================
# Convert Targets to Series
# ============================================================

y_train_series = pd.Series(
    y_train,
    index=X_train_raw.index
)


y_test_series = pd.Series(
    y_test,
    index=X_test_raw.index
)


# ============================================================
# Display Processed Data Information
# ============================================================

print(
    f"Original number of features: "
    f"{X_train_raw.shape[1]}"
)


print(
    f"Processed number of features: "
    f"{X_train_processed.shape[1]}"
)


# ============================================================
# Create Final Ordinal Logistic Regression Model
# ============================================================

model = OrderedModel(
    endog=y_train_series,
    exog=X_train_processed,
    distr="logit"
)


# ============================================================
# Train Final Model Using Best Method
# ============================================================

print("\n" + "=" * 70)
print(
    "TRAINING FINAL ORDINAL LOGISTIC REGRESSION MODEL"
)
print("=" * 70)


print(
    f"Selected optimisation method: "
    f"{best_method}"
)


# ------------------------------------------------------------
# Start Training Timer
# ------------------------------------------------------------

training_start_time = (
    time.perf_counter()
)


# ------------------------------------------------------------
# Final Model Error Handling
# ------------------------------------------------------------

try:

    result = model.fit(
        method=best_method,
        disp=False
    )


except Exception as e:

    final_error = (
        f"{type(e).__name__}: {str(e)}"
    )


    print(
        "\nERROR: The selected optimisation method "
        "failed while training the final model."
    )


    print(
        f"Method: {best_method}"
    )


    print(
        f"Reason: {final_error}"
    )


    raise RuntimeError(
        "Final Ordinal Logistic Regression model "
        "could not be trained using the selected "
        f"optimisation method '{best_method}'."
    ) from e


# ------------------------------------------------------------
# End Training Timer
# ------------------------------------------------------------

training_end_time = (
    time.perf_counter()
)


# ------------------------------------------------------------
# Calculate Final Training Time
# ------------------------------------------------------------

training_time = (
    training_end_time
    - training_start_time
)


# ============================================================
# Save Final Model
# ============================================================

result.save(
    FINAL_MODEL_PATH
)


# ============================================================
# Save Final Preprocessor
# ============================================================

joblib.dump(
    final_preprocessor,
    PREPROCESSOR_PATH
)


# ============================================================
# Save Feature Names
# ============================================================

joblib.dump(
    list(final_feature_names),
    FEATURES_PATH
)


print(
    f"\nFinal model saved to: "
    f"{FINAL_MODEL_PATH}"
)


print(
    f"Preprocessor saved to: "
    f"{PREPROCESSOR_PATH}"
)


print(
    f"Feature names saved to: "
    f"{FEATURES_PATH}"
)


# ============================================================
# Display Model Summary
# ============================================================

print("\n" + "=" * 70)
print(
    "ORDINAL LOGISTIC REGRESSION MODEL"
)
print("=" * 70)


# Uncomment if model summary is required.
# print(result.summary())


# ============================================================
# Predict Probabilities
# ============================================================

predicted_probabilities = (
    result.model.predict(
        result.params,
        exog=X_test_processed
    )
)


# ============================================================
# Predict Final Class
# ============================================================

y_pred = np.argmax(
    predicted_probabilities,
    axis=1
)


y_pred = y_pred.astype(int)


# ============================================================
# Model Evaluation
# ============================================================

accuracy = accuracy_score(
    y_test_series,
    y_pred
)


print("\n" + "=" * 70)
print(
    "FINAL MODEL PERFORMANCE"
)
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

class_names = [
    "Insufficient Weight",
    "Normal Weight",
    "Overweight Level I",
    "Overweight Level II",
    "Obesity Type I",
    "Obesity Type II",
    "Obesity Type III"
]


print(
    "\nClassification Report:"
)


print(
    classification_report(
        y_test_series,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test_series,
    y_pred
)


print(
    "\nConfusion Matrix:"
)


print(
    cm
)


# ============================================================
# Display Example Predictions
# ============================================================

class_names_dict = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}


print("\n" + "=" * 70)
print(
    "EXAMPLE PREDICTIONS"
)
print("=" * 70)


for i in range(
    min(10, len(y_test_series))
):

    actual_class = int(
        y_test_series.iloc[i]
    )


    predicted_class = int(
        y_pred[i]
    )


    print(
        f"\nActual:    "
        f"{class_names_dict[actual_class]}"
        f"\nPredicted: "
        f"{class_names_dict[predicted_class]}"
    )


    print(
        "Probabilities:"
    )


    for (
        class_index,
        probability
    ) in enumerate(
        predicted_probabilities[i]
    ):

        print(
            f"  "
            f"{class_names_dict[class_index]:25s}: "
            f"{probability:.4f}"
        )