from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("data/ObesityDataSet_raw_and_data_sinthetic.csv")

# ==========================
# Numerical Features
# ==========================
numerical_columns = [
    "Age",
    "Height",
    "Weight",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE"
]

results = []

for column in numerical_columns:

    mean = df[column].mean()
    std = df[column].std()

    # Calculate Z-score
    z_scores = (df[column] - mean) / std

    # Detect outliers
    outliers = df[np.abs(z_scores) > 3]

    results.append({
        "Feature": column,
        "Mean": round(mean, 2),
        "Std Dev": round(std, 2),
        "Maximum |Z-score|": round(np.abs(z_scores).max(), 2),
        "Number of Outliers": len(outliers)
    })

# Display results
outlier_table = pd.DataFrame(results)

print(outlier_table)

# Save to CSV
outlier_table.to_csv("Outlier_Detection_ZScore.csv", index=False)

# ==========================
# Z-score Distribution Plot
# ==========================

plt.figure(figsize=(16, 12))

for i, column in enumerate(numerical_columns, 1):

    # Calculate Z-score
    z_scores = (df[column] - df[column].mean()) / df[column].std()

    plt.subplot(4, 2, i)

    # Plot distribution
    sns.histplot(
        z_scores,
        bins=30,
        kde=True
    )

    # Add outlier boundaries
    plt.axvline(3, linestyle="--", label="Z = +3")
    plt.axvline(-3, linestyle="--", label="Z = -3")

    plt.title(f"Z-score Distribution of {column}")
    plt.xlabel("Z-score")
    plt.ylabel("Frequency")
    plt.legend()

plt.tight_layout()
plt.show()
