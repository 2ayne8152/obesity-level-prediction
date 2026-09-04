from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("data/ObesityDataSet_raw_and_data_sinthetic.csv")

# ==========================
# Select Numerical Features
# ==========================
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

results = []

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)

    outliers = ((df[column] < lower) | (df[column] > upper)).sum()

    results.append({
        "Feature": column,
        "Q1": round(Q1, 2),
        "Q3": round(Q3, 2),
        "IQR": round(IQR, 2),
        "Lower Boundary": round(lower, 2),
        "Upper Boundary": round(upper, 2),
        "Number of Outliers": outliers
    })

# ==========================
# Display Results
# ==========================
outlier_table = pd.DataFrame(results)

print(outlier_table)

# Save to CSV
outlier_table.to_csv("Outlier_Detection_IQR.csv", index=False)

# ==========================
# Plot Boxplots
# ==========================
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

axes = axes.flatten()

for i, column in enumerate(numerical_columns):

    sns.boxplot(
        y=df[column],
        ax=axes[i],
        color="skyblue"
    )

    axes[i].set_title(column, fontsize=12)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

plt.suptitle("Boxplots of Numerical Features", fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig("Numerical_Boxplots.png", dpi=300)

plt.show()