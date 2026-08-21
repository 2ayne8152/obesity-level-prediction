from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# ==========================
# Binary Variables
# ==========================
binary_mapping = {
    "Female": 0, "Male": 1,
    "no": 0, "yes": 1
}

df["Gender"] = df["Gender"].map(binary_mapping)
df["family_history_with_overweight"] = df["family_history_with_overweight"].map(binary_mapping)
df["FAVC"] = df["FAVC"].map(binary_mapping)
df["SMOKE"] = df["SMOKE"].map(binary_mapping)
df["SCC"] = df["SCC"].map(binary_mapping)

# ==========================
# Ordinal Variables
# ==========================

# Consumption of food between meals
df["CAEC"] = df["CAEC"].map({
    "no": 0,
    "Sometimes": 1,
    "Frequently": 2,
    "Always": 3
})

# Alcohol consumption
df["CALC"] = df["CALC"].map({
    "no": 0,
    "Sometimes": 1,
    "Frequently": 2,
    "Always": 3
})

# Target class (ordered by obesity severity)
df["NObeyesdad"] = df["NObeyesdad"].map({
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6
})

# ==========================
# Nominal Variable
# ==========================

# Transportation mode (arbitrary labels for visualization only)
df["MTRANS"] = df["MTRANS"].map({
    "Walking": 0,
    "Bike": 1,
    "Motorbike": 2,
    "Public_Transportation": 3,
    "Automobile": 4
})

# ==========================
# Correlation Matrix
# ==========================
corr_matrix = df.corr(numeric_only=True)

# ==========================
# Plot Heatmap
# ==========================
plt.figure(figsize=(14, 12))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Heatmap of Obesity Dataset")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig("correlation.png", dpi=300)
plt.show()