import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

RANDOM_STATE = 42

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
df = X.copy()
df["NObeyesdad"] = y.values

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
target_mapping = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6
}
df["NObeyesdad"] = df["NObeyesdad"].map(target_mapping)

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
})            # Target variable

# ==========================
# Separate Features and Target
# ==========================
X = df.drop("NObeyesdad", axis=1)   # Input features
y = df["NObeyesdad"]   

# ==========================
# 80:20 Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,      # 20% testing
    random_state=42,      # Ensures reproducibility
    stratify=y,           # Preserves class distribution
    shuffle=True          # Randomly shuffle before splitting
)

# ==========================
# Display Dataset Sizes
# ==========================
print("Original Dataset:", df.shape)
print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)

print("\nTraining Set Percentage: {:.1f}%".format(len(X_train)/len(df)*100))
print("Testing Set Percentage: {:.1f}%".format(len(X_test)/len(df)*100))