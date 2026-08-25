import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler


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
# Identify column types.
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

# Explicit category orders. For binary columns the order just fixes which
# label maps to 0 vs 1; for CAEC/CALC the order is the real severity order.
binary_categories = [["Female", "Male"], ["no", "yes"], ["no", "yes"], ["no", "yes"], ["no", "yes"]]
binary_categories = binary_categories[: len(binary_cols)]

ordinal_categories = [["no", "Sometimes", "Frequently", "Always"]] * len(ordinal_cols)

# Encode target labels in their natural CLINICAL order (kept consistent with
# LogisticRegression.py, even though XGBoost itself doesn't require an
# ordered target — this keeps class_names/report ordering identical across
# both scripts for easy side-by-side comparison).
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

# ColumnTransformer: ordinal-encode binary/ordinal columns, one-hot encode
# only the genuinely nominal column (MTRANS, drop="first"), and scale
# numeric columns. Same preprocessing as LogisticRegression.py.
preprocessor = ColumnTransformer(
    transformers=[
        ("bin", OrdinalEncoder(categories=binary_categories), binary_cols),
        ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
        (
            "nom",
            OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            nominal_cols,
        ),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Train/test split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)