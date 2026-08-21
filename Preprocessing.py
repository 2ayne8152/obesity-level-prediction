# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
def load_data():
    """
    Try loading via the official ucimlrepo package first. If that fails
    (e.g. no internet access), fall back to reading a local CSV named
    'ObesityDataSet_raw_and_data_sinthetic.csv' that you can download from:
    https://archive.ics.uci.edu/dataset/544
    """
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=544)
        X = dataset.data.features
        y = dataset.data.targets.squeeze()  # Series
        return X, y
    except Exception as e:
        print(f"ucimlrepo fetch failed ({e}); falling back to local CSV.")
        df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
        y = df["NObeyesdad"]
        X = df.drop(columns=["NObeyesdad"])
        return X, y


X, y = load_data()
print("Feature matrix shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# --------------------------------------------------------------------------
# 2. PREPROCESSING
# --------------------------------------------------------------------------
# Identify column types
binary_cols = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
nominal_cols = ["CAEC", "CALC", "MTRANS"]          # multi-category, unordered
numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Keep only columns that actually exist (robust to minor naming differences)
binary_cols = [c for c in binary_cols if c in X.columns]
nominal_cols = [c for c in nominal_cols if c in X.columns]
numeric_cols = [c for c in numeric_cols if c in X.columns]
categorical_cols = binary_cols + nominal_cols

# Encode target labels (7 obesity classes -> integers)
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print("\nClasses:", list(target_encoder.classes_))

# Train/test split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

# ColumnTransformer: one-hot encode categoricals, pass numeric columns through.
# Tree-based models like XGBoost don't need scaling, so numeric features
# are left as-is.
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)