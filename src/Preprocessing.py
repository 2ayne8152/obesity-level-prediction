import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42

def get_preprocessed_data():
    """Loads data, applies preprocessing, and returns train/test splits along with the preprocessor and encoder."""
    
    # Dynamically find the project root (assuming Preprocessing.py is in src/)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
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
            
            # Use pathlib to dynamically map to the data folder
            data_path = PROJECT_ROOT / "data" / "ObesityDataSet_raw_and_data_sinthetic.csv"
            df = pd.read_csv(data_path)
            
            y = df["NObeyesdad"]
            X = df.drop(columns=["NObeyesdad"])
            return X, y

    X, y = load_data()
    print("Feature matrix shape:", X.shape)
    
    # --------------------------------------------------------------------------
    # 2. PREPROCESSING
    # --------------------------------------------------------------------------
    binary_cols = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    ordinal_cols = ["CAEC", "CALC"]                    
    nominal_cols = ["MTRANS"]                          
    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

    binary_cols = [c for c in binary_cols if c in X.columns]
    ordinal_cols = [c for c in ordinal_cols if c in X.columns]
    nominal_cols = [c for c in nominal_cols if c in X.columns]
    numeric_cols = [c for c in numeric_cols if c in X.columns]

    binary_categories = [["Female", "Male"], ["no", "yes"], ["no", "yes"], ["no", "yes"], ["no", "yes"]]
    binary_categories = binary_categories[: len(binary_cols)]
    ordinal_categories = [["no", "Sometimes", "Frequently", "Always"]] * len(ordinal_cols)

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test, preprocessor, target_encoder