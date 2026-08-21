import pandas as pd

df = pd.read_csv("csv/ObesityDataSet_raw_and_data_sinthetic.csv")

categorical_columns = df.select_dtypes(include="object").columns
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

for col in categorical_columns:
    print(f"Unique values in {col}:")
    print(df[col].unique())
    print()

for col in numerical_columns:
    print(f"Summary statistics for {col}:")
    print(df[col].describe())
    print()