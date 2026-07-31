import pandas as pd

df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

categorical_columns = df.select_dtypes(include="object").columns

print(df.duplicated())

