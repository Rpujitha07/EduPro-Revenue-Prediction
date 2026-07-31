import pandas as pd

df = pd.read_csv("course_features.csv")

print("Columns in course_features.csv:\n")
print(df.columns.tolist())
import pandas as pd

df = pd.read_csv("course_features.csv")

print(df.select_dtypes(include="number").columns.tolist())