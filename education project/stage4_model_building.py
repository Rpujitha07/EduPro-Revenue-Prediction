# ============================================
# Stage 4 - Model Building
# EduPro Project
# ============================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv("course_features.csv")

print("="*60)
print("Dataset Loaded Successfully")
print("="*60)

print("Dataset Shape:", df.shape)

# ============================================
# Remove text columns
# ============================================

drop_columns = [
    "TransactionID",
    "UserID",
    "CourseID",
    "TransactionDate",
    "TeacherID",
    "UserName",
    "Email",
    "CourseName",
    "TeacherName",
    "Gender_y"          # Male/Female text column
]

df = df.drop(columns=drop_columns, errors="ignore")

# ============================================
# Target Variable
# ============================================

y = df["Amount"]

# Remove target from features
X = df.drop(columns=["Amount"])

# ============================================
# Keep ONLY numeric columns
# ============================================

X = X.select_dtypes(include=["number"])

print("\nFeatures Used:")
print(X.columns.tolist())

# ============================================
# Train Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ============================================
# Models
# ============================================

models = {

    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(),

    "Lasso Regression": Lasso(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        random_state=42
    )

}

results = []

best_model = None
best_name = ""
best_score = -999

# ============================================
# Train Models
# ============================================

for name, model in models.items():

    pipe = Pipeline([

        ("scaler", StandardScaler()),

        ("model", model)

    ])

    pipe.fit(X_train, y_train)

    predictions = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    print("\n"+"="*50)
    print(name)
    print("="*50)

    print("MAE :", round(mae,2))
    print("RMSE:", round(rmse,2))
    print("R2  :", round(r2,3))

    results.append({

        "Model":name,
        "MAE":round(mae,2),
        "RMSE":round(rmse,2),
        "R2":round(r2,3)

    })

    if r2 > best_score:

        best_score = r2
        best_model = pipe
        best_name = name

# ============================================
# Save Results
# ============================================

results_df = pd.DataFrame(results)

results_df.to_csv("model_results.csv", index=False)

with open("trained_models.pkl","wb") as f:

    pickle.dump(best_model,f)

print("\n")
print("="*60)
print("BEST MODEL :", best_name)
print("BEST R2 :", round(best_score,3))
print("="*60)

print("\nFiles Saved Successfully")

print("model_results.csv")

print("trained_models.pkl")

