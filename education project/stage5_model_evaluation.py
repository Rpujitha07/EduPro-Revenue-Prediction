
import pandas as pd
import matplotlib.pyplot as plt


# Load Model Results


results = pd.read_csv("model_results.csv")

print("=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)
print(results)


# Best Model


best_model = results.loc[results["R2"].idxmax()]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)
print(best_model)

# Plot R² Score Comparison


plt.figure(figsize=(10,5))
plt.bar(results["Model"], results["R2"])

plt.title("Model Comparison - R² Score")
plt.xlabel("Machine Learning Models")
plt.ylabel("R² Score")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("model_comparison_r2.png")

plt.show()


# Plot MAE Comparison


plt.figure(figsize=(10,5))
plt.bar(results["Model"], results["MAE"])

plt.title("Model Comparison - MAE")
plt.xlabel("Machine Learning Models")
plt.ylabel("Mean Absolute Error")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("model_comparison_mae.png")

plt.show()


# Plot RMSE Comparison


plt.figure(figsize=(10,5))
plt.bar(results["Model"], results["RMSE"])

plt.title("Model Comparison - RMSE")
plt.xlabel("Machine Learning Models")
plt.ylabel("Root Mean Squared Error")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("model_comparison_rmse.png")

plt.show()


# Save Best Model Details


best_model.to_frame().to_csv("best_model_summary.csv")

print("\n" + "=" * 60)
print("Evaluation Completed Successfully")
print("=" * 60)

print("\nFiles Created:")

print("✔ model_comparison_r2.png")
print("✔ model_comparison_mae.png")
print("✔ model_comparison_rmse.png")
print("✔ best_model_summary.csv")