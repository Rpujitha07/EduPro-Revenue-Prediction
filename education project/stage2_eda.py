
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Improve graph appearance
sns.set_style("whitegrid")


# Read the merged dataset

df = pd.read_csv("course_dataset.csv")


# Display Dataset

print("=" * 60)
print("FIRST 5 ROWS OF DATASET")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
df.info()

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

# GRAPH 1 : Course Price Distribution

plt.figure(figsize=(8,5))
plt.hist(df["CoursePrice"], bins=10)
plt.title("Course Price Distribution")
plt.xlabel("Course Price")
plt.ylabel("Number of Courses")
plt.savefig("course_price_distribution.png")
plt.show()

# GRAPH 2 : Payment Method Distribution

plt.figure(figsize=(7,5))
df["PaymentMethod"].value_counts().plot(kind="bar")
plt.title("Payment Method Distribution")
plt.xlabel("Payment Method")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("payment_method_distribution.png")
plt.show()

# GRAPH 3 : Course Category Distribution

plt.figure(figsize=(9,5))
df["CourseCategory"].value_counts().plot(kind="bar")
plt.title("Course Category Distribution")
plt.xlabel("Course Category")
plt.ylabel("Number of Courses")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("course_category_distribution.png")
plt.show()


# GRAPH 4 : Teacher Rating Distribution

plt.figure(figsize=(8,5))
plt.hist(df["TeacherRating"], bins=10)
plt.title("Teacher Rating Distribution")
plt.xlabel("Teacher Rating")
plt.ylabel("Number of Teachers")
plt.savefig("teacher_rating_distribution.png")
plt.show()


# GRAPH 5 : Course Rating Distribution

plt.figure(figsize=(8,5))
plt.hist(df["CourseRating"], bins=10)
plt.title("Course Rating Distribution")
plt.xlabel("Course Rating")
plt.ylabel("Number of Courses")
plt.savefig("course_rating_distribution.png")
plt.show()


# GRAPH 6 : Correlation Heatmap

numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()


# EDA Completed
print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("Graphs have been generated and saved.")
print("=" * 60)