import pandas as pd

# Read the Excel file
excel_file = "EduPro Online Platform.xlsx"

# Read all sheets
users = pd.read_excel(excel_file, sheet_name="Users")
teachers = pd.read_excel(excel_file, sheet_name="Teachers")
courses = pd.read_excel(excel_file, sheet_name="Courses")
transactions = pd.read_excel(excel_file, sheet_name="Transactions")

# Display the size of each dataset
print("Users Shape:", users.shape)
print("Teachers Shape:", teachers.shape)
print("Courses Shape:", courses.shape)
print("Transactions Shape:", transactions.shape)

# Display first 5 rows
print("\nUsers Data:")
print(users.head())

print("\nTeachers Data:")
print(teachers.head())

print("\nCourses Data:")
print(courses.head())

print("\nTransactions Data:")
print(transactions.head())

# Check missing values
print("\nMissing Values:")
print(users.isnull().sum())
print(teachers.isnull().sum())
print(courses.isnull().sum())
print(transactions.isnull().sum())

# Remove duplicate rows
users = users.drop_duplicates()
teachers = teachers.drop_duplicates()
courses = courses.drop_duplicates()
transactions = transactions.drop_duplicates()

print("\nDuplicate rows removed successfully.") 

print("\nUsers Columns:")
print(users.columns)

print("\nTeachers Columns:")
print(teachers.columns)

print("\nCourses Columns:")
print(courses.columns)

print("\nTransactions Columns:")
print(transactions.columns)

# Merge Transactions with Users

merged_data = pd.merge(
    transactions,
    users,
    on="UserID",
    how="left"
)


# Merge with Courses

merged_data = pd.merge(
    merged_data,
    courses,
    on="CourseID",
    how="left"
)


# Merge with Teachers

merged_data = pd.merge(
    merged_data,
    teachers,
    on="TeacherID",
    how="left"
)

# Display merged dataset
print("\nMerged Dataset:")
print(merged_data.head())

print("\nMerged Dataset Shape:")
print(merged_data.shape)

# Save merged dataset
merged_data.to_csv("course_dataset.csv", index=False)

print("\ncourse_dataset.csv created successfully!")