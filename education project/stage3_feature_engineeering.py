
# Import Libraries
import pandas as pd


# Read Dataset

df = pd.read_csv("course_dataset.csv")

print("="*60)
print("FIRST 5 ROWS")
print("="*60)
print(df.head())


# Feature 1 : Price Category


def price_category(price):
    if price < 100:
        return "Low"
    elif price < 300:
        return "Medium"
    else:
        return "High"

df["PriceCategory"] = df["CoursePrice"].apply(price_category)


# Feature 2 : Experience Level


def experience_level(exp):
    if exp < 5:
        return "Beginner"
    elif exp < 15:
        return "Intermediate"
    else:
        return "Expert"

df["ExperienceLevel"] = df["YearsOfExperience"].apply(experience_level)


# Feature 3 : Duration Category


def duration_category(duration):
    if duration < 10:
        return "Short"
    elif duration < 25:
        return "Medium"
    else:
        return "Long"

df["DurationCategory"] = df["CourseDuration"].apply(duration_category)




print("\nNew Features Created Successfully\n")

print(df[[
    "CoursePrice",
    "PriceCategory",
    "YearsOfExperience",
    "ExperienceLevel",
    "CourseDuration",
    "DurationCategory"
]].head())


# Convert Text Columns into Numbers


categorical_columns = [
    "Gender_x",
    "PaymentMethod",
    "CourseCategory",
    "CourseType",
    "CourseLevel",
    "PriceCategory",
    "ExperienceLevel",
    "DurationCategory",
    "Expertise"
]

df = pd.get_dummies(df, columns=categorical_columns)


# Save Feature Dataset


df.to_csv("course_features.csv", index=False)

print("\n===================================")
print("Feature Engineering Completed")
print("course_features.csv created successfully!")
print("===================================")

print("\nFinal Dataset Shape:")
print(df.shape)