import streamlit as st
import pandas as pd
import pickle
import os

# ============================================
# Load trained model
# ============================================

model_path = os.path.join(os.path.dirname(__file__), "trained_models.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# ============================================
# App Title
# ============================================

st.title("🎓 EduPro Revenue Prediction System")

st.write("Enter Course Details")

# ============================================
# User Inputs
# ============================================

age = st.number_input("Student Age", min_value=18, max_value=60, value=25)

course_price = st.number_input(
    "Course Price",
    min_value=50.0,
    max_value=1000.0,
    value=250.0
)

course_duration = st.number_input(
    "Course Duration (Hours)",
    min_value=1,
    max_value=100,
    value=20
)

course_rating = st.slider(
    "Course Rating",
    min_value=1.0,
    max_value=5.0,
    value=4.5
)

teacher_age = st.number_input(
    "Teacher Age",
    min_value=22,
    max_value=70,
    value=35
)

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=40,
    value=8
)

teacher_rating = st.slider(
    "Teacher Rating",
    min_value=1.0,
    max_value=5.0,
    value=4.7
)

# ============================================
# Prediction
# ============================================

if st.button("Predict Revenue"):

    input_data = pd.DataFrame({
        "Age_x": [age],
        "CoursePrice": [course_price],
        "CourseDuration": [course_duration],
        "CourseRating": [course_rating],
        "Age_y": [teacher_age],
        "YearsOfExperience": [experience],
        "TeacherRating": [teacher_rating]
    })

    # Get expected columns from the trained model
    training_columns = model.feature_names_in_

    # Add missing columns if any
    for col in training_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Arrange columns in the same order as training
    input_data = input_data[training_columns]

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Revenue: ₹ {prediction:.2f}")
