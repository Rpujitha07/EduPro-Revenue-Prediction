# ============================================
# EduPro Revenue Prediction Web App
# ============================================

import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("trained_models.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🎓 EduPro Revenue Prediction System")

st.write("Enter Course Details")

# -------------------------
# User Inputs
# -------------------------

age = st.number_input("Student Age", 18, 60, 25)

course_price = st.number_input("Course Price", 50.0, 1000.0, 250.0)

course_duration = st.number_input("Course Duration (Hours)", 1, 100, 20)

course_rating = st.slider("Course Rating", 1.0, 5.0, 4.5)

teacher_age = st.number_input("Teacher Age", 22, 70, 35)

experience = st.number_input("Years of Experience", 0, 40, 8)

teacher_rating = st.slider("Teacher Rating", 1.0, 5.0, 4.7)

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Revenue"):

    input_data = pd.DataFrame({
        "Age_x":[age],
        "CoursePrice":[course_price],
        "CourseDuration":[course_duration],
        "CourseRating":[course_rating],
        "Age_y":[teacher_age],
        "YearsOfExperience":[experience],
        "TeacherRating":[teacher_rating]
    })

    # Add missing feature columns expected by the model
    training_columns = model.feature_names_in_

    for col in training_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[training_columns]

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Revenue: ₹ {prediction:.2f}")
    