import streamlit as st
import pandas as pd
import joblib

# Load dataset (for slider/selectbox ranges)
df = pd.read_csv("student_habits_performance.csv")

# Load preprocessor and model
preprocessor = joblib.load("preprocessor.joblib")
model = joblib.load("linear_regression_model.joblib")

st.title("🎓 Student Performance Prediction App")

st.write("Adjust the inputs below to predict student performance.")

# Input form
age = st.slider("Age", min_value=int(df['age'].min()), max_value=int(df['age'].max()), value=int(df['age'].mean()))
gender = st.selectbox("Gender", df['gender'].unique())
study_hours_per_day = st.slider("Study Hours Per Day", min_value=float(df['study_hours_per_day'].min()), max_value=float(df['study_hours_per_day'].max()), value=float(df['study_hours_per_day'].mean()), step=0.1)
social_media_hours = st.slider("Social Media Hours", min_value=float(df['social_media_hours'].min()), max_value=float(df['social_media_hours'].max()), value=float(df['social_media_hours'].mean()), step=0.1)
netflix_hours = st.slider("Netflix Hours", min_value=float(df['netflix_hours'].min()), max_value=float(df['netflix_hours'].max()), value=float(df['netflix_hours'].mean()), step=0.1)
part_time_job = st.selectbox("Part-time Job", df['part_time_job'].unique())
attendance_percentage = st.slider("Attendance Percentage", min_value=float(df['attendance_percentage'].min()), max_value=float(df['attendance_percentage'].max()), value=float(df['attendance_percentage'].mean()), step=0.1)
sleep_hours = st.slider("Sleep Hours", min_value=float(df['sleep_hours'].min()), max_value=float(df['sleep_hours'].max()), value=float(df['sleep_hours'].mean()), step=0.1)
diet_quality = st.selectbox("Diet Quality", df['diet_quality'].unique())
exercise_frequency = st.slider("Exercise Frequency", min_value=int(df['exercise_frequency'].min()), max_value=int(df['exercise_frequency'].max()), value=int(df['exercise_frequency'].mean()))
parental_education_level = st.selectbox("Parental Education Level", df['parental_education_level'].unique())
internet_quality = st.selectbox("Internet Quality", df['internet_quality'].unique())
mental_health_rating = st.slider("Mental Health Rating (1-10)", min_value=int(df['mental_health_rating'].min()), max_value=int(df['mental_health_rating'].max()), value=int(df['mental_health_rating'].mean()))
extracurricular_participation = st.selectbox("Extracurricular Participation", df['extracurricular_participation'].unique())

# Create dataframe for input
input_data = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "study_hours_per_day": study_hours_per_day,
    "social_media_hours": social_media_hours,
    "netflix_hours": netflix_hours,
    "part_time_job": part_time_job,
    "attendance_percentage": attendance_percentage,
    "sleep_hours": sleep_hours,
    "diet_quality": diet_quality,
    "exercise_frequency": exercise_frequency,
    "parental_education_level": parental_education_level,
    "internet_quality": internet_quality,
    "mental_health_rating": mental_health_rating,
    "extracurricular_participation": extracurricular_participation
}])

# Predict button
if st.button("Predict Performance"):
    # Preprocess input
    X_processed = preprocessor.transform(input_data)
    # Predict
    prediction = model.predict(X_processed)
    st.success(f"🎯 Predicted Student Performance: **{prediction[0]:.2f}**")
