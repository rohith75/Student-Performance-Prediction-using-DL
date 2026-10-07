import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .score {
        font-size: 48px;
        font-weight: 700;
    }

    .verdict {
        font-size: 28px;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        color: #777;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "student_performance_model.keras"
PREPROCESSOR_PATH = BASE_DIR / "student_preprocessor.pkl"
DATASET_PATH = BASE_DIR / "student_performance_dataset.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

@st.cache_resource
def load_preprocessor():

    return joblib.load(PREPROCESSOR_PATH)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(DATASET_PATH)


# ============================================================
# LOAD PROJECT FILES
# ============================================================

try:

    model = load_model()
    preprocessor = load_preprocessor()
    df = load_dataset()

except Exception as e:

    st.error("❌ Error loading project files.")

    st.exception(e)

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎓📊 STUDENT PERFORMANCE PREDICTION</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Deep Learning Based Student Academic Performance Prediction'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# STUDENT INPUT
# ============================================================

st.header("👨‍🎓 Student Information")


col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    attendance = st.number_input(
        "Attendance Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=82.0,
        step=1.0
    )


    study_hours = st.number_input(
        "Study Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=4.5,
        step=0.5
    )


    previous_marks = st.number_input(
        "Previous Semester Marks",
        min_value=0.0,
        max_value=100.0,
        value=76.0,
        step=1.0
    )


    assignments = st.number_input(
        "Assignments Completed",
        min_value=0,
        max_value=20,
        value=8,
        step=1
    )


    sleep_hours = st.number_input(
        "Sleep Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )


    internet_access = st.selectbox(
        "Internet Access",
        ["Yes", "No"],
        index=0
    )


    extracurricular = st.selectbox(
        "Extra-Curricular Activities",
        ["No", "Yes"],
        index=1
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    parental_education = st.selectbox(
        "Parental Education",
        [
            "High School",
            "Diploma",
            "Bachelor",
            "Master"
        ],
        index=2
    )


    class_participation = st.number_input(
        "Class Participation",
        min_value=0.0,
        max_value=100.0,
        value=78.0,
        step=1.0
    )


    mock_tests = st.number_input(
        "Mock Tests Attended",
        min_value=0,
        max_value=20,
        value=6,
        step=1
    )


    disciplinary_records = st.number_input(
        "Disciplinary Records",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )


    family_income = st.selectbox(
        "Family Income",
        ["Low", "Middle", "High"],
        index=1
    )


    distance = st.number_input(
        "Distance From College (KM)",
        min_value=0.0,
        max_value=200.0,
        value=8.0,
        step=0.5
    )


    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=21,
        step=1
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("---")


predict_button = st.button(
    "🔮 Predict Student Performance",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ====================================================
        # CREATE USER DATAFRAME
        # ====================================================

        user_data = pd.DataFrame({

            "Attendance_Percentage": [attendance],

            "Study_Hours_Per_Day": [study_hours],

            "Previous_Semester_Marks": [previous_marks],

            "Assignments_Completed": [assignments],

            "Sleep_Hours": [sleep_hours],

            "Internet_Access": [internet_access],

            "Extra_Curricular_Activities": [extracurricular],

            "Parental_Education": [parental_education],

            "Class_Participation": [class_participation],

            "Mock_Tests_Attended": [mock_tests],

            "Disciplinary_Records": [disciplinary_records],

            "Family_Income": [family_income],

            "Distance_From_College_KM": [distance],

            "Age": [age]
        })


        # ====================================================
        # DEEP LEARNING MODEL PREDICTION
        # ====================================================

        user_processed = preprocessor.transform(user_data)


        # Convert to numpy float32
        user_processed = np.asarray(
            user_processed,
            dtype=np.float32
        )


        # Check model input size
        expected_features = model.input_shape[-1]

        actual_features = user_processed.shape[1]


        if actual_features != expected_features:

            st.error(
                f"❌ Feature mismatch: Model expects "
                f"{expected_features} features but received "
                f"{actual_features} features."
            )

            st.stop()


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(
            user_processed,
            verbose=0
        )


        raw_prediction = float(
            np.asarray(prediction).flatten()[0]
        )


        # ====================================================
        # CONVERT MODEL OUTPUT TO SCORE
        # ====================================================

        if 0 <= raw_prediction <= 1:

            model_score = raw_prediction * 100

        else:

            model_score = raw_prediction


        model_score = float(
            np.clip(model_score, 0, 100)
        )


        # ====================================================
        # PERFORMANCE EVALUATION
        # ====================================================
        #
        # The final category is determined using the student's
        # actual input values.
        #
        # Maximum score = 100
        #
        # Attendance       = 20
        # Study Hours      = 20
        # Previous Marks   = 20
        # Assignments      = 10
        # Sleep            = 5
        # Participation    = 10
        # Mock Tests       = 5
        # Discipline       = 10
        #
        # Total             = 100
        # ====================================================


        performance_score = 0


        # ----------------------------------------------------
        # ATTENDANCE
        # ----------------------------------------------------

        if attendance >= 85:

            performance_score += 20

        elif attendance >= 75:

            performance_score += 16

        elif attendance >= 60:

            performance_score += 12

        elif attendance >= 50:

            performance_score += 8

        else:

            performance_score += 3


        # ----------------------------------------------------
        # STUDY HOURS
        # ----------------------------------------------------

        if study_hours >= 6:

            performance_score += 20

        elif study_hours >= 4:

            performance_score += 16

        elif study_hours >= 2:

            performance_score += 12

        else:

            performance_score += 5


        # ----------------------------------------------------
        # PREVIOUS SEMESTER MARKS
        # ----------------------------------------------------

        if previous_marks >= 85:

            performance_score += 20

        elif previous_marks >= 70:

            performance_score += 16

        elif previous_marks >= 60:

            performance_score += 12

        elif previous_marks >= 50:

            performance_score += 8

        else:

            performance_score += 3


        # ----------------------------------------------------
        # ASSIGNMENTS
        # ----------------------------------------------------

        if assignments >= 9:

            performance_score += 10

        elif assignments >= 7:

            performance_score += 8

        elif assignments >= 5:

            performance_score += 5

        else:

            performance_score += 2


        # ----------------------------------------------------
        # SLEEP
        # ----------------------------------------------------

        if 7 <= sleep_hours <= 9:

            performance_score += 5

        elif 6 <= sleep_hours < 7 or 9 < sleep_hours <= 10:

            performance_score += 4

        else:

            performance_score += 2


        # ----------------------------------------------------
        # CLASS PARTICIPATION
        # ----------------------------------------------------

        if class_participation >= 85:

            performance_score += 10

        elif class_participation >= 70:

            performance_score += 8

        elif class_participation >= 50:

            performance_score += 5

        else:

            performance_score += 2


        # ----------------------------------------------------
        # MOCK TESTS
        # ----------------------------------------------------

        if mock_tests >= 8:

            performance_score += 5

        elif mock_tests >= 5:

            performance_score += 4

        elif mock_tests >= 3:

            performance_score += 2

        else:

            performance_score += 1


        # ----------------------------------------------------
        # DISCIPLINARY RECORDS
        # ----------------------------------------------------

        if disciplinary_records == 0:

            performance_score += 10

        elif disciplinary_records <= 2:

            performance_score += 5

        else:

            performance_score += 0


        # ====================================================
        # FINAL VERDICT
        # ====================================================

        if performance_score >= 85:

            icon = "🏆"

            verdict = "Excellent"

            recommendation = (
                "Excellent academic performance! "
                "Keep maintaining your current study habits."
            )


        elif performance_score >= 70:

            icon = "👍"

            verdict = "Good"

            recommendation = (
                "Good academic performance. "
                "Keep working consistently."
            )


        elif performance_score >= 50:

            icon = "⚠️"

            verdict = "Need to Improve"

            recommendation = (
                "Your performance can be improved. "
                "Focus more on attendance, study hours, "
                "assignments and mock tests."
            )


        else:

            icon = "❌"

            verdict = "Poor"

            recommendation = (
                "Performance needs significant improvement. "
                "Increase study time, attendance and "
                "academic participation."
            )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.markdown("---")

        st.markdown(
            f"""
            <div class="result-box">

                <div style="font-size:60px;">
                    {icon}
                </div>

                <div class="verdict">
                    {verdict} Performance
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # RESULT METRICS
        # ====================================================

        result_col1, result_col2, result_col3 = st.columns(3)


        with result_col1:

            st.metric(
                "Performance Score",
                f"{performance_score}/100"
            )


        with result_col2:

            st.metric(
                "Final Verdict",
                f"{icon} {verdict}"
            )


        with result_col3:

            st.metric(
                "Model Prediction",
                f"{model_score:.2f}%"
            )


        # ====================================================
        # RECOMMENDATION
        # ====================================================

        st.info(
            f"💡 Recommendation: {recommendation}"
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Developed by D Venkata Rohith
    </div>
    """,
    unsafe_allow_html=True
)