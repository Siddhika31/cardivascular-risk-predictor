import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('cardio_model.pkl')

# --- Page Setup ---
st.set_page_config(page_title="Heart Risk AI", page_icon="🫀", layout="wide")

# --- Title ---
st.markdown("<h1 style='text-align: center; color: crimson;'>🫀 Cardiovascular Disease Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>A Machine Learning Based Health Screening Tool</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("🧾 Enter Patient Details")
age = st.sidebar.slider("Age", 18, 100, 45)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 22.0)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
exercise = st.sidebar.radio("Exercise Regularly?", ["Yes", "No"])
smoking = st.sidebar.selectbox("Smoking History", ["Never", "Former", "Current"])
alcohol = st.sidebar.slider("Alcohol (drinks/week)", 0, 30, 2)
fruit = st.sidebar.slider("Fruit Intake (servings/day)", 0, 10, 3)
greens = st.sidebar.slider("Green Veg Intake (servings/day)", 0, 10, 4)
fried = st.sidebar.slider("Fried Potato Intake (servings/week)", 0, 10, 2)
checkup = st.sidebar.radio("Had recent health checkup?", ["Yes", "No"])

# --- Manual Encoding ---
sex_encoded = 1 if sex == "Male" else 0
exercise_encoded = 1 if exercise == "Yes" else 0
smoking_encoded = {"Never": 0, "Former": 1, "Current": 2}[smoking]
checkup_encoded = 1 if checkup == "Yes" else 0

# --- Input Array ---
input_data = np.array([[age, bmi, sex_encoded, exercise_encoded, smoking_encoded,
                        alcohol, fruit, greens, fried, checkup_encoded]])

# --- Prediction Section ---
col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🔍 Predict Risk"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        risk = "🛑 High Risk" if prediction == 1 else "✅ Low Risk"
        color = "#FF4B4B" if prediction == 1 else "#28A745"

        st.markdown(f"""
            <div style='background-color:{color};padding:15px;border-radius:10px'>
                <h2 style='color:white;text-align:center'>{risk}</h2>
            </div>
        """, unsafe_allow_html=True)

        st.metric(label="Model Confidence", value=f"{round(probability * 100, 2)}%")

        if prediction == 1:
            st.markdown("""### 💡 Suggested Tips
- Consult a cardiologist
- Reduce fried/high-sugar foods
- Exercise regularly under medical guidance
- Monitor blood pressure and cholesterol
- Avoid alcohol & smoking
""")

with col2:
    st.image("https://img.freepik.com/free-vector/doctor-with-stethoscope-heart_1308-65171.jpg", width=500)



# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; font-size: 14px;'>
    Developed as part of MCA Final Year Project by <b>Siddhika Belsare</b><br>
    Supervised by <b>Prof. Shubhangi Mahadik</b><br>
    Dataset: <a href='https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset' target='_blank'>Kaggle CVD Dataset</a>
</p>
""", unsafe_allow_html=True)
