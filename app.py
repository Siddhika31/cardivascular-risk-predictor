import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("cardio_model.pkl")

# Page config
st.set_page_config(page_title="Cardiovascular Disease Risk Predictor", layout="wide")

# Header
st.title("🫀 Cardiovascular Disease Risk Prediction App")
st.markdown("Use this AI-powered tool to estimate your risk for cardiovascular disease.")

# Layout with two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Enter Your Health Details")

    age = st.slider("Age", 18, 90, 30)
    bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 22.0)
    sex = st.radio("Sex", ("Male", "Female"))
    exercise = st.radio("Do you exercise regularly?", ("Yes", "No"))
    smoking = st.radio("Do you smoke?", ("Yes", "No"))
    alcohol = st.slider("Alcohol consumption (drinks/week)", 0.0, 20.0, 2.0)
    fruit = st.slider("Fruit consumption (servings/day)", 0.0, 10.0, 2.0)
    greens = st.slider("Green vegetables consumption (servings/day)", 0.0, 10.0, 2.0)
    fried = st.slider("Fried potato consumption (servings/week)", 0.0, 10.0, 1.0)
    checkup = st.radio("Regular health checkups?", ("Yes", "No"))

    if st.button("Predict Risk"):
        # Preprocess inputs
        sex = 1 if sex == "Male" else 0
        exercise = 1 if exercise == "Yes" else 0
        smoking = 1 if smoking == "Yes" else 0
        checkup = 1 if checkup == "Yes" else 0

        input_data = np.array([[
            age, bmi, sex, exercise, smoking,
            alcohol, fruit, greens, fried, checkup
        ]])

        # Predict
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("🔴 High Risk of Cardiovascular Disease!")
            st.markdown("### 🔍 Suggestions to Lower Risk:")
            st.markdown("""
            - 🩺 **Consult a cardiologist** for advanced screening  
            - 🥗 Improve your diet: more fiber, fruits, and greens  
            - 🚭 Stop smoking if applicable  
            - 🏃‍♂️ Exercise at least 30 mins/day  
            - 🧘 Reduce stress through meditation or yoga  
            - 🔬 Monitor blood pressure and cholesterol regularly  
            """)
        else:
            st.success("🟢 Low Risk of Cardiovascular Disease.")
            st.markdown("### ✅ Keep Up the Good Habits:")
            st.markdown("""
            - 🥗 Maintain a balanced diet  
            - 🏃‍♂️ Continue regular physical activity  
            - 🧘 Practice stress management (yoga/meditation)  
            - 🛌 Ensure adequate sleep and hydration  
            - 🩺 Regular health checkups help stay ahead  
            """)

with col2:
    st.image(
        "https://media.tenor.com/MEP9bkszV9gAAAAC/heart-health.gif",
        width=400,
        caption="Cardiovascular Monitoring",
    )
# --- Footer ---
st.markdown("---")
st.markdown("""
<p style='text-align: center; font-size: 14px;'>
Developed as part of MCA Final Year Project by <b>Siddhika Belsare</b><br>
Supervised by <b>Prof. Shubhangi Mahadik</b><br>
Dataset: <a href='https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset' target='_blank'>Kaggle CVD Dataset</a>
</p>
""", unsafe_allow_html=True)
