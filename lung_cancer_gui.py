import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.markdown(
    """
    <style>
    .centered-header {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    </style>
    <div class="centered-header">
        <h1>Lung Cancer Prediction</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader('Fill in the following details.')

col1, col2 = st.columns(2)
user_name = col1.text_input('Your Name', help="Please enter your full name.")
age = col2.number_input('Age', min_value=18, max_value=100, help="Please enter your age. Age must be between 18 and 100.")  # Part of model input for prediction

# Radio Options
gender = st.selectbox('Gender', ['Male', 'Female'], help="Please select your gender.")
smoking = st.selectbox('Do you smoke?', ['Yes', 'No'], help="Please indicate if you smoke.")
yellow_fingers = st.selectbox('Do you have Yellow Fingers?', ['Yes', 'No'], help="Please indicate if you have yellow fingers.")
anxiety = st.selectbox('Do you suffer from Anxiety?', ['Yes', 'No'], help="Please indicate if you suffer from anxiety.")
peer_pressure = st.selectbox('Do you have Peer Pressure?', ['Yes', 'No'], help="Please indicate if you experience peer pressure.")
chronic_disease = st.selectbox('Do you suffer from a Chronic Disease?', ['Yes', 'No'], help="Please indicate if you suffer from a chronic disease.")
fatigue = st.selectbox('Do you quite often feel Fatigued?', ['Yes', 'No'], help="Please indicate if you often feel fatigued.")
allergy = st.selectbox('Are you suffering from any Allergies?', ['Yes', 'No'], help="Please indicate if you suffer from any allergies.")
wheezing = st.selectbox('Do you have Wheezing problems?', ['Yes', 'No'], help="Please indicate if you have wheezing problems.")
alcohol = st.selectbox('Are you an Alcohol consumer?', ['Yes', 'No'], help="Please indicate if you consume alcohol.")
coughing = st.selectbox('Do you Cough frequently?', ['Yes', 'No'], help="Please indicate if you frequently cough.")
shortness_of_breath = st.selectbox('Do you suffer from Shortness of breath?', ['Yes', 'No'], help="Please indicate if you suffer from shortness of breath.")
swallow_difficulty = st.selectbox('Do you have Difficulty Swallowing food?', ['Yes', 'No'], help="Please indicate if you have difficulty swallowing food.")
chest_pain = st.selectbox('Do you suffer from Chest pain?', ['Yes', 'No'], help="Please indicate if you suffer from chest pain.")

# Creating a list for model
input_variables = [gender, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol, coughing, shortness_of_breath, swallow_difficulty, chest_pain]

model_input_dict = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0}
model_list = [model_input_dict[y] for y in input_variables]
model_list.insert(1, age / 100)

# Importing model and preparing dataframe for it
model = joblib.load('final_model.joblib')
columns = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY','PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING','ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH','SWALLOWING DIFFICULTY', 'CHEST PAIN']
df = pd.DataFrame(np.array([model_list]), columns=columns)

# Initialize session state for prediction result
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

def predict():
    prediction = model.predict(df)[0]
    st.session_state.prediction = prediction

st.button(label='Predict', on_click=predict, help="Click to predict your lung cancer risk based on the provided information.")

# Display the prediction result
if st.session_state.prediction is not None:
    if st.session_state.prediction == 1:
        st.header(f'{user_name}, you have lung cancer')
        st.subheader('Suggested Basic Treatments:')
        st.markdown("""
        - **Consult with an oncologist** to discuss your diagnosis and treatment options.
        - **Follow a healthy diet** and maintain a healthy weight.
        - **Avoid smoking** and limit exposure to secondhand smoke.
        - **Consider therapies** such as chemotherapy, radiation, or surgery as recommended by your doctor.
        - **Stay physically active** and engage in moderate exercise if possible.
        - **Seek support** from family, friends, or cancer support groups.
        """)
    else:
        st.header(f'{user_name}, you DO NOT have lung cancer')

# Footer
st.markdown(
    """
    <style>
    .footer {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 50px;
        margin-top: 50px;
        font-size: 14px;
        color: #777;
    }
    </style>
    <div class="footer">
        <p>&copy; 2024 | Developed by Juqiey</p>
    </div>
    """,
    unsafe_allow_html=True
)
