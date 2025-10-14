import streamlit as st
import numpy as np
import joblib

def header():
    st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="color: #4B8BBE;">Diabetes Prediction App</h1>
            <p>Enter your health details below to predict diabetes risk</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")



def load_model():
    selected_model_name = st.selectbox("Select Model",["LogisticRegression", "RandomForestClassifier", "XGBClassifier"])
    model_path = f'models/{selected_model_name}.pkl'
    model = joblib.load(open(model_path, 'rb'))
    return model

model = load_model()

def my_form():
    input_list = [
    ('Pregnancies', 0.0, 20.0),
    ('Glucose', 0.0, 200.0),
    ('Blood Pressure', 0.0, 150.0),
    ('Skin Thickness', 0.0, 100.0),
    ('Insulin', 0.0, 900.0),
    ('BMI', 0.0, 70.0),
    ('Diabetes Pedigree Function', 0.0, 2.5),
    ('Age', 0.0, 120.0)
]

    with st.form(key="diabetes_prediction"):
        cols = st.columns(2)
        input_data = []
        for idx, (label, min_val, max_val) in enumerate(input_list):
            with cols[idx % 2]:
                value = st.number_input(label=label, min_value=min_val, max_value=max_val, step=1.0)
                input_data.append(value)
        
        submit_btn = st.form_submit_button(label="Predict Diabetes")
    return submit_btn,input_data

submit_btn, input_data = my_form()

if submit_btn:
    
    input_array = np.array([input_data])
    
    prediction = model.predict(input_array)
    
    prob = model.predict_proba(input_array)[0][1]  
    st.text(float(prob))
    result = "Positive" if prediction[0] >= 0.5 else "Negative"
    # result = "Positive" if prediction[0] == 1 else "Negative"
    st.markdown("---")
    st.success(f"Diabetes Prediction: **{result}**")
