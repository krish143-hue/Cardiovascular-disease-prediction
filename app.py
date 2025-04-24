import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Load necessary files
model = pickle.load(open('best_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# Define input fields
def user_input():
    st.subheader("Enter Patient Information")
    age = st.slider('Age', 20, 100, 50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
    trestbps = st.slider('Resting Blood Pressure', 80, 200, 120)
    chol = st.slider('Serum Cholesterol (mg/dl)', 100, 400, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
    restecg = st.selectbox('Resting ECG Results', ['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'])
    thalach = st.slider('Max Heart Rate Achieved', 70, 210, 150)
    exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    oldpeak = st.slider('ST depression', 0.0, 6.0, 1.0)
    slope = st.selectbox('Slope of ST segment', ['upsloping', 'flat', 'downsloping'])
    ca = st.slider('No. of major vessels (0-3)', 0, 3, 0)
    thal = st.selectbox('Thalassemia', ['normal', 'fixed defect', 'reversable defect'])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return data

# Preprocess input for prediction
def preprocess_input(input_data):
    input_df = {key: [value] for key, value in input_data.items()}
    for col, le in label_encoders.items():
        input_df[col][0] = le.transform([input_df[col][0]])[0]
    input_array = np.array(list(input_df.values())).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    return scaled_input

# SHAP Explanation
def show_shap_explanation():
    st.subheader("🔍 SHAP Explanation")
    explainer = shap.Explainer(model)
    input_sample = np.zeros((1, len(label_encoders)))
    shap_values = explainer(input_sample)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.summary_plot(shap_values, feature_names=list(label_encoders.keys()), show=False)
    st.pyplot(bbox_inches='tight')

# Chatbot (placeholder)
def chatbot_interface():
    st.subheader("💬 Heart Health Chatbot")
    user_q = st.text_input("Ask a question about heart health")
    if user_q:
        st.write("🤖 Chatbot:", "This is a placeholder answer. More intelligent responses coming soon!")

# Navigation
st.sidebar.title("🔍 Navigation")
selection = st.sidebar.radio("Go to", ["About the App", "Predict", "SHAP Explanation", "Chatbot Q&A"])

# About Page
if selection == "About the App":
    st.title("❤️ Cardiovascular Disease Prediction App")
    st.markdown("""
    Welcome to **HeartGuard AI** – an intelligent system for heart disease prediction.

    - 🔬 Uses 9 ML models
    - 🐜 Feature selection via Ant Colony Optimization
    - ⚙️ Hyperparameter tuned models
    - 📊 SHAP for explainability
    - 🤖 Chatbot support
    """)

# Prediction Page
elif selection == "Predict":
    st.title("🩺 Make a Prediction")
    input_data = user_input()
    if st.button("Predict"):
        processed = preprocess_input(input_data)
        result = model.predict(processed)
        st.success("Prediction: " + ("💓 Disease Detected" if result[0] == 1 else "✅ No Disease"))

# SHAP Explanation
elif selection == "SHAP Explanation":
    show_shap_explanation()

# Chatbot Page
elif selection == "Chatbot Q&A":
    chatbot_interface()
