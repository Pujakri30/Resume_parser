import streamlit as st
import joblib

# Load saved model and tools
model = joblib.load("resume_parser_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Web App Title
st.title("🧠 Resume Parser: Experience Level Predictor")
st.write("Enter your resume details below:")

# Input fields
skills = st.text_input("Skills (comma-separated)", "Python, SQL, Excel")
designation = st.text_input("Designation", "Data Analyst")
education = st.text_input("Education", "B.Tech")

if st.button("Predict Experience Level"):
    input_text = skills + " " + designation + " " + education
    vector = vectorizer.transform([input_text])
    prediction = model.predict(vector)
    result = label_encoder.inverse_transform(prediction)[0]
    
    st.success(f"✅ Predicted Experience Level: **{result}**")
