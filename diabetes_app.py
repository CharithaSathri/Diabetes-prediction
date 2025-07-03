import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Prepare input and output
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Show app title
st.markdown("<h1 style='text-align: center; color: orange;'>Diabetes Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Predict the probability of having Diabetes</h3>", unsafe_allow_html=True)

# Input form
col1, col2, col3 = st.columns(3)

with col1:
    preg = st.number_input("Pregnancies", min_value=0, max_value=20, help="No. of Pregnancies")
    skin = st.number_input("SkinThickness", min_value=0, max_value=100)
    dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, format="%.3f")

with col2:
    glucose = st.number_input("Glucose", min_value=0, max_value=200, help="Glucose level in sugar")
    insulin = st.number_input("Insulin", min_value=0, max_value=900)
    age = st.number_input("Age", min_value=1, max_value=120)

with col3:
    bp = st.number_input("BloodPressure", min_value=0, max_value=140)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0)
    st.write("")
    st.write("")

# Prediction
if st.button("🧠 PREDICT PROBABILITY"):
    user_input = [[preg, glucose, bp, skin, insulin, bmi, dpf, age]]
    prediction = model.predict(user_input)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.success(f"Prediction: **{result}**")
