import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load Dataset
df = pd.read_csv("diabetes.csv")
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split and Train Models
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Models
dt_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
knn_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
svm_model = SVC(kernel='linear', probability=True).fit(X_train, y_train)

# Streamlit App
st.title("😷 Diabetes Prediction System")
st.write("Enter the patient's medical details below:")

# User Input
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=33)

input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
scaled_input = scaler.transform(input_data)

# Model Selection
model_choice = st.selectbox("Choose Model", ["Decision Tree", "K-Nearest Neighbors", "Support Vector Machine"])

# Prediction
if st.button("Predict"):
    if model_choice == "Decision Tree":
        prediction = dt_model.predict(scaled_input)[0]
    elif model_choice == "K-Nearest Neighbors":
        prediction = knn_model.predict(scaled_input)[0]
    else:
        prediction = svm_model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High chance of Diabetes.")
    else:
        st.success("✅ No Diabetes detected.")

st.markdown("Made with ❤️ by Dhaval Kanpariya")
