# Basic Libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt # we only need pyplot
sb.set() # set the default Seaborn style for graphics

cardioData = pd.read_csv('cardio_train_fixed.csv')
cardioData.head()

print("Data type : ", type(cardioData))
print("Data dims : ", cardioData.shape)

cardioData['height_metres'] = cardioData['height'] / 100 #Converts cm to m
cardioData['BMI'] = cardioData['weight'] / (cardioData['height_metres'] ** 2) #Calculates BMI using their height and weight

cardioData['age_years'] = np.round(cardioData['age'] / 365) #Age in days rounded to the nearest whole number

print(cardioData[['height_metres', 'weight', 'BMI', 'age_years']].head())

cardioNumData = pd.DataFrame(cardioData[['age_years', 'gender', 'height', 'weight', 'cholesterol', 'cardio', 'BMI', 'ap_hi', 'ap_lo']])
print(cardioNumData)

# Splitting data into train and test
from sklearn.model_selection import train_test_split

# Selecting the features and target variable
X = cardioNumData.drop(columns=['cardio', 'gender', 'height']) # Dropping columns that have low relevance
y = cardioNumData['cardio'] # Target variable

#Split 80% into training and 20% into testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Training by using a Random Forest Classifier - Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score

# Initializing and training the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predicitjg on test set
y_pred_rf = rf.predict(X_test)

print('RF Accuracy:', accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

import streamlit as st

# Function to make predictions
def predict_cvd(cardioNumData):
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([cardioNumData], columns=["age_years", "weight", "cholesterol", "BMI" "ap_hi", "ap_lo"])
    # Make prediction
    prediction = rf.predict(input_data)[0]
    return "Has Cardiovascular Disease" if prediction == 1 else "No Cardiovascular Disease"

# Streamlit UI
st.title("Cardiovascular Disease Prediction")
st.write("Enter patient details to predict the risk of cardiovascular disease.")

# Input fields
age_years = st.slider("Age (in years)", 10, 100, 30)
weight = st.number_input("Weight (kg)", 30, 150, 70)
cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])  # 1: Normal, 2: Above Normal, 3: Well Above Normal
BMI = st.number_input("BMI", 10, 50, 20)
ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", 90, 200, 120)
ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", 50, 130, 80)

#gluc = st.selectbox("Glucose Level", [1, 2, 3])  # 1: Normal, 2: Above Normal, 3: Well Above Normal
#smoke = st.radio("Do you smoke?", [0, 1])
#alco = st.radio("Do you consume alcohol?", [0, 1])
#active = st.radio("Are you physically active?", [0, 1])

# Predict button
if st.button("Predict"):
    result = predict_cvd(cardioNumData)
    st.subheader(f"Prediction: {result}")
