#Basic Libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
sb.set()

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

#Splitting data into train and test
from sklearn.model_selection import train_test_split

#Selecting the features and target variable
X = cardioNumData.drop(columns=['cardio', 'gender', 'height']) # Dropping columns that have low relevance
y = cardioNumData['cardio'] # Target variable

#Split 80% into training and 20% into testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Training by using a Random Forest Classifier - Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score

#Initializing and training the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

#Predicitjg on test set
y_pred_rf = rf.predict(X_test)

print('RF Accuracy:', accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

import streamlit as st

#Function to make predictions
def predict_cvd(cardioNumData):
    """Handle both dictionary and list inputs"""
    if isinstance(cardioNumData, dict):
        #Dictionary input
        df = pd.DataFrame([cardioNumData])
    else:
        #List/array input
        df = pd.DataFrame([cardioNumData], 
                         columns=["age_years", "weight", "cholesterol", "BMI", "ap_hi", "ap_lo"])
    return rf.predict(df)[0]

#Streamlit UI
st.title("Cardiovascular Risk Prediction")
st.write("Done by Chia Weng Choong, Julian Fun, Geraldine Lin, Jia Xuan Lim")
st.write("Enter patient details to predict the risk of cardiovascular disease.")

#Input fields
age_years = st.slider("Age", 10, 100, 50)
weight = st.number_input("Weight (kg)", 30, 150, 70)
cholesterol = st.selectbox("Cholesterol", [1, 2, 3])
BMI = st.number_input("BMI", 10, 50, 25)
ap_hi = st.number_input("Systolic BP", 90, 200, 120)
ap_lo = st.number_input("Diastolic BP", 50, 130, 80)

if st.button("Predict"):
    cardioNumData = {
        'age_years': age_years,
        'weight': weight,
        'cholesterol': cholesterol,
        'BMI': BMI,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo
    }
    
    try:
        result = predict_cvd(cardioNumData)
        if result == 1:  #High Risk
            st.error("High Risk of Cardiovascular Disease")  #Red box
        else:  #Low Risk
            st.success("Low Risk of Cardiovascular Disease")  #Green box
