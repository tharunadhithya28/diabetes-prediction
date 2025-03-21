# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 12:55:25 2025

@author: 91944
"""

import numpy as np 
import pandas as pd 
import joblib
import streamlit as st 
from sklearn.preprocessing import StandardScaler

#loading the saved model 
final_model = joblib.load("D:/Machine Learning/Deploying Model/final_model.pkl")

scaler = joblib.load("D:/Machine Learning/Deploying Model/scaler_data.pkl")

def diabetes_prediction(input_data):
    
    input_data_as_array = np.asarray(input_data,dtype=float).reshape(1,-1)
    print(input_data_as_array)

    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    input_df = pd.DataFrame(input_data_as_array, columns=feature_names)
    
    std_data = scaler.transform(input_df)

    prediction = final_model.predict(std_data)
    print(prediction)
    print(prediction)

    if(prediction[0] == 0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic"
    
def main():
    
    #giving a title 
    st.title('Diabetes')
    
    #getting input from user 
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness Value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPredigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age of the person")
    
    #code for Prediction 
    diagnosis = ''
    
    #creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose,BloodPressure,SkinThickness, Insulin,BMI,DiabetesPredigreeFunction, Age])
        
    st.success(diagnosis)
    



if __name__ == '__main__':
    main()    