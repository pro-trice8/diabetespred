# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 13:59:38 2025

@author: KIIT0001
"""

import numpy as np
import pickle
import streamlit as slt

loaded_model = pickle.load(open('C:/Users/KIIT0001/Documents/ML models/Diabetes/trained_model.sav', 'rb'))

def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
def main():
    
    slt.title("Diabetes Prediction")
    
    #creating input data
    Pregnancies = slt.text_input("Number of Pregnancies: ")
    Glucose =  slt.text_input("Glucose levels: ")
    BloodPressure = slt.text_input("Blood Pressure Levels: ")
    SkinThickness = slt.text_input("Thickness of the skin: ")
    Insulin =  slt.text_input("Insulin level in the body: ")
    BMI =  slt.text_input("BMI: ")
    DiabetesPedigreeFunction =  slt.text_input("Diabetes Pedigree Function value: ")
    Age = slt.text_input("Age: ")
    
    
    diagonis = ''
    #create button
    
    if slt.button("Diabetes Test Result"):
        diagonis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    slt.success(diagonis)
        
if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
        
        
        
        