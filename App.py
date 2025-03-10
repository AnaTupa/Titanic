import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.set_page_config(layout="wide")

st.title("Titanic Survivor Prediction app")

# reading all the pickle files
titanic_model_lr = pickle.load(open('Titanic_model_lr.pkl','rb')) # Logistic regression model
titanic_model_rf = pickle.load(open('Titanic_model_rf.pkl','rb')) # Logistic regression model
scaler = pickle.load(open('scaler.pkl', 'rb')) # scaler

# user need to define the input
st.header("Enter the input values to predict:")

PClass = st.number_input("Enter Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", min_value=1, max_value=3, step=1)
Sex = st.selectbox("Enter Gender (male, female)", ['male', 'female'])
Age = st.number_input("Enter Age (0 to 80)", min_value=0, max_value=80)
SibSp = st.number_input("Number of Siblings/Spouses aboard (0 to 8)", min_value=0, max_value=8)
Parch = st.number_input("Number of Parents/Children aboard (0 to 6)", min_value=0, max_value=6)
Fare = st.number_input("Enter Fare Paid (0 to 513)", min_value=0.0, max_value=513.0)
Embarked = st.selectbox("Enter Embarked Port (S, C, Q)", ['S', 'C', 'Q'])

# Convert categorical features to numerical
if Sex == 'male':
    Sex_numeric = 0
else:
    Sex_numeric = 1

Embarked_numeric = {'S': 0, 'C': 1, 'Q': 2}[Embarked]

# Create a DataFrame for the input data
input_data = pd.DataFrame([[PClass, Sex_numeric, Age, SibSp, Parch, Fare, Embarked_numeric]],
                          columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# Apply the same scaling as during training (scaling Age and Fare)
input_data_scaled = scaler.transform(input_data[['Age', 'Fare']])

# Update the DataFrame with the scaled values
input_data[['Age', 'Fare']] = input_data_scaled

# Make the prediction using the trained model

# User selects which model to use
selected_model = st.selectbox("Select one of the following models", ("Basic Model", "Random Forest Model"))

# Predict button for model selection
if st.button("Predict", key="predict_button"):  # Unique key added here
    if selected_model == "Basic Model":
        prediction = titanic_model_lr.predict(input_data)
        st.write("Prediction from Basic Model (Logistic Regression)")

    elif selected_model == "Random Forest Model":
        prediction = titanic_model_rf.predict(input_data)
        st.write("Prediction from Random Forest Model")

    # Display prediction result
    result = prediction[0]
    
    if result == 1:
        st.success("The passenger survived! :D")
    else:
        st.success("The passenger did not survive :(")