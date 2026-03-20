import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import tensorflow as tf
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#load the trained model
model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, 'model.keras'),
    compile=False
)
#load the lable encoder , scaler and onehot encoder

with open(os.path.join(BASE_DIR, 'lable_encoder_gender.pkl'), 'rb') as file:
    lable_encoder_gender = pickle.load(file)
with open(os.path.join(BASE_DIR, 'onehot_encoder_geo.pkl'), 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as file:
    scaler = pickle.load(file)


#streamlit app
st.title("Customer Churn Prediction")

#user input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", lable_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],          
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})
#one hot encode for geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#lable encoder to gender
input_data['Gender'] = lable_encoder_gender.transform(input_data['Gender']) 

#concatinate encoded geo data with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#scaleing the data
input_data_scaled = scaler.transform(input_data)

#st.write("Files in directory:", os.listdir(BASE_DIR))
#prediction churn
prdiction = model.predict(input_data_scaled)
prediction_proba = prdiction[0][0]

st.write(f"churn probability : {prediction_proba: .2f}")

if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")

