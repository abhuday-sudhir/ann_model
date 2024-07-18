import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

# Load model
try:
    model = tf.keras.models.load_model('ann_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    print(f"Error loading model: {e}")

# Load encoders and scaler
try:
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)

    with open('one_hot_encoder.pkl', 'rb') as file:
        ohe = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading encoders or scaler: {e}")
    print(f"Error loading encoders or scaler: {e}")

# Streamlit title
st.title("Churn Predictor")

# User input
geo = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 80)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

input_df = pd.DataFrame(input_data)

geo_encoded = ohe.transform([[geo]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

input_data = pd.concat([input_df, geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)

if prediction[0][0] > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")

st.write("Prediction probability",prediction[0][0] )