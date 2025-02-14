import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import streamlit as st
import pickle
import tensorflow as tf 


# Load the encoders and scalar
with open('/Users/pkadala/Documents/mystuff/AI/GenAI/myexercises/ann/onehotencoder_geo.pkl','rb') as file:
    onehotencoder_geo = pickle.load(file)

with open('/Users/pkadala/Documents/mystuff/AI/GenAI/myexercises/ann/labelencoder_gender.pkl', 'rb') as file:
    labelencoder_gender = pickle.load(file)

with open('/Users/pkadala/Documents/mystuff/AI/GenAI/myexercises/ann/scalar.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load trained model
model = tf.keras.models.load_model('/Users/pkadala/Documents/mystuff/AI/GenAI/myexercises/ann/ann_coassification_model.h5')
print(model.summary())


# Build the UI
st.title("Customer Churn Prediction")
geography = st.selectbox('Geography', onehotencoder_geo.categories_[0])
gender = st.selectbox('Gender', labelencoder_gender.classes_)
age = st.slider('Age', 18,100)
balance = st.number_input('Balance')
creditscore = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# Prepare input data
input_df = pd.DataFrame({
    'CreditScore' : [creditscore],
    'Gender' : [labelencoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_credit_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
}
)

print(input)

print(onehotencoder_geo.get_feature_names_out(['Geography']))

geo_encoded = onehotencoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotencoder_geo.get_feature_names_out(['Geography']))

input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

print(input_df)

## Scale the inpput data
input_df_scaled = scaler.transform(input_df)

## Predict the model
prediction = model.predict(input_df_scaled)
prediction_proba = prediction[0][0]

print(prediction_proba)

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likey to churn')
else:
    st.write('The customer is not likey to churn')    

