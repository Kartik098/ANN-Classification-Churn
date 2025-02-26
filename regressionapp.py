import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle


model = tf.keras.models.load_model('regression_model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Expected salary prediction')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenured = st.slider('Tenure', 0, 10)
num_of_products = st.number_input('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card',[0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.selectbox('Exited', [0, 1])

gender_encoded = label_encoder_gender.transform([gender])[0]
input_data = pd.DataFrame({
    'Geography': [geography],
    'Gender': [gender_encoded],
    'Age': [age],
    'Balance': [balance],
    'CreditScore': [credit_score],
    'Tenure': [tenured],
    'Exited': [exited],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

correct_feature_order = scaler.feature_names_in_
print("Correct feature order:", correct_feature_order)

geo_encoded = onehot_encoder_geo.transform(pd.DataFrame([[geography]], columns=['Geography'])).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out())

print("geo_encoded_df", geo_encoded_df)
input_data = pd.concat([input_data.drop("Geography", axis=1), geo_encoded_df], axis=1)

input_data = input_data[correct_feature_order]

# Scale the input data
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)

predicted_salary = prediction[0][0]
st.write(F"Predicted Estimated Salary: {predicted_salary:.2f}")



