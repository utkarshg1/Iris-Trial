import pandas as pd
import pickle
import streamlit as st
import numpy as np

# Title of the project
st.title("Iris Project - Utkarsh Gaikwad")

# Take input from user
sep_len = st.number_input("Sepal Length : ", min_value=0.00, step=0.01)
sep_wid = st.number_input("Sepal Width : ", min_value=0.00, step=0.01)
pet_len = st.number_input("Petal Length : ", min_value=0.00, step=0.01)
pet_wid = st.number_input("Petal Width : ", min_value=0.00, step=0.01)

# Submit button
submit = st.button("Predict Species")

# Predction subheader
st.subheader('Predictions :')

# function to predict species
def predict_species(model_path, pipeline_path):
    xnew = pd.DataFrame([sep_len, sep_wid, pet_len, pet_wid]).T 
    xnew.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    with open(model_path, 'rb') as file1:
        model = pickle.load(file1)
    with open(pipeline_path, 'rb') as file2:
        pipe = pickle.load(file2)
    xnew_pre = pipe.transform(xnew)
    pred = model.predict(xnew_pre)
    prob = model.predict_proba(xnew_pre)
    max_prob = np.max(prob)
    return pred, max_prob

if submit:
    model_path = 'notebook/model.pkl'
    pipeline_path = 'notebook/pipe.pkl'
    pred, max_prob = predict_species(model_path, pipeline_path)
    st.subheader(f"Predicted Species is : {pred[0]}")
    st.subheader(f"Probability of prediction : {max_prob:.4f}")
    st.progress(max_prob)
