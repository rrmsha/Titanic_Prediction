import pickle
import Orange
from Orange.data import Domain, DiscreteVariable
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd

#LOAD TRAINED MODEL
with open("titanic_tree.pkcls", "rb") as model:
    loaded_model = pickle.load(model)

#ADD COMPONENTS AND SIDEBAR
side_bar = st.sidebar
header = st.container()
body = st.container()
footer = st.container()

#HANDLE DATA AND GET INPUT
@st.cache_data
def get_data():
    data = pd.read_csv("titanic.tab",sep="\t")
    data = data.drop([0, 1])
    return data

def get_user_input():
    with side_bar:
        user_status = st.radio("status",("crew", "first", "second", "third"))
        user_age = st.radio("age", ("adult", "child"))
        user_sex = st.radio("sex", ("male", "female"))
    
    domain = Domain([DiscreteVariable("status", values=["crew", "first", "second", "third"]), DiscreteVariable("age", values=["adult", "child"]),\
                      DiscreteVariable("sex", values=["male", "female"])])
    status = ["crew", "first", "second", "third"]
    status_inp = status.index(user_status)
    age = ["adult", "child"]
    age_inp = age.index(user_age)
    sex = ["male", "female"]
    sex_inp = sex.index(user_sex)

    X = np.column_stack((status_inp, age_inp, sex_inp))
    data_table = Orange.data.Table(domain, X)
    return [user_status, user_age, user_sex], data_table

#INPUT
with side_bar:
    st.markdown("## Choose who is boarding the Titanic")
#DATASET INFO
with header:
    st.title("Titanic Survival Prediction")
    st.markdown("This is an app which predicts whether a passenger on the Titanic survived the famous shipwreck based on different features.")
    st.markdown("Features defined in our dataset are:- ")
    st.markdown("* Status")
    st.markdown("* Age")
    st.markdown("* Sex")

    with st.expander("Show Dataset"):
        st.dataframe(get_data())

    st.divider()
#USER INPUT 
with body:
    st.markdown("### Will your passenger survive the Titanic shipwreck?")
    user_input, data_table = get_user_input()
    input_col, result_col = st.columns(2)
    with input_col:
        st.markdown("##### Your Input: ")
        st.markdown(f"Status: `{user_input[0]}`")
        st.markdown(f"Age: `{user_input[1]}`")
        st.markdown(f"Sex: `{user_input[2]}`")
    with result_col:
        st.markdown("##### Prediction: ")
        prediction = loaded_model(data_table)
        if prediction:
            st.markdown("Your passenger **_will_** survive")
        else:
            st.markdown("Your passenger **_will not_** survive")
    st.divider()
#MODEL ACCURACY AND OTHER MEASURES 
with footer:
    st.markdown("### Model Accuracy: ")
    "The model is trained on Decision Tree and has had preprocessing done, on Orange Platform"
    st.markdown("> AUC score - `0.773`")
    st.markdown("> Classification Accuracy - `0.774`")