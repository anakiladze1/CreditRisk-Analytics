#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import joblib
import numpy as np

def feature_engineering(df):
    df = df.copy()
    df['interest_income'] = df['person_income'] * (df['loan_int_rate'] / 100)
    return df


@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")  
    return model

pipeline = load_artifacts()

st.title("💳 სესხის დეფოლტის პროგნოზირების სისტემა")
st.write("შეიყვანეთ კლიენტის მონაცემები, რათა გაიგოთ სესხის გაცემის რისკი.")


st.sidebar.header("კლიენტის პარამეტრები")

person_age = st.sidebar.number_input("ასაკი (person_age)", min_value=18, max_value=100, value=30)
person_income = st.sidebar.number_input("წლიური შემოსავალი (person_income)", min_value=0, value=50000)
person_emp_length = st.sidebar.number_input("სამუშაო სტაჟი წლებში (person_emp_length)", min_value=0.0, value=5.0)
loan_amnt = st.sidebar.number_input("სესხის ოდენობა (loan_amnt)", min_value=0, value=10000)
loan_int_rate = st.sidebar.number_input("სესხის საპროცენტო განაკვეთი (loan_int_rate)", min_value=0.0, value=10.0)
loan_percent_income = st.sidebar.number_input("სესხის წილი შემოსავალთან (loan_percent_income)", min_value=0.0, max_value=1.0, value=0.2)

person_home_ownership = st.sidebar.selectbox("საცხოვრებლის სტატუსი (home_ownership)", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_intent = st.sidebar.selectbox("სესხის აღების მიზანი (loan_intent)", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.sidebar.selectbox("სესხის კლასი (loan_grade)", ["A", "B", "C", "D", "E", "F", "G"])
cb_person_default_on_file = st.sidebar.selectbox("ისტორიაში დეფოლტი (cb_person_default_on_file)", ["Y", "N"])
cb_person_cred_hist_length = st.sidebar.number_input("საკრედიტო ისტორიის ხანგრძლივობა", min_value=0, value=3)


input_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_emp_length': [person_emp_length],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'person_home_ownership': [person_home_ownership],
    'loan_intent': [loan_intent],
    'loan_grade': [loan_grade],
    'cb_person_default_on_file': [cb_person_default_on_file],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length]
})

if st.button("პროგნოზის გენერაცია"):
    try:
        input_data_fe = feature_engineering(input_data)

        prediction_proba = pipeline.predict_proba(input_data_fe)
        default_probability = prediction_proba[0][1]


        custom_threshold = 0.3  


        is_default = 1 if default_probability >= custom_threshold else 0

        st.subheader("შედეგი:")
        if is_default == 1:
            st.error(f"⚠️ მაღალი რისკი: კლიენტი სავარაუდოდ ვერ გადაიხდის სესხს (Default).")
        else:
            st.success(f"✅ დაბალი რისკი: კლიენტი სანდოა.")

        st.write(f"**დეფოლტის ალბათობა:** {default_probability*100:.2f}%")

    except Exception as e:
        st.error(f"შეცდომა პროგნოზირებისას: {e}")

