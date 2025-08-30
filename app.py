import streamlit as st
import pandas as pd
import joblib

model = joblib.load('xgb_model.pkl')

st.title("Pelvic Fetal Predictive Tool")
st.write("Enter maternal and fetal parameters to predict delivery outcome.")

pelvic_inlet = st.slider("Pelvic Inlet Diameter (cm)", 10.0, 14.0, 12.0, step=0.01)
pelvic_outlet = st.slider("Pelvic Outlet Diameter (cm)", 8.0, 12.0, 10.0, step=0.01)
fetal_head = st.slider("Fetal Head Circumference (cm)", 30.0, 36.0, 33.0, step=0.01)
fetal_weight = st.slider("Fetal Weight (g)", 2500, 4500, 3500, step=50)
maternal_age = st.slider("Maternal Age (years)", 18, 45, 30)
parity = st.slider("Parity (previous births)", 0, 5, 1)

input_data = pd.DataFrame({
    'pelvic_inlet_cm': [pelvic_inlet],      
    'pelvic_outlet_cm': [pelvic_outlet],
    'fetal_head_cm': [fetal_head],
    'fetal_weight_g': [fetal_weight],
    'maternal_age': [maternal_age],
    'parity': [parity]
})


if st.button("Predict"):
    prob = model.predict_proba(input_data)[0, 1]  # Probability of vaginal delivery
    st.write(f"**Probability of Vaginal Delivery: {prob:.2%}**")
    if prob > 0.5:
        st.success("Likely Vaginal Delivery")
    else:
        st.warning("Likely Cesarean Section")