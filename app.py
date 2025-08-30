import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import numpy as np

model = joblib.load('xgb_model.pkl')
explainer = shap.TreeExplainer(model)

st.title("Pelvic Fetal Predictive Tool")
st.write("Enter maternal and fetal metrics to predict delivery outcome, see explanations, and visualize fetal passage.")

pelvic_inlet = st.slider("Pelvic Inlet Diameter (cm)", 10.0, 14.0, 12.0, step=0.1)
pelvic_outlet = st.slider("Pelvic Outlet Diameter (cm)", 8.0, 12.0, 10.0, step=0.1)
fetal_head = st.slider("Fetal Head Circumference (cm)", 30.0, 36.0, 33.0, step=0.1)
fetal_weight = st.slider("Fetal Weight (g)", 2500, 4500, 3500, step=50)
maternal_age = st.slider("Maternal Age (years)", 18, 45, 30)
parity = st.slider("Parity (previous births)", 0, 4, 1)

input_data = pd.DataFrame({
    'pelvic_inlet_cm': [pelvic_inlet],
    'pelvic_outlet_cm': [pelvic_outlet],
    'fetal_head_cm': [fetal_head],
    'fetal_weight_g': [fetal_weight],
    'maternal_age': [maternal_age],
    'parity': [parity]
})

if st.button("Predict"):
    prob = model.predict_proba(input_data)[0, 1]
    st.write(f"**Probability of Vaginal Delivery: {prob:.2%}**")
    if prob > 0.5:
        st.success("Likely Vaginal Delivery")
    else:
        st.warning("Likely Cesarean Section")

    shap_values = explainer(input_data)
    feature_names = input_data.columns
    shap_dict = dict(zip(feature_names, shap_values.values[0]))
    st.subheader("Factors Influencing Prediction (SHAP Values)")
    st.write("Positive = increases vaginal delivery chance; Negative = increases cesarean chance.")
    fig_shap = go.Figure(go.Bar(x=list(shap_dict.values()), y=list(shap_dict.keys()), orientation='h'))
    fig_shap.update_layout(title="SHAP Feature Importance", xaxis_title="Impact on Prediction")
    st.plotly_chart(fig_shap)

    st.subheader("Fetal Passage Visualization")
    fetal_head_diameter = fetal_head / np.pi
    theta = np.linspace(0, np.pi, 100)
    x_inlet = (pelvic_inlet / 2) * np.cos(theta)
    y_inlet = 2 + 0.5 * np.sin(theta)
    x_outlet = (pelvic_outlet / 2) * np.cos(theta)
    y_outlet = 0.5 * np.sin(theta)
    pelvis_x = np.concatenate([x_inlet, x_outlet[::-1]])
    pelvis_y = np.concatenate([y_inlet, y_outlet[::-1]])

    head_radius_x = fetal_head_diameter / 2 * (1.2 if prob < 0.5 else 1.0)
    head_radius_y = fetal_head_diameter / 2.5 * (1.2 if prob < 0.5 else 1.0)
    t = np.linspace(0, 2 * np.pi, 100)
    initial_head_x = head_radius_x * np.cos(t)
    initial_head_y = head_radius_y * np.sin(t) + 2.5

    end_y = -1.0 if prob > 0.5 else 1.5
    title = ("Model Prediction: Vaginal Delivery (Fetal Head Passing)"
            if prob > 0.5 else
            "Model Prediction: Cesarean Section - Cephalopelvic Disproportion (CPD)")

    frames = []
    for y_pos in np.linspace(2.5, end_y, 20):
        head_x = head_radius_x * np.cos(t)
        head_y = head_radius_y * np.sin(t) + y_pos
        frames.append(go.Frame(data=[
            go.Scatter(x=pelvis_x, y=pelvis_y, mode='lines', fill='toself', line_color='blue', name='Pelvis'),
            go.Scatter(x=head_x, y=head_y, mode='lines', line_color='red', name='Fetal Head')
        ]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pelvis_x, y=pelvis_y, mode='lines', fill='toself', line_color='blue', name='Pelvis'))
    fig.add_trace(go.Scatter(x=initial_head_x, y=initial_head_y, mode='lines', line_color='red', name='Fetal Head'))
    fig.update_layout(
        title=title,
        xaxis=dict(range=[-max(pelvic_inlet, fetal_head_diameter), max(pelvic_inlet, fetal_head_diameter)], title="Width (cm)"),
        yaxis=dict(range=[-1.5, 3], title="Height (cm)"),
        showlegend=True,
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
            }]
        }],
        transition={"duration": 100}
    )
    fig.update(frames=frames)
    st.plotly_chart(fig, use_container_width=True)