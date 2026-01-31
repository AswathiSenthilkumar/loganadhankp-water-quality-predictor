import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
from sklearn.preprocessing import StandardScaler
st.set_page_config(
    page_title="Advanced Water Quality Predictor",
    page_icon="💧",
    layout="wide"
)
@st.cache_resource
def load_artifacts():
    model = joblib.load("water_quality_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)

model, scaler = load_artifacts()
BASE_FEATURES = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
DERIVED_FEATURES = ['NO3_NO2', 'NH4_NO2', 'O2_per_Suspended']
ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES

SAFE_RANGES = {
    'NH4': (0, 1.5),
    'BSK5': (0, 6),
    'Suspended': (0, 50),
    'O2': (4, 14),
    'NO3': (0, 50),
    'NO2': (0, 1),
    'SO4': (0, 250),
    'PO4': (0, 5),
    'CL': (0, 250)
}
st.title("Advanced Water Quality Prediction System")
st.caption("Machine Learning–based potable water assessment with derived indicators")

st.sidebar.header("⚙️ Model Details")
st.sidebar.markdown("""
- **Algorithm:** Random Forest  
- **Input Features:** 9 raw + 3 engineered  
- **Scaling:** StandardScaler  
- **Accuracy:** ~93%  
""")
st.sidebar.markdown(f"📅 **Date:** {datetime.date.today()}")
st.subheader("Enter Water Parameters")

cols = st.columns(3)
user_input = {}

for i, feat in enumerate(BASE_FEATURES):
    min_v, max_v = SAFE_RANGES[feat]
    with cols[i % 3]:
        user_input[feat] = st.number_input(
            f"{feat} (mg/L)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            help=f"Recommended range: {min_v} – {max_v}"
        )
def engineer_features(d):
    return {
        'NO3_NO2': d['NO3'] / d['NO2'] if d['NO2'] != 0 else 0,
        'NH4_NO2': d['NH4'] + d['NO2'],
        'O2_per_Suspended': d['O2'] / d['Suspended'] if d['Suspended'] != 0 else 0
    }
st.divider()
if st.button("Predict Water Quality", use_container_width=True):

    derived = engineer_features(user_input)
    full_row = {**user_input, **derived}

    input_df = pd.DataFrame([full_row])[ALL_FEATURES]
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Outcome")

    if pred == 1:
        st.success(f"**Water is Safe to Drink**  \nConfidence: **{prob[1]*100:.2f}%**")
        st.balloons()
    else:
        st.error(f"**Water is NOT Safe to Drink**  \nRisk Probability: **{prob[0]*100:.2f}%**")
        st.warning("Treatment or filtration is strongly recommended.")
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=list(user_input.values()),
        theta=list(user_input.keys()),
        fill='toself'
    ))
    radar_fig.update_layout(
        title="Input Feature Radar",
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False
    )
    st.plotly_chart(radar_fig, use_container_width=True)
    st.subheader("Parameter Safety Check")
    status_df = []
    for k, v in user_input.items():
        lo, hi = SAFE_RANGES[k]
        status_df.append({
            "Feature": k,
            "Value": v,
            "Safe Range": f"{lo}–{hi}",
            "Status": "OK" if lo <= v <= hi else "Out of Range"
        })
    st.dataframe(pd.DataFrame(status_df), use_container_width=True)
    report = f"""
WATER QUALITY REPORT
Date: {datetime.date.today()}

Prediction: {'SAFE' if pred == 1 else 'NOT SAFE'}
Confidence: {max(prob)*100:.2f}%

Input Parameters:
{pd.Series(user_input).to_string()}

Derived Indicators:
{pd.Series(derived).to_string()}
"""
    st.download_button(
        "📄 Download Detailed Report",
        report,
        file_name=f"water_quality_report_{datetime.date.today()}.txt",
        mime="text/plain"
    )
with st.expander("Compare with Historical Dataset"):
    try:
        data = load_dataset("PB_All_2000_2021.csv")
        st.write(" Dataset Feature Averages")
        st.dataframe(data[BASE_FEATURES].mean().round(2))
    except:
        st.warning("Dataset not found in project directory.")
st.caption("Built with Streamlit | ML-powered Environmental Monitoring")
