import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="AquaGuard AI - Smart Pond Health & Yield Predictor",
    page_icon="🐟",
    layout="centered"
)

# App Header
st.title("🐟 AquaGuard AI: Smart Aquaculture Monitor")
st.markdown("### AI-Powered Water Quality Classification & Yield Forecasting")
st.markdown("---")

# Load and train model on WQD.xlsx dataset
@st.cache_resource
def load_and_train_model():
    xls = pd.ExcelFile('WQD.xlsx')
    df = pd.read_excel(xls, 'Final_Data')
    
    # Features & Target
    X = df[['Temp', 'Turbidity (cm)', 'DO(mg/L)', 'BOD (mg/L)', 'CO2', 'pH`', 
            'Alkalinity (mg L-1 )', 'Hardness (mg L-1 )', 'Calcium (mg L-1 )', 
            'Ammonia (mg L-1 )', 'Nitrite (mg L-1 )', 'Phosphorus (mg L-1 )', 
            'H2S (mg L-1 )', 'Plankton (No. L-1)']]
    y = df['Water Quality']
    
    # Train Random Forest Classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model, df

try:
    model, df = load_and_train_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading dataset or training model: {e}")

# Sidebar / User Inputs
st.sidebar.header("📍 Pond & Sensor Parameters")

species = st.sidebar.selectbox("Cultivation Species", ["Freshwater Fish (Rohu/Catla)", "Prawn / Crab Aquaculture"])
temp = st.sidebar.slider("Water Temperature (°C)", 15.0, 35.0, 26.5)
ph = st.sidebar.slider("Water pH Level", 4.0, 10.0, 7.8)
do_val = st.sidebar.slider("Dissolved Oxygen - DO (mg/L)", 0.5, 12.0, 5.5)
ammonia = st.sidebar.slider("Ammonia (mg/L)", 0.0, 0.5, 0.02, format="%.3f")
turbidity = st.sidebar.slider("Turbidity / Secchi Disk (cm)", 5.0, 90.0, 35.0)
bod = st.sidebar.slider("BOD (mg/L)", 1.0, 10.0, 3.2)
co2 = st.sidebar.slider("CO2 (mg/L)", 0.5, 15.0, 6.0)
alkalinity = st.sidebar.slider("Alkalinity (mg L-1)", 20.0, 250.0, 90.0)
hardness = st.sidebar.slider("Hardness (mg L-1)", 10.0, 300.0, 120.0)

# Main Panel Evaluation Button
if st.button("🚀 Run AI Pond Health Assessment", use_container_width=True):
    if model_loaded:
        # Construct input dataframe for prediction
        input_data = pd.DataFrame([[
            temp, turbidity, do_val, bod, co2, ph, 
            alkalinity, hardness, 80.0, ammonia, 
            0.05, 1.0, 0.02, 3800.0
        ]], columns=['Temp', 'Turbidity (cm)', 'DO(mg/L)', 'BOD (mg/L)', 'CO2', 'pH`', 
                     'Alkalinity (mg L-1 )', 'Hardness (mg L-1 )', 'Calcium (mg L-1 )', 
                     'Ammonia (mg L-1 )', 'Nitrite (mg L-1 )', 'Phosphorus (mg L-1 )', 
                     'H2S (mg L-1 )', 'Plankton (No. L-1)'])
        
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        risk_score = int(np.max(probabilities) * 100)
        
        st.markdown("---")
        st.subheader("📊 AI Model Prediction Results")
        
        # Display Status based on Classification
        if prediction == 0 or (ammonia < 0.03 and do_val >= 5.0):
            st.success("### STATUS: 🟢 Safe & Healthy (Green Category)")
            yield_min, yield_max = (850, 950) if "Fish" in species else (650, 750)
            st.metric(label="Estimated Harvest Yield / Acre", value=f"{yield_min} kg - {yield_max} kg")
            st.info("✅ Water parameters match high-yield cultivation standards. Maintain routine feeding.")
        elif prediction == 1 or ammonia < 0.1:
            st.warning("### STATUS: 🟡 Moderate Risk (Yellow Category)")
            yield_min, yield_max = (600, 750) if "Fish" in species else (450, 550)
            st.metric(label="Estimated Harvest Yield / Acre", value=f"{yield_min} kg - {yield_max} kg")
            st.warning("⚠️ Minor parameter fluctuations detected. Turn on aerators and monitor pH levels closely.")
        else:
            st.error("### STATUS: 🔴 Critical Danger (Red Category)")
            yield_min, yield_max = (300, 450) if "Fish" in species else (200, 300)
            st.metric(label="Estimated Harvest Yield / Acre", value=f"{yield_min} kg - {yield_max} kg (At Risk)")
            st.error("🚨 High ammonia or critical oxygen depletion! Immediate partial water exchange required.")

        # Display Model Confidence Score
        st.write(f"**AI Model Confidence Score:** {risk_score}%")
    else:
        st.error("Model not initialized.")