# =============================================================================
# HEART DISEASE WEARABLE MONITOR - PRODUCTION APP
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import time

# Page config
st.set_page_config(
    page_title="🏥 HeartGuard - AI Wearable Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models (YOUR COLAB MODELS)
@st.cache_resource
def load_models():
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    return model, scaler, features

model, scaler, features = load_models()

# User Database (Production: Use SQLite/PostgreSQL)
USER_DB = {
    "admin": hashlib.md5("admin123".encode()).hexdigest(),
    "patient1": hashlib.md5("patient123".encode()).hexdigest(),
    "doctor": hashlib.md5("doctor123".encode()).hexdigest()
}

# Session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'history' not in st.session_state:
    st.session_state.history = []

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem !important;
    font-weight: bold !important;
    color: #1f77b4 !important;
    text-align: center;
    margin-bottom: 2rem;
}
.risk-high {background-color: #ff4444; color: white; padding: 1rem; border-radius: 10px;}
.risk-medium {background-color: #ffaa00; color: white; padding: 1rem; border-radius: 10px;}
.risk-low {background-color: #44ff44; color: black; padding: 1rem; border-radius: 10px;}
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
              color: white; padding: 1rem; border-radius: 15px; text-align: center;}
</style>
""", unsafe_allow_html=True)

# Prediction function (YOUR MODEL)
def predict_risk(patient_data):
    input_data = np.array([patient_data])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    if probability > 0.7:
        risk_level = "HIGH RISK ⚠️"
        color = "risk-high"
    elif probability > 0.3:
        risk_level = "MEDIUM RISK ⚠️"
        color = "risk-medium"
    else:
        risk_level = "LOW RISK ✅"
        color = "risk-low"
    
    return {
        'prediction': 'Heart Disease Detected' if prediction == 1 else 'Healthy',
        'probability': probability,
        'risk_level': risk_level,
        'color': color
    }

# LOGIN PAGE
def login_page():
    st.markdown('<h1 class="main-header">🏥 HeartGuard AI</h1>', unsafe_allow_html=True)
    st.markdown("### Smart Wearable Heart Disease Monitor")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👤 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", type="primary"):
            pwd_hash = hashlib.md5(password.encode()).hexdigest()
            if username in USER_DB and USER_DB[username] == pwd_hash:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome {username}!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials!")
    
    with col2:
        st.subheader("📱 Demo Patients")
        st.info("**admin/admin123**")
        st.info("**patient1/patient123**")
        st.info("**doctor/doctor123**")

# MAIN DASHBOARD
def main_dashboard():
    # Header
    st.markdown('<h1 class="main-header">🏥 HeartGuard Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("👤 Profile")
        st.write(f"**Logged in as:** {st.session_state.username}")
        if st.button("🔓 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
        
        st.header("📊 Quick Stats")
        st.metric("Predictions Today", len(st.session_state.history))
        st.metric("High Risk Alerts", sum(1 for h in st.session_state.history if 'HIGH' in h['risk']))
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔬 Predict Risk", "📈 History", "📱 Smartwatch API"])
    
    with tab1:
        st.header("🔬 Real-time Heart Risk Prediction")
        
        # Input form (Smartwatch data)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Patient Vitals")
            age = st.slider("Age", 20, 90, 55)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
            trestbps = st.slider("Resting BP", 90, 200, 130)
            chol = st.slider("Cholesterol", 150, 400, 250)
        
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar >120mg/dl", [0, 1])
            restecg = st.selectbox("Resting ECG", [0, 1, 2])
            thalach = st.slider("Max Heart Rate", 70, 210, 150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1])
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
        
        col3, col4 = st.columns(2)
        with col3:
            slope = st.selectbox("ST Slope", [1, 2, 3])
            ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
        with col4:
            thal = st.selectbox("Thalassemia", [3, 6, 7])
        
        if st.button("🚀 PREDICT HEART RISK", type="primary", use_container_width=True):
            # Predict
            patient_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                          exang, oldpeak, slope, ca, thal]
            
            result = predict_risk(patient_data + [thalach/age, (chol/200)+(trestbps/120)+(age/70)])
            
            # Display result
            st.markdown(f"""
            <div class="{result['color']}">
                <h2>🎯 {result['risk_level']}</h2>
                <h3>{result['prediction']}</h3>
                <p>💉 Risk Probability: <strong>{result['probability']:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Save to history
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.history.append({
                'time': timestamp,
                'risk': result['risk_level'],
                'prob': result['probability'],
                'age': age
            })
            st.success("✅ Prediction saved to history!")
    
    with tab2:
        st.header("📈 Prediction History")
        if st.session_state.history:
            df_history = pd.DataFrame(st.session_state.history)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total", len(df_history))
            col2.metric("High Risk", len(df_history[df_history['risk'].str.contains('HIGH')]))
            col3.metric("Avg Risk", f"{df_history['prob'].mean():.1%}")
            col4.metric("Last Update", df_history['time'].iloc[-1])
            
            # Chart
            fig = px.histogram(df_history, x='prob', nbins=20, 
                             title="Risk Probability Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_history.tail(10))
        else:
            st.info("👆 Make a prediction to see history!")
    
    with tab3:
        st.header("📱 Smartwatch Integration")
        st.markdown("""
        ### API for Smartwatch/Wearable Devices
        ```
        POST https://yourapp.com/predict
        {
            "age": 55,
            "sex": 1,
            "thalach": 150,
            "trestbps": 130,
            ...
        }
        Response: {"risk": "HIGH RISK", "probability": 0.85}
        ```
        """)
        
        st.code("""
import requests
data = {"age": 65, "sex": 1, "thalach": 110, ...}
response = requests.post("YOUR_APP_URL/predict", json=data)
print(response.json())
        """)

# MAIN APP FLOW
def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()