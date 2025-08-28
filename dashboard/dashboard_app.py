"""
# Author: Wajid
# Bihar Crop Yield Prediction System

Streamlit Dashboard for Bihar Crop Forecasting
Interactive web application for crop yield predictions
"""

import os
import streamlit as st

# Dynamic API configuration for single service deployment
def get_api_base_url():
    """Get the API base URL based on environment"""
    
    # Check if we're running in a containerized environment (like Render)
    if 'PORT' in os.environ:
        # In production, API runs on the PORT provided by Render
        port = os.environ.get('PORT', '8000')
        # For Render deployment, use the same domain but API port
        api_url = f"http://localhost:{port}"
    else:
        # Local development
        api_url = "http://localhost:8000"
    
    return api_url

# Update the API_BASE_URL at the top of dashboard_app.py
API_BASE_URL = get_api_base_url()

# Test API connection and show status
def test_api_connection():
    """Test if API is accessible"""
    try:
        import requests
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, "Connected"
        else:
            return False, f"Error: {response.status_code}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

# Add this to your sidebar in dashboard_app.py
def show_api_status():
    """Show API connection status in sidebar"""
    is_connected, status = test_api_connection()
    
    if is_connected:
        st.sidebar.success(f"✅ API Status: {status}")
        st.sidebar.info(f"🔗 API URL: {API_BASE_URL}")
    else:
        st.sidebar.error(f"❌ API Status: {status}")
        st.sidebar.warning("Please ensure the API server is running")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, date

# Set up page configuration
st.set_page_config(
    page_title="Bihar Crop Yield Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
# Author: Wajid Raza
# Bihar Crop Yield Prediction System

<style>
.main-header {
    font-size: 3rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #4169E1;
    margin: 1rem 0;
}
</style>
"""
# Author: Your Name
# Bihar Crop Yield Prediction System
, unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🌾 Bihar Crop Yield Forecasting Dashboard</h1>', unsafe_allow_html=True)

import os

def get_api_base_url():
    """Get the API base URL based on environment"""
    if 'PORT' in os.environ:
        port = os.environ.get('PORT', '8000')
        api_url = f"http://localhost:{port}"
    else:
        api_url = "http://localhost:8000"
    return api_url

API_BASE_URL = get_api_base_url()


@st.cache_data
def get_districts():

    try:
        response = requests.get(f"{API_BASE_URL}/districts")
        if response.status_code == 200:
            return response.json()["districts"]
        else:
            return ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga"]
    except:
        return ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga"]

@st.cache_data
def get_crops():

    try:
        response = requests.get(f"{API_BASE_URL}/crops")
        if response.status_code == 200:
            return response.json()["crops"]
        else:
            return {
                'rice': ['kharif'],
                'wheat': ['rabi'],
                'maize': ['kharif', 'rabi'],
                'sugarcane': ['kharif'],
                'jute': ['kharif']
            }
    except:
        return {
            'rice': ['kharif'],
            'wheat': ['rabi'],
            'maize': ['kharif', 'rabi'],
            'sugarcane': ['kharif'],
            'jute': ['kharif']
        }

def make_prediction(request_data):

    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=request_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the FastAPI server is running on localhost:8000")
        return None
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Sidebar for inputs
st.sidebar.markdown('<h2 class="sub-header">🔧 Input Parameters</h2>', unsafe_allow_html=True)

# Get available districts and crops
districts = get_districts()
crops_data = get_crops()

# Location and Crop Selection
st.sidebar.subheader("📍 Location & Crop")
selected_district = st.sidebar.selectbox("Select District", districts)
selected_crop = st.sidebar.selectbox("Select Crop", list(crops_data.keys()))

# Update season options based on selected crop
available_seasons = crops_data.get(selected_crop, ['kharif'])
selected_season = st.sidebar.selectbox("Select Season", available_seasons)

# Year selection
current_year = datetime.now().year
selected_year = st.sidebar.slider("Select Year", 2024, 2030, current_year)

# Weather Data Input
st.sidebar.subheader("🌤️ Weather Data")
temp_max = st.sidebar.slider("Maximum Temperature (°C)", 15.0, 45.0, 35.0, 0.1)
temp_min = st.sidebar.slider("Minimum Temperature (°C)", 5.0, 35.0, 22.0, 0.1)
rainfall = st.sidebar.slider("Total Rainfall (mm)", 0.0, 2000.0, 1000.0, 10.0)
humidity = st.sidebar.slider("Average Humidity (%)", 30.0, 95.0, 70.0, 1.0)
solar_radiation = st.sidebar.slider("Solar Radiation", 10.0, 30.0, 20.0, 0.1)

# Satellite Data Input
st.sidebar.subheader("🛰️ Satellite Data")
ndvi_mean = st.sidebar.slider("Average NDVI", 0.0, 1.0, 0.7, 0.01)
ndvi_max = st.sidebar.slider("Maximum NDVI", ndvi_mean, 1.0, 0.85, 0.01)
lai_mean = st.sidebar.slider("Average LAI", 0.0, 10.0, 3.5, 0.1)
lai_max = st.sidebar.slider("Maximum LAI", lai_mean, 10.0, 5.0, 0.1)

# Soil Data Input
st.sidebar.subheader("🏞️ Soil Data")
soil_ph = st.sidebar.slider("Soil pH", 4.0, 10.0, 7.0, 0.1)
organic_carbon = st.sidebar.slider("Organic Carbon (%)", 0.0, 5.0, 0.8, 0.1)
nitrogen = st.sidebar.slider("Available Nitrogen (kg/ha)", 50.0, 400.0, 200.0, 5.0)
phosphorus = st.sidebar.slider("Available Phosphorus (kg/ha)", 5.0, 100.0, 25.0, 1.0)
potassium = st.sidebar.slider("Available Potassium (kg/ha)", 50.0, 300.0, 150.0, 5.0)

# Prediction button
if st.sidebar.button("🚀 Get Prediction", type="primary"):
    # Prepare request data
    request_data = {
        "district": selected_district,
        "crop": selected_crop,
        "year": selected_year,
        "season": selected_season,
        "weather": {
            "temp_max_c_mean": temp_max,
            "temp_min_c_mean": temp_min,
            "rainfall_mm_sum": rainfall,
            "humidity_percent_mean": humidity,
            "solar_radiation_mean": solar_radiation
        },
        "satellite": {
            "ndvi_mean": ndvi_mean,
            "ndvi_max": ndvi_max,
            "lai_mean": lai_mean,
            "lai_max": lai_max
        },
        "soil": {
            "ph": soil_ph,
            "organic_carbon_percent": organic_carbon,
            "nitrogen_kg_per_hectare": nitrogen,
            "phosphorus_kg_per_hectare": phosphorus,
            "potassium_kg_per_hectare": potassium
        }
    }

    # Make prediction
    with st.spinner("Making prediction..."):
        result = make_prediction(request_data)

    if result:
        st.session_state['prediction_result'] = result
        st.session_state['request_data'] = request_data

# Display prediction results
if 'prediction_result' in st.session_state:
    result = st.session_state['prediction_result']

    st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)

    # Display main results
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Predicted Yield",
            f"{result['predicted_yield']:.0f} kg/ha",
            delta=None
        )

    with col2:
        st.metric(
            "Lower Bound",
            f"{result['confidence_interval'][0]:.0f} kg/ha",
            delta=None
        )

    with col3:
        st.metric(
            "Upper Bound",
            f"{result['confidence_interval'][1]:.0f} kg/ha",
            delta=None
        )

    with col4:
        st.metric(
            "Model Used",
            result['model_used'].title(),
            delta=None
        )

    # Visualization
    st.markdown('<h3 class="sub-header">📈 Yield Prediction Visualization</h3>', unsafe_allow_html=True)

    # Create gauge chart for yield prediction
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = result['predicted_yield'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{result['crop'].title()} Yield (kg/ha)"},
        delta = {'reference': (result['confidence_interval'][0] + result['confidence_interval'][1]) / 2},
        gauge = {
            'axis': {'range': [None, result['confidence_interval'][1] * 1.2]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, result['confidence_interval'][0]], 'color': "lightgray"},
                {'range': [result['confidence_interval'][0], result['confidence_interval'][1]], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': result['predicted_yield']
            }
        }
    ))

    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Confidence interval chart
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=['Lower Bound', 'Predicted', 'Upper Bound'],
        y=[result['confidence_interval'][0], result['predicted_yield'], result['confidence_interval'][1]],
        marker_color=['lightcoral', 'darkgreen', 'lightcoral'],
        text=[f"{result['confidence_interval'][0]:.0f}", f"{result['predicted_yield']:.0f}", f"{result['confidence_interval'][1]:.0f}"],
        textposition='auto',
    ))

    fig_bar.update_layout(
        title=f"Yield Prediction Range for {result['crop'].title()} in {result['district']}",
        xaxis_title="Prediction Type",
        yaxis_title="Yield (kg/hectare)",
        height=400
    )

    st.plotly_chart(fig_bar, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
# Author: Wajid Raza
# Bihar Crop Yield Prediction System

<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🌾 Bihar Crop Yield Forecasting System v1.0</p>
    <p>Built with Streamlit, FastAPI, and Machine Learning</p>
</div>
"""
, unsafe_allow_html=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.info("""
# Author: Wajid Raza
# Bihar Crop Yield Prediction System

This dashboard provides AI-powered crop yield predictions for Bihar districts using:
- Weather data
- Satellite imagery (NDVI, LAI)
- Soil characteristics
- Machine learning models (XGBoost, LightGBM, Random Forest)
"""
)

if st.sidebar.button("📊 API Health Check"):
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            st.sidebar.success(f"✅ API Status: {health_data['status']}")
            st.sidebar.info(f"Models loaded: {health_data['models_loaded']}")
        else:
            st.sidebar.error("❌ API not responding")
    except:
        st.sidebar.error("❌ Cannot connect to API")
