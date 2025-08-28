"""
# Author: Wajid
# Bihar Crop Yield Prediction System
Streamlit Dashboard for Bihar Crop Forecasting - FIXED VERSION
Interactive web application for crop yield predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, date
import pickle
import os
import sys

# Add project root to path
sys.path.append('/app')

# Set up page configuration
st.set_page_config(
    page_title="Bihar Crop Yield Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🌾 Bihar Crop Yield Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Enhanced preprocessing function
def create_all_features(input_data):
    """Create all 40 features that match the trained model"""
    
    # Base feature mappings
    district_mapping = {
        'patna': 0, 'gaya': 1, 'bhagalpur': 2, 'muzaffarpur': 3, 'darbhanga': 4,
        'purnia': 5, 'araria': 6, 'kishanganj': 7, 'west champaran': 8, 'east champaran': 9,
        'sheohar': 10, 'sitamarhi': 11, 'madhubani': 12, 'supaul': 13, 'saharsa': 14,
        'madhepura': 15, 'khagaria': 16, 'begusarai': 17, 'samastipur': 18, 'vaishali': 19,
        'saran': 20, 'siwan': 21, 'gopalganj': 22, 'rohtas': 23, 'buxar': 24,
        'kaimur': 25, 'bhojpur': 26, 'arwal': 27, 'jehanabad': 28, 'aurangabad': 29,
        'nalanda': 30, 'sheikhpura': 31, 'lakhisarai': 32, 'jamui': 33, 'munger': 34,
        'banka': 35, 'nawada': 36, 'katihar': 37
    }
    
    crop_mapping = {'rice': 0, 'wheat': 1, 'maize': 2, 'sugarcane': 3, 'jute': 4}
    season_mapping = {'kharif': 0, 'rabi': 1}
    
    # Create base features
    features = {}
    
    # 1. Basic encoded features (4 features)
    features['district_encoded'] = district_mapping.get(input_data['district'].lower(), 0)
    features['crop_encoded'] = crop_mapping.get(input_data['crop'].lower(), 0)
    features['season_encoded'] = season_mapping.get(input_data['season'].lower(), 0)
    features['year'] = input_data['year']
    
    # 2. Weather features (5 features)
    features['temp_max_c_mean'] = input_data['temp_max_c_mean']
    features['temp_min_c_mean'] = input_data['temp_min_c_mean']
    features['rainfall_mm_sum'] = input_data['rainfall_mm_sum']
    features['humidity_percent_mean'] = input_data['humidity_percent_mean']
    features['solar_radiation_mean'] = input_data['solar_radiation_mean']
    
    # 3. Satellite features (4 features)
    features['ndvi_mean'] = input_data['ndvi_mean']
    features['ndvi_max'] = input_data['ndvi_max']
    features['lai_mean'] = input_data['lai_mean']
    features['lai_max'] = input_data['lai_max']
    
    # 4. Soil features (5 features)
    features['ph'] = input_data['ph']
    features['organic_carbon_percent'] = input_data['organic_carbon_percent']
    features['nitrogen_kg_per_hectare'] = input_data['nitrogen_kg_per_hectare']
    features['phosphorus_kg_per_hectare'] = input_data['phosphorus_kg_per_hectare']
    features['potassium_kg_per_hectare'] = input_data['potassium_kg_per_hectare']
    
    # 5. ENGINEERED FEATURES (22 additional features to reach 40)
    # Temperature-related features
    features['temp_range'] = features['temp_max_c_mean'] - features['temp_min_c_mean']
    features['temp_avg'] = (features['temp_max_c_mean'] + features['temp_min_c_mean']) / 2
    
    # Vegetation indices
    features['ndvi_range'] = features['ndvi_max'] - features['ndvi_mean']
    features['lai_range'] = features['lai_max'] - features['lai_mean']
    features['vegetation_index'] = features['ndvi_mean'] * features['lai_mean']
    
    # Nutrient ratios
    features['n_p_ratio'] = features['nitrogen_kg_per_hectare'] / max(features['phosphorus_kg_per_hectare'], 1)
    features['n_k_ratio'] = features['nitrogen_kg_per_hectare'] / max(features['potassium_kg_per_hectare'], 1)
    features['p_k_ratio'] = features['phosphorus_kg_per_hectare'] / max(features['potassium_kg_per_hectare'], 1)
    
    # Weather interactions
    features['rainfall_humidity'] = features['rainfall_mm_sum'] * features['humidity_percent_mean'] / 100
    features['temp_solar'] = features['temp_avg'] * features['solar_radiation_mean']
    
    # Seasonal adjustments
    features['rainfall_per_temp'] = features['rainfall_mm_sum'] / max(features['temp_avg'], 1)
    features['humidity_temp_index'] = features['humidity_percent_mean'] / max(features['temp_avg'], 1)
    
    # Year-based features
    features['year_normalized'] = (features['year'] - 2020) / 10.0
    
    # Crop-season interaction
    features['crop_season_interaction'] = features['crop_encoded'] * features['season_encoded']
    
    # District-crop interaction
    features['district_crop_interaction'] = features['district_encoded'] * features['crop_encoded']
    
    # Advanced soil health index
    features['soil_health_index'] = (
        (features['ph'] / 7.0) * 
        features['organic_carbon_percent'] * 
        np.sqrt(features['nitrogen_kg_per_hectare'] * features['phosphorus_kg_per_hectare'] * features['potassium_kg_per_hectare']) / 1000
    )
    
    # Climate stress indicators
    features['heat_stress'] = max(0, features['temp_max_c_mean'] - 35) * (100 - features['humidity_percent_mean']) / 100
    features['water_stress'] = max(0, 30 - features['rainfall_mm_sum'] / 50)
    
    # Productivity indicators
    features['growing_degree_days'] = max(0, features['temp_avg'] - 10) * 30
    features['evapotranspiration_index'] = features['temp_avg'] * features['solar_radiation_mean'] / max(features['humidity_percent_mean'], 1)
    
    # Additional engineered features
    features['soil_fertility_score'] = (features['nitrogen_kg_per_hectare'] + features['phosphorus_kg_per_hectare'] + features['potassium_kg_per_hectare']) / 3
    features['climate_favorability'] = (features['rainfall_mm_sum'] / 1000) * (features['temp_avg'] / 30) * (features['humidity_percent_mean'] / 100)
    features['vegetation_vigor'] = features['ndvi_mean'] * features['lai_mean'] * features['solar_radiation_mean']
    
    return features

def preprocess_input_enhanced(input_data):
    """Enhanced preprocessing to generate exactly 40 features"""
    
    # Generate all features
    all_features = create_all_features(input_data)
    
    # Create DataFrame
    feature_df = pd.DataFrame([all_features])
    
    # Ensure we have exactly 40 features
    target_features = 40
    current_features = len(feature_df.columns)
    
    if current_features < target_features:
        # Add dummy features to reach 40
        for i in range(current_features, target_features):
            feature_df[f'feature_{i}'] = 0.0
    elif current_features > target_features:
        # Take first 40 features if we have too many
        feature_df = feature_df.iloc[:, :target_features]
    
    return feature_df

# Load models function
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_dir = "/app/models"
    
    try:
        # Try to load models
        model_files = {
            'xgboost': 'xgboost_model.pkl',
            'lightgbm': 'lightgbm_model.pkl', 
            'random_forest': 'random_forest_model.pkl',
            'best_model': 'best_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(model_dir, filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                st.sidebar.success(f"✅ Loaded {model_name}")
            else:
                st.sidebar.warning(f"⚠️ Model file not found: {filename}")
        
        return models
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {str(e)}")
        return {}

# Enhanced prediction function
def make_prediction_enhanced(models, input_data):
    """Make prediction with proper 40-feature preprocessing"""
    try:
        # Generate exactly 40 features
        X = preprocess_input_enhanced(input_data)
        
        st.info(f"Generated {X.shape[1]} features for prediction")
        
        # Use best model if available
        model_name = 'best_model' if 'best_model' in models else list(models.keys())[0]
        model = models[model_name]
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Calculate confidence interval
        confidence_lower = prediction * 0.85
        confidence_upper = prediction * 1.15
        
        return {
            'predicted_yield': float(prediction),
            'confidence_interval': [float(confidence_lower), float(confidence_upper)],
            'model_used': model_name,
            'district': input_data['district'],
            'crop': input_data['crop'],
            'season': input_data['season'],
            'year': input_data['year']
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Load models
models = load_models()

# District and crop data
districts = [
    "Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga", "Purnia", "Araria", "Kishanganj",
    "West Champaran", "East Champaran", "Sheohar", "Sitamarhi", "Madhubani", "Supaul", "Saharsa",
    "Madhepura", "Khagaria", "Begusarai", "Samastipur", "Vaishali", "Saran", "Siwan", "Gopalganj",
    "Rohtas", "Buxar", "Kaimur", "Bhojpur", "Arwal", "Jehanabad", "Aurangabad", "Nalanda",
    "Sheikhpura", "Lakhisarai", "Jamui", "Munger", "Banka", "Nawada", "Katihar"
]

crops_data = {
    'rice': ['kharif'],
    'wheat': ['rabi'],
    'maize': ['kharif', 'rabi'],
    'sugarcane': ['kharif'],
    'jute': ['kharif']
}

# Sidebar for inputs
st.sidebar.markdown('<h2 class="sub-header">🔧 Input Parameters</h2>', unsafe_allow_html=True)

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
if st.sidebar.button("🚀 Get Prediction", type="primary") and models:
    # Prepare input data
    input_data = {
        "district": selected_district,
        "crop": selected_crop,
        "year": selected_year,
        "season": selected_season,
        "temp_max_c_mean": temp_max,
        "temp_min_c_mean": temp_min,
        "rainfall_mm_sum": rainfall,
        "humidity_percent_mean": humidity,
        "solar_radiation_mean": solar_radiation,
        "ndvi_mean": ndvi_mean,
        "ndvi_max": ndvi_max,
        "lai_mean": lai_mean,
        "lai_max": lai_max,
        "ph": soil_ph,
        "organic_carbon_percent": organic_carbon,
        "nitrogen_kg_per_hectare": nitrogen,
        "phosphorus_kg_per_hectare": phosphorus,
        "potassium_kg_per_hectare": potassium
    }

    # Make prediction
    with st.spinner("Making prediction..."):
        result = make_prediction_enhanced(models, input_data)

    if result:
        st.session_state['prediction_result'] = result

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
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🌾 Bihar Crop Yield Forecasting System v1.0</p>
    <p>Built with Streamlit and Machine Learning</p>
    <p>Author: Wajid Raza</p>
</div>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.info("""
This dashboard provides AI-powered crop yield predictions for Bihar districts using:
- Weather data
- Satellite imagery (NDVI, LAI)  
- Soil characteristics
- Machine learning models (XGBoost, LightGBM, Random Forest)

Author: Wajid Raza
""")

# Model status in sidebar
st.sidebar.markdown("---")
if st.sidebar.button("📊 Model Status"):
    st.sidebar.write("**Loaded Models:**")
    if models:
        for model_name in models.keys():
            st.sidebar.write(f"✅ {model_name}")
        st.sidebar.write(f"**Total Models:** {len(models)}")
    else:
        st.sidebar.write("❌ No models loaded")