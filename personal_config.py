# Author: Wajid
# Project: Bihar Crop Forecasting
# Created: August 2025

# My custom model preferences
MY_MODEL_PREFERENCES = {
    "primary_model": "xgboost",
    "backup_model": "lightgbm", 
    "ensemble_method": "weighted_voting"
}

# My data processing preferences
DATA_SETTINGS = {
    "handle_outliers": True,
    "use_feature_scaling": True,
    "validation_split": 0.2,
    "random_seed": 42  # My lucky number!
}

# My API settings
API_CONFIG = {
    "port": 8000,
    "host": "0.0.0.0",
    "debug_mode": False,
    "max_requests_per_minute": 100
}

# Personal notes
DEVELOPMENT_NOTES = """
This project was developed as part of my MLOps learning journey.
Key challenges I solved:
1. Feature engineering for agricultural data
2. Handling seasonal variations in crop yields  
3. Building production-ready API with FastAPI
4. Creating interactive dashboard with Streamlit

Future improvements I want to add:
- Real-time data integration
- More sophisticated ensemble methods
- Mobile app integration
- If anyone want to enhance it then do it
"""
