"""
# Author: Wajid
# Bihar Crop Yield Prediction System

Data Ingestion Module for Bihar Crop Forecasting
Handles loading and initial processing of agricultural data
"""


import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional
import os

# Custom logging setup - Modified by me
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class DataIngestion:

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[INFO] Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"[ERROR] Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            print(f"[ERROR] Error parsing YAML file: {e}")
            raise

    def load_agri_data(self) -> pd.DataFrame:
        try:
            file_path = self.config['data_sources']['agricultural']['file']
            df = pd.read_csv(file_path)
            print(f"[INFO] Loaded agricultural data: {df.shape[0]} records")

            # Basic data validation
            required_columns = ['year', 'district', 'crop', 'yield_kg_per_hectare']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            return df
        except Exception as e:
            print(f"[ERROR] Error loading agricultural data: {e}")
            raise

    def load_weather_df(self) -> pd.DataFrame:
        try:
            file_path = self.config['data_sources']['weather']['file']
            df = pd.read_csv(file_path)
            print(f"[INFO] Loaded weather data: {df.shape[0]} records")

            # Validate weather data ranges
            validation_config = self.config.get('validation', {})
            if 'temperature_range' in validation_config:
                temp_min, temp_max = validation_config['temperature_range']
                temp_cols = ['temp_max_c', 'temp_min_c', 'temp_avg_c']
                for col in temp_cols:
                    if col in df.columns:
                        df[col] = df[col].clip(temp_min, temp_max)

            return df
        except Exception as e:
            print(f"[ERROR] Error loading weather data: {e}")
            raise

    def load_sat_data(self) -> pd.DataFrame:
        try:
            file_path = self.config['data_sources']['satellite']['file']
            df = pd.read_csv(file_path)
            print(f"[INFO] Loaded satellite data: {df.shape[0]} records")

            # Validate NDVI range
            if 'ndvi' in df.columns:
                df['ndvi'] = df['ndvi'].clip(0, 1)

            return df
        except Exception as e:
            print(f"[ERROR] Error loading satellite data: {e}")
            raise

    def load_soil_data(self) -> pd.DataFrame:
        try:
            file_path = self.config['data_sources']['soil']['file']
            df = pd.read_csv(file_path)
            print(f"[INFO] Loaded soil data: {df.shape[0]} records")
            return df
        except Exception as e:
            print(f"[ERROR] Error loading soil data: {e}")
            raise

    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        logger.info("Loading all datasets...")

        agri_data = self.load_agri_data()
        weather_df = self.load_weather_df()
        sat_data = self.load_sat_data()
        soil_data = self.load_soil_data()

        logger.info("All datasets loaded successfully")

        return agri_data, weather_df, sat_data, soil_data

def main():

    try:
        # Initialize data ingestion
        ingestion = DataIngestion()

        # Load all data
        agri_data, weather_df, sat_data, soil_data = ingestion.load_all_data()

        # Print basic statistics
        print("\n📊 Data Loading Summary:")
        print(f"Agricultural data: {agri_data.shape}")
        print(f"Weather data: {weather_df.shape}")
        print(f"Satellite data: {sat_data.shape}")
        print(f"Soil data: {soil_data.shape}")

        return True

    except Exception as e:
        print(f"[ERROR] Data ingestion failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Data ingestion completed successfully!")
    else:
        print("❌ Data ingestion failed!")
