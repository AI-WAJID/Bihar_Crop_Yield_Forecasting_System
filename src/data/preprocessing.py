"""
# Author: Wajid
# Bihar Crop Yield Prediction System

Data Preprocessing Module for Bihar Crop Forecasting
Handles data cleaning, feature engineering, and preparation for ML models
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml
import logging
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        
        self.config_path = config_path
        self.config = self._load_config()
        self.scalers = {}
        self.encoders = {}

    def _load_config(self) -> Dict:

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def merge_datasets(self, agri_df: pd.DataFrame, weather_df: pd.DataFrame, 
                      satellite_df: pd.DataFrame, soil_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Merging datasets...")

        # Start with agricultural data as base
        merged_df = agri_df.copy()

        # Merge weather data (monthly aggregations)
        weather_agg = self._aggregate_weather_df(weather_df)
        merged_df = pd.merge(merged_df, weather_agg, 
                            on=['year', 'district', 'season'], how='left')

        # Merge satellite data (seasonal aggregations)
        satellite_agg = self._aggregate_sat_data(satellite_df)
        merged_df = pd.merge(merged_df, satellite_agg,
                            on=['year', 'district', 'season'], how='left')

        # Merge soil data (static by district)
        merged_df = pd.merge(merged_df, soil_df, on='district', how='left')

        print(f"[INFO] Merged dataset shape: {merged_df.shape}")

        return merged_df

    def _aggregate_weather_df(self, weather_df: pd.DataFrame) -> pd.DataFrame:

        weather_df = weather_df.copy()

        # Map months to seasons
        season_mapping = {
            1: 'rabi', 2: 'rabi', 3: 'rabi', 4: 'kharif',
            5: 'kharif', 6: 'kharif', 7: 'kharif', 8: 'kharif',
            9: 'kharif', 10: 'kharif', 11: 'rabi', 12: 'rabi'
        }
        weather_df['season'] = weather_df['month'].map(season_mapping)

        # Aggregate by year, district, season
        agg_functions = {
            'temp_max_c': ['mean', 'max'],
            'temp_min_c': ['mean', 'min'],
            'temp_avg_c': 'mean',
            'rainfall_mm': ['sum', 'mean'],
            'humidity_percent': 'mean',
            'solar_radiation': 'mean'
        }

        weather_agg = weather_df.groupby(['year', 'district', 'season']).agg(agg_functions)

        # Flatten column names
        weather_agg.columns = ['_'.join(col).strip() for col in weather_agg.columns]
        weather_agg = weather_agg.reset_index()

        return weather_agg

    def _aggregate_sat_data(self, satellite_df: pd.DataFrame) -> pd.DataFrame:
        satellite_df = satellite_df.copy()

        # Map months to seasons
        season_mapping = {
            1: 'rabi', 2: 'rabi', 3: 'rabi', 4: 'kharif',
            5: 'kharif', 6: 'kharif', 7: 'kharif', 8: 'kharif',
            9: 'kharif', 10: 'kharif', 11: 'rabi', 12: 'rabi'
        }
        satellite_df['season'] = satellite_df['month'].map(season_mapping)

        # Aggregate by year, district, season
        agg_functions = {
            'ndvi': ['mean', 'max', 'std'],
            'lai': ['mean', 'max'],
            'evi': ['mean', 'max']
        }

        satellite_agg = satellite_df.groupby(['year', 'district', 'season']).agg(agg_functions)

        # Flatten column names
        satellite_agg.columns = ['_'.join(col).strip() for col in satellite_agg.columns]
        satellite_agg = satellite_agg.reset_index()

        return satellite_agg

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Creating engineered features...")

        df = df.copy()

        # 1. Time-based features
        df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())

        # 2. Weather-derived features
        if 'temp_max_c_mean' in df.columns and 'temp_min_c_mean' in df.columns:
            df['temp_range'] = df['temp_max_c_mean'] - df['temp_min_c_mean']
            df['growing_degree_days'] = np.maximum((df['temp_max_c_mean'] + df['temp_min_c_mean']) / 2 - 10, 0)

        # 3. Rainfall intensity
        if 'rainfall_mm_sum' in df.columns:
            df['rainfall_intensity'] = df['rainfall_mm_sum'] / 4

        # 4. Vegetation health indicators
        if 'ndvi_mean' in df.columns and 'lai_mean' in df.columns:
            df['vegetation_health'] = df['ndvi_mean'] * df['lai_mean']

        # 5. Soil fertility index
        soil_cols = ['nitrogen_kg_per_hectare', 'phosphorus_kg_per_hectare', 'potassium_kg_per_hectare']
        if all(col in df.columns for col in soil_cols):
            df['soil_fertility_index'] = (
                df['nitrogen_kg_per_hectare'] / 250 + 
                df['phosphorus_kg_per_hectare'] / 50 + 
                df['potassium_kg_per_hectare'] / 150
            ) / 3

        # 6. Crop-specific features
        df['is_rice'] = (df['crop'] == 'rice').astype(int)
        df['is_wheat'] = (df['crop'] == 'wheat').astype(int)
        df['is_kharif_crop'] = df['season'].map({'kharif': 1, 'rabi': 0})

        # 7. Lag features (previous year yield)
        df = df.sort_values(['district', 'crop', 'year'])
        df['yield_lag_1'] = df.groupby(['district', 'crop'])['yield_kg_per_hectare'].shift(1)
        df['yield_lag_2'] = df.groupby(['district', 'crop'])['yield_kg_per_hectare'].shift(2)

        # 8. Yield trend
        df['yield_trend'] = df.groupby(['district', 'crop'])['yield_kg_per_hectare'].pct_change()

        print(f"[INFO] Features created. Dataset shape: {df.shape}")

        return df

    def clean_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Handling missing values...")

        df = df.copy()

        # Fill missing values with appropriate strategies
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns

        # For numeric columns, use median imputation
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

        # For categorical columns, use mode imputation
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
                df[col].fillna(mode_val, inplace=True)

        return df

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        logger.info("Encoding categorical features...")

        df = df.copy()

        # List of categorical columns to encode
        categorical_cols = ['district', 'crop', 'season', 'soil_type']

        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    encoder = LabelEncoder()
                    df[f'{col}_encoded'] = encoder.fit_transform(df[col].astype(str))
                    self.encoders[col] = encoder
                else:
                    if col in self.encoders:
                        df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))

        return df

    def scale_features(self, df: pd.DataFrame, target_col: str = 'yield_kg_per_hectare', 
                      fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Scaling features...")

        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]

        X = df[feature_cols].copy()
        y = df[target_col].copy() if target_col in df.columns else None

        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['features'] = scaler
        else:
            if 'features' in self.scalers:
                X_scaled = self.scalers['features'].transform(X)
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers['features'] = scaler

        # Convert back to DataFrame
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

        # Add non-numeric columns back
        for col in df.columns:
            if col not in feature_cols and col != target_col:
                X_scaled_df[col] = df[col]

        return X_scaled_df, y

    def prepare_training_data(self, merged_df: pd.DataFrame, 
                             target_col: str = 'yield_kg_per_hectare') -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Starting complete preprocessing pipeline...")

        # 1. Create features
        df = self.create_features(merged_df)

        # 2. Handle missing values
        df = self.clean_missing_data(df)

        # 3. Encode categorical features
        df = self.encode_categorical_features(df, fit=True)

        # 4. Scale features
        X, y = self.scale_features(df, target_col, fit=True)

        print(f"[INFO] Preprocessing completed. Final dataset shape: {X.shape}")

        return X, y

def main():

    from src.data.ingestion import DataIngestion

    try:
        # Load data
        ingestion = DataIngestion()
        agri_data, weather_df, sat_data, soil_data = ingestion.load_all_data()

        # Initialize preprocessor
        preprocessor = DataPreprocessor()

        # Merge datasets
        merged_df = preprocessor.merge_datasets(agri_data, weather_df, sat_data, soil_data)

        # Prepare training data
        X, y = preprocessor.prepare_training_data(merged_df)

        print("\n📊 Preprocessing Summary:")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature columns: {X.select_dtypes(include=[np.number]).shape[1]} numeric features")

        # Save processed data
        X.to_csv('data/processed/features.csv', index=False)
        y.to_csv('data/processed/target.csv', index=False)

        print("✅ Processed data saved to data/processed/")

        return True

    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Data preprocessing completed successfully!")
    else:
        print("❌ Data preprocessing failed!")
