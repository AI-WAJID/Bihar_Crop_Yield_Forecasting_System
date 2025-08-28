"""
# Author: Wajid
# Bihar Crop Yield Prediction System

Model Training Module for Bihar Crop Forecasting
Implements XGBoost, LightGBM, Random Forest models with MLflow tracking
"""


import pandas as pd
import numpy as np
import yaml
import logging
import joblib
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# MLOps
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class CropYieldPredictor:

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        
        self.config_path = config_path
        self.config = self._load_config()
        self.models = {}
        self.best_model = None
        self.best_score = float('inf')

        # Initialize MLflow
        try:
            mlflow.set_experiment("Bihar_Crop_Yield_Forecasting")
        except:
            logger.warning("MLflow not available, continuing without tracking")

    def _load_config(self) -> Dict:

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
        }
        return metrics

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, 
                     X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:

        logger.info("Training XGBoost model...")

        try:
            with mlflow.start_run(run_name="XGBoost_Model", nested=True):
                # Get XGBoost configuration
                xgb_config = self.config['xgboost']

                # Initialize model
                model = xgb.XGBRegressor(**xgb_config)

                # Train model
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )

                # Make predictions
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)

                # Calculate metrics
                train_metrics = self.get_metrics(y_train, train_pred)
                val_metrics = self.get_metrics(y_val, val_pred)

                # Log parameters and metrics
                mlflow.log_params(xgb_config)
                mlflow.log_metrics({
                    'train_rmse': train_metrics['rmse'],
                    'train_mae': train_metrics['mae'],
                    'train_r2': train_metrics['r2_score'],
                    'val_rmse': val_metrics['rmse'],
                    'val_mae': val_metrics['mae'],
                    'val_r2': val_metrics['r2_score']
                })

                # Log model
                mlflow.xgboost.log_model(model, "xgboost_model")
        except:
            # Continue without MLflow if it fails
            xgb_config = self.config['xgboost']
            model = xgb.XGBRegressor(**xgb_config)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_metrics = self.get_metrics(y_train, train_pred)
            val_metrics = self.get_metrics(y_val, val_pred)

        # Save model locally
        joblib.dump(model, 'models/xgboost_model.pkl')

        # Update best model if this performs better
        if val_metrics['rmse'] < self.best_score:
            self.best_model = model
            self.best_score = val_metrics['rmse']
            print(f"[INFO] New best model: XGBoost (RMSE: {val_metrics['rmse']:.2f})")

        self.models['xgboost'] = {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }

        return self.models['xgboost']

    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:

        logger.info("Training LightGBM model...")

        try:
            with mlflow.start_run(run_name="LightGBM_Model", nested=True):
                # Get LightGBM configuration
                lgb_config = self.config['lightgbm']

                # Initialize model
                model = lgb.LGBMRegressor(**lgb_config)

                # Train model
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )

                # Make predictions
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)

                # Calculate metrics
                train_metrics = self.get_metrics(y_train, train_pred)
                val_metrics = self.get_metrics(y_val, val_pred)

                # Log parameters and metrics
                mlflow.log_params(lgb_config)
                mlflow.log_metrics({
                    'train_rmse': train_metrics['rmse'],
                    'val_rmse': val_metrics['rmse'],
                    'val_r2': val_metrics['r2_score']
                })

                # Log model
                mlflow.lightgbm.log_model(model, "lightgbm_model")
        except:
            # Continue without MLflow if it fails
            lgb_config = self.config['lightgbm']
            model = lgb.LGBMRegressor(**lgb_config)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_metrics = self.get_metrics(y_train, train_pred)
            val_metrics = self.get_metrics(y_val, val_pred)

        # Save model locally
        joblib.dump(model, 'models/lightgbm_model.pkl')

        # Update best model if this performs better
        if val_metrics['rmse'] < self.best_score:
            self.best_model = model
            self.best_score = val_metrics['rmse']
            print(f"[INFO] New best model: LightGBM (RMSE: {val_metrics['rmse']:.2f})")

        self.models['lightgbm'] = {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }

        return self.models['lightgbm']

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        logger.info("Training Random Forest model...")

        try:
            with mlflow.start_run(run_name="RandomForest_Model", nested=True):
                # Get Random Forest configuration
                rf_config = self.config['random_forest']

                # Initialize model
                model = RandomForestRegressor(**rf_config)

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)

                # Calculate metrics
                train_metrics = self.get_metrics(y_train, train_pred)
                val_metrics = self.get_metrics(y_val, val_pred)

                # Log parameters and metrics
                mlflow.log_params(rf_config)
                mlflow.log_metrics({
                    'val_rmse': val_metrics['rmse'],
                    'val_r2': val_metrics['r2_score']
                })

                # Log model
                mlflow.sklearn.log_model(model, "random_forest_model")
        except:
            # Continue without MLflow if it fails
            rf_config = self.config['random_forest']
            model = RandomForestRegressor(**rf_config)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_metrics = self.get_metrics(y_train, train_pred)
            val_metrics = self.get_metrics(y_val, val_pred)

        # Save model locally
        joblib.dump(model, 'models/random_forest_model.pkl')

        # Update best model if this performs better
        if val_metrics['rmse'] < self.best_score:
            self.best_model = model
            self.best_score = val_metrics['rmse']
            print(f"[INFO] New best model: Random Forest (RMSE: {val_metrics['rmse']:.2f})")

        self.models['random_forest'] = {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }

        return self.models['random_forest']

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        logger.info("Starting training of all models...")

        try:
            with mlflow.start_run(run_name="All_Models_Training"):
                # Train individual models
                self.train_xgboost(X_train, y_train, X_val, y_val)
                self.train_lightgbm(X_train, y_train, X_val, y_val)
                self.train_random_forest(X_train, y_train, X_val, y_val)

                # Log best model info
                mlflow.log_metrics({'best_model_rmse': self.best_score})
        except:
            # Train models without MLflow
            self.train_xgboost(X_train, y_train, X_val, y_val)
            self.train_lightgbm(X_train, y_train, X_val, y_val)
            self.train_random_forest(X_train, y_train, X_val, y_val)

        # Save best model
        if self.best_model is not None:
            joblib.dump(self.best_model, 'models/best_model.pkl')

        results = {
            'models': self.models,
            'best_model': self.best_model,
            'best_score': self.best_score
        }

        return results

def main():

    try:
        # Load processed data
        X = pd.read_csv('data/processed/features.csv')
        y = pd.read_csv('data/processed/target.csv').squeeze()

        # Select only numeric features for modeling
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_features]

        print(f"[INFO] Using {len(numeric_features)} features for modeling")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=0.2, random_state=42
        )

        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42
        )

        print(f"[INFO] Training set: {X_train_split.shape}")
        print(f"[INFO] Validation set: {X_val.shape}")
        print(f"[INFO] Test set: {X_test.shape}")

        # Initialize predictor
        predictor = CropYieldPredictor()

        # Train all models
        results = predictor.train_all_models(X_train_split, y_train_split, X_val, y_val)

        # Test the best model
        if predictor.best_model is not None:
            test_pred = predictor.best_model.predict(X_test)
            test_metrics = predictor.get_metrics(y_test, test_pred)

            print("\n🏆 Best Model Test Results:")
            print(f"RMSE: {test_metrics['rmse']:.2f} kg/hectare")
            print(f"MAE: {test_metrics['mae']:.2f} kg/hectare")
            print(f"R²: {test_metrics['r2_score']:.3f}")
            print(f"MAPE: {test_metrics['mape']:.2f}%")

        print("\n✅ Model training completed successfully!")
        print(f"Best model RMSE: {predictor.best_score:.2f} kg/hectare")

        return True

    except Exception as e:
        print(f"[ERROR] Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ All models trained successfully!")
    else:
        print("❌ Model training failed!")
