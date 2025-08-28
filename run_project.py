
import os
import sys
import logging
import subprocess
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors"""
    logger.info(f"Starting: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False

def setup_environment():
    """Set up the environment and install dependencies"""
    logger.info("🚀 Setting up environment...")

    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False

    # Set up MLflow
    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'

    return True

def run_data_pipeline():
    """Run the complete data pipeline"""
    logger.info("📊 Running data pipeline...")

    # Step 1: Data ingestion
    if not run_command("python -m src.data.ingestion", "Data ingestion"):
        return False

    # Step 2: Data preprocessing
    if not run_command("python -m src.data.preprocessing", "Data preprocessing"):
        return False

    logger.info("✅ Data pipeline completed successfully")
    return True

def run_model_training():
    """Run model training pipeline"""
    logger.info("🤖 Training machine learning models...")

    if not run_command("python -m src.models.train_model", "Model training"):
        return False

    logger.info("✅ Model training completed successfully")
    return True

def start_api_server():
    """Start the FastAPI server"""
    logger.info("🚀 Starting FastAPI server...")

    # Start the API server in background
    try:
        process = subprocess.Popen([
            'uvicorn', 'src.deployment.api:app', 
            '--host', '0.0.0.0', 
            '--port', '8000',
            '--reload'
        ])

        # Wait a moment for server to start
        time.sleep(10)

        logger.info("✅ API server started successfully")
        logger.info("🌐 API available at: http://localhost:8000")
        logger.info("📚 API documentation at: http://localhost:8000/docs")
        return process

    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        return None

def start_dashboard():
    """Start the Streamlit dashboard"""
    logger.info("📊 Starting Streamlit dashboard...")

    try:
        process = subprocess.Popen([
            'streamlit', 'run', 'dashboard/dashboard_app.py',
            '--server.port', '8501'
        ])

        time.sleep(5)
        logger.info("✅ Dashboard started successfully")
        logger.info("📊 Dashboard available at: http://localhost:8501")
        return process

    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        return None

def main():
    """Main execution function"""
    print("=" * 60)
    print("🌾 BIHAR CROP YIELD FORECASTING SYSTEM")
    print("=" * 60)

    try:
        # Step 1: Environment setup
        if not setup_environment():
            logger.error("Environment setup failed")
            sys.exit(1)

        # Step 2: Run data pipeline
        if not run_data_pipeline():
            logger.error("Data pipeline failed")
            sys.exit(1)

        # Step 3: Train models
        if not run_model_training():
            logger.error("Model training failed")
            sys.exit(1)

        # Step 4: Start API server
        api_process = start_api_server()
        if api_process is None:
            logger.error("Failed to start API server")
            sys.exit(1)

        # Step 5: Start dashboard
        dashboard_process = start_dashboard()
        if dashboard_process is None:
            logger.warning("Failed to start dashboard, but continuing...")

        print("\n" + "=" * 60)
        print("🎉 BIHAR CROP FORECASTING SYSTEM IS READY!")
        print("=" * 60)
        print("📍 Services running:")
        print("   🔌 API Server: http://localhost:8000")
        print("   📚 API Docs: http://localhost:8000/docs")
        if dashboard_process:
            print("   📊 Dashboard: http://localhost:8501")
        print("\n💡 Usage:")
        print("   1. Visit the dashboard for interactive predictions")
        print("   2. Use the API for programmatic access")
        print("   3. Check /docs for API documentation")
        print("\n⏹️  Press Ctrl+C to stop all services")
        print("=" * 60)

        # Keep processes running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down services...")

            if api_process:
                api_process.terminate()
                logger.info("✅ API server stopped")

            if dashboard_process:
                dashboard_process.terminate()
                logger.info("✅ Dashboard stopped")

            logger.info("👋 Goodbye!")

    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
