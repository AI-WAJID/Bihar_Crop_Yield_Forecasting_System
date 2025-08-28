from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import subprocess
import os
import signal
import threading
import time

# Your existing API imports and setup
from src.deployment.api import app as original_app

# Create new app that includes dashboard
app = FastAPI(
    title="Bihar Crop Yield Forecasting System",
    description="Complete system with API and Dashboard",
    version="1.0.0"
)

# Start Streamlit in background
streamlit_process = None

def start_streamlit():
    global streamlit_process
    try:
        # Start Streamlit on a different port internally
        streamlit_process = subprocess.Popen([
            "streamlit", "run", "dashboard/dashboard_app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false"
        ])
        print("✅ Streamlit started successfully")
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")

# Start Streamlit in background thread
threading.Thread(target=start_streamlit, daemon=True).start()
time.sleep(3)  # Wait for Streamlit to start

# Include all original API routes
app.mount("/api", original_app)

# Serve dashboard via proxy
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return """
    <html>
        <head>
            <title>Bihar Crop Forecasting Dashboard</title>
            <style>
                body { margin: 0; padding: 0; }
                iframe { width: 100%; height: 100vh; border: none; }
            </style>
        </head>
        <body>
            <iframe src="http://localhost:8501" title="Dashboard"></iframe>
        </body>
    </html>
    """

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Bihar Crop Yield Forecasting System",
        "api": "/api/docs",
        "dashboard": "/dashboard",
        "status": "healthy"
    }

# Cleanup on shutdown
def cleanup():
    global streamlit_process
    if streamlit_process:
        streamlit_process.terminate()
        streamlit_process.wait()

import atexit
atexit.register(cleanup)