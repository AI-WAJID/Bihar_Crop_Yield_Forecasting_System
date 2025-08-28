#!/bin/bash

echo "🚀 Starting Bihar Crop Forecasting System..."

# Get port from environment (Render provides PORT variable)
API_PORT=${PORT:-8000}
DASHBOARD_PORT=$((API_PORT + 1))

echo "📡 Starting FastAPI server on port $API_PORT..."
# Start FastAPI server in background
uvicorn src.deployment.api:app --host 0.0.0.0 --port $API_PORT &
API_PID=$!

# Wait a moment for API to start
sleep 5

echo "📊 Starting Streamlit dashboard on port $DASHBOARD_PORT..."
# Start Streamlit dashboard
streamlit run dashboard/dashboard_app.py \
  --server.port $DASHBOARD_PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false &
DASHBOARD_PID=$!

echo "✅ Both services started!"
echo "🔗 API available on port $API_PORT"
echo "🌐 Dashboard available on port $DASHBOARD_PORT"

# Function to handle shutdown
cleanup() {
    echo "🛑 Shutting down services..."
    kill $API_PID $DASHBOARD_PID
    exit 0
}

# Handle shutdown signals
trap cleanup SIGTERM SIGINT

# Keep the script running
wait
