FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/
COPY data/ ./data/
COPY dashboard/ ./dashboard/

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Create directories
RUN mkdir -p /app/mlruns

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/_stcore/health || exit 1

# Run Streamlit ONLY
CMD streamlit run dashboard/dashboard_app.py --server.port=${PORT:-8000} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false
