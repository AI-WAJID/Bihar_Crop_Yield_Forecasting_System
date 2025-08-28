FROM python:3.9-slim

WORKDIR /app

# Install essential system dependencies for building Python packages
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

# Upgrade pip, setuptools, and wheel to latest versions for better compatibility
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/
COPY data/ ./data/
COPY dashboard/ ./dashboard/
COPY start_services.sh .

# Make the startup script executable
RUN chmod +x start_services.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Create directories for MLflow
RUN mkdir -p /app/mlruns

# Expose ports (Render will map these automatically)
EXPOSE 8000 8001

# Health check for API
HEALTHCHECK --interval=30s --timeout=30s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run both services using the startup script
CMD ["./start_services.sh"]
