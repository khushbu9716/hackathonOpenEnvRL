# Dockerfile
# E-Commerce Customer Support Environment
#
# Build : docker build -t ecommerce-support-env .
# Run   : docker run -p 7860:7860 ecommerce-support-env
# Test  : curl http://localhost:7860/health

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY models.py .
COPY main.py .
COPY openenv.yaml .
COPY env/ ./env/
COPY tasks/ ./tasks/
COPY data/ ./data/
COPY baseline/ ./baseline/
COPY config/ ./config/

# Create non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

# Expose port — HuggingFace Spaces uses 7860
EXPOSE 7860

# Health check — used by HuggingFace Spaces and openenv validate
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
