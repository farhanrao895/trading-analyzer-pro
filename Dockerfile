# Railway Dockerfile for Python Backend
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Expose port (Railway sets PORT env var)
EXPOSE $PORT

# Start FastAPI server
CMD python -m uvicorn main:app --host 0.0.0.0 --port $PORT

