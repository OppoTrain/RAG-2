# Use a multi-stage build
# Stage 1: Build the backend (FastAPI)
FROM python:3.9-slim as backend

# Set working directory for the backend
WORKDIR /app

# Copy backend requirements
COPY app/requirements.txt ./

# Install backend dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY app/ ./

# Expose FastAPI port
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 2: Build the frontend (Streamlit)
FROM python:3.9-slim as frontend

# Set working directory for the frontend
WORKDIR /frontend

# Copy frontend requirements
COPY frontend/requirements.txt ./

# Install frontend dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the frontend code
COPY frontend/ ./

# Expose Streamlit port
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "chatbot_interface.py", "--server.port=8501", "--server.address=0.0.0.0"]