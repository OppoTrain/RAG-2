# Streamlit Dockerfile
FROM python:3.9
WORKDIR /frontend
COPY /requirements.txt /frontend/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /frontend
CMD ["streamlit", "run", "frontend/chatbot_interface.py", "--server.port=8501", "--server.address=0.0.0.0"]
