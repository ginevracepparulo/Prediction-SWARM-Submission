FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./frontend ./frontend
COPY ./backend ./backend

COPY ./devcontainer ./devcontainer

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py"]