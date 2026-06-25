FROM python:3.10-slim

WORKDIR /app

# Dependencies install karo
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt
# Code copy karo
COPY src/ ./src/
COPY models/ ./models/
COPY outputs/ ./outputs/

# Port expose karo
EXPOSE 8000

# FastAPI start karo
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]