FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt .
COPY wheelhouse/ ./wheelhouse/

RUN pip install --no-cache-dir --no-index --find-links=./wheelhouse -r requirements-docker.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]