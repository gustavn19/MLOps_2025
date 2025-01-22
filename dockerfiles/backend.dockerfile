# Base image
FROM python:3.12-slim

EXPOSE $PORT

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY requirements_backend.txt /app/requirements_backend.txt
COPY src/pokedec/backend.py /app/backend.py
COPY models/model_best.onnx models/model_best.onnx

RUN pip install -r requirements_backend.txt --no-cache-dir
RUN pip install pydantic

CMD exec uvicorn backend:app --port $PORT --host 0.0.0.0
