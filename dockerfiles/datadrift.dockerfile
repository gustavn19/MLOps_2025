FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

RUN pip install fastapi evidently numpy pandas google-cloud-storage --no-cache-dir

COPY src/pokedec/data_drift.py /app/data_drift.py
COPY src/pokedec/image_analysis.py /app/image_analysis.py

EXPOSE $PORT

# CMD exec uvicorn data_drift:app --port $PORT --host 0.0.0.0

ENTRYPOINT ["sh", "-c", "uvicorn data_drift:app --server.port $PORT --server.address= 0.0.0.0"]
