FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY requirements_frontend.txt /app/requirements_frontend.txt
COPY src/my_project/frontend.py /app/frontend.py
COPY pokemon_labels.json /app/pokemon_labels.json

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend.txt

EXPOSE $PORT

ENTRYPOINT ["sh", "-c", "streamlit run frontend.py --server.port $PORT --server.address=0.0.0.0"]
