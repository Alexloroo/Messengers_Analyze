FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for asyncpg and other native packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Telethon session and SQLite DB are stored here by default.
RUN mkdir -p /app/telethon_data

CMD ["python", "/app/listener.py"]
