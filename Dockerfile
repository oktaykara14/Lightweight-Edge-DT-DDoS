FROM python:3.11-slim

# System networking and monitoring utilities
RUN apt-get update && apt-get install -y \
    procps \
    tcpdump \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Create standard directory structure
RUN mkdir -p data/raw data/processed models figures src

COPY . .

CMD ["tail", "-f", "/dev/null"]
