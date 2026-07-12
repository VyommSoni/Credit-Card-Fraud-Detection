FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 3. Upgrade pip to the latest version (fixing that 23.0.1 -> 26.1.2 notice) and install packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your app code
COPY . .

EXPOSE 8080

CMD ["python", "app.py"]