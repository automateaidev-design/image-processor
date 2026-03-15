FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV U2NET_HOME=/app/models

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models

# Pre-download BRIA RMBG-1.4 — same model as rembg.com
RUN python -c "from rembg import new_session; new_session('bria-rmbg')"

COPY app.py .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
