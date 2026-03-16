FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8000

CMD ["sh", "-c", "gunicorn app:app -k uvicorn.workers.UvicornWorker -w 4 -t 120 -b 0.0.0.0:${PORT}"]
