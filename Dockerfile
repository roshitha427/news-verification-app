FROM node:24-alpine AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_ENV=production COOKIE_SECURE=true DATABASE_BACKEND=mongo
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt
COPY backend ./backend
COPY ml ./ml
COPY --from=web /web/dist ./frontend/dist
RUN useradd --create-home appuser && mkdir -p /app/data /app/models && chown -R appuser:appuser /app
RUN install -d -m 0700 -o appuser -g appuser /home/appuser/.ssh
USER appuser
EXPOSE 5000
# Mount model artifacts read-only at /app/models; provide secrets through the environment.
CMD ["waitress-serve", "--listen=0.0.0.0:5000", "--threads=8", "--call", "backend.app:create_server_app"]
