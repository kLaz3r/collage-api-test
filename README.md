# Collage Maker API (Concise)

FastAPI service to create photo collages. Docker-first. Celery + Redis for background jobs.

## Deploy (Docker)

Requirements: Docker, Docker Compose, `.env` (set at least `APP_REDIS_URL`).

Compose (API + worker + Redis):

```yaml
version: "3.8"
services:
    api:
        build: .
        ports: ["8000:8000"]
        env_file: .env
        depends_on: [redis]
        restart: unless-stopped
    worker:
        build: .
        command: celery -A celery_app.celery_app worker -l info
        env_file: .env
        depends_on: [redis]
        restart: unless-stopped
    redis:
        image: redis:7
        ports: ["6379:6379"]
        restart: unless-stopped
```

Run:

```bash
docker build -t collage-api .
docker compose up -d
```

Health: `GET /health` • Metrics: `GET /metrics` • Docs: `/docs`

## Dev (simple)

```bash
# Redis
docker run -d --name redis -p 6379:6379 redis:7
# API (hot reload)
uvicorn server:app --reload --host 0.0.0.0 --port 8000
# Worker
celery -A celery_app.celery_app worker -l info
```

## Minimal Usage

```bash
# Create job
curl -X POST "http://localhost:8000/api/collage/create" \
  -F "files=@a.jpg" -F "files=@b.jpg"
# Status
curl "http://localhost:8000/api/collage/status/<job_id>"
# Download
curl -OJ "http://localhost:8000/api/collage/download/<job_id>"
```

## Key Endpoints

-   `POST /api/collage/create` → { job_id, status }
-   `GET /api/collage/status/{job_id}` → job info
-   `GET /api/collage/download/{job_id}` → image
-   `GET /api/collage/jobs` → list jobs
-   `DELETE /api/collage/cleanup/{job_id}`

## Configuration (env)

Common:

-   `APP_REDIS_URL` (e.g., `redis://localhost:6379/0`)
-   `APP_CORS_ALLOW_ORIGINS` (JSON-like list)
-   Limits: `APP_MAX_IMAGE_SIZE`, `APP_MAX_TOTAL_SIZE`, `APP_MAX_CANVAS_PIXELS`
-   Preflight: `APP_PREFLIGHT_ENABLED`, `APP_PREFLIGHT_MAX_TOTAL_SOURCE_PIXELS`
-   Pre-resize: `APP_PRE_RESIZE_ENABLED`, `APP_PRE_RESIZE_MAX_DIM`
-   Logging: `APP_LOG_TO_FILE`, `APP_LOG_FILE_PATH`

Notes: EXIF orientation normalized; optional pre-resize for huge inputs; Redis TTL cleanup; strict security headers by default.
