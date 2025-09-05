# Deployment (Docker) and Dev Quickstart

## Docker (Production)

-   Requirements: Docker, Docker Compose, `.env` (set at least `APP_REDIS_URL`)

### docker-compose.yml (API + Celery worker + Redis)

```yaml
over  sion: "3.8"
services:
    api:
        build: .
        ports:
            - "8000:8000"
        env_file: .env
        depends_on:
            - redis
        restart: unless-stopped

    worker:
        build: .
        command: celery -A celery_app.celery_app worker -l info
        env_file: .env
        depends_on:
            - redis
        restart: unless-stopped

    redis:
        image: redis:7
        ports:
            - "6379:6379"
        restart: unless-stopped
```

Run:

```bash
# Build image
docker build -t collage-api .

# Start services
docker compose up -d

# Tail API logs
docker compose logs -f api
```

Notes:

-   The container exposes port 8000 and includes a HEALTHCHECK on `/health`.
-   Gunicorn is the default process (see `gunicorn.conf.py`).
-   Metrics at `http://localhost:8000/metrics`.

## Dev (simple)

-   Start Redis via Docker:

```bash
docker run -d --name redis -p 6379:6379 redis:7
```

-   Start API with auto-reload:

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

-   Start Celery worker:

```bash
celery -A celery_app.celery_app worker -l info
```

Minimal `.env` example:

```bash
APP_REDIS_URL=redis://localhost:6379/0
APP_CORS_ALLOW_ORIGINS=["http://localhost","http://localhost:3000"]
```
