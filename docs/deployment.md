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

## Windows Development

On Windows, Celery workers may encounter permission errors with multiprocessing. The application automatically detects Windows and uses the `solo` worker pool to avoid these issues.

### Windows-specific configuration

You can override the default Windows configuration by setting these environment variables:

```bash
# Use threads instead of solo pool (alternative for Windows)
APP_CELERY_WORKER_POOL=threads
APP_CELERY_WORKER_CONCURRENCY=4

# Or force prefork pool (may cause permission errors)
APP_CELERY_WORKER_POOL=prefork
APP_CELERY_WORKER_CONCURRENCY=2
```

### Troubleshooting Windows Issues

If you encounter `PermissionError: [WinError 5] Access is denied` errors:

1. **Use the default configuration**: The application automatically uses `solo` pool on Windows
2. **Run as Administrator**: Try running your terminal/IDE as Administrator
3. **Use threads pool**: Set `APP_CELERY_WORKER_POOL=threads` in your `.env` file
4. **Check antivirus**: Some antivirus software may block process creation

### Windows-specific `.env` example:

```bash
APP_REDIS_URL=redis://localhost:6379/0
APP_CORS_ALLOW_ORIGINS=["http://localhost","http://localhost:3000"]
# Windows-specific Celery configuration
APP_CELERY_WORKER_POOL=solo
APP_CELERY_WORKER_CONCURRENCY=1
```
