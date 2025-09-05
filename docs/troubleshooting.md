# Troubleshooting (Concise)

## Common issues

### 1) Unregistered Celery task

Error: `Received unregistered task 'tasks.generate_collage_task'`

-   Ensure worker imports tasks: `include=['tasks']` in `celery_app.py`.
-   Restart worker: `pkill -f 'celery.*collage_worker' || true && celery -A celery_app.celery_app worker -l info`

### 2) Event loop is closed (Redis async in worker)

-   Use synchronous Redis client inside Celery tasks (already implemented in `tasks.py`).

### 3) 400 invalid image / file too large

-   Confirm file is a real image; limits: 10MB per file, 500MB total (configurable).
-   Transparent PNG: set `background_color=#00000000` and `output_format=png`.

### 4) Collage not ready yet

-   Status is not `completed`. Wait for worker; check `/api/collage/status/{job_id}`.

### 5) Docker healthcheck failing

-   Check logs: `docker compose logs -f api`.
-   Verify Redis connectivity (`APP_REDIS_URL`).

### 6) CORS blocked

-   Set `APP_CORS_ALLOW_ORIGINS=["https://yourapp.com"]`.

### 7) High memory usage

-   Enable pre-resize (default); reduce DPI/size; limit images.
