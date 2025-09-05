# Developer Guide (Concise)

## Architecture

-   FastAPI app (`server.py`)
-   Celery worker (`tasks.py`) with Redis broker/backend (`celery_app.py`)
-   Collage generation: `CollageGenerator`, `MasonryPacker`, `GridPacker`
-   Jobs stored in Redis with TTL; periodic cleanup removes stale files
-   Logging: JSON-ish events + optional rotating file handler
-   Metrics: Prometheus at `/metrics`

## Add a new layout

1. Create a packer class with `pack_images(...) -> List[ImageBlock]`.
2. Add layout to `LayoutStyle` and selection switch.

## Add an image effect

1. Implement method on `CollageGenerator`.
2. Add a config flag; apply during `generate`.

## Local run

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
celery -A celery_app.celery_app worker -l info
```

## Tests

```bash
pytest -q
```

## Code pointers

-   Upload path & validation: `create_collage()` in `server.py`
-   Generation pipeline: `CollageGenerator.generate()`
-   Redis helpers: `save_job`, `update_job`, `get_job`, `list_all_jobs`
