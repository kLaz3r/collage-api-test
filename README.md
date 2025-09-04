# Collage Maker API

A high-performance FastAPI application for creating beautiful photo collages with multiple layout styles and customizable options.

## Features

-   **Multiple Layout Styles**: Masonry and Grid layouts
-   **High-Resolution Output**: Configurable DPI up to 300 for print-quality results
-   **Smart Image Processing**: Aspect ratio preservation, intelligent cropping, and shadow effects
-   **Background Jobs via Celery**: Offloaded collage generation with Redis broker
-   **Redis-backed Job Status**: Durable job progress with TTL cleanup
-   **RESTful API**: Clean, well-documented endpoints with OpenAPI/Swagger support
-   **File Management**: Automatic cleanup and size validation
-   **CORS Support**: Ready for web application integration
-   **🔒 Security First**: Magic number validation, rate limiting, and security headers
-   **📊 Production Ready**: Health checks and structured settings via environment

## Quick Start

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Configure (optional)

Create a `.env` file or export env vars (defaults shown):

```bash
# App
APP_APP_NAME="Collage Maker API"
APP_APP_VERSION="1.0.0"
APP_HOST=0.0.0.0
APP_PORT=8000

# Paths
APP_UPLOAD_DIR=uploads
APP_OUTPUT_DIR=outputs
APP_TEMP_DIR=temp

# Limits
APP_MAX_IMAGE_SIZE=10485760          # 10 MB
APP_MAX_TOTAL_SIZE=524288000         # 500 MB
APP_MAX_CANVAS_PIXELS=250000000

# Rate limiting
APP_RATE_LIMIT_REQUESTS=10
APP_RATE_LIMIT_WINDOW_SECONDS=60

# Redis (for jobs)
APP_REDIS_URL=redis://localhost:6379/0   # or set APP_REDIS_HOST/PORT/DB
APP_REDIS_HOST=localhost
APP_REDIS_PORT=6379
APP_REDIS_DB=0
APP_JOB_TTL_SECONDS=3600
APP_CLEANUP_INTERVAL_SECONDS=600

# CORS
APP_CORS_ALLOW_ORIGINS=["*"]
APP_CORS_ALLOW_CREDENTIALS=true
APP_CORS_ALLOW_METHODS=["*"]
APP_CORS_ALLOW_HEADERS=["*"]
```

### 3) Run Redis and the worker

```bash
# Start Redis (example)
docker run -p 6379:6379 redis:7

# Start Celery worker
./venv/bin/celery -A celery_app.celery_app worker -l info
```

### 4) Run the API

```bash
python server.py
# or
./venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
```

API available at `http://localhost:8000`.

## Basic Usage

```bash
# Create a collage with default settings
curl -X POST "http://localhost:8000/api/collage/create" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"

# Response: {"job_id":"<uuid>","status":"pending"}

# Check job status
curl "http://localhost:8000/api/collage/status/<job_id>"

# Download when completed
curl -OJ "http://localhost:8000/api/collage/download/<job_id>"
```

## API Endpoints

-   `GET /` — API information
-   `POST /api/collage/create` — Create a new collage
-   `GET /api/collage/status/{job_id}` — Check job status
-   `GET /api/collage/download/{job_id}` — Download completed collage
-   `GET /api/collage/jobs` — List jobs
-   `DELETE /api/collage/cleanup/{job_id}` — Cleanup job files
-   `GET /health` — Health status

## Documentation

-   Interactive Docs: `http://localhost:8000/docs`
-   ReDoc: `http://localhost:8000/redoc`
-   OpenAPI JSON: `http://localhost:8000/openapi.json`

For detailed guides, see `docs/`.

## License

MIT License - see `LICENSE`.
