# API Reference (Concise)

Base URL: `http://localhost:8000`

## Core Endpoints

-   `POST /api/collage/create`

    -   Form-data: `files[]` (2–200 images). Optional: `width_mm`, `height_mm`, `dpi`, `layout_style` (masonry|grid), `spacing`, `background_color`, `maintain_aspect_ratio`, `apply_shadow`, `output_format` (jpeg|png|tiff)
    -   Returns (CreateCollageResponse): `{ job_id, status, message }`

-   `GET /api/collage/status/{job_id}` → (CollageJobPublic)
-   `GET /api/collage/download/{job_id}` → image (with `Content-Disposition`, `ETag`, `Cache-Control`)
-   `GET /api/collage/jobs` → `List[CollageJobPublic]`
-   `DELETE /api/collage/cleanup/{job_id}` → `{ message }`
-   `GET /health` → service health
-   `GET /metrics` → Prometheus metrics

## Minimal Examples

Create:

```bash
curl -X POST "http://localhost:8000/api/collage/create" \
  -F "files=@a.jpg" -F "files=@b.jpg" \
  -F "layout_style=masonry" -F "output_format=jpeg"
```

Status:

```bash
curl "http://localhost:8000/api/collage/status/<job_id>"
```

Download:

```bash
curl -OJ "http://localhost:8000/api/collage/download/<job_id>"
```

## Notes

-   Upload limits: per file 10MB, total 500MB (defaults; configurable).
-   Preflight pixel budget and optional pre-resize protect memory usage.
-   PNG transparent background: use `background_color=#00000000`.
