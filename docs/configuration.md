# Configuration (Concise)

## Request Parameters (common)

-   `layout_style`: `masonry` (default) | `grid`
-   `width_mm`, `height_mm`: physical size (defaults 304.8 × 457.2)
-   `dpi`: 72–300 (default 150)
-   `spacing`: 0–100 (default 40.0)
-   `background_color`: hex (#RRGGBB or #RRGGBBAA); use `#00000000` for PNG transparency
-   `maintain_aspect_ratio`: true/false
-   `apply_shadow`: true/false
-   `output_format`: `jpeg` | `png` | `tiff`

## Environment Variables (essential)

-   `APP_REDIS_URL`: e.g., `redis://localhost:6379/0`
-   `APP_CORS_ALLOW_ORIGINS`: JSON-like list of allowed origins
-   Limits: `APP_MAX_IMAGE_SIZE`, `APP_MAX_TOTAL_SIZE`, `APP_MAX_CANVAS_PIXELS`
-   Preflight: `APP_PREFLIGHT_ENABLED`, `APP_PREFLIGHT_MAX_TOTAL_SOURCE_PIXELS`
-   Pre-resize: `APP_PRE_RESIZE_ENABLED`, `APP_PRE_RESIZE_MAX_DIM`
-   Logging: `APP_LOG_TO_FILE`, `APP_LOG_FILE_PATH`, `APP_LOG_LEVEL`

## Tips

-   For PNG transparency, set `background_color=#00000000` and `output_format=png`.
-   For large inputs, enable pre-resize (default on) to keep memory stable.
