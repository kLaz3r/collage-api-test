# API Reference (Detailed)

Base URL: `http://localhost:8000`

## Data Types

### Enums

#### LayoutStyle

```json
{
    "type": "string",
    "enum": ["masonry", "grid"],
    "description": "Layout algorithm for photo arrangement"
}
```

#### OutputFormat

```json
{
    "type": "string",
    "enum": ["jpeg", "png", "tiff"],
    "description": "Output image format"
}
```

#### JobStatus

```json
{
    "type": "string",
    "enum": ["pending", "processing", "completed", "failed"],
    "description": "Status of collage generation job"
}
```

### Request Models

#### CollageConfig (for mm-based dimensions)

```json
{
    "width_mm": {
        "type": "number",
        "format": "float",
        "minimum": 50.0,
        "maximum": 1219.2,
        "default": 304.8,
        "description": "Canvas width in millimeters (2-48 inches)"
    },
    "height_mm": {
        "type": "number",
        "format": "float",
        "minimum": 50.0,
        "maximum": 1219.2,
        "default": 457.2,
        "description": "Canvas height in millimeters (2-48 inches)"
    },
    "dpi": {
        "type": "integer",
        "minimum": 72,
        "maximum": 300,
        "default": 150,
        "description": "Resolution in dots per inch"
    },
    "layout_style": {
        "$ref": "#/definitions/LayoutStyle",
        "default": "masonry"
    },
    "spacing": {
        "type": "number",
        "format": "float",
        "minimum": 0.0,
        "maximum": 100.0,
        "default": 40.0,
        "description": "Spacing as percentage of canvas dimensions"
    },
    "background_color": {
        "type": "string",
        "pattern": "^#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?$",
        "default": "#FFFFFF",
        "description": "Hex color code with optional alpha (#RRGGBB or #RRGGBBAA)"
    },
    "maintain_aspect_ratio": {
        "type": "boolean",
        "default": true,
        "description": "Whether to preserve photo aspect ratios"
    },
    "apply_shadow": {
        "type": "boolean",
        "default": false,
        "description": "Add drop shadows to photos"
    },
    "output_format": {
        "$ref": "#/definitions/OutputFormat",
        "default": "jpeg"
    },
    "face_aware_cropping": {
        "type": "boolean",
        "default": false,
        "description": "Enable face-aware cropping using MediaPipe"
    },
    "face_margin": {
        "type": "number",
        "format": "float",
        "minimum": 0.0,
        "maximum": 0.3,
        "default": 0.08,
        "description": "Margin around detected faces (0.0-0.3)"
    },
    "pretrim_borders": {
        "type": "boolean",
        "default": false,
        "description": "Trim solid uniform borders from photos"
    }
}
```

#### CollagePixelConfig (for pixel-based dimensions)

```json
{
    "width_px": {
        "type": "integer",
        "minimum": 320,
        "maximum": 20000,
        "default": 1920,
        "description": "Canvas width in pixels"
    },
    "height_px": {
        "type": "integer",
        "minimum": 320,
        "maximum": 20000,
        "default": 1080,
        "description": "Canvas height in pixels"
    },
    "dpi": {
        "type": "integer",
        "minimum": 72,
        "maximum": 300,
        "default": 96,
        "description": "Resolution in dots per inch"
    },
    "layout_style": {
        "$ref": "#/definitions/LayoutStyle",
        "default": "masonry"
    },
    "spacing": {
        "type": "number",
        "format": "float",
        "minimum": 0.0,
        "maximum": 100.0,
        "default": 40.0,
        "description": "Spacing as percentage of canvas dimensions"
    },
    "background_color": {
        "type": "string",
        "pattern": "^#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?$",
        "default": "#FFFFFF",
        "description": "Hex color code with optional alpha (#RRGGBB or #RRGGBBAA)"
    },
    "maintain_aspect_ratio": {
        "type": "boolean",
        "default": true,
        "description": "Whether to preserve photo aspect ratios"
    },
    "apply_shadow": {
        "type": "boolean",
        "default": false,
        "description": "Add drop shadows to photos"
    },
    "output_format": {
        "$ref": "#/definitions/OutputFormat",
        "default": "jpeg"
    },
    "face_aware_cropping": {
        "type": "boolean",
        "default": false,
        "description": "Enable face-aware cropping using MediaPipe"
    },
    "face_margin": {
        "type": "number",
        "format": "float",
        "minimum": 0.0,
        "maximum": 0.3,
        "default": 0.08,
        "description": "Margin around detected faces (0.0-0.3)"
    },
    "pretrim_borders": {
        "type": "boolean",
        "default": false,
        "description": "Trim solid uniform borders from photos"
    }
}
```

### Response Models

#### CreateCollageResponse

```json
{
    "job_id": {
        "type": "string",
        "format": "uuid",
        "description": "Unique identifier for the collage job"
    },
    "status": {
        "type": "string",
        "enum": ["pending", "processing", "completed", "failed"],
        "description": "Current job status"
    },
    "message": {
        "type": "string",
        "description": "Human-readable status message"
    }
}
```

#### CollageJobPublic

```json
{
    "job_id": {
        "type": "string",
        "format": "uuid",
        "description": "Unique identifier for the job"
    },
    "status": {
        "$ref": "#/definitions/JobStatus"
    },
    "created_at": {
        "type": "string",
        "format": "date-time",
        "description": "Job creation timestamp"
    },
    "completed_at": {
        "type": "string",
        "format": "date-time",
        "nullable": true,
        "description": "Job completion timestamp"
    },
    "output_file": {
        "type": "string",
        "nullable": true,
        "description": "Name of the generated collage file"
    },
    "error_message": {
        "type": "string",
        "nullable": true,
        "description": "Error message if job failed"
    },
    "progress": {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "default": 0,
        "description": "Job completion percentage"
    }
}
```

#### CleanupResponse

```json
{
    "message": {
        "type": "string",
        "description": "Success message"
    }
}
```

#### HealthCheckResponse

```json
{
    "status": {
        "type": "string",
        "enum": ["healthy", "unhealthy"],
        "description": "Overall health status"
    },
    "timestamp": {
        "type": "string",
        "format": "date-time",
        "description": "Timestamp of health check"
    },
    "version": {
        "type": "string",
        "description": "Application version"
    },
    "checks": {
        "type": "object",
        "properties": {
            "filesystem": {
                "type": "object",
                "properties": {
                    "temp_dir": { "type": "string" },
                    "temp_space_gb": { "type": "number" },
                    "output_dir": { "type": "string" },
                    "output_space_gb": { "type": "number" },
                    "healthy": { "type": "boolean" }
                }
            },
            "jobs": {
                "type": "object",
                "properties": {
                    "total_jobs": { "type": "integer" },
                    "active_jobs": { "type": "integer" },
                    "healthy": { "type": "boolean" }
                }
            },
            "dependencies": {
                "type": "object",
                "properties": {
                    "magic_available": { "type": "boolean" },
                    "redis_connected": { "type": "boolean" },
                    "healthy": { "type": "boolean" }
                }
            }
        }
    }
}
```

## Endpoints

### Create Collage (mm-based dimensions)

**Endpoint:** `POST /api/collage/create`

Creates a new collage using millimeter-based dimensions. Supports advanced features like face-aware cropping and background trimming.

#### Parameters

| Parameter               | Type    | Required | Default   | Description                           |
| ----------------------- | ------- | -------- | --------- | ------------------------------------- |
| `files`                 | File[]  | Yes      | -         | Array of image files (min 2, max 200) |
| `width_mm`              | number  | No       | 304.8     | Canvas width in mm (50.0-1219.2)      |
| `height_mm`             | number  | No       | 457.2     | Canvas height in mm (50.0-1219.2)     |
| `dpi`                   | integer | No       | 150       | Image resolution (72-300)             |
| `layout_style`          | string  | No       | "masonry" | "masonry" or "grid"                   |
| `spacing`               | number  | No       | 40.0      | Spacing as percentage (0.0-100.0)     |
| `background_color`      | string  | No       | "#FFFFFF" | Hex color (#RRGGBB or #RRGGBBAA)      |
| `maintain_aspect_ratio` | boolean | No       | true      | Preserve photo aspect ratios          |
| `apply_shadow`          | boolean | No       | false     | Add drop shadows to photos            |
| `output_format`         | string  | No       | "jpeg"    | "jpeg", "png", or "tiff"              |
| `face_aware_cropping`   | boolean | No       | false     | Enable MediaPipe face detection       |
| `face_margin`           | number  | No       | 0.08      | Margin around faces (0.0-0.3)         |
| `pretrim_borders`       | boolean | No       | false     | Trim solid uniform borders            |

#### Responses

**Success Response (200 OK)**

```json
{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "pending",
    "message": "Collage generation started"
}
```

**Error Responses**

-   `400 Bad Request` - Invalid parameters or missing files
-   `413 Payload Too Large` - File size limits exceeded
-   `422 Unprocessable Entity` - Validation errors
-   `500 Internal Server Error` - Server error

### Create Collage (pixel-based dimensions)

**Endpoint:** `POST /api/collage/create-pixels`

Creates a new collage using pixel-based dimensions. Optimized for digital displays and web usage.

#### Parameters

| Parameter               | Type    | Required | Default   | Description                           |
| ----------------------- | ------- | -------- | --------- | ------------------------------------- |
| `files`                 | File[]  | Yes      | -         | Array of image files (min 2, max 200) |
| `width_px`              | integer | No       | 1920      | Canvas width in pixels (320-20000)    |
| `height_px`             | integer | No       | 1080      | Canvas height in pixels (320-20000)   |
| `dpi`                   | integer | No       | 96        | Image resolution (72-300)             |
| `layout_style`          | string  | No       | "masonry" | "masonry" or "grid"                   |
| `spacing`               | number  | No       | 40.0      | Spacing as percentage (0.0-100.0)     |
| `background_color`      | string  | No       | "#FFFFFF" | Hex color (#RRGGBB or #RRGGBBAA)      |
| `maintain_aspect_ratio` | boolean | No       | true      | Preserve photo aspect ratios          |
| `apply_shadow`          | boolean | No       | false     | Add drop shadows to photos            |
| `output_format`         | string  | No       | "jpeg"    | "jpeg", "png", or "tiff"              |
| `face_aware_cropping`   | boolean | No       | false     | Enable MediaPipe face detection       |
| `face_margin`           | number  | No       | 0.08      | Margin around faces (0.0-0.3)         |
| `pretrim_borders`       | boolean | No       | false     | Trim solid uniform borders            |

### Get Job Status

**Endpoint:** `GET /api/collage/status/{job_id}`

Retrieves the current status and progress of a collage generation job.

#### Parameters

| Parameter | Type          | Required   | Description           |
| --------- | ------------- | ---------- | --------------------- |
| `job_id`  | string (UUID) | Yes (path) | Unique job identifier |

**Error Responses**

-   `404 Not Found` - Job not found

### Download Collage

**Endpoint:** `GET /api/collage/download/{job_id}`

Downloads the completed collage image file.

#### Parameters

| Parameter | Type          | Required   | Description           |
| --------- | ------------- | ---------- | --------------------- |
| `job_id`  | string (UUID) | Yes (path) | Unique job identifier |

#### Response Headers

-   `Content-Type` - Depends on output format (image/jpeg, image/png, or image/tiff)
-   `Content-Disposition` - Attachment with filename
-   `Content-Length` - File size in bytes
-   `ETag` - Weak ETag based on file mtime and size
-   `Cache-Control` - Set to "private, max-age=31536000, immutable"

**Error Responses**

-   `400 Bad Request` - Job not ready or failed
-   `404 Not Found` - Job not found or file missing

### List Jobs

**Endpoint:** `GET /api/collage/jobs`

Returns a list of all collage generation jobs currently stored in the system.

### Clean Up Job

**Endpoint:** `DELETE /api/collage/cleanup/{job_id}`

Removes temporary files and output for a completed job to free up disk space.

#### Parameters

| Parameter | Type          | Required   | Description           |
| --------- | ------------- | ---------- | --------------------- |
| `job_id`  | string (UUID) | Yes (path) | Unique job identifier |

**Error Responses**

-   `404 Not Found` - Job not found

### Health Check

**Endpoint:** `GET /health`

Comprehensive health check endpoint for monitoring service status.

### Metrics

**Endpoint:** `GET /metrics`

Exposes Prometheus-style metrics for monitoring.

**Response Format**
Plain text format compatible with Prometheus monitoring.

### Optimize Grid Layout

**Endpoint:** `POST /api/collage/optimize-grid`

Calculates optimal grid dimensions and recommends how many images to add or remove for a perfect grid.

#### Parameters

| Parameter    | Type    | Required | Default | Description              |
| ------------ | ------- | -------- | ------- | ------------------------ |
| `num_images` | integer | Yes      | -       | Number of images (2-200) |
| `width_mm`   | number  | No       | 304.8   | Canvas width in mm       |
| `height_mm`  | number  | No       | 457.2   | Canvas height in mm      |
| `dpi`        | integer | No       | 150     | Image resolution         |
| `spacing`    | number  | No       | 40.0    | Spacing percentage       |

### Analyze Masonry Layout

**Endpoint:** `POST /api/collage/analyze-masonry`

Analyzes how the masonry layout algorithm would distribute images without generating the actual collage.

#### Parameters

Same as optimize-grid endpoint.

### Root Endpoint

**Endpoint:** `GET /`

Provides basic API information and available endpoints.

## Notes

-   Upload limits: per file 10MB, total 500MB (defaults; configurable).
-   Preflight pixel budget and optional pre-resize protect memory usage.
-   PNG transparent background: use `background_color=#00000000`.
