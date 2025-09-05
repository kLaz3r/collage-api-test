# API Reference

Complete reference for the Collage Maker API endpoints, parameters, and responses.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required for this API.

## Endpoints

### GET /

Get API information and available endpoints.

**Response:**

```json
{
    "name": "Collage Maker API",
    "version": "1.0.0",
    "endpoints": {
        "create_collage": "/api/collage/create",
        "get_status": "/api/collage/status/{job_id}",
        "download": "/api/collage/download/{job_id}",
        "list_jobs": "/api/collage/jobs",
        "optimize_grid": "/api/collage/optimize-grid"
    }
}
```

### POST /api/collage/create

Create a new collage from uploaded images.

**Parameters:**

| Parameter             | Type    | Required | Default | Description                                                                                              |
| --------------------- | ------- | -------- | ------- | -------------------------------------------------------------------------------------------------------- |
| files                 | File[]  | Yes      | -       | Image files to include in collage (2-200 files)                                                          |
| width_mm              | float   | No       | 304.8   | Width of output collage in millimeters (50-1219.2)                                                       |
| height_mm             | float   | No       | 457.2   | Height of output collage in millimeters (50-1219.2)                                                      |
| dpi                   | int     | No       | 150     | Resolution in dots per inch (72-300)                                                                     |
| layout_style          | string  | No       | masonry | Layout algorithm: `masonry`, `grid`                                                                      |
| spacing               | float   | No       | 40.0    | Spacing between images and from canvas edges as percentage of canvas (0-100%, where 100% = 5% of canvas) |
| background_color      | string  | No       | #FFFFFF | Background color as hex code                                                                             |
| maintain_aspect_ratio | boolean | No       | true    | Preserve original image aspect ratios                                                                    |
| apply_shadow          | boolean | No       | false   | Add drop shadow effects to images                                                                        |
| output_format         | string  | No       | jpeg    | Output format: `jpeg`, `png`, `tiff`                                                                     |

**Request Examples:**

**Basic JPEG Collage:**

```bash
curl -X POST "http://localhost:8000/api/collage/create" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "width_mm=406.4" \
  -F "height_mm=508" \
  -F "layout_style=masonry" \
  -F "background_color=#F0F0F0"
```

**PNG with Transparency:**

```bash
curl -X POST "http://localhost:8000/api/collage/create" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "output_format=png" \
  -F "background_color=#00000000"
```

**High-Quality TIFF:**

```bash
curl -X POST "http://localhost:8000/api/collage/create" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "output_format=tiff" \
  -F "dpi=300"
```

**Success Response (200):** (CreateCollageResponse)

```json
{
    "job_id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "pending",
    "message": "Collage generation started"
}
```

**Error Responses:**

-   **400 Bad Request** - Invalid parameters or file constraints

    ```json
    {
        "detail": "At least 2 images required"
    }
    ```

-   **400 Bad Request** - File size limits exceeded

    ```json
    {
        "detail": "File image.jpg exceeds 10MB limit"
    }
    ```

-   **400 Bad Request** - Invalid file type
    ```json
    {
        "detail": "File document.pdf is not an image"
    }
    ```

### GET /api/collage/status/{job_id}

Get the status of a collage generation job.

**Parameters:**

| Parameter | Type   | Required | Description             |
| --------- | ------ | -------- | ----------------------- |
| job_id    | string | Yes      | UUID of the collage job |

**Success Response (200):** (CollageJobPublic)

```json
{
    "job_id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "processing",
    "created_at": "2025-08-31T12:30:45.123456",
    "completed_at": null,
    "output_file": null,
    "error_message": null,
    "progress": 50
}
```

**Status Values:**

-   `pending` - Job queued for processing
-   `processing` - Job currently being processed
-   `completed` - Job finished successfully
-   `failed` - Job failed with error

**Error Response (404):**

```json
{
    "detail": "Job not found"
}
```

### GET /api/collage/download/{job_id}

Download the completed collage image.

**Parameters:**

| Parameter | Type   | Required | Description                       |
| --------- | ------ | -------- | --------------------------------- |
| job_id    | string | Yes      | UUID of the completed collage job |

**Success Response (200):** JPEG image file

**Error Responses:**

-   **404 Not Found** - Job not found

    ```json
    {
        "detail": "Job not found"
    }
    ```

-   **400 Bad Request** - Collage not ready

    ```json
    {
        "detail": "Collage not ready yet"
    }
    ```

-   **404 Not Found** - Output file missing
    ```json
    {
        "detail": "Output file not found"
    }
    ```

### GET /api/collage/jobs

List all collage generation jobs.

**Success Response (200):** (List[CollageJobPublic])

```json
[
    {
        "job_id": "123e4567-e89b-12d3-a456-426614174000",
        "status": "completed",
        "created_at": "2025-08-31T12:30:45.123456",
        "completed_at": "2025-08-31T12:31:23.654321",
        "output_file": "collage_123e4567-e89b-12d3-a456-426614174000.jpg",
        "error_message": null,
        "progress": 100
    },
    {
        "job_id": "456e7890-e89b-12d3-a456-426614174001",
        "status": "failed",
        "created_at": "2025-08-31T12:35:12.345678",
        "completed_at": null,
        "output_file": null,
        "error_message": "Invalid image format",
        "progress": 0
    }
]
```

### DELETE /api/collage/cleanup/{job_id}

Clean up temporary files for a completed job.

**Parameters:**

| Parameter | Type   | Required | Description                 |
| --------- | ------ | -------- | --------------------------- |
| job_id    | string | Yes      | UUID of the job to clean up |

**Success Response (200):** (CleanupResponse)

### GET /metrics

Prometheus metrics endpoint.

**Response:** `text/plain; version=0.0.4`

```json
{
    "message": "Job cleaned up successfully"
}
```

**Error Response (404):**

```json
{
    "detail": "Job not found"
}
```

### GET /health

Comprehensive health check endpoint that monitors system status, disk space, active jobs, and dependencies.

**Success Response (200):**

```json
{
    "status": "healthy",
    "timestamp": "2025-08-31T12:30:45.123456",
    "version": "1.0.0",
    "checks": {
        "filesystem": {
            "temp_dir": "/path/to/temp",
            "temp_space_gb": 15.2,
            "output_dir": "/path/to/outputs",
            "output_space_gb": 25.8,
            "healthy": true
        },
        "jobs": {
            "total_jobs": 5,
            "active_jobs": 2,
            "healthy": true
        },
        "dependencies": {
            "magic_available": true,
            "healthy": true
        }
    }
}
```

**Response Fields:**

-   `status`: Overall health status (`"healthy"` or `"unhealthy"`)
-   `timestamp`: ISO 8601 timestamp of the health check
-   `version`: API version
-   `checks`: Detailed health checks for different components
    -   `filesystem`: Disk space monitoring (requires 1GB+ free space)
    -   `jobs`: Active job monitoring (warns if >50 active jobs)
    -   `dependencies`: Required dependency availability

### POST /api/collage/optimize-grid

Calculate optimal grid dimensions and provide recommendations for perfect grid layout.

**Parameters:**

| Parameter  | Type  | Required | Default | Range     | Description                          |
| ---------- | ----- | -------- | ------- | --------- | ------------------------------------ |
| num_images | int   | Yes      | -       | 2-200     | Number of images to analyze          |
| width_mm   | float | No       | 304.8   | 50-1219.2 | Canvas width in millimeters          |
| height_mm  | float | No       | 457.2   | 50-1219.2 | Canvas height in millimeters         |
| dpi        | int   | No       | 150     | 72-300    | Resolution in dots per inch          |
| spacing    | float | No       | 40.0    | 0-100     | Spacing between images as percentage |

**Request Example:**

```bash
curl -X POST "http://localhost:8000/api/collage/optimize-grid" \
  -F "num_images=13" \
  -F "width_mm=304.8" \
  -F "height_mm=457.2" \
  -F "dpi=150" \
  -F "spacing=40.0"
```

**Success Response (200):**

```json
{
    "success": true,
    "optimization": {
        "current_grid": {
            "total_images": 13,
            "cols": 2,
            "rows": 7,
            "images_in_last_row": 1,
            "is_perfect": false
        },
        "closest_perfect_grid": {
            "type": "remove_images",
            "total_images": 12,
            "cols": 2,
            "rows": 6,
            "images_needed": 0,
            "images_to_remove": 1
        },
        "recommendations": {
            "add_images": [
                {
                    "images_needed": 3,
                    "total_images": 16,
                    "cols": 2,
                    "rows": 8
                }
            ],
            "remove_images": [
                {
                    "images_to_remove": 1,
                    "total_images": 12,
                    "cols": 2,
                    "rows": 6
                }
            ]
        },
        "canvas_info": {
            "width": 1800,
            "height": 2700,
            "spacing": 40.0
        }
    },
    "message": "Grid optimization calculated successfully"
}
```

**Use Case:** This endpoint helps frontend applications determine how many images to add or remove to achieve a perfect even grid without incomplete rows, improving the aesthetic quality of grid collages.

## Data Models

### CollageConfig

Configuration object for collage generation.

```python
{
  "width_mm": 304.8,           # 50-1219.2 mm (2-48 inches)
  "height_mm": 457.2,          # 50-1219.2 mm (2-48 inches)
  "dpi": 150,                  # 72-300 DPI
  "layout_style": "masonry",   # "masonry" | "grid"
  "spacing": 40.0,             # 0-100% of canvas dimensions (where 100% = 5% of canvas)
  "background_color": "#FFFFFF", # Hex color code
  "maintain_aspect_ratio": true, # boolean
  "apply_shadow": false        # boolean
}
```

### CollageJob

Job status and metadata.

```python
{
  "job_id": "string",          # UUID
  "status": "pending",         # "pending" | "processing" | "completed" | "failed"
  "created_at": "datetime",    # ISO 8601 timestamp
  "completed_at": "datetime",  # ISO 8601 timestamp or null
  "output_file": "string",     # Filename or null
  "error_message": "string",   # Error description or null
  "progress": 0                # 0-100 percentage
}
```

## File Constraints

-   **Minimum files:** 2 images
-   **Maximum files:** 200 images
-   **Individual file size:** Maximum 10MB per image
-   **Total size:** Maximum 500MB for all files combined
-   **Supported formats:** JPEG, PNG, GIF, BMP, TIFF, WebP
-   **Image dimensions:** No specific limits, but very large images may impact processing time

## Rate Limiting

Rate limiting is implemented to prevent abuse:

-   **Limit:** 100 requests per minute per IP address
-   **Window:** 60 seconds sliding window
-   **Response:** HTTP 429 (Too Many Requests) when limit exceeded

```json
{
    "error": "Rate limit exceeded. Please try again later."
}
```

## Error Handling

All endpoints return appropriate HTTP status codes and JSON error messages. Common error patterns:

-   **400 Bad Request:** Invalid input parameters or file constraints
-   **404 Not Found:** Resource not found (job, file)
-   **429 Too Many Requests:** Rate limit exceeded
-   **500 Internal Server Error:** Server-side processing errors

## Security Features

The API includes several security measures:

### File Upload Security

-   **Magic Number Validation:** Files are validated using magic numbers to ensure they are actual images
-   **Filename Sanitization:** Dangerous characters are removed from filenames to prevent path traversal
-   **Size Limits:** Individual files limited to 10MB, total upload limited to 500MB
-   **Format Validation:** Only supported image formats (JPEG, PNG, GIF, BMP, TIFF, WebP) are accepted

### Request Security

-   **Rate Limiting:** 100 requests per minute per IP address
-   **Security Headers:** Automatic addition of security headers on all responses:
    -   `X-Content-Type-Options: nosniff`
    -   `X-Frame-Options: DENY`
    -   `X-XSS-Protection: 1; mode=block`
    -   `Referrer-Policy: strict-origin-when-cross-origin`
    -   `Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'`

### Input Validation

-   **Parameter Validation:** All input parameters are validated using Pydantic models
-   **Hex Color Validation:** Background colors must be valid 6-digit hex codes
-   **Range Validation:** Numeric parameters are constrained to reasonable ranges

## Content Types

-   **Request:** `multipart/form-data` for file uploads, `application/json` for other requests
-   **Response:** `application/json` for API responses, `image/jpeg` for collage downloads
