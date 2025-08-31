# Collage Maker API

A high-performance FastAPI application for creating beautiful photo collages with multiple layout styles and customizable options.

## Features

-   **Multiple Layout Styles**: Masonry, Grid, Random, and Spiral layouts
-   **High-Resolution Output**: Configurable DPI up to 300 for print-quality results
-   **Smart Image Processing**: Aspect ratio preservation, intelligent cropping, and shadow effects
-   **Asynchronous Processing**: Background job processing for large collage generation
-   **RESTful API**: Clean, well-documented endpoints with OpenAPI/Swagger support
-   **File Management**: Automatic cleanup and size validation
-   **CORS Support**: Ready for web application integration

## Quick Start

### Installation

1. Clone the repository:

```bash
git clone https://github.com/kLaz3r/collage-api-test.git
cd collage-api-test
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
python server.py
```

The API will be available at `http://localhost:8000`

### Basic Usage

```bash
# Create a collage with default settings
curl -X POST "http://localhost:8000/api/collage/create" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"

# Response: {"job_id": "123e4567-e89b-12d3-a456-426614174000", "status": "pending"}
```

## API Endpoints

### Core Endpoints

-   `GET /` - API information and available endpoints
-   `POST /api/collage/create` - Create a new collage
-   `GET /api/collage/status/{job_id}` - Check job status
-   `GET /api/collage/download/{job_id}` - Download completed collage
-   `GET /api/collage/jobs` - List all jobs
-   `DELETE /api/collage/cleanup/{job_id}` - Clean up job files

### Health Check

-   `GET /health` - Service health status

## Documentation

### 📚 Complete Documentation Suite

-   **[Interactive Documentation](./docs/index.html)** - Live API explorer with status monitoring
-   **[API Reference](./docs/api-reference.md)** - Complete endpoint reference with examples
-   **[Configuration Guide](./docs/configuration.md)** - Layout styles and customization options
-   **[Usage Examples](./docs/examples.md)** - Code samples in Python, JavaScript, PHP, and cURL
-   **[Developer Guide](./docs/developer-guide.md)** - Architecture and extension guide
-   **[Deployment Guide](./docs/deployment.md)** - Production deployment and scaling
-   **[Troubleshooting](./docs/troubleshooting.md)** - Common issues and solutions

### 🚀 Interactive API Explorer

When running the server, visit:

-   **Swagger UI**: `http://localhost:8000/docs` - Interactive API testing
-   **ReDoc**: `http://localhost:8000/redoc` - Clean documentation
-   **OpenAPI JSON**: `http://localhost:8000/openapi.json` - API specification

## License

MIT License - see [LICENSE](LICENSE) file for details.
