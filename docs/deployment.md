# Deployment Guide

Complete guide for deploying the Collage Maker API to production environments.

## Local Development Setup

### Prerequisites

-   Python 3.8+
-   pip package manager
-   Git

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/kLaz3r/collage-api-test.git
cd collage-api-test
```

2. **Create virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the application:**

```bash
python server.py
```

The API will be available at `http://localhost:8000`

### Development with Hot Reload

For development with automatic reloading:

```bash
# Install additional dev dependencies
pip install -r requirements.txt

# Run with uvicorn reload
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

## Production Deployment

### Option 1: Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs temp

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "server.py"]
```

#### Docker Compose

```yaml
version: "3.8"

services:
    collage-api:
        build: .
        ports:
            - "8000:8000"
        volumes:
            - ./outputs:/app/outputs
            - ./uploads:/app/uploads
        environment:
            - ENVIRONMENT=production
        restart: unless-stopped
        healthcheck:
            test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
            interval: 30s
            timeout: 10s
            retries: 3
```

#### Build and Run

```bash
# Build the image
docker build -t collage-api .

# Run with Docker Compose
docker-compose up -d

# Or run directly
docker run -p 8000:8000 -v $(pwd)/outputs:/app/outputs collage-api
```

### Option 2: Systemd Service

#### Create Systemd Service File

```bash
sudo nano /etc/systemd/system/collage-api.service
```

```ini
[Unit]
Description=Collage Maker API
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/path/to/collage-api
Environment=PATH=/path/to/venv/bin
ExecStart=/path/to/venv/bin/python server.py
Restart=always
RestartSec=5

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=/path/to/collage-api/uploads /path/to/collage-api/outputs /path/to/collage-api/temp

[Install]
WantedBy=multi-user.target
```

#### Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable collage-api

# Start service
sudo systemctl start collage-api

# Check status
sudo systemctl status collage-api

# View logs
sudo journalctl -u collage-api -f
```

### Option 3: Nginx + Gunicorn

#### Gunicorn Configuration

Create `gunicorn.conf.py`:

```python
# gunicorn.conf.py
bind = "127.0.0.1:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 30
keepalive = 2
user = "www-data"
group = "www-data"
tmp_upload_dir = None
```

#### Nginx Configuration

```nginx
# /etc/nginx/sites-available/collage-api
server {
    listen 80;
    server_name your-domain.com;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # File upload size
    client_max_body_size 500M;

    # Proxy to Gunicorn
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeout settings
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Static file serving (optional)
    location /static/ {
        alias /path/to/collage-api/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

#### SSL Configuration (Let's Encrypt)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Test renewal
sudo certbot renew --dry-run
```

#### Start Services

```bash
# Start Gunicorn
gunicorn server:app -c gunicorn.conf.py

# Enable Nginx site
sudo ln -s /etc/nginx/sites-available/collage-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Production Configuration

### Environment Variables

Create a `.env` file for production settings:

```bash
# .env
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-secret-key-here

# Database (if using)
DATABASE_URL=postgresql://user:password@localhost/collage_api

# Redis (for job storage)
REDIS_URL=redis://localhost:6379

# File storage
UPLOAD_DIR=/var/lib/collage-api/uploads
OUTPUT_DIR=/var/lib/collage-api/outputs
TEMP_DIR=/tmp/collage-api

# API settings
MAX_WORKERS=4
MAX_REQUESTS_PER_WORKER=1000
REQUEST_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/collage-api/app.log

# CORS
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### Production Server Configuration

Update `server.py` for production:

```python
import os
from pathlib import Path

# Production configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT != "production"

# Directory configuration
if ENVIRONMENT == "production":
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/var/lib/collage-api/uploads"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/var/lib/collage-api/outputs"))
    TEMP_DIR = Path(os.getenv("TEMP_DIR", "/tmp/collage-api"))
else:
    UPLOAD_DIR = Path("uploads")
    OUTPUT_DIR = Path("outputs")
    TEMP_DIR = Path("temp")

# Create directories
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# CORS configuration
if ENVIRONMENT == "production":
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
else:
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Scaling and Performance

### Horizontal Scaling with Load Balancer

#### HAProxy Configuration

```haproxy
# /etc/haproxy/haproxy.cfg
frontend http_front
    bind *:80
    default_backend api_backend

backend api_backend
    balance roundrobin
    server api1 127.0.0.1:8001 check
    server api2 127.0.0.1:8002 check
    server api3 127.0.0.1:8003 check
```

#### Multiple Application Instances

```bash
# Start multiple instances
gunicorn server:app -b 127.0.0.1:8001 -w 4 &
gunicorn server:app -b 127.0.0.1:8002 -w 4 &
gunicorn server:app -b 127.0.0.1:8003 -w 4 &
```

### Redis for Job Storage

Replace in-memory job storage with Redis:

```python
import redis
import json

# Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_job_status(job_id: str) -> dict:
    """Get job status from Redis"""
    data = redis_client.get(f"job:{job_id}")
    return json.loads(data) if data else None

def set_job_status(job_id: str, status: dict):
    """Set job status in Redis"""
    redis_client.setex(f"job:{job_id}", 3600, json.dumps(status))  # 1 hour expiry
```

### File Storage Optimization

#### Shared File Storage

For multiple instances, use shared storage:

```bash
# NFS mount
sudo mount -t nfs 192.168.1.100:/shared/collage-files /mnt/collage-files

# Or use cloud storage (AWS S3, Google Cloud Storage)
```

#### CDN Integration

Serve output files via CDN:

```python
from botocore.client import Config
import boto3

# S3 client
s3 = boto3.client('s3', config=Config(signature_version='s3v4'))

def upload_to_cdn(file_path, job_id):
    """Upload file to CDN"""
    bucket = 'your-cdn-bucket'
    key = f"collages/{job_id}.jpg"

    s3.upload_file(file_path, bucket, key)

    # Generate presigned URL
    url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=3600  # 1 hour
    )

    return url
```

## Monitoring and Logging

### Application Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.getenv("LOG_FILE", "app.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log requests
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
    )

    return response
```

### Health Checks and Metrics

#### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()

    response = await call_next(request)

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(time.time() - start_time)

    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

#### Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(),
        "checks": {
            "database": check_database(),
            "file_system": check_file_system(),
            "memory": check_memory_usage(),
            "disk_space": check_disk_space()
        }
    }

    # Determine overall health
    if any(not check["healthy"] for check in health_status["checks"].values()):
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail=health_status)

    return health_status
```

## Security Considerations

### Input Validation

```python
from pydantic import validator

class SecureCollageConfig(BaseModel):
    width_inches: float = Field(default=12, ge=4, le=48)
    height_inches: float = Field(default=18, ge=4, le=48)
    dpi: int = Field(default=150, ge=72, le=300)

    @validator('background_color')
    def validate_color(cls, v):
        """Validate hex color format"""
        if not re.match(r'^#[0-9A-Fa-f]{6}$', v):
            raise ValueError('Invalid hex color format')
        return v
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

@app.post("/api/collage/create")
@limiter.limit("10/minute")
async def create_collage(request: Request, ...):
    # Rate limited endpoint
    pass
```

### File Upload Security

```python
import magic

def validate_image_file(file_path: str) -> bool:
    """Validate that file is actually an image"""
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)

    allowed_types = [
        'image/jpeg',
        'image/png',
        'image/gif',
        'image/bmp',
        'image/tiff',
        'image/webp'
    ]

    return file_type in allowed_types

# Check file type after upload
if not validate_image_file(str(file_path)):
    raise HTTPException(status_code=400, detail="Invalid file type")
```

## Backup and Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/var/backups/collage-api"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database (if using)
pg_dump collage_api > $BACKUP_DIR/database_$DATE.sql

# Backup uploaded files
tar -czf $BACKUP_DIR/uploads_$DATE.tar.gz /var/lib/collage-api/uploads/

# Backup outputs (optional, as they can be regenerated)
tar -czf $BACKUP_DIR/outputs_$DATE.tar.gz /var/lib/collage-api/outputs/

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
```

### Recovery Procedures

```bash
#!/bin/bash
# restore.sh

BACKUP_DATE="20231201_120000"

# Restore database
psql collage_api < /var/backups/collage-api/database_$BACKUP_DATE.sql

# Restore files
tar -xzf /var/backups/collage-api/uploads_$BACKUP_DATE.tar.gz -C /

echo "Restore completed"
```

## Troubleshooting Production Issues

### Common Issues

#### High Memory Usage

```bash
# Monitor memory usage
ps aux --sort=-%mem | head

# Check for memory leaks
import tracemalloc
tracemalloc.start()
# ... run some operations ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

#### Slow Response Times

```bash
# Profile application
python -m cProfile -s time server.py

# Check system resources
top
iostat -x 1
free -h
```

#### File System Issues

```bash
# Check disk space
df -h

# Check file permissions
ls -la /var/lib/collage-api/

# Monitor file system events
inotifywait -r /var/lib/collage-api/
```

### Performance Tuning

#### Gunicorn Optimization

```python
# gunicorn.conf.py - optimized settings
workers = multiprocessing.cpu_count() * 2 + 1
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 30
keepalive = 2
worker_class = "uvicorn.workers.UvicornWorker"
```

#### System Tuning

```bash
# Increase file descriptors
echo "fs.file-max = 65536" >> /etc/sysctl.conf
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Apply changes
sysctl -p
```

## Cloud Deployment

### AWS Deployment

#### Elastic Beanstalk

```python
# .ebextensions/01_app.config
option_settings:
  aws:elasticbeanstalk:application:environment:
    ENVIRONMENT: production
  aws:autoscaling:launchconfiguration:
    InstanceType: t3.medium
    IamInstanceProfile: aws-elasticbeanstalk-ec2-role
  aws:elasticbeanstalk:healthreporting:system:
    SystemType: enhanced
```

#### ECS with Fargate

```yaml
# docker-compose.yml for ECS
version: "3.8"
services:
    collage-api:
        image: your-registry/collage-api:latest
        environment:
            - ENVIRONMENT=production
        ports:
            - "8000:8000"
        logging:
            driver: awslogs
            options:
                awslogs-group: /ecs/collage-api
                awslogs-region: us-east-1
```

### Google Cloud Run

```yaml
# cloudbuild.yaml
steps:
    - name: "gcr.io/cloud-builders/docker"
      args: ["build", "-t", "gcr.io/$PROJECT_ID/collage-api", "."]
    - name: "gcr.io/cloud-builders/docker"
      args: ["push", "gcr.io/$PROJECT_ID/collage-api"]
    - name: "gcr.io/google.com/cloudsdktool/cloud-sdk"
      entrypoint: gcloud
      args:
          - run
          - deploy
          - collage-api
          - --image=gcr.io/$PROJECT_ID/collage-api
          - --platform=managed
          - --port=8000
          - --memory=1Gi
          - --cpu=1
```

## Maintenance Tasks

### Log Rotation

```bash
# /etc/logrotate.d/collage-api
/var/log/collage-api/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload collage-api
    endscript
}
```

### Database Maintenance

```bash
# Vacuum and analyze database weekly
0 2 * * 0 psql -d collage_api -c "VACUUM ANALYZE;"

# Backup database daily
0 1 * * * /path/to/backup.sh
```

### Monitoring Alerts

Set up alerts for:

-   High CPU usage (>80%)
-   High memory usage (>85%)
-   Low disk space (<10% free)
-   Failed health checks
-   High error rates (>5%)
-   Slow response times (>5 seconds)

This deployment guide provides a comprehensive foundation for running the Collage Maker API in production with proper scaling, monitoring, and security measures.
