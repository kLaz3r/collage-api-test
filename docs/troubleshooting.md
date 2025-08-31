# Troubleshooting Guide

Common issues and solutions for the Collage Maker API.

## API Issues

### Job Stuck in Processing

**Symptoms:**

-   Job status remains "processing" indefinitely
-   No collage output generated
-   Progress doesn't update

**Possible Causes & Solutions:**

1. **Memory Issues:**

    ```bash
    # Check system memory
    free -h

    # Monitor process memory usage
    ps aux --sort=-%mem | grep python
    ```

    **Solution:** Increase available memory or reduce image sizes/DPI

2. **Large Image Files:**

    ```python
    # Check image dimensions before processing
    from PIL import Image
    with Image.open('large_image.jpg') as img:
        print(f"Size: {img.size}, File size: {len(img.tobytes())}")
    ```

    **Solution:** Resize large images before upload or reduce DPI setting

3. **Background Task Failure:**

    ```python
    # Check for exceptions in background task
    import logging
    logging.basicConfig(level=logging.DEBUG)
    ```

    **Solution:** Review application logs for error messages

### File Upload Errors

**Error:** `File exceeds 10MB limit`

**Solutions:**

```python
# Compress images before upload
from PIL import Image
import io

def compress_image(image_path, max_size_mb=9):
    with Image.open(image_path) as img:
        # Calculate compression ratio
        current_size = len(img.tobytes())
        target_size = max_size_mb * 1024 * 1024

        if current_size > target_size:
            # Compress image
            quality = int(95 * target_size / current_size)
            quality = max(quality, 10)  # Minimum quality

            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            output.seek(0)
            return output
```

**Error:** `Invalid file type`

**Solutions:**

-   Verify file extension matches actual format
-   Use proper MIME types in upload requests
-   Check file isn't corrupted

### CORS Errors

**Error:** `Access-Control-Allow-Origin` header missing

**Solutions:**

1. **Update CORS configuration:**

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. **For development:**

```python
# Allow all origins in development
allow_origins = ["*"] if ENVIRONMENT == "development" else ["https://yourdomain.com"]
```

### Rate Limiting Issues

**Error:** `429 Too Many Requests`

**Solutions:**

-   Implement exponential backoff in client code
-   Reduce request frequency
-   Contact administrator for rate limit increase

```python
import time
import requests

def create_collage_with_retry(images, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post('http://localhost:8000/api/collage/create',
                                   files=images)
            if response.status_code == 429:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited, waiting {wait_time} seconds")
                time.sleep(wait_time)
                continue
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1)
    raise Exception("Max retries exceeded")
```

## Image Processing Issues

### Poor Image Quality

**Symptoms:**

-   Blurry or pixelated output
-   Colors don't match original
-   Artifacts in final collage

**Solutions:**

1. **Increase DPI:**

```json
{
    "dpi": 300,
    "width_inches": 16,
    "height_inches": 20
}
```

2. **Use higher quality source images:**

```python
# Check source image quality
from PIL import Image

def check_image_quality(image_path):
    with Image.open(image_path) as img:
        print(f"Format: {img.format}")
        print(f"Mode: {img.mode}")
        print(f"Size: {img.size}")

        # Check if image is highly compressed
        if hasattr(img, 'info') and 'progressive' in img.info:
            print("Progressive JPEG detected")
```

3. **Avoid multiple compression:**

-   Upload original images when possible
-   Avoid re-saving compressed images

### Layout Problems

**Issue:** Images don't fit properly in collage

**Debugging:**

```python
# Test layout algorithm separately
from server import MasonryPacker

packer = MasonryPacker(2000, 3000, 20)
blocks = packer.pack_images(['image1.jpg', 'image2.jpg'])

for i, block in enumerate(blocks):
    print(f"Block {i}: Position({block.x}, {block.y}) Size({block.width}, {block.height})")
```

**Common Solutions:**

-   Adjust spacing parameter
-   Try different layout style
-   Check image aspect ratios
-   Modify canvas dimensions

### Memory Errors

**Error:** `MemoryError` or `PIL.Image.DecompressionBombError`

**Solutions:**

1. **Limit image sizes:**

```python
from PIL import Image, ImageFile

# Set limits to prevent decompression bomb
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 100000000  # 100MP limit
```

2. **Process images in chunks:**

```python
def process_large_image(image_path, max_dimension=4000):
    with Image.open(image_path) as img:
        # Resize if too large
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Process resized image
        return img
```

3. **Monitor memory usage:**

```python
import psutil
import os

def check_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

    if memory_mb > 1000:  # 1GB threshold
        print("WARNING: High memory usage detected")
```

## Network and Connectivity Issues

### Connection Timeouts

**Symptoms:**

-   Requests hang indefinitely
-   `ConnectionError` exceptions
-   Slow response times

**Solutions:**

1. **Increase timeout values:**

```python
import requests

response = requests.post(
    'http://localhost:8000/api/collage/create',
    files=files,
    timeout=(10, 300)  # (connect timeout, read timeout)
)
```

2. **Check server status:**

```bash
# Test basic connectivity
curl -v http://localhost:8000/health

# Check if server is running
ps aux | grep python
```

3. **Network diagnostics:**

```bash
# Check network connectivity
ping localhost

# Test port availability
netstat -tlnp | grep :8000

# Check firewall rules
sudo ufw status
```

### SSL/TLS Issues

**Error:** `SSL: CERTIFICATE_VERIFY_FAILED`

**Solutions:**

```python
# Disable SSL verification (development only)
response = requests.post(url, files=files, verify=False)

# Or provide custom CA bundle
response = requests.post(url, files=files, verify='/path/to/ca-bundle.crt')
```

## File System Issues

### Permission Errors

**Error:** `PermissionError: [Errno 13] Permission denied`

**Solutions:**

1. **Check file permissions:**

```bash
# Check directory permissions
ls -la /path/to/collage-api/

# Fix permissions
sudo chown -R www-data:www-data /path/to/collage-api/uploads/
sudo chown -R www-data:www-data /path/to/collage-api/outputs/
sudo chmod 755 /path/to/collage-api/uploads/
sudo chmod 755 /path/to/collage-api/outputs/
```

2. **SELinux/AppArmor issues:**

```bash
# Check SELinux status
sestatus

# Temporarily disable SELinux for testing
sudo setenforce 0
```

### Disk Space Issues

**Error:** `OSError: [Errno 28] No space left on device`

**Solutions:**

1. **Check disk usage:**

```bash
# Check disk space
df -h

# Find large files
du -sh /path/to/collage-api/* | sort -hr | head -10

# Clean up old files
find /path/to/collage-api/outputs/ -name "*.jpg" -mtime +7 -delete
```

2. **Implement cleanup:**

```python
import os
from pathlib import Path
from datetime import datetime, timedelta

def cleanup_old_files(directory, days_old=7):
    """Remove files older than specified days"""
    cutoff_date = datetime.now() - timedelta(days=days_old)
    directory_path = Path(directory)

    for file_path in directory_path.glob("*"):
        if file_path.is_file():
            file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_modified < cutoff_date:
                file_path.unlink()
                print(f"Removed old file: {file_path}")
```

## Performance Issues

### Slow Processing Times

**Symptoms:**

-   Long wait times for collage generation
-   High CPU usage
-   Memory spikes during processing

**Optimization Solutions:**

1. **Profile performance:**

```python
import cProfile
import pstats

def profile_collage_creation():
    profiler = cProfile.Profile()
    profiler.enable()

    # Run collage creation
    create_collage(images)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 time-consuming functions
```

2. **Optimize image processing:**

```python
# Use faster image operations
from PIL import Image, ImageFilter

def fast_resize(img, size):
    """Fast image resizing with optimization"""
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Use faster resampling filter
    return img.resize(size, Image.Resampling.BILINEAR)
```

3. **Implement caching:**

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_image_dimensions(image_path):
    """Cache image dimension lookups"""
    with Image.open(image_path) as img:
        return img.size
```

### High CPU Usage

**Solutions:**

1. **Reduce concurrent jobs:**

```python
# Limit concurrent background tasks
import asyncio
from asyncio import Semaphore

semaphore = Semaphore(3)  # Max 3 concurrent jobs

async def process_collage_limited(job_id, image_paths, config):
    async with semaphore:
        await process_collage(job_id, image_paths, config)
```

2. **Optimize algorithms:**

```python
# Use numpy for mathematical operations
import numpy as np

def fast_calculations(width, height, num_images):
    """Use vectorized operations"""
    cols = np.sqrt(num_images * width / height)
    rows = np.ceil(num_images / cols)
    return int(cols), int(rows)
```

## Database and Caching Issues

### Redis Connection Problems

**Error:** `ConnectionError: Error 111 connecting to localhost:6379`

**Solutions:**

1. **Check Redis status:**

```bash
# Check if Redis is running
sudo systemctl status redis

# Start Redis if stopped
sudo systemctl start redis

# Test connection
redis-cli ping
```

2. **Connection configuration:**

```python
import redis

# Configure Redis connection with retry
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True,
    max_connections=20
)
```

### Job Status Persistence

**Issue:** Job status lost after server restart

**Solutions:**

1. **Use persistent storage:**

```python
# Save to disk periodically
import json
import atexit

def save_job_status():
    """Save job status to disk"""
    with open('job_status.json', 'w') as f:
        json.dump(job_status, f)

def load_job_status():
    """Load job status from disk"""
    try:
        with open('job_status.json', 'r') as f:
            global job_status
            job_status = json.load(f)
    except FileNotFoundError:
        pass

# Register save function
atexit.register(save_job_status)
```

2. **Use database:**

```python
# PostgreSQL job storage
import psycopg2

def save_job_to_db(job_id, status):
    conn = psycopg2.connect("dbname=collage_api user=api_user")
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO jobs (job_id, status, created_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (job_id) DO UPDATE SET
        status = EXCLUDED.status,
        updated_at = NOW()
    """, (job_id, status))

    conn.commit()
    cur.close()
    conn.close()
```

## Logging and Debugging

### Enable Debug Logging

```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Log PIL operations
logging.getLogger('PIL').setLevel(logging.DEBUG)
```

### Debug Image Processing

```python
def debug_image_processing(image_path):
    """Debug image loading and processing"""
    print(f"Processing image: {image_path}")

    try:
        with Image.open(image_path) as img:
            print(f"Format: {img.format}")
            print(f"Size: {img.size}")
            print(f"Mode: {img.mode}")
            print(f"Info: {img.info}")

            # Test basic operations
            resized = img.resize((100, 100), Image.Resampling.LANCZOS)
            print(f"Resize successful: {resized.size}")

    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
```

### Monitor System Resources

```python
import psutil
import time

def monitor_resources(duration=60):
    """Monitor system resources during processing"""
    print("Monitoring system resources...")

    for _ in range(duration):
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        print(f"CPU: {cpu}%, Memory: {memory.percent}%, Disk: {disk.percent}%")

        # Check for high usage
        if cpu > 90:
            print("WARNING: High CPU usage detected")
        if memory.percent > 85:
            print("WARNING: High memory usage detected")
        if disk.percent > 90:
            print("WARNING: Low disk space detected")

# Usage
monitor_resources(30)  # Monitor for 30 seconds
```

## Getting Help

### Log Analysis

```bash
# Search for errors in logs
grep -i "error\|exception\|failed" /var/log/collage-api/app.log

# Check recent log entries
tail -f /var/log/collage-api/app.log

# Analyze log patterns
awk '/ERROR/ {print $0}' /var/log/collage-api/app.log | head -10
```

### Health Check Script

```python
#!/usr/bin/env python3
"""
Health check script for Collage Maker API
"""

import requests
import sys
from datetime import datetime

def check_api_health():
    """Comprehensive health check"""
    base_url = "http://localhost:8000"

    checks = {
        "api_root": f"{base_url}/",
        "health_endpoint": f"{base_url}/health",
        "api_docs": f"{base_url}/docs",
        "job_list": f"{base_url}/api/collage/jobs"
    }

    results = {}

    for name, url in checks.items():
        try:
            response = requests.get(url, timeout=10)
            results[name] = {
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            results[name] = {
                "status": "ERROR",
                "error": str(e)
            }

    return results

def main():
    print(f"Collage API Health Check - {datetime.now()}")
    print("=" * 50)

    results = check_api_health()

    all_pass = True
    for name, result in results.items():
        status = result["status"]
        if status != "PASS":
            all_pass = False

        print(f"{name:15}: {status}")
        if "error" in result:
            print(f"{'':15}  Error: {result['error']}")
        elif "status_code" in result:
            print(f"{'':15}  Status: {result['status_code']}, Time: {result['response_time']:.2f}s")

    print("=" * 50)
    if all_pass:
        print("✅ All health checks passed")
        sys.exit(0)
    else:
        print("❌ Some health checks failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Support Information

For additional help:

1. **Check the documentation:**

    - API Reference (`docs/api-reference.md`)
    - Configuration Guide (`docs/configuration.md`)
    - Developer Guide (`docs/developer-guide.md`)

2. **Gather diagnostic information:**

```bash
# System information
uname -a
python --version
pip list | grep -E "(fastapi|uvicorn|pillow)"

# Application logs
tail -50 /var/log/collage-api/app.log

# Running processes
ps aux | grep python
```

3. **Test with minimal example:**

```bash
# Test with small images
curl -X POST "http://localhost:8000/api/collage/create" \
  -F "files=@small_image1.jpg" \
  -F "files=@small_image2.jpg" \
  -F "width_inches=8" \
  -F "height_inches=10" \
  -F "dpi=72"
```

This troubleshooting guide covers the most common issues and provides practical solutions for maintaining a healthy Collage Maker API deployment.
