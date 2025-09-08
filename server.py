"""
FastAPI Collage Maker Application
A web API for creating high-resolution photo collages with masonry layout
"""

import io
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"
os.environ["absl_logging_minloglevel"] = "2"
import hashlib
import uuid
import json
import random
import asyncio
import re
import tempfile
import shutil
import logging
import time
from typing import List, Optional, Dict, Tuple, Literal
from datetime import datetime
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Request, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np
from config import AppSettings
from redis.asyncio import Redis as AsyncRedis
from celery_app import celery_app
import aiofiles
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Security imports
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

from collections import defaultdict
from logging.handlers import RotatingFileHandler

# MediaPipe lazy import flags (to allow env vars to be set before import)
_MEDIAPIPE_IMPORT_TRIED = False
_MEDIAPIPE_AVAILABLE = False
mp = None  # type: ignore
_FACE_DETECTOR = None  # type: ignore

def _configure_logging():
    handlers = [logging.StreamHandler()]
    if settings.log_to_file:
        handlers.append(RotatingFileHandler(
            settings.log_file_path,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count
        ))
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

# Load settings
settings = AppSettings()

# Configure logging (after settings)
logger = _configure_logging()

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Create beautiful photo collages with masonry layout",
    version=settings.app_version
)
# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'path', 'status']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency (seconds)',
    ['method', 'path']
)
ACTIVE_JOBS = Gauge('collage_active_jobs', 'Number of active jobs (pending+processing)')
TOTAL_JOBS = Gauge('collage_total_jobs', 'Number of jobs currently stored')

# Startup/Shutdown events: init redis and start cleanup loop
@app.on_event("startup")
async def on_startup():
    global redis_client
    # Initialize Redis client
    if settings.redis_url:
        redis_client = AsyncRedis.from_url(settings.redis_url, decode_responses=True)
    else:
        redis_client = AsyncRedis(host=settings.redis_host, port=settings.redis_port, db=settings.redis_db, decode_responses=True)
    # Fire-and-forget cleanup loop
    asyncio.create_task(_cleanup_loop())

@app.on_event("shutdown")
async def on_shutdown():
    global redis_client
    if redis_client is not None:
        await redis_client.close()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://college-maker-frontend.vercel.app"],
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Configuration
UPLOAD_DIR = settings.upload_dir
OUTPUT_DIR = settings.output_dir
TEMP_DIR = settings.temp_dir
MAX_IMAGE_SIZE = settings.max_image_size
MAX_TOTAL_SIZE = settings.max_total_size
MAX_CANVAS_PIXELS = settings.max_canvas_pixels
STREAM_CHUNK_SIZE = 1024 * 1024  # 1MB per chunk

# Create directories
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# Guard against decompression bombs for very large images
Image.MAX_IMAGE_PIXELS = MAX_CANVAS_PIXELS

# Redis client (initialized on startup)
redis_client: AsyncRedis | None = None

# Job key helpers
def _job_key(job_id: str) -> str:
    return f"job:{job_id}"

async def _get_redis() -> AsyncRedis:
    global redis_client
    if redis_client is None:
        # Lazy init if startup not called (e.g., tests)
        if settings.redis_url:
            redis_client = AsyncRedis.from_url(settings.redis_url, decode_responses=True)
        else:
            redis_client = AsyncRedis(host=settings.redis_host, port=settings.redis_port, db=settings.redis_db, decode_responses=True)
    return redis_client

async def save_job(job: 'CollageJob') -> None:
    client = await _get_redis()
    payload = {
        'job_id': job.job_id,
        'status': job.status.value if isinstance(job.status, JobStatus) else job.status,
        'created_at': job.created_at.isoformat() if isinstance(job.created_at, datetime) else str(job.created_at),
        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
        'output_file': job.output_file,
        'error_message': job.error_message,
        'progress': job.progress,
    }
    await client.set(_job_key(job.job_id), json.dumps(payload), ex=settings.job_ttl_seconds)

async def update_job(job_id: str, updates: Dict) -> None:
    client = await _get_redis()
    existing = await client.get(_job_key(job_id))
    if existing:
        data = json.loads(existing)
    else:
        data = {'job_id': job_id, 'status': JobStatus.PENDING.value, 'created_at': datetime.now().isoformat(), 'progress': 0}
    # Coerce enum values to strings if provided
    if 'status' in updates and isinstance(updates['status'], JobStatus):
        updates = {**updates, 'status': updates['status'].value}
    data.update(updates)
    await client.set(_job_key(job_id), json.dumps(data), ex=settings.job_ttl_seconds)

async def get_job(job_id: str) -> Optional[Dict]:
    client = await _get_redis()
    raw = await client.get(_job_key(job_id))
    return json.loads(raw) if raw else None

async def delete_job(job_id: str) -> None:
    client = await _get_redis()
    await client.delete(_job_key(job_id))

async def list_all_jobs() -> List[Dict]:
    client = await _get_redis()
    cursor = 0
    keys: List[str] = []
    while True:
        cursor, batch = await client.scan(cursor=cursor, match='job:*', count=200)
        keys.extend(batch)
        if cursor == 0:
            break
    if not keys:
        return []
    values = await client.mget(keys)
    return [json.loads(v) for v in values if v]

async def count_total_jobs() -> int:
    client = await _get_redis()
    cursor = 0
    total = 0
    while True:
        cursor, batch = await client.scan(cursor=cursor, match='job:*', count=500)
        total += len(batch)
        if cursor == 0:
            break
    return total

async def count_active_jobs() -> int:
    jobs = await list_all_jobs()
    return sum(1 for j in jobs if j.get('status') in [JobStatus.PENDING.value, JobStatus.PROCESSING.value])

async def is_redis_connected() -> bool:
    try:
        client = await _get_redis()
        pong = await client.ping()
        return bool(pong)
    except Exception:
        return False

async def cleanup_stale_files() -> None:
    """Delete files for jobs that no longer exist in Redis (expired/cleaned)."""
    client = await _get_redis()
    # Cleanup temp files: pattern {job_id}_filename
    for file in TEMP_DIR.glob('*_*'):
        try:
            job_id = file.name.split('_', 1)[0]
            exists = await client.exists(_job_key(job_id))
            if not exists:
                file.unlink()
        except Exception:
            continue
    # Cleanup outputs: pattern collage_{job_id}.ext
    for file in OUTPUT_DIR.glob('collage_*'):
        try:
            stem = file.stem  # e.g., collage_<jobid>
            if not stem.startswith('collage_'):
                continue
            job_id = stem.split('collage_', 1)[1]
            exists = await client.exists(_job_key(job_id))
            if not exists:
                file.unlink()
        except Exception:
            continue

async def _cleanup_loop():
    while True:
        await asyncio.sleep(settings.cleanup_interval_seconds)
        await cleanup_stale_files()

# Enums
class LayoutStyle(str, Enum):
    MASONRY = "masonry"
    GRID = "grid"

class OutputFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Pydantic models
class CollageConfig(BaseModel):
    width_mm: float = Field(default=304.8, ge=50, le=1219.2)  # 12 inches = 304.8 mm, 2-48 inches = 50-1219.2 mm
    height_mm: float = Field(default=457.2, ge=50, le=1219.2)  # 18 inches = 457.2 mm, 2-48 inches = 50-1219.2 mm
    dpi: int = Field(default=150, ge=72, le=300)
    layout_style: LayoutStyle = LayoutStyle.MASONRY
    spacing: float = Field(default=40.0, ge=0.0, le=100.0)  # Spacing as percentage of canvas dimensions (0-100%, where 100% = 5% of canvas)
    background_color: str = Field(default="#FFFFFF")
    maintain_aspect_ratio: bool = True
    apply_shadow: bool = False
    output_format: OutputFormat = OutputFormat.JPEG
    # Face-aware options
    face_aware_cropping: bool = False
    face_margin: float = Field(default=0.08, ge=0.0, le=0.3)
    pretrim_borders: bool = False

    @validator('background_color')
    def validate_color(cls, v):
        """Validate hex color format - supports #RRGGBB and #RRGGBBAA (with alpha)"""
        if not re.match(r'^#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?$', v):
            raise ValueError('Invalid hex color format - must be #RRGGBB or #RRGGBBAA (with alpha)')
        return v

class CollagePixelConfig(BaseModel):
    width_px: int = Field(default=1920, ge=320, le=20000)
    height_px: int = Field(default=1080, ge=320, le=20000)
    dpi: int = Field(default=96, ge=72, le=300)
    layout_style: LayoutStyle = LayoutStyle.MASONRY
    spacing: float = Field(default=40.0, ge=0.0, le=100.0)
    background_color: str = Field(default="#FFFFFF")
    maintain_aspect_ratio: bool = True
    apply_shadow: bool = False
    output_format: OutputFormat = OutputFormat.JPEG
    # Face-aware options
    face_aware_cropping: bool = False
    face_margin: float = Field(default=0.08, ge=0.0, le=0.3)
    pretrim_borders: bool = False

    @validator('background_color')
    def validate_color(cls, v):
        if not re.match(r'^#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?$', v):
            raise ValueError('Invalid hex color format - must be #RRGGBB or #RRGGBBAA (with alpha)')
        return v

class CollageJob(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    progress: int = Field(default=0, ge=0, le=100)

# Public response models
class CreateCollageResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    message: str

class CollageJobPublic(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    progress: int

class CleanupResponse(BaseModel):
    message: str

class ImageBlock:
    """Represents a single image block in the collage"""
    def __init__(self, x: int, y: int, width: int, height: int, image_path: str = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.image_path = image_path
        self.image = None

class MasonryPacker:
    """Implements dynamic masonry layout algorithm for optimal canvas filling"""

    def __init__(self, canvas_width: int, canvas_height: int, spacing_percent: float = 2.0):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.spacing_percent = spacing_percent
        self.spacing_pixels = int(min(canvas_width, canvas_height) * (spacing_percent / 100.0) * 0.05)
        self.blocks = []

    def pack_images(self, image_paths: List[str], maintain_aspect: bool = True) -> List[ImageBlock]:
        """Pack images using the new dynamic masonry algorithm"""
        if not image_paths:
            return []

        # Get image dimensions and aspect ratios
        images_info = []
        for path in image_paths:
            with Image.open(path) as img:
                images_info.append({
                    'path': path,
                    'width': img.width,
                    'height': img.height,
                    'aspect': img.width / img.height
                })

        # Calculate optimal column count
        optimal_columns = self._calculate_optimal_columns(len(images_info))
        
        # Distribute photos across columns
        distribution = self._distribute_photos(len(images_info), optimal_columns)
        
        # Calculate photo layout
        layout = self._calculate_photo_layout(images_info, distribution, optimal_columns)
        
        # Refine layout for better fit
        layout = self._refine_masonry_layout(layout, images_info)
        
        # Convert layout to ImageBlock objects
        blocks = []
        for item in layout:
            block = ImageBlock(
                x=int(item['x']),
                y=int(item['y']),
                width=int(item['width']),
                height=int(item['height']),
                image_path=item['photo']['path']
            )
            blocks.append(block)
        
        return blocks

    def _calculate_optimal_columns(self, photo_count: int) -> int:
        """Calculate optimal number of columns for best canvas filling"""
        # Start with a reasonable estimate based on canvas aspect ratio
        min_columns = max(1, int((photo_count * self.canvas_width / self.canvas_height) ** 0.5))
        max_columns = min(photo_count, max(1, self.canvas_width // 200))  # min 200px per column
        
        best_columns = min_columns
        best_score = float('inf')
        
        # Test different column counts to find the best fit
        for cols in range(min_columns, max_columns + 1):
            score = self._evaluate_layout(photo_count, cols)
            if score < best_score:
                best_score = score
                best_columns = cols
        
        return best_columns

    def _evaluate_layout(self, photo_count: int, columns: int) -> float:
        """Evaluate layout quality based on column count"""
        column_width = (self.canvas_width - self.spacing_pixels * (columns + 1)) / columns
        photos_per_column = (photo_count + columns - 1) // columns  # Ceiling division
        total_spacing_per_column = self.spacing_pixels * (photos_per_column + 1)
        available_height_per_column = self.canvas_height - total_spacing_per_column
        avg_photo_height = available_height_per_column / photos_per_column
        
        # Score based on how well photos fit (penalize extreme aspect ratios)
        aspect_ratio = column_width / avg_photo_height
        aspect_penalty = abs(aspect_ratio - 1.0)  # prefer square-ish photos
        
        # Penalty for uneven distribution
        remainder = photo_count % columns
        unevenness_penalty = remainder / columns if remainder > 0 else 0
        
        return aspect_penalty + unevenness_penalty

    def _distribute_photos(self, photo_count: int, columns: int) -> List[int]:
        """Distribute photos evenly across columns"""
        distribution = [0] * columns
        photos_per_column = photo_count // columns
        remainder = photo_count % columns
        
        # Distribute evenly, then add remainder to first columns
        for i in range(columns):
            distribution[i] = photos_per_column + (1 if i < remainder else 0)
        
        return distribution

    def _calculate_photo_layout(self, images_info: List[dict], distribution: List[int], columns: int) -> List[dict]:
        """Calculate photo dimensions and positions"""
        column_width = (self.canvas_width - self.spacing_pixels * (columns + 1)) / columns
        
        layout = []
        photo_index = 0
        
        for col in range(columns):
            photos_in_this_column = distribution[col]
            total_spacing = self.spacing_pixels * (photos_in_this_column + 1)
            available_height = self.canvas_height - total_spacing
            photo_height = available_height / photos_in_this_column
            
            x = self.spacing_pixels + col * (column_width + self.spacing_pixels)
            y = self.spacing_pixels
            
            for row in range(photos_in_this_column):
                layout.append({
                    'photo': images_info[photo_index],
                    'x': x,
                    'y': y,
                    'width': column_width,
                    'height': photo_height
                })
                
                y += photo_height + self.spacing_pixels
                photo_index += 1
        
        return layout

    def _refine_masonry_layout(self, layout: List[dict], images_info: List[dict]) -> List[dict]:
        """Refine layout for better fit and aspect ratios"""
        # Adjust for photo aspect ratios
        for item in layout:
            photo = item['photo']
            if photo['aspect']:
                ideal_height = item['width'] / photo['aspect']
                # Slightly adjust height while maintaining total coverage
                item['height'] = min(item['height'] * 1.2, ideal_height)
        
        # Redistribute remaining vertical space
        columns = max(1, len(set(int(item['x'] / (item['width'] + self.spacing_pixels)) for item in layout)))
        self._redistribute_vertical_space(layout, columns)
        
        return layout

    def _redistribute_vertical_space(self, layout: List[dict], columns: int):
        """Redistribute vertical space to balance column heights"""
        for col in range(columns):
            # Find items in this column
            column_items = [
                item for item in layout 
                if int(item['x'] / (item['width'] + self.spacing_pixels)) == col
            ]
            
            if not column_items:
                continue
            
            # Sort by y position
            column_items.sort(key=lambda x: x['y'])
            
            total_current_height = sum(item['height'] for item in column_items)
            total_spacing = self.spacing_pixels * (len(column_items) + 1)
            available_height = self.canvas_height - total_spacing
            
            if total_current_height > 0:
                scale_factor = available_height / total_current_height
                
                current_y = self.spacing_pixels
                for item in column_items:
                    item['height'] *= scale_factor
                    item['y'] = current_y
                    current_y += item['height'] + self.spacing_pixels


class GridPacker:
    """Implements grid layout for images"""

    def __init__(self, canvas_width: int, canvas_height: int, spacing_percent: float = 2.0):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.spacing_percent = spacing_percent
        self.spacing_pixels = int(min(canvas_width, canvas_height) * (spacing_percent / 100.0) * 0.05)

    def pack_images(self, image_paths: List[str]) -> List[ImageBlock]:
        """Pack images in a uniform grid"""
        blocks = []
        num_images = len(image_paths)
        if num_images == 0:
            return blocks

        # Calculate grid dimensions
        computed_cols = int(np.sqrt(num_images * (self.canvas_width / self.canvas_height)))
        cols = max(1, computed_cols)
        rows = max(1, int(np.ceil(num_images / cols)))

        # Add edge spacing to all four sides
        edge_spacing = self.spacing_pixels
        
        # Calculate cell dimensions with edge spacing
        cell_width = (self.canvas_width - (cols - 1) * self.spacing_pixels - 2 * edge_spacing) // cols
        cell_height = (self.canvas_height - (rows - 1) * self.spacing_pixels - 2 * edge_spacing) // rows

        for i, path in enumerate(image_paths):
            row = i // cols
            col = i % cols

            x = edge_spacing + col * (cell_width + self.spacing_pixels)
            y = edge_spacing + row * (cell_height + self.spacing_pixels)

            block = ImageBlock(x, y, cell_width, cell_height, path)
            blocks.append(block)

        return blocks

    def calculate_optimal_grid(self, num_images: int) -> dict:
        """
        Calculate optimal grid dimensions and provide recommendations for perfect grid
        
        Returns:
            dict: Contains optimal grid info and recommendations
        """
        # Calculate current grid dimensions
        computed_cols = int(np.sqrt(num_images * (self.canvas_width / self.canvas_height)))
        cols = max(1, computed_cols)
        rows = max(1, int(np.ceil(num_images / cols)))
        
        # Calculate how many images would fit in a complete grid
        complete_grid_images = cols * rows
        
        # Calculate how many images are in the last row
        images_in_last_row = num_images % cols if cols > 0 else 0
        
        # Find the next perfect grid sizes
        next_perfect_grids = []
        prev_perfect_grids = []
        
        # Look for next perfect grids (add images)
        for i in range(1, 11):  # Check next 10 possibilities
            test_images = num_images + i
            if test_images > 200:  # Don't suggest more than 200 images
                break
            test_cols = int(np.sqrt(test_images * (self.canvas_width / self.canvas_height)))
            test_rows = int(np.ceil(test_images / test_cols))
            if test_images == test_cols * test_rows:  # Perfect grid
                next_perfect_grids.append({
                    'images_needed': i,
                    'total_images': test_images,
                    'cols': test_cols,
                    'rows': test_rows
                })
                if len(next_perfect_grids) >= 3:  # Limit to 3 suggestions
                    break
        
        # Look for previous perfect grids (remove images)
        for i in range(1, min(11, num_images)):  # Check previous possibilities
            test_images = num_images - i
            if test_images < 2:  # Minimum 2 images required
                break
            test_cols = int(np.sqrt(test_images * (self.canvas_width / self.canvas_height)))
            test_rows = int(np.ceil(test_images / test_cols))
            if test_images == test_cols * test_rows:  # Perfect grid
                prev_perfect_grids.append({
                    'images_to_remove': i,
                    'total_images': test_images,
                    'cols': test_cols,
                    'rows': test_rows
                })
                if len(prev_perfect_grids) >= 3:  # Limit to 3 suggestions
                    break
        
        # Find the closest perfect grid
        closest_perfect = None
        min_diff = float('inf')
        
        # Check current grid
        if num_images == complete_grid_images:
            closest_perfect = {
                'type': 'perfect',
                'total_images': num_images,
                'cols': cols,
                'rows': rows,
                'images_needed': 0,
                'images_to_remove': 0
            }
        else:
            # Check next perfect grids
            for grid in next_perfect_grids:
                if grid['images_needed'] < min_diff:
                    min_diff = grid['images_needed']
                    closest_perfect = {
                        'type': 'add_images',
                        'total_images': grid['total_images'],
                        'cols': grid['cols'],
                        'rows': grid['rows'],
                        'images_needed': grid['images_needed'],
                        'images_to_remove': 0
                    }
            
            # Check previous perfect grids
            for grid in prev_perfect_grids:
                if grid['images_to_remove'] < min_diff:
                    min_diff = grid['images_to_remove']
                    closest_perfect = {
                        'type': 'remove_images',
                        'total_images': grid['total_images'],
                        'cols': grid['cols'],
                        'rows': grid['rows'],
                        'images_needed': 0,
                        'images_to_remove': grid['images_to_remove']
                    }
        
        return {
            'current_grid': {
                'total_images': num_images,
                'cols': cols,
                'rows': rows,
                'images_in_last_row': images_in_last_row,
                'is_perfect': num_images == complete_grid_images
            },
            'closest_perfect_grid': closest_perfect,
            'recommendations': {
                'add_images': next_perfect_grids[:3],
                'remove_images': prev_perfect_grids[:3]
            },
            'canvas_info': {
                'width': self.canvas_width,
                'height': self.canvas_height,
                'spacing': self.spacing_pixels
            }
        }


# Rate limiting (simple in-memory implementation)
rate_limit_store = defaultdict(list)
RATE_LIMIT_REQUESTS = settings.rate_limit_requests  # requests per window
RATE_LIMIT_WINDOW = settings.rate_limit_window_seconds  # seconds

def check_rate_limit(client_ip: str) -> bool:
    """Simple rate limiting check"""
    now = time.time()
    # Clean old requests
    rate_limit_store[client_ip] = [
        req_time for req_time in rate_limit_store[client_ip]
        if now - req_time < RATE_LIMIT_WINDOW
    ]

    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False

    rate_limit_store[client_ip].append(now)
    return True

def validate_image_file(file_path: str) -> bool:
    """Validate that file is actually an image using magic numbers"""
    if not MAGIC_AVAILABLE:
        # Fallback to basic PIL validation
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    try:
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
    except Exception:
        return False

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal"""
    # Remove any path separators and dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Limit length
    filename = filename[:100]
    return filename

 







class CollageGenerator:
    """Generates the final collage image"""
    
    def __init__(self, config: CollageConfig):
        self.config = config
        # Convert mm to pixels: 1 inch = 25.4 mm, so mm / 25.4 = inches, then * dpi = pixels
        self.canvas_width = int((config.width_mm / 25.4) * config.dpi)
        self.canvas_height = int((config.height_mm / 25.4) * config.dpi)
        # Enforce safe canvas bounds
        if self.canvas_width * self.canvas_height > MAX_CANVAS_PIXELS:
            raise ValueError(
                f"Canvas too large: {self.canvas_width*self.canvas_height} pixels exceeds limit {MAX_CANVAS_PIXELS}"
            )
    
    def generate(self, image_blocks: List[ImageBlock], output_path: str) -> str:
        """Generate the final collage image"""
        # Create canvas (support RGBA when background has alpha)
        r, g, b, a = self._parse_color_rgba(self.config.background_color)
        if a < 255:
            canvas = Image.new('RGBA', (self.canvas_width, self.canvas_height), (r, g, b, a))
        else:
            canvas = Image.new('RGB', (self.canvas_width, self.canvas_height), (r, g, b))
        
        # Process each image block
        for block in image_blocks:
            try:
                with Image.open(block.image_path) as img:
                    # Normalize EXIF orientation (rotate/transpose to upright)
                    try:
                        img = ImageOps.exif_transpose(img)
                    except Exception:
                        pass
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Optional: pre-trim borders (e.g., black bars from screenshots)
                    if getattr(self.config, 'pretrim_borders', False):
                        try:
                            img = self._trim_borders(img)
                        except Exception:
                            # Fail safe: ignore border trim errors
                            pass
                    
                    # Resize to fit block
                    img_resized = self._smart_resize(img, block.width, block.height)
                    
                    # Apply effects
                    if self.config.apply_shadow:
                        img_resized = self._add_shadow(img_resized)
                    
                    # Paste onto canvas (respect alpha if present)
                    if img_resized.mode == 'RGBA':
                        canvas.paste(img_resized, (block.x, block.y), img_resized)
                    else:
                        canvas.paste(img_resized, (block.x, block.y))
            except Exception as e:
                print(f"Error processing image {block.image_path}: {e}")
                continue
        
        # Save the final image in the specified format
        if self.config.output_format == OutputFormat.JPEG:
            # JPEG does not support alpha
            if canvas.mode == 'RGBA':
                canvas = canvas.convert('RGB')
            canvas.save(output_path, 'JPEG', quality=95, dpi=(self.config.dpi, self.config.dpi))
        elif self.config.output_format == OutputFormat.PNG:
            # Save as-is; if alpha present, canvas is already RGBA
            canvas.save(output_path, 'PNG', dpi=(self.config.dpi, self.config.dpi))
        elif self.config.output_format == OutputFormat.TIFF:
            canvas.save(output_path, 'TIFF', dpi=(self.config.dpi, self.config.dpi), compression='tiff_lzw')
        
        return output_path
    
    def _smart_resize(self, img: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """Resize without distortion using scale-to-cover and center crop.

        If maintain_aspect_ratio is False in config, allow direct resize (stretching).
        """
        # If stretching is explicitly allowed, do a direct resize
        if not self.config.maintain_aspect_ratio:
            return img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Scale to cover the target box while preserving aspect ratio
        src_w, src_h = img.width, img.height
        scale = max(target_width / src_w, target_height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Face-aware crop if enabled
        if getattr(self.config, 'face_aware_cropping', False):
            try:
                faces = self._detect_faces(img)
            except Exception:
                faces = []
            if faces:
                # Weighted center by area and score
                weights = []
                centers_x = []
                centers_y = []
                for (x1, y1, x2, y2, score) in faces:
                    area = max(1, (x2 - x1)) * max(1, (y2 - y1))
                    w = max(1e-3, area * max(1e-3, score))
                    cx = x1 + (x2 - x1) / 2.0
                    cy = y1 + (y2 - y1) / 2.0
                    weights.append(w)
                    centers_x.append(cx * w)
                    centers_y.append(cy * w)
                sum_w = sum(weights)
                if sum_w > 0:
                    center_x = sum(centers_x) / sum_w
                    center_y = sum(centers_y) / sum_w
                else:
                    center_x = new_w / 2.0
                    center_y = new_h / 2.0

                # Compute crop window around weighted center
                left = int(round(center_x - (target_width / 2.0)))
                top = int(round(center_y - (target_height / 2.0)))
                left = max(0, min(left, new_w - target_width))
                top = max(0, min(top, new_h - target_height))
                right = left + target_width
                bottom = top + target_height
                return img.crop((left, top, right, bottom))

        # Fallback: Center-crop to the exact target size
        left = max(0, (new_w - target_width) // 2)
        top = max(0, (new_h - target_height) // 2)
        right = left + target_width
        bottom = top + target_height
        return img.crop((left, top, right, bottom))
    
    def _add_shadow(self, img: Image.Image) -> Image.Image:
        """Add a soft, offset drop shadow based on spacing.

        Offset is 50% of spacing (both x and y). Shadow is blurred and slightly spread for a natural look.
        """
        # Calculate spacing in pixels to derive shadow parameters
        spacing_pixels = int(min(self.canvas_width, self.canvas_height) * (self.config.spacing / 100.0) * 0.05)
        offset = max(1, int(round(spacing_pixels * 0.12)))
        blur_radius = max(3, int(round(spacing_pixels * 0.6)))
        spread_px = max(0, int(round(spacing_pixels * 0.08)))
        alpha_base = max(50, min(100, 80 + spacing_pixels // 8))

        # Ensure the original is RGBA for correct compositing
        base_rgb = img
        base = img.convert('RGBA')

        # Pad left/top only if blur would otherwise clip there (keep image anchored at top-left)
        left_top_pad = max(0, blur_radius - offset)
        right_bottom_pad = blur_radius + spread_px + offset

        shadow_w = base.width + left_top_pad + right_bottom_pad
        shadow_h = base.height + left_top_pad + right_bottom_pad
        # Build rounded-rectangle alpha mask for softer corners
        corner_radius = max(2, int(round(min(base.width, base.height) * 0.03)))
        mask = Image.new('L', (base.width, base.height), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, base.width, base.height], radius=corner_radius, fill=255)

        # Create alpha canvas, paste mask at offset, then blur for softness
        alpha_canvas = Image.new('L', (shadow_w, shadow_h), 0)
        alpha_canvas.paste(mask, (left_top_pad + offset, left_top_pad + offset))
        alpha_blurred = alpha_canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Apply a directional gradient to reduce visibility on top/left (less overlap, less rectangular look)
        alpha_arr = (np.array(alpha_blurred, dtype=np.float32) / 255.0)
        h, w = alpha_arr.shape
        yy, xx = np.mgrid[0:h, 0:w]
        grad_x = np.clip(xx / max(1, w - 1), 0.0, 1.0) ** 0.6
        grad_y = np.clip(yy / max(1, h - 1), 0.0, 1.0) ** 0.6
        grad = grad_x * grad_y
        alpha_scaled = np.clip(alpha_arr * grad * (alpha_base / 255.0), 0.0, 1.0)
        alpha_final = (alpha_scaled * 255).astype(np.uint8)

        # Build shadow from computed alpha
        shadow = Image.new('RGBA', (shadow_w, shadow_h), (0, 0, 0, 0))
        shadow.putalpha(Image.fromarray(alpha_final))

        # Composite: place the original image at the top-left anchor (account for left/top pad only)
        result = Image.new('RGBA', (shadow_w, shadow_h), (255, 255, 255, 0))
        result.paste(shadow, (0, 0))
        result.paste(base_rgb, (left_top_pad, left_top_pad))

        # Draw a subtle inner white border for added contrast
        border_thickness = min(12, max(2, int(round(spacing_pixels * 0.14))))
        if border_thickness > 0:
            x0 = left_top_pad
            y0 = left_top_pad
            x1 = x0 + base.width
            y1 = y0 + base.height
            border_draw = ImageDraw.Draw(result)
            # Top edge
            border_draw.rectangle([x0, y0, x1 - 1, y0 + border_thickness - 1], fill=(255, 255, 255, 255))
            # Bottom edge
            border_draw.rectangle([x0, y1 - border_thickness, x1 - 1, y1 - 1], fill=(255, 255, 255, 255))
            # Left edge
            border_draw.rectangle([x0, y0, x0 + border_thickness - 1, y1 - 1], fill=(255, 255, 255, 255))
            # Right edge
            border_draw.rectangle([x1 - border_thickness, y0, x1 - 1, y1 - 1], fill=(255, 255, 255, 255))

        return result
    
    def _parse_color(self, color_str: str) -> Tuple[int, int, int]:
        """Parse color string to RGB tuple"""
        if color_str.startswith('#'):
            color_str = color_str[1:]
            return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))
        return (255, 255, 255)  # Default white

    def _parse_color_rgba(self, color_str: str) -> Tuple[int, int, int, int]:
        """Parse #RRGGBB or #RRGGBBAA to RGBA tuple."""
        if isinstance(color_str, str) and color_str.startswith('#'):
            hex_str = color_str[1:]
            if len(hex_str) == 8:
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                a = int(hex_str[6:8], 16)
                return (r, g, b, a)
            elif len(hex_str) == 6:
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                return (r, g, b, 255)
        return (255, 255, 255, 255)

    def _detect_faces(self, img: Image.Image) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using MediaPipe; returns list of (x1, y1, x2, y2, score) in pixel coords.

        Runs on the provided image dimensions. Expects RGB PIL image.
        """
        global _MEDIAPIPE_IMPORT_TRIED, _MEDIAPIPE_AVAILABLE, mp, _FACE_DETECTOR
        if not _MEDIAPIPE_IMPORT_TRIED:
            # Reduce TensorFlow/absl/glog verbosity prior to importing mediapipe
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all,1=INFO,2=WARNING,3=ERROR
            os.environ.setdefault("GLOG_minloglevel", "3")       # 0=INFO,1=WARNING,2=ERROR,3=FATAL
            os.environ.setdefault("absl_logging_minloglevel", "3")
            try:
                import mediapipe as mp  # type: ignore
                try:
                    from absl import logging as absl_logging  # type: ignore
                    absl_logging.set_verbosity(absl_logging.ERROR)
                except Exception:
                    pass
                _MEDIAPIPE_AVAILABLE = True
            except Exception:
                mp = None  # type: ignore
                _MEDIAPIPE_AVAILABLE = False
            _MEDIAPIPE_IMPORT_TRIED = True

        if not _MEDIAPIPE_AVAILABLE or mp is None:  # type: ignore
            return []
        np_img = np.asarray(img)
        # Ensure 3-channel RGB
        if np_img.ndim == 2:
            np_img = np.stack([np_img] * 3, axis=-1)
        if np_img.shape[2] == 4:
            np_img = np_img[:, :, :3]

        h, w = np_img.shape[0], np_img.shape[1]
        detections_out: List[Tuple[int, int, int, int, float]] = []

        # Reuse a single detector instance to avoid repeated init (and repeated warnings)
        if _FACE_DETECTOR is None:
            try:
                _FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(  # type: ignore
                    model_selection=1,
                    min_detection_confidence=0.5,
                )
            except Exception:
                _FACE_DETECTOR = None
                return []

        results = None
        try:
            results = _FACE_DETECTOR.process(np_img)  # type: ignore
        except Exception:
            # Try to recreate once if detector got into a bad state
            try:
                _FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(  # type: ignore
                    model_selection=1,
                    min_detection_confidence=0.5,
                )
                results = _FACE_DETECTOR.process(np_img)  # type: ignore
            except Exception:
                return []

        if not results or not getattr(results, 'detections', None):
            return []

        for det in results.detections:  # type: ignore
            score = float(det.score[0]) if det.score else 0.0
            rel = det.location_data.relative_bounding_box
            x = max(0.0, float(rel.xmin)) * w
            y = max(0.0, float(rel.ymin)) * h
            bw = max(0.0, float(rel.width)) * w
            bh = max(0.0, float(rel.height)) * h
            x1 = int(max(0, round(x)))
            y1 = int(max(0, round(y)))
            x2 = int(min(w, round(x + bw)))
            y2 = int(min(h, round(y + bh)))
            if x2 > x1 and y2 > y1:
                # Expand a bit using face_margin but keep inside image
                margin = getattr(self.config, 'face_margin', 0.08)
                mx = int(round((x2 - x1) * margin))
                my = int(round((y2 - y1) * margin))
                x1 = max(0, x1 - mx)
                y1 = max(0, y1 - my)
                x2 = min(w, x2 + mx)
                y2 = min(h, y2 + my)
                detections_out.append((x1, y1, x2, y2, score))
        return detections_out

    def _trim_borders(self, img: Image.Image) -> Image.Image:
        """Trim solid uniform borders (e.g., black bars) from edges conservatively.

        Detects near-constant top/bottom rows and left/right columns and crops them
        if they exceed a small fraction of the dimension.
        """
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        h, w = arr.shape[0], arr.shape[1]
        if h < 40 or w < 40:
            return img

        # Luminance
        lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
        std_thresh = 2.0
        dark_thresh = 16.0
        bright_thresh = 239.0
        min_frac = 0.04

        def scan_rows(start: int, step: int) -> int:
            count = 0
            i = start
            while 0 <= i < h:
                row = lum[i, :]
                row_std = float(row.std())
                row_mean = float(row.mean())
                if row_std < std_thresh and (row_mean <= dark_thresh or row_mean >= bright_thresh):
                    count += 1
                    i += step
                else:
                    break
            return count

        def scan_cols(start: int, step: int) -> int:
            count = 0
            j = start
            while 0 <= j < w:
                col = lum[:, j]
                col_std = float(col.std())
                col_mean = float(col.mean())
                if col_std < std_thresh and (col_mean <= dark_thresh or col_mean >= bright_thresh):
                    count += 1
                    j += step
                else:
                    break
            return count

        top_bar = scan_rows(0, 1)
        bottom_bar = scan_rows(h - 1, -1)
        left_bar = scan_cols(0, 1)
        right_bar = scan_cols(w - 1, -1)

        # Require minimum fractional size to avoid accidental trims
        min_rows = int(h * min_frac)
        min_cols = int(w * min_frac)
        top_bar = top_bar if top_bar >= min_rows else 0
        bottom_bar = bottom_bar if bottom_bar >= min_rows else 0
        left_bar = left_bar if left_bar >= min_cols else 0
        right_bar = right_bar if right_bar >= min_cols else 0

        new_left = left_bar
        new_top = top_bar
        new_right = w - right_bar
        new_bottom = h - bottom_bar

        if new_left < new_right and new_top < new_bottom and (left_bar or right_bar or top_bar or bottom_bar):
            return img.crop((new_left, new_top, new_right, new_bottom))
        return img

class CollageGeneratorPixels(CollageGenerator):
    """Generator variant that accepts pixel-based dimensions directly."""

    def __init__(self, config: CollagePixelConfig):
        self.config = config
        self.canvas_width = int(config.width_px)
        self.canvas_height = int(config.height_px)
        if self.canvas_width * self.canvas_height > MAX_CANVAS_PIXELS:
            raise ValueError(
                f"Canvas too large: {self.canvas_width*self.canvas_height} pixels exceeds limit {MAX_CANVAS_PIXELS}"
            )

async def process_collage(job_id: str, image_paths: List[str], config: CollageConfig):
    """Background task to process collage generation"""
    try:
        # Update status in Redis
        await update_job(job_id, {"status": JobStatus.PROCESSING.value, "progress": 10})
        
        # Initialize generator
        generator = CollageGenerator(config)
        
        # Choose packer based on layout style
        if config.layout_style == LayoutStyle.MASONRY:
            packer = MasonryPacker(
                generator.canvas_width,
                generator.canvas_height,
                config.spacing
            )
            blocks = packer.pack_images(image_paths, config.maintain_aspect_ratio)
        elif config.layout_style == LayoutStyle.GRID:
            packer = GridPacker(
                generator.canvas_width,
                generator.canvas_height,
                config.spacing
            )
            blocks = packer.pack_images(image_paths)
        else:
            # Default to masonry
            packer = MasonryPacker(
                generator.canvas_width,
                generator.canvas_height,
                config.spacing
            )
            blocks = packer.pack_images(image_paths, config.maintain_aspect_ratio)
        
        await update_job(job_id, {"progress": 50})
        
        # Generate collage
        file_extension = config.output_format.value
        output_filename = f"collage_{job_id}.{file_extension}"
        output_path = OUTPUT_DIR / output_filename
        generator.generate(blocks, str(output_path))

        # Update job status
        await update_job(job_id, {
            "status": JobStatus.COMPLETED.value,
            "completed_at": datetime.now().isoformat(),
            "output_file": output_filename,
            "progress": 100,
        })
        
    except Exception as e:
        await update_job(job_id, {"status": JobStatus.FAILED.value, "error_message": str(e), "progress": 0})

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "endpoints": {
            "create_collage": "/api/collage/create",
            "create_collage_pixels": "/api/collage/create-pixels",
            "get_status": "/api/collage/status/{job_id}",
            "download": "/api/collage/download/{job_id}",
            "list_jobs": "/api/collage/jobs"
        }
    }

@app.post("/api/collage/create", response_model=CreateCollageResponse)
async def create_collage(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    width_mm: float = Form(default=304.8, ge=50, le=1219.2),  # 12 inches = 304.8 mm, 2-48 inches = 50-1219.2 mm
    height_mm: float = Form(default=457.2, ge=50, le=1219.2),  # 18 inches = 457.2 mm, 2-48 inches = 50-1219.2 mm
    dpi: int = Form(default=150, ge=72, le=300),
    layout_style: LayoutStyle = Form(default=LayoutStyle.MASONRY),
    spacing: float = Form(default=40.0, ge=0.0, le=100.0),
    background_color: str = Form(default="#FFFFFF"),
    maintain_aspect_ratio: bool = Form(default=True),
    apply_shadow: bool = Form(default=False),
    output_format: OutputFormat = Form(default=OutputFormat.JPEG),
    face_aware_cropping: bool = Form(default=False),
    face_margin: float = Form(default=0.08, ge=0.0, le=0.3),
    pretrim_borders: bool = Form(default=False)
):
    """Create a new collage from uploaded images"""

    # Log incoming request details for debugging
    file_details = ", ".join([f"{f.filename} ({f.size if hasattr(f, 'size') else 'unknown size'})" for f in files if f.filename])
    logger.info(f"Incoming request: {len(files)} files received - Parameters: width={width_mm}mm, height={height_mm}mm, dpi={dpi}, layout={layout_style.value}, spacing={spacing}% (scaled), bg_color={background_color}, maintain_ratio={maintain_aspect_ratio}, shadow={apply_shadow}, format={output_format.value}")
    logger.info(f"Files details: {file_details}")

    logger.info(f"Creating collage with {len(files)} files, layout: {layout_style}")

    # Validate file count
    if len(files) < 2:
        logger.warning("Collage creation failed: insufficient files")
        raise HTTPException(status_code=400, detail="At least 2 images required")
    if len(files) > 200:
        logger.warning("Collage creation failed: too many files")
        raise HTTPException(status_code=400, detail="Maximum 200 images allowed")

    # Create job
    job_id = str(uuid.uuid4())
    job = CollageJob(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.now(),
        progress=0
    )
    await save_job(job)

    # Save uploaded files
    image_paths = []
    total_size = 0

    for file in files:
        # Sanitize filename
        safe_filename = sanitize_filename(file.filename or "image.jpg")

        # Stream file to temp storage in chunks
        file_path = TEMP_DIR / f"{job_id}_{safe_filename}"
        bytes_written = 0
        async with aiofiles.open(file_path, 'wb') as out_f:
            while True:
                chunk = await file.read(STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                total_size += len(chunk)

                # Per-file limit
                if bytes_written > MAX_IMAGE_SIZE:
                    await out_f.close()
                    try:
                        file_path.unlink()
                    except Exception:
                        pass
                    logger.warning(f"File {safe_filename} exceeds size limit")
                    raise HTTPException(status_code=400, detail=f"File {safe_filename} exceeds 10MB limit")

                # Total limit
                if total_size > MAX_TOTAL_SIZE:
                    await out_f.close()
                    try:
                        file_path.unlink()
                    except Exception:
                        pass
                    logger.warning("Total file size exceeds limit")
                    raise HTTPException(status_code=400, detail="Total file size exceeds 500MB limit")

                await out_f.write(chunk)

        # Validate file is actually an image
        if not validate_image_file(str(file_path)):
            file_path.unlink()  # Clean up invalid file
            logger.warning(f"Invalid image file: {safe_filename}")
            raise HTTPException(status_code=400, detail=f"File {safe_filename} is not a valid image")

        # Optional preflight: accumulate total source pixels and optionally pre-resize overly large images
        try:
            with Image.open(file_path) as im:
                src_w, src_h = im.width, im.height
                # Track cumulative pixel count
                if settings.preflight_enabled:
                    # keep an in-request accumulator on the function scope via closure (use list to be mutable)
                    if 'preflight_total_pixels' not in locals():
                        preflight_total_pixels = 0
                    preflight_total_pixels += (src_w * src_h)
                    total_pixels = preflight_total_pixels
                    if total_pixels > settings.preflight_max_total_source_pixels:
                        logger.warning("Preflight pixel budget exceeded")
                        raise HTTPException(status_code=400, detail="Total image pixels too large; reduce image sizes or count")

                # Optional pre-resize to cap max dimensions for very large sources
                if settings.pre_resize_enabled and max(src_w, src_h) > settings.pre_resize_max_dim:
                    scale = settings.pre_resize_max_dim / float(max(src_w, src_h))
                    new_w = max(1, int(src_w * scale))
                    new_h = max(1, int(src_h * scale))
                    im = ImageOps.exif_transpose(im)
                    im = im.convert('RGB') if im.mode != 'RGB' else im
                    im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    im.save(file_path, format='JPEG', quality=92)
        except HTTPException:
            raise
        except Exception:
            # If we cannot inspect image, fail safe
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Failed to process {safe_filename}")

        image_paths.append(str(file_path))

    # Create config once after files are processed
    config = CollageConfig(
            width_mm=width_mm,
            height_mm=height_mm,
            dpi=dpi,
            layout_style=layout_style,
            spacing=spacing,
            background_color=background_color,
            maintain_aspect_ratio=maintain_aspect_ratio,
            apply_shadow=apply_shadow,
            output_format=output_format,
            face_aware_cropping=face_aware_cropping,
            face_margin=face_margin,
            pretrim_borders=pretrim_borders
        )

    logger.info(f"Collage job {job_id} created with {len(image_paths)} images")

    # Enqueue background processing via Celery
    config_payload = {
        "width_mm": config.width_mm,
        "height_mm": config.height_mm,
        "dpi": config.dpi,
        "layout_style": config.layout_style.value,
        "spacing": config.spacing,
        "background_color": config.background_color,
        "maintain_aspect_ratio": config.maintain_aspect_ratio,
        "apply_shadow": config.apply_shadow,
        "output_format": config.output_format.value,
        "face_aware_cropping": config.face_aware_cropping,
        "face_margin": config.face_margin,
        "pretrim_borders": config.pretrim_borders,
    }
    celery_app.send_task(
        "tasks.generate_collage_task",
        args=[job_id, image_paths, config_payload],
    )

    return CreateCollageResponse(job_id=job_id, status="pending", message="Collage generation started")

@app.post("/api/collage/create-pixels", response_model=CreateCollageResponse)
async def create_collage_pixels(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    width_px: int = Form(default=1920, ge=320, le=20000),
    height_px: int = Form(default=1080, ge=320, le=20000),
    dpi: int = Form(default=96, ge=72, le=300),
    layout_style: LayoutStyle = Form(default=LayoutStyle.MASONRY),
    spacing: float = Form(default=40.0, ge=0.0, le=100.0),
    background_color: str = Form(default="#FFFFFF"),
    maintain_aspect_ratio: bool = Form(default=True),
    apply_shadow: bool = Form(default=False),
    output_format: OutputFormat = Form(default=OutputFormat.JPEG),
    face_aware_cropping: bool = Form(default=False),
    face_margin: float = Form(default=0.08, ge=0.0, le=0.3),
    pretrim_borders: bool = Form(default=False)
):
    """Create a new collage from uploaded images using pixel dimensions."""

    file_details = ", ".join([f"{f.filename} ({f.size if hasattr(f, 'size') else 'unknown size'})" for f in files if f.filename])
    logger.info(f"Incoming pixel-based request: {len(files)} files - Parameters: width={width_px}px, height={height_px}px, dpi={dpi}, layout={layout_style.value}, spacing={spacing}% (scaled), bg_color={background_color}, maintain_ratio={maintain_aspect_ratio}, shadow={apply_shadow}, format={output_format.value}")
    logger.info(f"Files details: {file_details}")

    if len(files) < 2:
        logger.warning("Collage creation (pixels) failed: insufficient files")
        raise HTTPException(status_code=400, detail="At least 2 images required")
    if len(files) > 200:
        logger.warning("Collage creation (pixels) failed: too many files")
        raise HTTPException(status_code=400, detail="Maximum 200 images allowed")

    job_id = str(uuid.uuid4())
    job = CollageJob(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.now(),
        progress=0
    )
    await save_job(job)

    image_paths = []
    total_size = 0

    for file in files:
        safe_filename = sanitize_filename(file.filename or "image.jpg")

        file_path = TEMP_DIR / f"{job_id}_{safe_filename}"
        bytes_written = 0
        async with aiofiles.open(file_path, 'wb') as out_f:
            while True:
                chunk = await file.read(STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                total_size += len(chunk)

                if bytes_written > MAX_IMAGE_SIZE:
                    await out_f.close()
                    try:
                        file_path.unlink()
                    except Exception:
                        pass
                    logger.warning(f"File {safe_filename} exceeds size limit")
                    raise HTTPException(status_code=400, detail=f"File {safe_filename} exceeds 10MB limit")

                if total_size > MAX_TOTAL_SIZE:
                    await out_f.close()
                    try:
                        file_path.unlink()
                    except Exception:
                        pass
                    logger.warning("Total file size exceeds limit")
                    raise HTTPException(status_code=400, detail="Total file size exceeds 500MB limit")

                await out_f.write(chunk)

        if not validate_image_file(str(file_path)):
            file_path.unlink()
            logger.warning(f"Invalid image file: {safe_filename}")
            raise HTTPException(status_code=400, detail=f"File {safe_filename} is not a valid image")

        try:
            with Image.open(file_path) as im:
                src_w, src_h = im.width, im.height
                if settings.preflight_enabled:
                    if 'preflight_total_pixels' not in locals():
                        preflight_total_pixels = 0
                    preflight_total_pixels += (src_w * src_h)
                    total_pixels = preflight_total_pixels
                    if total_pixels > settings.preflight_max_total_source_pixels:
                        logger.warning("Preflight pixel budget exceeded")
                        raise HTTPException(status_code=400, detail="Total image pixels too large; reduce image sizes or count")

                if settings.pre_resize_enabled and max(src_w, src_h) > settings.pre_resize_max_dim:
                    scale = settings.pre_resize_max_dim / float(max(src_w, src_h))
                    new_w = max(1, int(src_w * scale))
                    new_h = max(1, int(src_h * scale))
                    im = ImageOps.exif_transpose(im)
                    im = im.convert('RGB') if im.mode != 'RGB' else im
                    im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    im.save(file_path, format='JPEG', quality=92)
        except HTTPException:
            raise
        except Exception:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Failed to process {safe_filename}")

        image_paths.append(str(file_path))

    config = CollagePixelConfig(
        width_px=width_px,
        height_px=height_px,
        dpi=dpi,
        layout_style=layout_style,
        spacing=spacing,
        background_color=background_color,
        maintain_aspect_ratio=maintain_aspect_ratio,
        apply_shadow=apply_shadow,
        output_format=output_format,
        face_aware_cropping=face_aware_cropping,
        face_margin=face_margin,
        pretrim_borders=pretrim_borders
    )

    logger.info(f"Collage job {job_id} (pixels) created with {len(image_paths)} images")

    config_payload = {
        "width_px": config.width_px,
        "height_px": config.height_px,
        "dpi": config.dpi,
        "layout_style": config.layout_style.value,
        "spacing": config.spacing,
        "background_color": config.background_color,
        "maintain_aspect_ratio": config.maintain_aspect_ratio,
        "apply_shadow": config.apply_shadow,
        "output_format": config.output_format.value,
        "face_aware_cropping": config.face_aware_cropping,
        "face_margin": config.face_margin,
        "pretrim_borders": config.pretrim_borders,
    }
    celery_app.send_task(
        "tasks.generate_collage_pixels_task",
        args=[job_id, image_paths, config_payload],
    )

    return CreateCollageResponse(job_id=job_id, status="pending", message="Collage generation started")

@app.get("/api/collage/status/{job_id}", response_model=CollageJobPublic)
async def get_status(job_id: str):
    """Get the status of a collage generation job"""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Coerce to model (enums as strings handled by Pydantic)
    return CollageJobPublic(
        job_id=job.get('job_id'),
        status=JobStatus(job.get('status')) if isinstance(job.get('status'), str) else job.get('status'),
        created_at=datetime.fromisoformat(job.get('created_at')) if isinstance(job.get('created_at'), str) else job.get('created_at'),
        completed_at=datetime.fromisoformat(job.get('completed_at')) if isinstance(job.get('completed_at'), str) and job.get('completed_at') else job.get('completed_at'),
        output_file=job.get('output_file'),
        error_message=job.get('error_message'),
        progress=int(job.get('progress') or 0),
    )

@app.get("/api/collage/download/{job_id}")
async def download_collage(job_id: str):
    """Download the generated collage"""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.get('status') != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Collage not ready yet")
    
    file_path = OUTPUT_DIR / job['output_file']
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    # Determine media type and filename based on file extension
    file_extension = file_path.suffix.lower()
    if file_extension == '.png':
        media_type = 'image/png'
    elif file_extension == '.tiff' or file_extension == '.tif':
        media_type = 'image/tiff'
    else:
        media_type = 'image/jpeg'
    
    # Add download headers: Content-Disposition, ETag, Cache-Control
    # Compute a weak ETag based on file mtime and size
    stat = file_path.stat()
    etag_src = f"{stat.st_mtime_ns}-{stat.st_size}".encode()
    etag = hashlib.md5(etag_src).hexdigest()  # nosec - not for security, only caching tag

    resp = FileResponse(
        path=file_path,
        media_type=media_type,
        filename=job['output_file']
    )
    resp.headers['Content-Disposition'] = f"attachment; filename=\"{job['output_file']}\""
    resp.headers['ETag'] = f"W/\"{etag}\""
    resp.headers['Cache-Control'] = 'private, max-age=31536000, immutable'
    return resp



@app.get("/api/collage/jobs", response_model=List[CollageJobPublic])
async def list_jobs():
    """List all collage generation jobs"""
    jobs = await list_all_jobs()
    results: List[CollageJobPublic] = []
    for job in jobs:
        results.append(
            CollageJobPublic(
                job_id=job.get('job_id'),
                status=JobStatus(job.get('status')) if isinstance(job.get('status'), str) else job.get('status'),
                created_at=datetime.fromisoformat(job.get('created_at')) if isinstance(job.get('created_at'), str) else job.get('created_at'),
                completed_at=datetime.fromisoformat(job.get('completed_at')) if isinstance(job.get('completed_at'), str) and job.get('completed_at') else job.get('completed_at'),
                output_file=job.get('output_file'),
                error_message=job.get('error_message'),
                progress=int(job.get('progress') or 0),
            )
        )
    return results

@app.delete("/api/collage/cleanup/{job_id}", response_model=CleanupResponse)
async def cleanup_job(job_id: str):
    """Clean up temporary files for a job"""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Clean up temp files
    for file in TEMP_DIR.glob(f"{job_id}_*"):
        file.unlink()
    
    # Clean up output file if exists
    if job.get('output_file'):
        output_file = OUTPUT_DIR / job['output_file']
        if output_file.exists():
            output_file.unlink()
    
    # Remove from Redis
    await delete_job(job_id)
    
    return CleanupResponse(message="Job cleaned up successfully")

@app.post("/api/collage/optimize-grid")
async def optimize_grid(
    num_images: int = Form(..., ge=2, le=200),
    width_mm: float = Form(default=304.8, ge=50, le=1219.2),  # 12 inches = 304.8 mm, 2-48 inches = 50-1219.2 mm
    height_mm: float = Form(default=457.2, ge=50, le=1219.2),  # 18 inches = 457.2 mm, 2-48 inches = 50-1219.2 mm
    dpi: int = Form(default=150, ge=72, le=300),
    spacing: float = Form(default=40.0, ge=0.0, le=100.0)
):
    """
    Calculate optimal grid dimensions and provide recommendations for perfect grid layout
    
    This endpoint helps frontend applications determine how many images to add or remove
    to achieve a perfect even grid without incomplete rows.
    """
    try:
        # Calculate canvas dimensions
        # Convert mm to pixels: 1 inch = 25.4 mm, so mm / 25.4 = inches, then * dpi = pixels
        canvas_width = int((width_mm / 25.4) * dpi)
        canvas_height = int((height_mm / 25.4) * dpi)
        
        # Create GridPacker instance
        packer = GridPacker(canvas_width, canvas_height, spacing)
        
        # Get optimization recommendations
        optimization = packer.calculate_optimal_grid(num_images)
        
        return {
            "success": True,
            "optimization": optimization,
            "message": "Grid optimization calculated successfully"
        }
        
    except Exception as e:
        logger.error(f"Grid optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Grid optimization failed: {str(e)}")

@app.post("/api/collage/analyze-masonry")
async def analyze_masonry_layout(
    num_images: int = Form(..., ge=2, le=200),
    width_mm: float = Form(default=304.8, ge=50, le=1219.2),
    height_mm: float = Form(default=457.2, ge=50, le=1219.2),
    dpi: int = Form(default=150, ge=72, le=300),
    spacing: float = Form(default=40.0, ge=0.0, le=100.0)
):
    """
    Analyze how the masonry layout algorithm would distribute photos
    
    This endpoint helps understand the layout algorithm without creating actual images.
    Returns detailed information about column distribution, photo sizing, and layout optimization.
    """
    try:
        # Calculate canvas dimensions
        canvas_width = int((width_mm / 25.4) * dpi)
        canvas_height = int((height_mm / 25.4) * dpi)
        
        # Create MasonryPacker instance
        packer = MasonryPacker(canvas_width, canvas_height, spacing)
        
        # Calculate optimal columns
        optimal_columns = packer._calculate_optimal_columns(num_images)
        
        # Distribute photos
        distribution = packer._distribute_photos(num_images, optimal_columns)
        
        # Calculate layout metrics
        column_width = (canvas_width - packer.spacing_pixels * (optimal_columns + 1)) / optimal_columns
        max_photos_per_column = max(distribution)
        min_photos_per_column = min(distribution)
        
        # Calculate spacing and coverage
        total_spacing_width = packer.spacing_pixels * (optimal_columns + 1)
        total_spacing_height = packer.spacing_pixels * (max_photos_per_column + 1)
        available_width = canvas_width - total_spacing_width
        available_height = canvas_height - total_spacing_height
        
        # Calculate efficiency metrics
        width_utilization = (available_width / canvas_width) * 100
        height_utilization = (available_height / canvas_height) * 100
        
        # Calculate average photo dimensions
        avg_photo_width = column_width
        avg_photo_height = available_height / max_photos_per_column
        avg_aspect_ratio = avg_photo_width / avg_photo_height
        
        # Generate sample layout preview
        sample_layout = []
        photo_index = 0
        for col in range(optimal_columns):
            photos_in_column = distribution[col]
            x = packer.spacing_pixels + col * (column_width + packer.spacing_pixels)
            y = packer.spacing_pixels
            
            for row in range(photos_in_column):
                sample_layout.append({
                    'column': col,
                    'row': row,
                    'x': int(x),
                    'y': int(y),
                    'width': int(column_width),
                    'height': int(avg_photo_height),
                    'photo_index': photo_index
                })
                y += avg_photo_height + packer.spacing_pixels
                photo_index += 1
        
        return {
            "success": True,
            "analysis": {
                "canvas": {
                    "width_px": canvas_width,
                    "height_px": canvas_height,
                    "width_mm": width_mm,
                    "height_mm": height_mm,
                    "dpi": dpi
                },
                "layout": {
                    "optimal_columns": optimal_columns,
                    "column_width": int(column_width),
                    "spacing_pixels": packer.spacing_pixels,
                    "spacing_percent": spacing
                },
                "distribution": {
                    "total_photos": num_images,
                    "photos_per_column": distribution,
                    "max_photos_per_column": max_photos_per_column,
                    "min_photos_per_column": min_photos_per_column,
                    "distribution_evenness": "even" if max_photos_per_column == min_photos_per_column else "uneven"
                },
                "efficiency": {
                    "width_utilization_percent": round(width_utilization, 2),
                    "height_utilization_percent": round(height_utilization, 2),
                    "overall_coverage": round((width_utilization + height_utilization) / 2, 2)
                },
                "photo_metrics": {
                    "average_width": int(avg_photo_width),
                    "average_height": int(avg_photo_height),
                    "average_aspect_ratio": round(avg_aspect_ratio, 3),
                    "aspect_ratio_quality": "square" if 0.8 <= avg_aspect_ratio <= 1.2 else "wide" if avg_aspect_ratio > 1.2 else "tall"
                },
                "sample_layout": sample_layout
            },
            "algorithm_features": {
                "dynamic_column_calculation": True,
                "even_distribution": True,
                "full_canvas_coverage": True,
                "aspect_ratio_consideration": True,
                "flexible_spacing": True
            },
            "message": "Masonry layout analysis completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Masonry analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Masonry analysis failed: {str(e)}")

def _log_json(event: str, **kwargs):
    try:
        record = {"event": event, **kwargs}
        logger.info(json.dumps(record, default=str))
    except Exception:
        # Fallback to plain logging if JSON serialization fails
        logger.info(f"{event} | {kwargs}")


# Request logging + Request ID middleware
@app.middleware("http")
async def log_requests(request, call_next):
    # Correlation/Request ID
    req_id = request.headers.get("X-Request-ID") or request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    request.state.request_id = req_id

    # Client IP for rate limiting
    client_ip = request.client.host if request.client else "unknown"

    # Rate limit check
    if not check_rate_limit(client_ip):
        _log_json(
            "rate_limit_exceeded",
            request_id=req_id,
            method=request.method,
            path=request.url.path,
            client_ip=client_ip,
        )
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Please try again later.", "request_id": req_id}
        )

    start_time = datetime.now()
    _log_json(
        "request_start",
        request_id=req_id,
        method=request.method,
        path=request.url.path,
        client_ip=client_ip,
    )

    response = await call_next(request)
    elapsed = (datetime.now() - start_time).total_seconds()
    process_time_ms = elapsed * 1000

    # Add response headers
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    # Modern headers
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-site"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Minimal, self-only CSP by default (adjust if serving UI/assets)
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"

    _log_json(
        "request_end",
        request_id=req_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(process_time_ms, 2),
        client_ip=client_ip,
    )

    # Prometheus metrics
    try:
        REQUEST_COUNT.labels(method=request.method, path=request.url.path, status=response.status_code).inc()
        REQUEST_LATENCY.labels(method=request.method, path=request.url.path).observe(elapsed)
    except Exception:
        pass

    return response


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    try:
        # Update job gauges
        ACTIVE_JOBS.set(await count_active_jobs())
        TOTAL_JOBS.set(await count_total_jobs())
    except Exception:
        # If Redis is unavailable, keep previous values
        pass
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Health check
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check file system
        temp_space = shutil.disk_usage(TEMP_DIR)
        output_space = shutil.disk_usage(OUTPUT_DIR)

        # Check active jobs (from Redis)
        active_jobs = await count_active_jobs()

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": settings.app_version,
            "checks": {
                "filesystem": {
                    "temp_dir": str(TEMP_DIR),
                    "temp_space_gb": temp_space.free / (1024**3),
                    "output_dir": str(OUTPUT_DIR),
                    "output_space_gb": output_space.free / (1024**3),
                    "healthy": temp_space.free > 1024**3 and output_space.free > 1024**3  # 1GB free
                },
                "jobs": {
                    "total_jobs": await count_total_jobs(),
                    "active_jobs": active_jobs,
                    "healthy": active_jobs < 50  # Reasonable limit
                },
                "dependencies": {
                    "magic_available": MAGIC_AVAILABLE,
                    "redis_connected": await is_redis_connected(),
                    "healthy": True
                }
            }
        }

        # Determine overall health
        all_checks_healthy = all(
            check.get("healthy", False)
            for check in health_status["checks"].values()
        )

        if not all_checks_healthy:
            health_status["status"] = "unhealthy"
            logger.warning("Health check failed", extra=health_status)

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
