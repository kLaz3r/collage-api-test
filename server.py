"""
FastAPI Collage Maker Application
A web API for creating high-resolution photo collages with masonry layout
"""

import io
import os
import uuid
import json
import random
import asyncio
import re
import tempfile
import shutil
import logging
import time
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Request, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# Security imports
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Collage Maker API",
    description="Create beautiful photo collages with masonry layout",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB per image
MAX_TOTAL_SIZE = 500 * 1024 * 1024  # 500MB total

# Create directories
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# Store job status (in production, use Redis)
job_status = {}

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

    @validator('background_color')
    def validate_color(cls, v):
        """Validate hex color format - supports #RRGGBB and #RRGGBBAA (with alpha)"""
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

        # Calculate grid dimensions
        cols = int(np.sqrt(num_images * (self.canvas_width / self.canvas_height)))
        rows = int(np.ceil(num_images / cols))

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
        cols = int(np.sqrt(num_images * (self.canvas_width / self.canvas_height)))
        rows = int(np.ceil(num_images / cols))
        
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
RATE_LIMIT_REQUESTS = 100  # requests per window
RATE_LIMIT_WINDOW = 60  # seconds

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Collage Maker API",
    description="Create beautiful photo collages with masonry layout",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB per image
MAX_TOTAL_SIZE = 500 * 1024 * 1024  # 500MB total

# Create directories
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)







class CollageGenerator:
    """Generates the final collage image"""
    
    def __init__(self, config: CollageConfig):
        self.config = config
        # Convert mm to pixels: 1 inch = 25.4 mm, so mm / 25.4 = inches, then * dpi = pixels
        self.canvas_width = int((config.width_mm / 25.4) * config.dpi)
        self.canvas_height = int((config.height_mm / 25.4) * config.dpi)
    
    def generate(self, image_blocks: List[ImageBlock], output_path: str) -> str:
        """Generate the final collage image"""
        # Create canvas
        bg_color = self._parse_color(self.config.background_color)
        canvas = Image.new('RGB', (self.canvas_width, self.canvas_height), bg_color)
        
        # Process each image block
        for block in image_blocks:
            try:
                with Image.open(block.image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to fit block
                    img_resized = self._smart_resize(img, block.width, block.height)
                    
                    # Apply effects
                    if self.config.apply_shadow:
                        img_resized = self._add_shadow(img_resized)
                    
                    # Paste onto canvas
                    canvas.paste(img_resized, (block.x, block.y))
            except Exception as e:
                print(f"Error processing image {block.image_path}: {e}")
                continue
        
        # Save the final image in the specified format
        if self.config.output_format == OutputFormat.JPEG:
            canvas.save(output_path, 'JPEG', quality=95, dpi=(self.config.dpi, self.config.dpi))
        elif self.config.output_format == OutputFormat.PNG:
            # Convert to RGBA for PNG to support transparency if needed
            if self.config.background_color == "#00000000":  # Transparent background
                canvas = canvas.convert('RGBA')
                # Make background transparent
                data = np.array(canvas)
                # Find white pixels and make them transparent
                white_pixels = np.all(data[:, :, :3] == [255, 255, 255], axis=2)
                data[white_pixels, 3] = 0
                canvas = Image.fromarray(data)
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

        # Center-crop to the exact target size
        left = max(0, (new_w - target_width) // 2)
        top = max(0, (new_h - target_height) // 2)
        right = left + target_width
        bottom = top + target_height
        return img.crop((left, top, right, bottom))
    
    def _add_shadow(self, img: Image.Image) -> Image.Image:
        """Add drop shadow effect to image"""
        # Create shadow
        shadow = Image.new('RGBA', (img.width + 20, img.height + 20), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rectangle([10, 10, img.width + 10, img.height + 10], fill=(0, 0, 0, 128))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=5))
        
        # Composite
        result = Image.new('RGBA', shadow.size, (255, 255, 255, 0))
        result.paste(shadow, (0, 0))
        result.paste(img, (0, 0))
        
        return result.convert('RGB')
    
    def _parse_color(self, color_str: str) -> Tuple[int, int, int]:
        """Parse color string to RGB tuple"""
        if color_str.startswith('#'):
            color_str = color_str[1:]
            return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))
        return (255, 255, 255)  # Default white

async def process_collage(job_id: str, image_paths: List[str], config: CollageConfig):
    """Background task to process collage generation"""
    try:
        # Update status
        job_status[job_id]['status'] = JobStatus.PROCESSING
        job_status[job_id]['progress'] = 10
        
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
        
        job_status[job_id]['progress'] = 50
        
        # Generate collage
        file_extension = config.output_format.value
        output_filename = f"collage_{job_id}.{file_extension}"
        output_path = OUTPUT_DIR / output_filename
        generator.generate(blocks, str(output_path))

        # Update job status
        job_status[job_id]['status'] = JobStatus.COMPLETED
        job_status[job_id]['completed_at'] = datetime.now()
        job_status[job_id]['output_file'] = output_filename
        job_status[job_id]['progress'] = 100
        
    except Exception as e:
        job_status[job_id]['status'] = JobStatus.FAILED
        job_status[job_id]['error_message'] = str(e)
        job_status[job_id]['progress'] = 0

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Collage Maker API",
        "version": "1.0.0",
        "endpoints": {
            "create_collage": "/api/collage/create",
            "get_status": "/api/collage/status/{job_id}",
            "download": "/api/collage/download/{job_id}",
            "list_jobs": "/api/collage/jobs"
        }
    }

@app.post("/api/collage/create")
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
    output_format: OutputFormat = Form(default=OutputFormat.JPEG)
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
    job_status[job_id] = job.dict()

    # Save uploaded files
    image_paths = []
    total_size = 0

    for file in files:
        # Sanitize filename
        safe_filename = sanitize_filename(file.filename or "image.jpg")

        # Read file
        contents = await file.read()
        total_size += len(contents)

        # Check size limits
        if len(contents) > MAX_IMAGE_SIZE:
            logger.warning(f"File {safe_filename} exceeds size limit")
            raise HTTPException(status_code=400, detail=f"File {safe_filename} exceeds 10MB limit")
        if total_size > MAX_TOTAL_SIZE:
            logger.warning("Total file size exceeds limit")
            raise HTTPException(status_code=400, detail="Total file size exceeds 500MB limit")

        # Save file temporarily
        file_path = TEMP_DIR / f"{job_id}_{safe_filename}"
        with open(file_path, 'wb') as f:
            f.write(contents)

        # Validate file is actually an image
        if not validate_image_file(str(file_path)):
            file_path.unlink()  # Clean up invalid file
            logger.warning(f"Invalid image file: {safe_filename}")
            raise HTTPException(status_code=400, detail=f"File {safe_filename} is not a valid image")

        image_paths.append(str(file_path))

            # Create config
        config = CollageConfig(
            width_mm=width_mm,
            height_mm=height_mm,
            dpi=dpi,
            layout_style=layout_style,
            spacing=spacing,
            background_color=background_color,
            maintain_aspect_ratio=maintain_aspect_ratio,
            apply_shadow=apply_shadow,
            output_format=output_format
        )

    logger.info(f"Collage job {job_id} created with {len(image_paths)} images")

    # Start background processing
    background_tasks.add_task(process_collage, job_id, image_paths, config)

    return {"job_id": job_id, "status": "pending", "message": "Collage generation started"}

@app.get("/api/collage/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a collage generation job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status[job_id]

@app.get("/api/collage/download/{job_id}")
async def download_collage(job_id: str):
    """Download the generated collage"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    
    if job['status'] != JobStatus.COMPLETED:
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
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=job['output_file']
    )



@app.get("/api/collage/jobs")
async def list_jobs():
    """List all collage generation jobs"""
    return list(job_status.values())

@app.delete("/api/collage/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up temporary files for a job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Clean up temp files
    for file in TEMP_DIR.glob(f"{job_id}_*"):
        file.unlink()
    
    # Clean up output file if exists
    if job_status[job_id].get('output_file'):
        output_file = OUTPUT_DIR / job_status[job_id]['output_file']
        if output_file.exists():
            output_file.unlink()
    
    # Remove from job status
    del job_status[job_id]
    
    return {"message": "Job cleaned up successfully"}

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

# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    # Get client IP for rate limiting
    client_ip = request.client.host if request.client else "unknown"

    # Check rate limit
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Please try again later."}
        )

    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds() * 1000

    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms - IP: {client_ip}"
    )

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' https://fastapi.tiangolo.com"

    return response

# Health check
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check file system
        temp_space = shutil.disk_usage(TEMP_DIR)
        output_space = shutil.disk_usage(OUTPUT_DIR)

        # Check active jobs
        active_jobs = sum(1 for job in job_status.values()
                         if job['status'] in [JobStatus.PENDING, JobStatus.PROCESSING])

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "checks": {
                "filesystem": {
                    "temp_dir": str(TEMP_DIR),
                    "temp_space_gb": temp_space.free / (1024**3),
                    "output_dir": str(OUTPUT_DIR),
                    "output_space_gb": output_space.free / (1024**3),
                    "healthy": temp_space.free > 1024**3 and output_space.free > 1024**3  # 1GB free
                },
                "jobs": {
                    "total_jobs": len(job_status),
                    "active_jobs": active_jobs,
                    "healthy": active_jobs < 50  # Reasonable limit
                },
                "dependencies": {
                    "magic_available": MAGIC_AVAILABLE,
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
