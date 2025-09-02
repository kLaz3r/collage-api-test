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
    """Implements masonry/bin packing algorithm for image layout"""

    def __init__(self, canvas_width: int, canvas_height: int, spacing_percent: float = 2.0):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.spacing_percent = spacing_percent
        self.spacing_pixels = int(min(canvas_width, canvas_height) * (spacing_percent / 100.0) * 0.05)
        self.blocks = []

    def pack_images(self, image_paths: List[str], maintain_aspect: bool = True) -> List[ImageBlock]:
        """Pack images using masonry layout algorithm"""
        blocks = []

        # Get image dimensions
        images_info = []
        for path in image_paths:
            with Image.open(path) as img:
                images_info.append({
                    'path': path,
                    'width': img.width,
                    'height': img.height,
                    'aspect': img.width / img.height
                })

        # Sort by area (largest first) for better packing
        images_info.sort(key=lambda x: x['width'] * x['height'], reverse=True)

        # Simple masonry layout - column-based approach
        num_columns = self._calculate_columns(len(images_info))

        # Add edge spacing to left and right sides
        edge_spacing = self.spacing_pixels
        total_spacing_width = (num_columns - 1) * self.spacing_pixels + 2 * edge_spacing

        # Use the absolute maximum available width - no unused space
        if len(images_info) > 80:
            # For 100 images, maximize space utilization
            available_width = self.canvas_width - total_spacing_width
            column_width = available_width // num_columns  # Use every available pixel
            if column_width < 30:  # Too small? Use fixed small size
                column_width = 30
        else:
            # Less aggressive for smaller counts
            available_width = self.canvas_width - total_spacing_width
            column_width = available_width // num_columns

        column_heights = [edge_spacing] * num_columns

        # First pass: place images in columns
        for img_info in images_info:
            # Calculate dimensions first
            if maintain_aspect:
                aspect = img_info['aspect']
                width = column_width
                height = int(width / aspect)
            else:
                # Instead of random heights, use proportional height based on image aspect ratio
                # but scaled to fit the column while filling available space more effectively
                aspect = img_info['aspect']
                width = column_width
                # Use a more reasonable height range that still varies but avoids extremes
                base_height = int(width / aspect)
                # Adjust to be more proportional to image content rather than random
                height = max(80, min(base_height, int(width * 1.2))) # Allow some variation but stay reasonable

            # Try to find a column where this image fits
            placed = False
            for attempt in range(num_columns * 2):  # Try multiple times
                min_col = column_heights.index(min(column_heights))
                y = column_heights[min_col]

                # If it fits in current position
                if y + height <= self.canvas_height:
                    x = edge_spacing + min_col * (column_width + self.spacing_pixels)
                    block = ImageBlock(x, y, width, height, img_info['path'])
                    blocks.append(block)
                    column_heights[min_col] += height + self.spacing_pixels
                    placed = True
                    break
                else:
                    # Make space by adjusting this column height
                    column_heights[min_col] = self.canvas_height - height - self.spacing_pixels

        # Second pass: fill remaining gaps
        remaining_images = [img for img in images_info if all(
            block.image_path != img['path'] for block in blocks
        )]

        # Sort remaining images by size (smallest first to fill gaps)
        remaining_images.sort(key=lambda x: x['width'] * x['height'])

        for img_info in remaining_images:
            # Calculate dimensions
            if maintain_aspect:
                aspect = img_info['aspect']
                width = column_width
                height = int(width / aspect)
            else:
                # Use proportional height based on image content instead of random
                aspect = img_info['aspect']
                width = column_width
                base_height = int(width / aspect)
                height = max(80, min(base_height, int(width * 1.2)))

            # Find the column with most space available
            column_spaces = []
            for col in range(num_columns):
                y = column_heights[col]
                available_height = self.canvas_height - y - self.spacing_pixels
                if height <= available_height:
                    column_spaces.append((col, available_height))

            if column_spaces:
                # Place in column with most available space
                column_spaces.sort(key=lambda x: x[1], reverse=True)
                target_col = column_spaces[0][0]  # Column with most space

                x = edge_spacing + target_col * (column_width + self.spacing_pixels)
                y = column_heights[target_col]
                block = ImageBlock(x, y, width, height, img_info['path'])
                blocks.append(block)
                column_heights[target_col] += height + self.spacing_pixels

        # Third pass: force fit any remaining images by scaling them down, prioritizing bottom filling
        still_remaining = [img for img in images_info if all(
            block.image_path != img['path'] for block in blocks
        )]

        # Sort by remaining space to prioritize columns that need more filling
        column_info = [(col, self.canvas_height - column_heights[col]) for col in range(num_columns)]
        column_info.sort(key=lambda x: x[1])  # Sort by remaining space (smallest first - need most filling)

        for img_info in still_remaining:
            # Try to fill the shortest column first to balance heights
            column_spaces = []
            for col in range(num_columns):
                available_height = self.canvas_height - column_heights[col]
                if available_height > 30:  # Lower minimum to fill more small spaces
                    column_spaces.append((col, available_height))

            if column_spaces:
                # Prioritize columns with least remaining space to balance heights
                column_spaces.sort(key=lambda x: x[1])  # Smallest remaining space first
                target_col = column_spaces[0][0]
                available_height = column_spaces[0][1] - self.spacing_pixels

                # Scale image to fit available space
                x = edge_spacing + target_col * (column_width + self.spacing_pixels)
                y = column_heights[target_col]

                # Fit to available space while maintaining aspect ratio
                scaled_height = min(available_height, int(column_width / img_info['aspect']))
                scaled_width = int(scaled_height * img_info['aspect'])

                # Ensure width fits in column
                if scaled_width > column_width:
                    scaled_width = column_width
                    scaled_height = int(scaled_width / img_info['aspect'])

                # Don't allow images smaller than 20px to avoid being too tiny
                if scaled_height < 20:
                    scaled_height = min(20, available_height)
                    scaled_width = int(scaled_height * img_info['aspect'])
                    if scaled_width > column_width:
                        scaled_width = column_width

                block = ImageBlock(x, y, scaled_width, scaled_height, img_info['path'])
                blocks.append(block)
                column_heights[target_col] += scaled_height + self.spacing_pixels

        return blocks

    def _calculate_columns(self, num_images: int) -> int:
        """Calculate optimal number of columns for vertical filling efficiency"""
        # For vertical filling, prioritize wider columns over many narrow ones
        if num_images <= 4:
            return 2
        elif num_images <= 9:
            return 3
        elif num_images <= 16:
            return 4
        elif num_images <= 25:
            return 5
        elif num_images <= 36:
            return 6
        elif num_images <= 49:
            return 7
        elif num_images <= 64:
            return 8
        elif num_images <= 81:
            return 9  # 9 columns for 64-81 images - optimal vertical filling
        elif num_images <= 100:
            return 10  # 10 columns for 82-100 images - balances width/height
        else:
            # For very high counts, still keep reasonable number
            return min(12, max(8, int(num_images ** 0.4) + 2))


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
        """Resize image with even cropping to maintain composition"""
        img_aspect = img.width / img.height
        target_aspect = target_width / target_height

        if abs(img_aspect - target_aspect) < 0.1:
            # Aspects are similar, simple resize
            return img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Enhanced cropping for more balanced crop distribution
        if img_aspect > target_aspect:
            # Image is wider - need to crop horizontally (sides)
            new_width = int(img.height * target_aspect)

            # Distribute crop more evenly from both sides instead of just center-cropping
            total_crop = img.width - new_width
            left_crop = total_crop // 3  # Crop more from left
            right_crop = total_crop - left_crop  # Crop less from right to preserve composition

            left = left_crop
            right = left + new_width
            img = img.crop((left, 0, right, img.height))

        else:
            # Image is taller - need to crop vertically (top/bottom)
            new_height = int(img.width / target_aspect)

            # Distribute crop more evenly from top and bottom instead of more from bottom
            total_crop = img.height - new_height
            top_crop = total_crop // 2       # Equal distribution from top
            bottom_crop = total_crop // 2     # Equal distribution from bottom

            top = top_crop
            bottom = top + new_height
            img = img.crop((0, top, img.width, bottom))

        return img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
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
