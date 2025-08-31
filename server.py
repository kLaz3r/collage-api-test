# main.py
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
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

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
    RANDOM = "random"
    SPIRAL = "spiral"

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Pydantic models
class CollageConfig(BaseModel):
    width_inches: float = Field(default=12, ge=4, le=48)
    height_inches: float = Field(default=18, ge=4, le=48)
    dpi: int = Field(default=150, ge=72, le=300)
    layout_style: LayoutStyle = LayoutStyle.MASONRY
    spacing: int = Field(default=10, ge=0, le=50)
    background_color: str = Field(default="#FFFFFF")
    maintain_aspect_ratio: bool = True
    apply_shadow: bool = False

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
    
    def __init__(self, canvas_width: int, canvas_height: int, spacing: int = 10):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.spacing = spacing
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
        column_width = (self.canvas_width - (num_columns - 1) * self.spacing) // num_columns
        column_heights = [0] * num_columns
        
        for img_info in images_info:
            # Find shortest column
            min_col = column_heights.index(min(column_heights))
            
            # Calculate position
            x = min_col * (column_width + self.spacing)
            y = column_heights[min_col]
            
            # Calculate dimensions
            if maintain_aspect:
                aspect = img_info['aspect']
                width = column_width
                height = int(width / aspect)
            else:
                width = column_width
                height = random.randint(150, 400)  # Random height for variety
            
            # Check if it fits
            if y + height <= self.canvas_height:
                block = ImageBlock(x, y, width, height, img_info['path'])
                blocks.append(block)
                column_heights[min_col] += height + self.spacing
        
        return blocks
    
    def _calculate_columns(self, num_images: int) -> int:
        """Calculate optimal number of columns based on image count"""
        if num_images <= 6:
            return 2
        elif num_images <= 15:
            return 3
        elif num_images <= 30:
            return 4
        elif num_images <= 60:
            return 5
        else:
            return 6

class GridPacker:
    """Implements grid layout for images"""
    
    def __init__(self, canvas_width: int, canvas_height: int, spacing: int = 10):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.spacing = spacing
    
    def pack_images(self, image_paths: List[str]) -> List[ImageBlock]:
        """Pack images in a uniform grid"""
        blocks = []
        num_images = len(image_paths)
        
        # Calculate grid dimensions
        cols = int(np.sqrt(num_images * (self.canvas_width / self.canvas_height)))
        rows = int(np.ceil(num_images / cols))
        
        # Calculate cell dimensions
        cell_width = (self.canvas_width - (cols - 1) * self.spacing) // cols
        cell_height = (self.canvas_height - (rows - 1) * self.spacing) // rows
        
        for i, path in enumerate(image_paths):
            row = i // cols
            col = i % cols
            
            x = col * (cell_width + self.spacing)
            y = row * (cell_height + self.spacing)
            
            block = ImageBlock(x, y, cell_width, cell_height, path)
            blocks.append(block)
        
        return blocks

class CollageGenerator:
    """Generates the final collage image"""
    
    def __init__(self, config: CollageConfig):
        self.config = config
        self.canvas_width = int(config.width_inches * config.dpi)
        self.canvas_height = int(config.height_inches * config.dpi)
    
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
        
        # Save the final image
        canvas.save(output_path, 'JPEG', quality=95, dpi=(self.config.dpi, self.config.dpi))
        return output_path
    
    def _smart_resize(self, img: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """Resize image with smart cropping to maintain composition"""
        img_aspect = img.width / img.height
        target_aspect = target_width / target_height
        
        if abs(img_aspect - target_aspect) < 0.1:
            # Aspects are similar, simple resize
            return img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Smart crop and resize
        if img_aspect > target_aspect:
            # Image is wider, crop width
            new_width = int(img.height * target_aspect)
            left = (img.width - new_width) // 2
            img = img.crop((left, 0, left + new_width, img.height))
        else:
            # Image is taller, crop height
            new_height = int(img.width / target_aspect)
            top = (img.height - new_height) // 4  # Crop more from bottom
            img = img.crop((0, top, img.width, top + new_height))
        
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
        output_filename = f"collage_{job_id}.jpg"
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
    width_inches: float = Query(default=12, ge=4, le=48),
    height_inches: float = Query(default=18, ge=4, le=48),
    dpi: int = Query(default=150, ge=72, le=300),
    layout_style: LayoutStyle = Query(default=LayoutStyle.MASONRY),
    spacing: int = Query(default=10, ge=0, le=50),
    background_color: str = Query(default="#FFFFFF"),
    maintain_aspect_ratio: bool = Query(default=True),
    apply_shadow: bool = Query(default=False)
):
    """Create a new collage from uploaded images"""
    
    # Validate file count
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 images required")
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images allowed")
    
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
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
        
        # Read file
        contents = await file.read()
        total_size += len(contents)
        
        # Check size limits
        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds 10MB limit")
        if total_size > MAX_TOTAL_SIZE:
            raise HTTPException(status_code=400, detail="Total file size exceeds 500MB limit")
        
        # Save file
        file_path = TEMP_DIR / f"{job_id}_{file.filename}"
        with open(file_path, 'wb') as f:
            f.write(contents)
        image_paths.append(str(file_path))
    
    # Create config
    config = CollageConfig(
        width_inches=width_inches,
        height_inches=height_inches,
        dpi=dpi,
        layout_style=layout_style,
        spacing=spacing,
        background_color=background_color,
        maintain_aspect_ratio=maintain_aspect_ratio,
        apply_shadow=apply_shadow
    )
    
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
    
    return FileResponse(
        path=file_path,
        media_type='image/jpeg',
        filename=f"collage_{job_id}.jpg"
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

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)