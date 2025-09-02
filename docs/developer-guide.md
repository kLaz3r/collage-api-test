# Developer Guide

Comprehensive guide for developers working with or extending the Collage Maker API.

## Architecture Overview

### System Components

The Collage Maker API is built with a modular architecture consisting of several key components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│  Background Jobs │────│   File System   │
│                 │    │                  │    │                 │
│ • REST Endpoints│    │ • Async Tasks    │    │ • Uploads/      │
│ • Request       │    │ • Job Tracking   │    │   Temp Files    │
│   Validation    │    │ • Progress       │    │ • Outputs       │
│ • CORS Support  │    │   Updates        │    │ • Cleanup       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────────────┐
                    │  Image Processing  │
                    │                    │
                    │ • Layout Algorithms│
                    │ • Image Resizing   │
                    │ • Shadow Effects   │
                    │ • Format Conversion│
                    └────────────────────┘
```

### Core Classes

#### `CollageGenerator`

The main orchestrator for collage creation:

```python
class CollageGenerator:
    def __init__(self, config: CollageConfig):
        self.config = config
        # Convert mm to pixels: 1 inch = 25.4 mm, so mm / 25.4 = inches, then * dpi = pixels
        self.canvas_width = int((config.width_mm / 25.4) * config.dpi)
        self.canvas_height = int((config.height_mm / 25.4) * config.dpi)

    def generate(self, image_blocks: List[ImageBlock], output_path: str) -> str:
        # Creates the final collage image
        pass
```

**Key Methods:**

-   `generate()`: Main collage creation method
-   `_smart_resize()`: Intelligent image resizing with cropping
-   `_add_shadow()`: Drop shadow effect application
-   `_parse_color()`: Color format parsing

#### Layout Packers

Three layout algorithms implement the `pack_images()` interface:

```python
class MasonryPacker:
    def pack_images(self, image_paths: List[str], maintain_aspect: bool = True) -> List[ImageBlock]:
        # Masonry layout algorithm
        pass

class GridPacker:
    def pack_images(self, image_paths: List[str]) -> List[ImageBlock]:
        # Grid layout algorithm
        pass

class RandomPacker:
    def pack_images(self, image_paths: List[str]) -> List[ImageBlock]:
        # Random layout algorithm
        pass
```

#### `ImageBlock`

Represents a single image placement in the collage:

```python
@dataclass
class ImageBlock:
    x: int          # X coordinate
    y: int          # Y coordinate
    width: int      # Block width
    height: int     # Block height
    image_path: str # Source image path
    image: Image = None  # PIL Image object (loaded on demand)
```

## Job Processing Flow

### Request Lifecycle

1. **Request Validation**

    ```python
    # Validate file count, sizes, and types
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 images required")
    ```

2. **File Upload & Storage**

    ```python
    # Save uploaded files to temp directory
    file_path = TEMP_DIR / f"{job_id}_{file.filename}"
    with open(file_path, 'wb') as f:
        f.write(contents)
    ```

3. **Job Creation**

    ```python
    job = CollageJob(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.now(),
        progress=0
    )
    job_status[job_id] = job.dict()
    ```

4. **Background Processing**
    ```python
    background_tasks.add_task(process_collage, job_id, image_paths, config)
    ```

### Background Task Processing

The `process_collage` function handles the actual image processing:

```python
async def process_collage(job_id: str, image_paths: List[str], config: CollageConfig):
    try:
        # Update status to processing
        job_status[job_id]['status'] = JobStatus.PROCESSING
        job_status[job_id]['progress'] = 10

        # Initialize generator and packer
        generator = CollageGenerator(config)

        # Choose layout algorithm
        if config.layout_style == LayoutStyle.MASONRY:
            packer = MasonryPacker(...)
        # ... other layout types

        # Pack images into layout
        blocks = packer.pack_images(image_paths, config.maintain_aspect_ratio)

        # Update progress
        job_status[job_id]['progress'] = 50

        # Generate final collage
        output_path = OUTPUT_DIR / f"collage_{job_id}.jpg"
        generator.generate(blocks, str(output_path))

        # Mark as completed
        job_status[job_id]['status'] = JobStatus.COMPLETED
        job_status[job_id]['completed_at'] = datetime.now()
        job_status[job_id]['output_file'] = output_filename
        job_status[job_id]['progress'] = 100

    except Exception as e:
        # Handle errors
        job_status[job_id]['status'] = JobStatus.FAILED
        job_status[job_id]['error_message'] = str(e)
```

## Layout Algorithms

### Masonry Layout Implementation

The masonry layout uses a bin-packing approach:

```python
def pack_images(self, image_paths: List[str], maintain_aspect: bool = True) -> List[ImageBlock]:
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

    # Column-based approach
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
            width = column_width
            height = int(width / img_info['aspect'])
        else:
            width = column_width
            height = random.randint(150, 400)

        # Create block
        block = ImageBlock(x, y, width, height, img_info['path'])
        blocks.append(block)

        # Update column height
        column_heights[min_col] += height + self.spacing

    return blocks
```

### Grid Layout Implementation

Uniform grid with equal cell sizes:

```python
def pack_images(self, image_paths: List[str]) -> List[ImageBlock]:
    num_images = len(image_paths)

    # Calculate grid dimensions
    cols = int(np.sqrt(num_images * (self.canvas_width / self.canvas_height)))
    rows = int(np.ceil(num_images / cols))

    # Calculate cell dimensions
    cell_width = (self.canvas_width - (cols - 1) * self.spacing) // cols
    cell_height = (self.canvas_height - (rows - 1) * self.spacing) // rows

    blocks = []
    for i, path in enumerate(image_paths):
        row = i // cols
        col = i % cols

        x = col * (cell_width + self.spacing)
        y = row * (cell_height + self.spacing)

        block = ImageBlock(x, y, cell_width, cell_height, path)
        blocks.append(block)

    return blocks
```

## Image Processing Pipeline

### Smart Resizing Algorithm

The `_smart_resize` method implements intelligent cropping:

```python
def _smart_resize(self, img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    img_aspect = img.width / img.height
    target_aspect = target_width / target_height

    if abs(img_aspect - target_aspect) < 0.1:
        # Simple resize for similar aspects
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
```

### Shadow Effect Implementation

Drop shadow with configurable parameters:

```python
def _add_shadow(self, img: Image.Image) -> Image.Image:
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
```

## File Management

### Directory Structure

```
project/
├── uploads/          # Temporary uploaded files
├── outputs/          # Generated collages
├── temp/            # Processing temporary files
└── server.py        # Main application
```

### Cleanup Strategy

Automatic cleanup is implemented for job completion:

```python
@app.delete("/api/collage/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    # Clean up temp files
    for file in TEMP_DIR.glob(f"{job_id}_*"):
        file.unlink()

    # Clean up output file
    if job_status[job_id].get('output_file'):
        output_file = OUTPUT_DIR / job_status[job_id]['output_file']
        if output_file.exists():
            output_file.unlink()

    # Remove from job status
    del job_status[job_id]
```

## Error Handling

### Exception Hierarchy

```python
class CollageError(Exception):
    """Base exception for collage operations"""
    pass

class ValidationError(CollageError):
    """Input validation errors"""
    pass

class ProcessingError(CollageError):
    """Image processing errors"""
    pass

class FileSystemError(CollageError):
    """File system operation errors"""
    pass
```

### Error Recovery

The system implements graceful error handling:

```python
try:
    # Processing logic
    pass
except PIL.UnidentifiedImageError:
    job_status[job_id]['status'] = JobStatus.FAILED
    job_status[job_id]['error_message'] = "Invalid image format"
except OSError as e:
    job_status[job_id]['status'] = JobStatus.FAILED
    job_status[job_id]['error_message'] = f"File system error: {e}"
except Exception as e:
    job_status[job_id]['status'] = JobStatus.FAILED
    job_status[job_id]['error_message'] = f"Unexpected error: {e}"
```

## Performance Optimization

### Memory Management

-   Images are loaded on-demand and released after processing
-   Large images are processed in chunks when possible
-   Temporary files are cleaned up immediately after use

### Caching Strategy

```python
# Image dimension caching
image_cache = {}

def get_image_info(path):
    if path not in image_cache:
        with Image.open(path) as img:
            image_cache[path] = {
                'width': img.width,
                'height': img.height,
                'aspect': img.width / img.height
            }
    return image_cache[path]
```

### Concurrent Processing

The system supports multiple concurrent jobs:

```python
# Background task management
background_tasks = BackgroundTasks()

@app.post("/api/collage/create")
async def create_collage(background_tasks: BackgroundTasks, ...):
    # Start background processing
    background_tasks.add_task(process_collage, job_id, image_paths, config)
```

## Testing Strategy

### Unit Tests

```python
def test_masonry_packer():
    packer = MasonryPacker(1000, 1000, 10)

    # Test with sample images
    blocks = packer.pack_images(['test1.jpg', 'test2.jpg', 'test3.jpg'])

    assert len(blocks) == 3
    assert all(block.width > 0 for block in blocks)
    assert all(block.height > 0 for block in blocks)
```

### Integration Tests

```python
def test_full_collage_creation():
    # Create test images
    # Make API request
    # Verify output file
    # Check cleanup
    pass
```

## Extending the API

### Adding New Layout Styles

1. Create a new packer class implementing the packer interface
2. Add the layout style to the `LayoutStyle` enum
3. Update the layout selection logic in `process_collage`

```python
class CustomPacker:
    def __init__(self, canvas_width, canvas_height, spacing):
        # Initialize custom layout

    def pack_images(self, image_paths, maintain_aspect=True):
        # Implement custom packing algorithm
        pass
```

### Adding Image Effects

1. Create effect methods in `CollageGenerator`
2. Add configuration parameters
3. Update the processing pipeline

```python
def _add_custom_effect(self, img: Image.Image) -> Image.Image:
    # Implement custom image effect
    pass
```

## Monitoring and Logging

### Job Tracking

```python
# Job status storage (in production, use Redis)
job_status = {}

def update_job_progress(job_id, progress, message=None):
    job_status[job_id]['progress'] = progress
    if message:
        job_status[job_id]['message'] = message
```

### Performance Metrics

```python
import time

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}

    def start_operation(self, operation_id):
        self.metrics[operation_id] = {
            'start_time': time.time(),
            'operation': operation_id
        }

    def end_operation(self, operation_id):
        if operation_id in self.metrics:
            duration = time.time() - self.metrics[operation_id]['start_time']
            print(f"Operation {operation_id} took {duration:.2f} seconds")
```

## Production Considerations

### Scalability

-   Use Redis for job status storage in production
-   Implement job queuing with Celery
-   Add load balancing for multiple instances
-   Consider CDN for output file delivery

### Security

-   Implement authentication and authorization
-   Add rate limiting
-   Validate file uploads thoroughly
-   Use secure file storage

### Reliability

-   Add comprehensive error handling
-   Implement retry mechanisms
-   Add health checks and monitoring
-   Create backup and recovery procedures
