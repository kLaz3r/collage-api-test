# Configuration Guide

Detailed guide to configuring collage generation parameters, layout styles, and optimization options.

## Layout Styles

The API supports two different layout algorithms, each with unique characteristics and use cases.

### Masonry Layout (Default)

**Description:** A Pinterest-style layout where images are arranged in columns with varying heights. Images are packed efficiently to minimize empty space while maintaining visual flow.

**Best for:**

-   Mixed image sizes and aspect ratios
-   Natural, organic appearance
-   Maximum space utilization
-   Photo galleries and portfolios

**Algorithm Details:**

-   Images sorted by area (largest first) for optimal packing
-   Dynamic column count based on image count (2-6 columns)
-   Maintains aspect ratios by default
-   Intelligent cropping when aspect ratios differ significantly

**Example Configuration:**

```json
{
    "layout_style": "masonry",
    "maintain_aspect_ratio": true,
    "spacing": 40.0
}
```

### Grid Layout

**Description:** A uniform grid where all images are resized to fit equally-sized cells. Creates a structured, organized appearance.

**Best for:**

-   Consistent, professional look
-   Same-sized images
-   Catalog-style presentations
-   Maximum predictability

**Algorithm Details:**

-   Calculates optimal grid dimensions (rows × columns)
-   All images resized to fit grid cells
-   Maintains consistent spacing
-   Ignores original aspect ratios for uniformity

**Example Configuration:**

```json
{
    "layout_style": "grid",
    "maintain_aspect_ratio": false,
    "spacing": 20.0
}
```

## Size and Resolution Settings

### Physical Dimensions

Control the output size in real-world measurements.

**Parameters:**

-   `width_mm`: Width in millimeters (50-1219.2, default: 304.8)
-   `height_mm`: Height in millimeters (50-1219.2, default: 457.2)

**Common Print Sizes:**

```json
// A4 Portrait
{"width_mm": 210, "height_mm": 297}

// Letter Portrait
{"width_mm": 216, "height_mm": 279}

// Poster
{"width_mm": 610, "height_mm": 914}

// Instagram Post
{"width_mm": 183, "height_mm": 183} // ~7.2 inches at 150 DPI

// Small Square (new minimum)
{"width_mm": 50, "height_mm": 50} // ~2 inches at 150 DPI
```

### Resolution (DPI)

Control the print quality and file size.

**DPI Options:**

-   `72`: Web/screen use (smallest file size)
-   `150`: Standard print quality (default)
-   `300`: High-quality print (largest file size)

**Impact on Output:**

-   Higher DPI = sharper prints but larger files
-   Lower DPI = smaller files but pixelated prints
-   150 DPI is suitable for most home printers

**File Size Estimation:**

```
File Size (MB) ≈ (width_mm × height_mm × dpi²) / (25.4² × 8 × 1024²)
```

## Spacing and Positioning

### Spacing Parameter

Controls the gap between images in pixels.

**Range:** 0-100% of canvas dimensions (default: 40.0%, where 100% = 5% of canvas)

**Effects:**

-   `0.0`: Images touch each other and canvas edges (no gaps)
-   `40.0`: Comfortable spacing between images and from canvas edges (2% of canvas)
-   `100.0`: Generous spacing between images and from canvas edges (5% of canvas)

**Note:** Spacing is applied both between images and at the canvas edges for consistent visual balance.

### Output Format

Controls the file format of the generated collage.

**Options:**

-   `jpeg` - JPEG format (default, good compression, no transparency)
-   `png` - PNG format (lossless, supports transparency)
-   `tiff` - TIFF format (high quality, lossless, good for printing)

**Default:** `jpeg`

**Use Cases:**

-   **JPEG**: Web use, email, general sharing (smaller file size)
-   **PNG**: When transparency is needed, web graphics, logos
-   **TIFF**: High-quality printing, archival purposes, professional use

**Special Features:**

-   **PNG**: Supports transparent backgrounds when using `#00000000` as background color
-   **TIFF**: Uses LZW compression for optimal file size while maintaining quality

### Background Color

Set the canvas background color.

**Format:** Hex color codes (e.g., "#FFFFFF", "#000000", "#F0F0F0")

**Common Options:**

-   `"#FFFFFF"`: White (default)
-   `"#000000"`: Black
-   `"#F5F5F5"`: Light gray
-   `"#2C2C2C"`: Dark gray

## Image Processing Options

### Aspect Ratio Preservation

Control whether original image proportions are maintained.

**Options:**

-   `true` (default): Maintains original aspect ratios
-   `false`: Allows stretching to fit layout requirements

**When to use `maintain_aspect_ratio: false`:**

-   Grid layouts requiring uniform cells
-   When exact sizing is more important than proportion
-   Creating uniform visual elements

### Shadow Effects

Add drop shadow effects to images for depth.

**Options:**

-   `false` (default): No shadows
-   `true`: Apply drop shadows

**Shadow Characteristics:**

-   Semi-transparent black shadow
-   10-pixel offset from image
-   5-pixel blur radius
-   Works best with light backgrounds

## Advanced Configuration Examples

### High-Quality Print Collage

```json
{
    "width_mm": 406.4,
    "height_mm": 508,
    "dpi": 300,
    "layout_style": "masonry",
    "spacing": 15,
    "background_color": "#FFFFFF",
    "maintain_aspect_ratio": true,
    "apply_shadow": false,
    "output_format": "tiff"
}
```

### Web Gallery Layout

```json
{
    "width_mm": 203.2,
    "height_mm": 135.5,
    "dpi": 72,
    "layout_style": "grid",
    "spacing": 5,
    "background_color": "#F8F9FA",
    "maintain_aspect_ratio": false,
    "apply_shadow": true,
    "output_format": "jpeg"
}
```

### Minimalist Design

```json
{
    "width_mm": 304.8,
    "height_mm": 406.4,
    "dpi": 150,
    "layout_style": "masonry",
    "spacing": 20,
    "background_color": "#FFFFFF",
    "maintain_aspect_ratio": true,
    "apply_shadow": false,
    "output_format": "jpeg"
}
```

### PNG with Transparency

```json
{
    "width_mm": 203.2,
    "height_mm": 203.2,
    "dpi": 150,
    "layout_style": "grid",
    "spacing": 40,
    "background_color": "#00000000",
    "maintain_aspect_ratio": true,
    "apply_shadow": false,
    "output_format": "png"
}
```

## Performance Considerations

### Image Count Impact

-   **2-10 images:** Fast processing, high quality
-   **11-30 images:** Moderate processing time
-   **31-60 images:** Slower processing, consider optimization
-   **61-200 images:** Longest processing time, monitor memory usage

### Resolution Guidelines

-   **Web use:** 72-96 DPI
-   **Standard print:** 150 DPI
-   **Professional print:** 300 DPI
-   **Large format:** 150-200 DPI (balance quality vs. file size)

### Memory Usage

-   Higher DPI significantly increases memory requirements
-   Large images (>5MB each) may require more processing power
-   Consider system resources when processing many high-resolution images

## Troubleshooting Configuration Issues

### Common Problems

**Images not fitting properly:**

-   Reduce spacing or adjust canvas size
-   Check aspect ratio settings
-   Try different layout style

**Poor image quality:**

-   Increase DPI setting
-   Ensure source images are high quality
-   Check for compression artifacts

**Processing timeouts:**

-   Reduce image count or resolution
-   Use smaller canvas dimensions
-   Optimize source image sizes

**Unexpected layout:**

-   Verify layout_style parameter
-   Check spacing values
-   Review aspect ratio settings

### Optimization Tips

1. **For speed:** Use lower DPI (72-96) and fewer images
2. **For quality:** Use higher DPI (150-300) and optimize source images
3. **For web:** Balance file size with visual quality
4. **For print:** Prioritize resolution over processing speed
