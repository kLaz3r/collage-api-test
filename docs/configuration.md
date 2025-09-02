# Configuration Guide

Detailed guide to configuring collage generation parameters, layout styles, and optimization options.

## Layout Styles

The API supports four different layout algorithms, each with unique characteristics and use cases.

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
    "spacing": 10
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
    "spacing": 5
}
```

### Random Layout

**Description:** Images are placed randomly across the canvas with overlapping and varied positioning for artistic effect.

**Best for:**

-   Creative, artistic collages
-   Abstract compositions
-   When unpredictability is desired
-   Decorative purposes

**Algorithm Details:**

-   Random positioning within canvas bounds
-   Potential overlapping of images
-   Variable image sizes and rotations
-   Non-deterministic results

**Example Configuration:**

```json
{
    "layout_style": "random",
    "spacing": 0,
    "apply_shadow": true
}
```

### Spiral Layout

**Description:** Images arranged in a spiral pattern from the center outward, creating a radial composition.

**Best for:**

-   Circular or focal compositions
-   Creating visual flow from center
-   Artistic arrangements
-   When a central focal point is desired

**Algorithm Details:**

-   Images placed in spiral pattern
-   Center-weighted composition
-   Maintains aspect ratios
-   Creates natural viewing flow

**Example Configuration:**

```json
{
    "layout_style": "spiral",
    "maintain_aspect_ratio": true,
    "background_color": "#000000"
}
```

## Size and Resolution Settings

### Physical Dimensions

Control the output size in real-world measurements.

**Parameters:**

-   `width_inches`: Width in inches (4-48, default: 12)
-   `height_inches`: Height in inches (4-48, default: 18)

**Common Print Sizes:**

```json
// A4 Portrait
{"width_inches": 8.27, "height_inches": 11.69}

// Letter Portrait
{"width_inches": 8.5, "height_inches": 11}

// Poster
{"width_inches": 24, "height_inches": 36}

// Instagram Post
{"width_inches": 1080/150, "height_inches": 1080/150} // ~7.2 inches at 150 DPI
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
File Size (MB) ≈ (width_inches × height_inches × dpi²) / (8 × 1024²)
```

## Spacing and Positioning

### Spacing Parameter

Controls the gap between images in pixels.

**Range:** 0-50 pixels (default: 10)

**Effects:**

-   `0`: Images touch each other (no gaps)
-   `10`: Comfortable spacing for most layouts
-   `20+`: Generous spacing for clean, minimal look

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
    "width_inches": 16,
    "height_inches": 20,
    "dpi": 300,
    "layout_style": "masonry",
    "spacing": 15,
    "background_color": "#FFFFFF",
    "maintain_aspect_ratio": true,
    "apply_shadow": false
}
```

### Web Gallery Layout

```json
{
  "width_inches": 1200/150,
  "height_inches": 800/150,
  "dpi": 72,
  "layout_style": "grid",
  "spacing": 5,
  "background_color": "#F8F9FA",
  "maintain_aspect_ratio": false,
  "apply_shadow": true
}
```

### Artistic Composition

```json
{
    "width_inches": 18,
    "height_inches": 24,
    "dpi": 150,
    "layout_style": "random",
    "spacing": 0,
    "background_color": "#000000",
    "maintain_aspect_ratio": true,
    "apply_shadow": true
}
```

### Minimalist Design

```json
{
    "width_inches": 12,
    "height_inches": 16,
    "dpi": 150,
    "layout_style": "masonry",
    "spacing": 20,
    "background_color": "#FFFFFF",
    "maintain_aspect_ratio": true,
    "apply_shadow": false
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
