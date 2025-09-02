# Grid Optimization API

The Grid Optimization API helps frontend applications determine how many images to add or remove to achieve a perfect even grid layout without incomplete rows.

## Problem

When creating grid collages, the current algorithm can result in incomplete rows at the bottom, leaving white space that is not aesthetically pleasing. For example:

-   7 images → 2×4 grid (last row has only 1 image)
-   13 images → 2×7 grid (last row has only 1 image)
-   19 images → 3×7 grid (last row has only 1 image)

## Solution

The Grid Optimization API provides recommendations for:

1. **Adding images** to reach the next perfect grid
2. **Removing images** to reach the previous perfect grid
3. **Current grid analysis** showing the current layout and completeness

## API Endpoint

### POST `/api/collage/optimize-grid`

Calculate optimal grid dimensions and provide recommendations for perfect grid layout.

#### Request Parameters

| Parameter    | Type  | Required | Default | Range        | Description                          |
| ------------ | ----- | -------- | ------- | ------------ | ------------------------------------ |
| `num_images` | int   | Yes      | -       | 2-200        | Number of images to analyze          |
| `width_mm`   | float | No       | 304.8   | 101.6-1219.2 | Canvas width in millimeters          |
| `height_mm`  | float | No       | 457.2   | 101.6-1219.2 | Canvas height in millimeters         |
| `dpi`        | int   | No       | 150     | 72-300       | Resolution in dots per inch          |
| `spacing`    | float | No       | 40.0    | 0-100        | Spacing between images as percentage |

#### Example Request

```bash
curl -X POST "http://localhost:8000/api/collage/optimize-grid" \
  -F "num_images=13" \
  -F "width_mm=304.8" \
  -F "height_mm=457.2" \
  -F "dpi=150" \
  -F "spacing=40.0"
```

#### Response Format

```json
{
    "success": true,
    "optimization": {
        "current_grid": {
            "total_images": 13,
            "cols": 2,
            "rows": 7,
            "images_in_last_row": 1,
            "is_perfect": false
        },
        "closest_perfect_grid": {
            "type": "remove_images",
            "total_images": 12,
            "cols": 2,
            "rows": 6,
            "images_needed": 0,
            "images_to_remove": 1
        },
        "recommendations": {
            "add_images": [
                {
                    "images_needed": 3,
                    "total_images": 16,
                    "cols": 2,
                    "rows": 8
                }
            ],
            "remove_images": [
                {
                    "images_to_remove": 1,
                    "total_images": 12,
                    "cols": 2,
                    "rows": 6
                }
            ]
        },
        "canvas_info": {
            "width": 1800,
            "height": 2700,
            "spacing": 40.0
        }
    },
    "message": "Grid optimization calculated successfully"
}
```

## Response Fields

### `current_grid`

-   `total_images`: Current number of images
-   `cols`: Number of columns in current grid
-   `rows`: Number of rows in current grid
-   `images_in_last_row`: Number of images in the last row (0 = complete row)
-   `is_perfect`: Whether the current grid is perfect (no incomplete rows)

### `closest_perfect_grid`

-   `type`: One of `"perfect"`, `"add_images"`, or `"remove_images"`
-   `total_images`: Total images in the perfect grid
-   `cols`: Number of columns in the perfect grid
-   `rows`: Number of rows in the perfect grid
-   `images_needed`: Images to add (only if type is "add_images")
-   `images_to_remove`: Images to remove (only if type is "remove_images")

### `recommendations`

-   `add_images`: Array of up to 3 suggestions for adding images
-   `remove_images`: Array of up to 3 suggestions for removing images

## Usage Examples

### Frontend Integration

```javascript
async function getGridOptimization(
    numImages,
    widthMm = 304.8, // 12 inches = 304.8 mm
    heightMm = 457.2 // 18 inches = 457.2 mm
) {
    const formData = new FormData();
    formData.append("num_images", numImages);
    formData.append("width_mm", widthMm);
    formData.append("height_mm", heightMm);

    const response = await fetch("/api/collage/optimize-grid", {
        method: "POST",
        body: formData,
    });

    const result = await response.json();
    return result.optimization;
}

// Example usage
const optimization = await getGridOptimization(13);
if (!optimization.current_grid.is_perfect) {
    if (optimization.closest_perfect_grid.type === "add_images") {
        console.log(
            `Add ${optimization.closest_perfect_grid.images_needed} images for perfect grid`
        );
    } else {
        console.log(
            `Remove ${optimization.closest_perfect_grid.images_to_remove} images for perfect grid`
        );
    }
}
```

### Common Scenarios

#### Scenario 1: Add Images

-   **Current**: 7 images → 2×4 grid (incomplete last row)
-   **Recommendation**: Add 1 image → 8 images → 2×4 grid (perfect)
-   **Action**: Frontend suggests adding 1 more image

#### Scenario 2: Remove Images

-   **Current**: 13 images → 2×7 grid (incomplete last row)
-   **Recommendation**: Remove 1 image → 12 images → 2×6 grid (perfect)
-   **Action**: Frontend suggests removing 1 image

#### Scenario 3: Perfect Grid

-   **Current**: 91 images → 7×13 grid (complete)
-   **Recommendation**: Already perfect, no changes needed
-   **Action**: Frontend shows "Perfect grid!" message

## Algorithm Details

The optimization algorithm:

1. **Calculates current grid dimensions** using the same formula as the GridPacker
2. **Identifies incomplete rows** by checking if `num_images % cols != 0`
3. **Finds next perfect grids** by testing adding 1-10 images
4. **Finds previous perfect grids** by testing removing 1-10 images
5. **Recommends the closest option** (fewest images to add/remove)
6. **Limits suggestions** to 3 options in each direction

## Benefits

-   **Eliminates white space** in grid collages
-   **Improves aesthetics** with complete, uniform rows
-   **Provides clear guidance** for frontend applications
-   **Maintains consistency** with existing grid layout algorithm
-   **Flexible recommendations** for both adding and removing images

## Integration Notes

-   The API uses the same grid calculation logic as the collage generation
-   Recommendations are based on the current canvas dimensions and spacing
-   The algorithm prioritizes the option requiring the fewest changes
-   All recommendations maintain the minimum requirement of 2 images
-   The API is designed to be called before collage generation to optimize the input
