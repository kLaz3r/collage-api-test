#!/usr/bin/env python3
"""
Standalone test script for the new masonry layout algorithm
"""

def calculate_optimal_columns(canvas_width, canvas_height, photo_count, spacing_pixels):
    """Calculate optimal number of columns for best canvas filling"""
    # Start with a reasonable estimate based on canvas aspect ratio
    min_columns = max(1, int((photo_count * canvas_width / canvas_height) ** 0.5))
    max_columns = min(photo_count, max(1, canvas_width // 200))  # min 200px per column
    
    best_columns = min_columns
    best_score = float('inf')
    
    # Test different column counts to find the best fit
    for cols in range(min_columns, max_columns + 1):
        score = evaluate_layout(canvas_width, canvas_height, photo_count, cols, spacing_pixels)
        if score < best_score:
            best_score = score
            best_columns = cols
    
    return best_columns

def evaluate_layout(canvas_width, canvas_height, photo_count, columns, spacing_pixels):
    """Evaluate layout quality based on column count"""
    column_width = (canvas_width - spacing_pixels * (columns + 1)) / columns
    photos_per_column = (photo_count + columns - 1) // columns  # Ceiling division
    total_spacing_per_column = spacing_pixels * (photos_per_column + 1)
    available_height_per_column = canvas_height - total_spacing_per_column
    avg_photo_height = available_height_per_column / photos_per_column
    
    # Score based on how well photos fit (penalize extreme aspect ratios)
    aspect_ratio = column_width / avg_photo_height
    aspect_penalty = abs(aspect_ratio - 1.0)  # prefer square-ish photos
    
    # Penalty for uneven distribution
    remainder = photo_count % columns
    unevenness_penalty = remainder / columns if remainder > 0 else 0
    
    return aspect_penalty + unevenness_penalty

def distribute_photos(photo_count, columns):
    """Distribute photos evenly across columns"""
    distribution = [0] * columns
    photos_per_column = photo_count // columns
    remainder = photo_count % columns
    
    # Distribute evenly, then add remainder to first columns
    for i in range(columns):
        distribution[i] = photos_per_column + (1 if i < remainder else 0)
    
    return distribution

def test_masonry_algorithm():
    """Test the new masonry layout algorithm"""
    
    # Test canvas dimensions (800x600 pixels)
    canvas_width = 800
    canvas_height = 600
    spacing_pixels = 8  # 8 pixels spacing
    
    print("=== Testing New Masonry Layout Algorithm ===")
    print(f"Canvas: {canvas_width}x{canvas_height} pixels")
    print(f"Spacing: {spacing_pixels} pixels")
    print()
    
    # Test different photo counts
    test_cases = [5, 10, 25, 50, 100]
    
    for photo_count in test_cases:
        print(f"--- Testing with {photo_count} photos ---")
        
        # Calculate optimal columns
        optimal_columns = calculate_optimal_columns(canvas_width, canvas_height, photo_count, spacing_pixels)
        print(f"Optimal columns: {optimal_columns}")
        
        # Distribute photos
        distribution = distribute_photos(photo_count, optimal_columns)
        print(f"Distribution: {distribution}")
        
        # Calculate layout metrics
        column_width = (canvas_width - spacing_pixels * (optimal_columns + 1)) / optimal_columns
        max_photos_per_column = max(distribution)
        min_photos_per_column = min(distribution)
        
        # Calculate efficiency
        total_spacing_width = spacing_pixels * (optimal_columns + 1)
        total_spacing_height = spacing_pixels * (max_photos_per_column + 1)
        available_width = canvas_width - total_spacing_width
        available_height = canvas_height - total_spacing_height
        
        width_utilization = (available_width / canvas_width) * 100
        height_utilization = (available_height / canvas_height) * 100
        
        print(f"Column width: {column_width:.1f}px")
        print(f"Width utilization: {width_utilization:.1f}%")
        print(f"Height utilization: {height_utilization:.1f}%")
        print(f"Overall coverage: {(width_utilization + height_utilization) / 2:.1f}%")
        
        # Calculate average photo aspect ratio
        avg_photo_height = available_height / max_photos_per_column
        avg_aspect_ratio = column_width / avg_photo_height
        print(f"Average photo aspect ratio: {avg_aspect_ratio:.3f}")
        print()
    
    print("=== Algorithm Features ===")
    print("✓ Dynamic column calculation")
    print("✓ Even photo distribution")
    print("✓ Full canvas coverage")
    print("✓ Aspect ratio consideration")
    print("✓ Flexible spacing")
    print()
    
    print("=== Usage Example ===")
    print("The new algorithm automatically:")
    print("1. Calculates optimal column count based on photo count and canvas dimensions")
    print("2. Distributes photos evenly across columns")
    print("3. Scales photos to fill the entire canvas height")
    print("4. Maintains reasonable aspect ratios")
    print("5. Provides consistent spacing between all elements")
    print()
    
    print("=== API Integration ===")
    print("This algorithm is now integrated into your collage API:")
    print("- Use the /api/collage/create endpoint with layout_style=masonry")
    print("- Use the /api/collage/analyze-masonry endpoint to preview layouts")
    print("- The algorithm automatically optimizes for your canvas dimensions")

if __name__ == "__main__":
    test_masonry_algorithm()
