from server import MasonryPacker, GridPacker


def test_masonry_never_zero_or_negative():
    packer = MasonryPacker(canvas_width=1000, canvas_height=800, spacing_percent=20.0)
    # Fake image paths won't be opened here; we only test internal layout helper with counts
    cols = packer._calculate_optimal_columns(10)
    assert cols >= 1
    dist = packer._distribute_photos(10, cols)
    assert sum(dist) == 10
    layout = packer._calculate_photo_layout(
        images_info=[{'path': 'p', 'width': 100, 'height': 100, 'aspect': 1.0} for _ in range(10)],
        distribution=dist,
        columns=cols,
    )
    for item in layout:
        assert item['width'] > 0
        assert item['height'] > 0


def test_grid_dimensions_valid():
    packer = GridPacker(canvas_width=1200, canvas_height=900, spacing_percent=20.0)
    info = packer.calculate_optimal_grid(13)
    assert info['current_grid']['cols'] >= 1
    assert info['current_grid']['rows'] >= 1

