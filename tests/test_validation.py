from io import BytesIO
from PIL import Image


def _make_image_bytes(size=(100, 100), color=(255, 0, 0)) -> bytes:
    img = Image.new('RGB', size, color)
    buf = BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def test_create_collage_requires_min_two_images(client):
    files = {
        'files': ('a.jpg', _make_image_bytes(), 'image/jpeg'),
    }
    resp = client.post('/api/collage/create', files=files)
    assert resp.status_code == 400
    assert 'At least 2 images required' in resp.text


def test_create_collage_rejects_non_image(client):
    files = [
        ('files', ('a.jpg', _make_image_bytes(), 'image/jpeg')),
        ('files', ('b.txt', b'not an image', 'text/plain')),
    ]
    resp = client.post('/api/collage/create', files=files)
    assert resp.status_code == 400
    assert 'not a valid image' in resp.text


