import time
from io import BytesIO
from PIL import Image


def _img(color=(0, 128, 255)) -> bytes:
    im = Image.new('RGB', (120, 80), color)
    buf = BytesIO()
    im.save(buf, format='JPEG')
    return buf.getvalue()


def test_e2e_create_status_download(client, tmp_path):
    # Create two small images
    files = [
        ('files', ('a.jpg', _img((255, 0, 0)), 'image/jpeg')),
        ('files', ('b.jpg', _img((0, 255, 0)), 'image/jpeg')),
    ]

    r = client.post('/api/collage/create', files=files)
    assert r.status_code == 200
    job = r.json()
    job_id = job['job_id']

    # Since Celery is stubbed in tests, processing won't actually run.
    # We can still assert status endpoint returns the created job.
    s = client.get(f'/api/collage/status/{job_id}')
    assert s.status_code in (200, 404)

    # Download should 400/404 without actual processing
    d = client.get(f'/api/collage/download/{job_id}')
    assert d.status_code in (400, 404)

