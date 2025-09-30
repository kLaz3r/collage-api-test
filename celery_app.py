import platform
from celery import Celery
from config import AppSettings


settings = AppSettings()


def _build_broker_url() -> str:
    if settings.redis_url:
        return settings.redis_url
    return f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"


celery_app = Celery(
    'collage_worker',
    broker=_build_broker_url(),
    backend=_build_broker_url(),
    include=['tasks'],  # ensure tasks module is imported and tasks are registered
)

# Sensible defaults
celery_config = {
    'task_ignore_result': False,
    'task_acks_late': True,
    'worker_prefetch_multiplier': 1,
    'result_expires': settings.job_ttl_seconds,
}

# Platform-specific configuration
if settings.celery_worker_pool:
    celery_config['worker_pool'] = settings.celery_worker_pool
elif platform.system() == 'Windows':
    celery_config.update({
        'worker_pool': 'solo',  # Use solo pool on Windows to avoid multiprocessing issues
        'worker_concurrency': 1,  # Single worker process
    })
else:
    # Linux/macOS configuration
    celery_config.update({
        'worker_pool': 'prefork',  # Default multiprocessing pool
        'worker_concurrency': 4,  # Multiple worker processes
    })

# Override concurrency if specified
if settings.celery_worker_concurrency:
    celery_config['worker_concurrency'] = settings.celery_worker_concurrency

celery_app.conf.update(celery_config)

# Also explicitly import tasks for environments where include is ignored
try:
    import tasks  # noqa: F401
except Exception:
    pass


