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
celery_app.conf.update(
    task_ignore_result=False,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=settings.job_ttl_seconds,
)

# Also explicitly import tasks for environments where include is ignored
try:
    import tasks  # noqa: F401
except Exception:
    pass


