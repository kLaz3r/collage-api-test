from __future__ import annotations

from typing import List, Dict
from datetime import datetime
import json

from celery import shared_task
from redis import Redis as SyncRedis

from config import AppSettings

# Import generation utilities from server module
from server import (
    CollageConfig,
    CollagePixelConfig,
    LayoutStyle,
    OutputFormat,
    MasonryPacker,
    GridPacker,
    CollageGenerator,
    CollageGeneratorPixels,
    OUTPUT_DIR,
)


settings = AppSettings()


def _to_layout_style(value: str) -> LayoutStyle:
    return LayoutStyle(value)


def _to_output_format(value: str) -> OutputFormat:
    return OutputFormat(value)


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _get_sync_redis() -> SyncRedis:
    if settings.redis_url:
        return SyncRedis.from_url(settings.redis_url, decode_responses=True)
    return SyncRedis(host=settings.redis_host, port=settings.redis_port, db=settings.redis_db, decode_responses=True)


def _update_job_sync(job_id: str, updates: Dict) -> None:
    client = _get_sync_redis()
    raw = client.get(_job_key(job_id))
    data = json.loads(raw) if raw else {"job_id": job_id, "status": "pending", "created_at": datetime.now().isoformat(), "progress": 0}
    data.update(updates)
    client.set(_job_key(job_id), json.dumps(data), ex=settings.job_ttl_seconds)


@shared_task(name="tasks.generate_collage_task")
def generate_collage_task(job_id: str, image_paths: List[str], config_data: Dict) -> str:
    """Celery task: generate collage image and update job in Redis (sync)."""
    # Ensure enums are properly reconstructed
    config = CollageConfig(
        width_mm=float(config_data.get("width_mm")),
        height_mm=float(config_data.get("height_mm")),
        dpi=int(config_data.get("dpi")),
        layout_style=_to_layout_style(config_data.get("layout_style")),
        spacing=float(config_data.get("spacing")),
        background_color=str(config_data.get("background_color")),
        maintain_aspect_ratio=bool(config_data.get("maintain_aspect_ratio")),
        apply_shadow=bool(config_data.get("apply_shadow")),
        output_format=_to_output_format(config_data.get("output_format")),
    )

    try:
        _update_job_sync(job_id, {"status": "processing", "progress": 10})

        generator = CollageGenerator(config)

        if config.layout_style == LayoutStyle.MASONRY:
            packer = MasonryPacker(generator.canvas_width, generator.canvas_height, config.spacing)
            blocks = packer.pack_images(image_paths, config.maintain_aspect_ratio)
        else:
            packer = GridPacker(generator.canvas_width, generator.canvas_height, config.spacing)
            blocks = packer.pack_images(image_paths)

        _update_job_sync(job_id, {"progress": 50})

        file_extension = config.output_format.value
        output_filename = f"collage_{job_id}.{file_extension}"
        output_path = OUTPUT_DIR / output_filename

        generator.generate(blocks, str(output_path))

        _update_job_sync(job_id, {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "output_file": output_filename,
            "progress": 100,
        })

        return output_filename
    except Exception as exc:  # pragma: no cover - defensive
        _update_job_sync(job_id, {"status": "failed", "error_message": str(exc), "progress": 0})
        raise


@shared_task(name="tasks.generate_collage_pixels_task")
def generate_collage_pixels_task(job_id: str, image_paths: List[str], config_data: Dict) -> str:
    """Celery task: generate collage image (pixel-based canvas) and update job in Redis (sync)."""
    config = CollagePixelConfig(
        width_px=int(config_data.get("width_px")),
        height_px=int(config_data.get("height_px")),
        dpi=int(config_data.get("dpi")),
        layout_style=_to_layout_style(config_data.get("layout_style")),
        spacing=float(config_data.get("spacing")),
        background_color=str(config_data.get("background_color")),
        maintain_aspect_ratio=bool(config_data.get("maintain_aspect_ratio")),
        apply_shadow=bool(config_data.get("apply_shadow")),
        output_format=_to_output_format(config_data.get("output_format")),
    )

    try:
        _update_job_sync(job_id, {"status": "processing", "progress": 10})

        generator = CollageGeneratorPixels(config)

        if config.layout_style == LayoutStyle.MASONRY:
            packer = MasonryPacker(generator.canvas_width, generator.canvas_height, config.spacing)
            blocks = packer.pack_images(image_paths, config.maintain_aspect_ratio)
        else:
            packer = GridPacker(generator.canvas_width, generator.canvas_height, config.spacing)
            blocks = packer.pack_images(image_paths)

        _update_job_sync(job_id, {"progress": 50})

        file_extension = config.output_format.value
        output_filename = f"collage_{job_id}.{file_extension}"
        output_path = OUTPUT_DIR / output_filename

        generator.generate(blocks, str(output_path))

        _update_job_sync(job_id, {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "output_file": output_filename,
            "progress": 100,
        })

        return output_filename
    except Exception as exc:  # pragma: no cover - defensive
        _update_job_sync(job_id, {"status": "failed", "error_message": str(exc), "progress": 0})
        raise


