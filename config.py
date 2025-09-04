from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application configuration loaded from environment variables or .env.

    Env prefix: APP_
    Example: APP_UPLOAD_DIR=/data/uploads
    """

    # App
    app_name: str = Field(default="Collage Maker API")
    app_version: str = Field(default="1.0.0")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # Paths
    upload_dir: Path = Field(default=Path("uploads"))
    output_dir: Path = Field(default=Path("outputs"))
    temp_dir: Path = Field(default=Path("temp"))

    # Limits
    max_image_size: int = Field(default=10 * 1024 * 1024)  # 10 MB
    max_total_size: int = Field(default=500 * 1024 * 1024)  # 500 MB
    max_canvas_pixels: int = Field(default=250_000_000)

    # Rate limiting
    rate_limit_requests: int = Field(default=10)
    rate_limit_window_seconds: int = Field(default=60)

    # Redis
    redis_url: str | None = Field(default=None)
    redis_host: str = Field(default="192.168.3.2")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    job_ttl_seconds: int = Field(default=1 * 60 * 60)  # 1h
    cleanup_interval_seconds: int = Field(default=600)  # 10 minutes

    # CORS
    cors_allow_origins: List[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: List[str] = Field(default_factory=lambda: ["*"])
    cors_allow_headers: List[str] = Field(default_factory=lambda: ["*"])

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


