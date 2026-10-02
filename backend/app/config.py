"""Application settings for the foot-scanner backend."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FOOT_", env_file=".env", extra="ignore")

    data_dir: Path = Path(__file__).resolve().parent.parent / "data" / "scans"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"
    default_target_frames: int = 70
    max_upload_mb: int = 512
    thumbnail_max_width: int = 320
    blur_laplacian_min: float = 40.0


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
