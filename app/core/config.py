"""Application configuration via Pydantic BaseSettings.

All runtime values come from environment variables (or a .env file).
No hardcoded credentials anywhere in the application code.
"""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to the repository root (two levels up from this file).
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Central settings object.  Populated from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str

    # ── ML artifacts ─────────────────────────────────────────────────────────
    artifacts_dir: Path = PROJECT_ROOT / "app" / "artifacts" / "v1"

    # ── Inference ─────────────────────────────────────────────────────────────
    confidence_threshold: float = 0.60

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: str

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_origins(cls, v: object) -> list[str]:
        """Allow a comma-separated string or a list."""
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v  # type: ignore[return-value]

    @field_validator("confidence_threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        """Confidence must be in (0, 1]."""
        if not 0.0 < v <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        return v


settings = Settings()
