"""Application configuration via Pydantic BaseSettings.

All runtime values come from environment variables (or a .env file).
No hardcoded credentials anywhere in the application code.

The .env file is the single source of truth for ANTHROPIC_API_KEY and
ANTHROPIC_MODEL.  Change them there — nowhere else in the codebase.
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
    cors_origins: str = "*"

    # ── Anthropic (Layer 1 + Layer 3 of hybrid pipeline) ─────────────────────
    # Change these values in .env only — never hardcode them in source files.
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"

    @field_validator("confidence_threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        """Confidence must be in (0, 1]."""
        if not 0.0 < v <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        return v

    @field_validator("anthropic_api_key")
    @classmethod
    def _validate_api_key(cls, v: str) -> str:
        """Fail fast at startup if the Anthropic API key is not configured."""
        if not v:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file locally or to your Railway/Render "
                "environment variables."
            )
        return v


settings = Settings()
