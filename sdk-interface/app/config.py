import logging
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with environment variable support.

    Reads from SDK__ prefixed environment variables from root .env file.
    Falls back to non-prefixed vars for backwards compatibility.
    """

    model_config = SettingsConfigDict(
        # Look for .env in parent directory (project root)
        env_file="../.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # SDK__ prefix is automatically stripped by pydantic when reading env vars
        env_prefix="SDK__",
    )

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str = "http://localhost:8090"
    openrouter_site_name: str = "Open WebUI Stack"
    models_cache_ttl: int = 60
    api_keys: str = ""
    environment: str = "DEV"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    detailed_request_logging: bool = False
    # CORS settings
    cors_origins: str = ""  # Comma-separated list of allowed origins

    interaction_poll_interval: int = 30

    @field_validator("port", mode="before")
    @classmethod
    def make_port(cls, v: str | int):
        if isinstance(v, str):
            return int(v.split(":", 1)[0])
        return v


settings = Settings()
