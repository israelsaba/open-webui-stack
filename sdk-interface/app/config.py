from pathlib import Path
from pydantic import SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    grok_api_key: SecretStr | None = None
    api_keys: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    detailed_request_logging: bool = False
    migrations_path: Path = Path("migrations")
    db_path: Path = Path("data/db.sqlite3")
    
    interaction_poll_interval: int = 30


settings = Settings()
