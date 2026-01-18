from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    anthropic_api_key: SecretStr
    google_api_key: SecretStr | None = None
    api_keys: str = ""  # Format: username1:token1;username2:token2;...
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


settings = Settings()
