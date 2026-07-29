"""Application configuration sourced from environment variables."""

from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with secure defaults suitable for a small HTTP service."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    allowed_origins: str = "http://localhost:8501"
    max_upload_mb: int = Field(default=20, ge=1, le=100)
    max_total_upload_mb: int = Field(default=50, ge=1, le=250)
    corpus_ttl_minutes: int = Field(default=120, ge=5, le=1_440)
    api_auth_token: SecretStr | None = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "document_chunks"
    embedding_dimensions: int = Field(default=1536, ge=1)

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def require_production_auth(self) -> "Settings":
        if self.app_env.lower() == "production" and not self.api_auth_token:
            raise ValueError("API_AUTH_TOKEN is required when APP_ENV=production.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
