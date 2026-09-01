from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SUPPORT_", extra="ignore")

    app_name: str = "Support Case Classification Engine"
    environment: str = "development"
    model_version: str = "production-0.1.0"
    artifact_dir: str = "models/production"
    results_dir: str = "results"
    persist_case_text: bool = False
    first_stage_threshold: float = 0.62
    margin_threshold: float = 0.12
    enable_llm: bool = False
    llama_endpoint: str | None = None
    llama_model: str = "llama-3.3-70b"
    max_batch_size: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
