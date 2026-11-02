from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Security & Rate Limiting
    cache_api_key: Optional[str] = Field(default=None, alias="CACHE_API_KEY")
    rate_limit: int = Field(default=120, alias="RATE_LIMIT")
    max_entries: int = Field(default=500, alias="MAX_ENTRIES")
    default_ttl: int = Field(default=86400, alias="TTL")
    max_query_length: int = Field(default=4096, alias="MAX_QUERY_LENGTH")

    # Thresholds & Matching
    rerank_min: float = Field(default=0.6, alias="RERANK_MIN")
    rerank_enabled: bool = Field(default=True, alias="RERANK")
    qdrant_collection: str = Field(default="semantic_cache", alias="QDRANT_COLLECTION")
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")
    cost_per_1k_tokens: float = Field(default=0.0005, alias="COST_PER_1K_TOKENS")
    base_threshold_code: float = Field(default=0.93, alias="BASE_THRESHOLD_CODE")
    base_threshold_general: float = Field(default=0.88, alias="BASE_THRESHOLD_GENERAL")

    # Storage URLs
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")
    qdrant_url: Optional[str] = Field(default=None, alias="QDRANT_URL")
    qdrant_path: str = Field(default="qdrant_data", alias="QDRANT_PATH")
    sqlite_db_path: str = Field(default="cache.db", alias="SQLITE_DB_PATH")

    # LLM Settings & Fallback
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")
    gemini_fallback_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_FALLBACK_MODEL")
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES")

    # Observability
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


settings = Settings()

