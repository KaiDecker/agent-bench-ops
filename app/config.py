from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置。"""

    app_name: str = "AgentBenchOps"
    app_env: str = "local"
    debug: bool = False
    api_prefix: str = "/api/v1"

    database_url: str = "postgresql+asyncpg://agentbench:agentbench@localhost:5432/agentbench"

    # 留空时自动从 database_url 派生 Psycopg 连接地址。
    checkpoint_database_url: str | None = None

    redis_url: str = "redis://localhost:6379/0"
    sql_echo: bool = False

    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """读取并缓存应用配置。"""
    return Settings()


settings = get_settings()
