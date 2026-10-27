"""
全局配置 —— 从 .env 加载，用 pydantic-settings 做类型校验
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 绝对路径定位 .env（相对于 config.py 文件向上两级到项目根）
_ENV_FILE = str(Path(__file__).resolve().parent.parent.parent / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # ====== DeepSeek ======
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # ====== MySQL ======
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "customer_service"

    @property
    def mysql_dsn(self) -> str:
        """SQLAlchemy 异步连接串"""
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    # ====== Redis ======
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_user: str = ""      # 留空表示用默认用户（Redis 6.0+ ACL 才需要）
    redis_password: str = ""
    redis_db: int = 0

    # ====== Qdrant ======
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    qdrant_collection: str = "product_faqs"

    # ====== Embedding ======
    embedding_model_path: str = "BAAI/bge-base-zh-v1.5"
    embedding_dimension: int = 768

    # ====== Server ======
    server_host: str = "0.0.0.0"
    server_port: int = 8000


settings = Settings()
