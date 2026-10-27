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
    # 知识库切片 collection：写入具体版本，读侧可切别名
    kb_collection: str = "kb_chunks_v1"
    kb_collection_alias: str = "kb_chunks"

    # ====== Embedding ======
    # 后端切换：local = 本机加载模型；http = 调用服务器上的 embedding API
    embedding_backend: str = "local"
    embedding_model_path: str = "BAAI/bge-base-zh-v1.5"
    embedding_dimension: int = 768
    # http 后端：通过 SSH 隧道访问服务器服务，如 http://127.0.0.1:8000
    embedding_api_url: str = ""
    embedding_api_path: str = "/embed"
    embedding_api_key: str = ""
    # BGE 查询指令：只加在 query 侧，入库/文档侧保持原文；置空可关闭
    embedding_query_instruction: str = "为这个句子生成表示以用于检索相关文章："

    # ====== Rerank ======
    rerank_model_path: str = "BAAI/bge-reranker-base"
    rerank_max_length: int = 512

    # ====== Server ======
    server_host: str = "0.0.0.0"
    server_port: int = 8000


settings = Settings()
