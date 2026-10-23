"""
把 data/faqs.csv 里的 FAQ 写入 Qdrant
运行: python -m src.rag.init_faqs
"""
from __future__ import annotations
import csv
import os
from pathlib import Path

from src.core.logging import log


def load_faqs(csv_path: str) -> list[dict]:
    faqs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            faqs.append({
                "id": row["id"],
                "text": row["text"],
                "metadata": {"source": row.get("source", "faq")},
            })
    return faqs


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "faqs.csv")
    csv_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        log.error(f"❌ FAQ 文件不存在: {csv_path}")
        return

    faqs = load_faqs(csv_path)
    log.info(f"📄 读取到 {len(faqs)} 条 FAQ")

    from src.rag.vector_store import get_vector_store
    vs = get_vector_store()

    # 先清一下（MVP 阶段简化处理）
    log.info("🗑️  清空 collection 后重建...")
    from qdrant_client import models
    vs.client.delete_collection(vs.collection_name)
    vs._ensure_collection()

    vs.upsert(faqs)
    log.info(f"✅ 完成！Qdrant 中已有 {vs.count()} 条向量")


if __name__ == "__main__":
    main()
