"""
意图样本库入库 —— knowledge_base.csv → kb_intent_v1

与 kb_ingest 的区别：
- kb_ingest 的 extra_metadata 是请求级（整个文件一套），意图样本要求行级标签
  （每行一个 L2，一行的问题就是一条问句样本），所以不走切分，直接逐行构造。
- 每行 = 1 个 intent_sample 切片，payload 自带完整溯源元数据：
    l1 / l2_code / l2_name / l2_remark  ← 意图判定信号
    biz_category(一级类别) / biz_category_name(二级类别名称)  ← 检索可过滤
    breadcrumb(层级) / loc(行号)  ← 溯源
- 独立 kb_id=kb_intent_v1，agent_scope=intent_agent，与 FAQ 库隔离。

用法：
    python -m src.intent.kb ingest --csv data/knowledge_base.csv [--kb kb_intent_v1]
    python -m src.intent.kb search "快递到哪了" [--kb kb_intent_v1] [--top 10]
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

from src.core.config import settings
from src.core.logging import log
from src.rag.kb_embed import get_embedder
from src.rag.kb_store import KbChunkPoint, KbStore, get_kb_store

DEFAULT_KB_ID = "kb_intent_v1"
AGENT_SCOPE = "intent_agent"
DOC_ID = "knowledge_base"
STRATEGY = "row"  # 每行一条样本，不做切分
CHUNK_TYPE = "intent_sample"


def load_intent_samples(csv_path: str) -> list[dict]:
    """读取 knowledge_base.csv，返回带元数据的样本列表。

    CSV 列（5 列）：一级类别 / 二级类别编码 / 二级类别名称 / 二级类别备注 / 问题
    """
    path = os.path.abspath(csv_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV 不存在: {path}")
    samples: list[dict] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row_no, row in enumerate(reader, start=2):  # 行号从 2 起（表头占 1）
            text = (row.get("问题") or "").strip()
            l2 = (row.get("二级类别编码") or "").strip().upper()
            if not text or not l2:
                log.warning(f"跳过空样本: 第 {row_no} 行 (l2={l2!r})")
                continue
            samples.append(
                {
                    "text": text,
                    "l1": (row.get("一级类别") or "").strip(),
                    "l2_code": l2,
                    "l2_name": (row.get("二级类别名称") or "").strip(),
                    "l2_remark": (row.get("二级类别备注") or "").strip(),
                    "row_no": row_no,
                }
            )
    if not samples:
        raise ValueError(f"CSV 无有效样本: {path}")
    return samples


def build_chunks(
    samples: list[dict], kb_id: str = DEFAULT_KB_ID, index_version: int = 1
) -> list[KbChunkPoint]:
    """为每条样本构造 KbChunkPoint（元数据注入的关键位置）。

    元数据注入点设计：
    - chunk_id 稳定可溯：{l2_code}_{三位序号}，如 GOODS_SPEC_001；
    - parent_id = {l2_code}（父块=二级意图），检索端按 parent 归并即得到意图证据簇；
    - parent_text = 该 L2 的 '名称｜备注' 摘要，随每条样本冗余存储，检索时可直接读到意图定义；
    - loc = {"row": 行号, "source": 文件名} 溯源到 CSV 原始行。
    """
    by_l2: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_l2[s["l2_code"]].append(s)

    chunks: list[KbChunkPoint] = []
    seq: dict[str, int] = defaultdict(int)
    for s in samples:
        l2 = s["l2_code"]
        seq[l2] += 1
        parent_l2 = by_l2[l2]
        # 父块定义为该意图的「定义摘要」：名称 + 备注（问题语句本身不作父文本，避免冗余）
        parent_text = f"{s['l1']}｜{l2}｜{s['l2_name']}（{s['l2_remark']}）"
        _ = parent_l2  # 保留使用，防止误删

        payload = {
            "text": s["text"],
            "parent_id": f"intent_{l2}",
            "parent_text": parent_text,
            "kb_id": kb_id,
            "tenant_id": "default",
            "agent_scope": [AGENT_SCOPE],
            "doc_id": DOC_ID,
            "doc_name": "knowledge_base.csv",
            "strategy": STRATEGY,
            "chunk_type": CHUNK_TYPE,
            "breadcrumb": [s["l1"], l2, s["l2_name"]],
            "loc": {"row": s["row_no"], "source": "knowledge_base.csv"},
            # 意图判定信号（LLM 证据聚合 + 追溯用）
            "l1": s["l1"],
            "l2_code": l2,
            "l2_name": s["l2_name"],
            "l2_remark": s["l2_remark"],
            # 兼容检索侧 biz_category / biz_category_name 过滤字段
            "biz_category": s["l1"],
            "biz_category_name": l2,
        }
        chunks.append(
            KbChunkPoint(
                chunk_id=f"{l2}_{seq[l2]:03d}",
                kb_id=kb_id,
                vector=[],  # 由 ingest 填充
                payload=payload,
                index_version=index_version,
                status="active",
            )
        )
    return chunks


def ingest(
    csv_path: str = "data/knowledge_base.csv",
    kb_id: str = DEFAULT_KB_ID,
    index_version: int = 1,
) -> int:
    """入库主流程：读 CSV → 构造 chunk → 向量化 → upsert。"""
    samples = load_intent_samples(csv_path)
    chunks = build_chunks(samples, kb_id=kb_id, index_version=index_version)

    log.info(f"意图样本: {len(samples)} 条 → {len(chunks)} 个切片 (kb={kb_id})")

    embedder = get_embedder()
    texts = [c.payload["text"] for c in chunks]
    result = embedder.encode_documents(texts)
    for chunk, vector in zip(chunks, result.vectors):
        chunk.vector = vector

    store = get_kb_store()
    written = store.upsert(chunks, wait=True)
    log.info(f"写入完成: {written} 条 (kb={kb_id})")

    # 校验：按二级编码统计入库数量
    stat: dict[str, int] = defaultdict(int)
    for s in samples:
        stat[s["l2_code"]] += 1
    log.info(f"样本分布: {len(stat)} 个 L2，每类 {dict(list(stat.items())[:3])} ...")
    return written


def search(
    query: str,
    kb_id: str = DEFAULT_KB_ID,
    top_k: int = 10,
    agent_role: str = AGENT_SCOPE,
    score_threshold: float | None = None,
) -> None:
    """检索意图样本库（演示）。"""
    embedder = get_embedder()
    store = get_kb_store()

    qv = embedder.encode_query([query]).vectors[0]
    hits = store.search(
        qv,
        [kb_id],
        agent_role,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    print(f"\nquery: {query}  (kb={kb_id}, top={top_k})\n")
    if not hits:
        print("无命中")
        return
    for i, hit in enumerate(hits, 1):
        p = hit.payload
        print(
            f"[{i}] score={hit.score:.4f} | {p.get('l2_code')} {p.get('l2_name')} "
            f"| row={p.get('loc', {}).get('row')}\n"
            f"    {hit.text[:60]}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="意图样本库入库/检索")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="入库 knowledge_base.csv")
    p_ingest.add_argument("--csv", default="data/knowledge_base.csv")
    p_ingest.add_argument("--kb", default=DEFAULT_KB_ID)
    p_ingest.add_argument("--version", type=int, default=1)

    p_search = sub.add_parser("search", help="检索意图样本")
    p_search.add_argument("query")
    p_search.add_argument("--kb", default=DEFAULT_KB_ID)
    p_search.add_argument("--top", type=int, default=10)
    p_search.add_argument("--agent", default=AGENT_SCOPE)
    p_search.add_argument("--score-threshold", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "ingest":
        written = ingest(csv_path=args.csv, kb_id=args.kb, index_version=args.version)
        print(f"OK: 写入 {written} 条到 kb={args.kb}")
        return 0
    search(
        query=args.query,
        kb_id=args.kb,
        top_k=args.top,
        agent_role=args.agent,
        score_threshold=args.score_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())