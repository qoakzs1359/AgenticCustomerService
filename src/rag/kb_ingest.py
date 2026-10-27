"""
知识入库编排：切分 → 质检 → 装配 payload → 向量化 → staging → active。

设计要点：
- 复用 `src.tools.kb_chunking` 的切分产物（Parent + Chunk），不在此重复切分逻辑；
- 先写 staging，校验条数后再切 active；失败停在 staging，不污染线上；
- 每个切片都带完整定位元数据与 parent_text，检索命中后可还原原文并给出引用；
- 版本冲突、覆盖空洞、空切片直接拒绝；硬切与坏单元按比例卡阈值。

用法：
    python -m src.rag.kb_ingest data/intent_rag_kb.csv \
        --kb kb_faq_v1 --agent customer_service --doc-id intent_rag_kb
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.core.logging import log
from src.rag.kb_embed import EmbeddingError, Embedder, get_embedder
from src.rag.kb_store import KbChunkPoint, KbStore, get_kb_store
from src.tools.kb_chunking import ChunkBundle, chunk_file

# payload 中 parent 原文上限：32KB，超出截断并告警
MAX_PARENT_CHARS = 32 * 1024
# 硬切占比阈值
HARD_CUT_REJECT = 0.05
HARD_CUT_WARN = 0.01
# 坏单元占比阈值
BAD_UNIT_REJECT = 0.20

# 切分策略 -> 面向检索的语义类型，用于软过滤/加权，避免再造一套分类
CHUNK_TYPE_BY_STRATEGY = {
    "qa_pair": "faq",
    "section": "doc_section",
    "table_row": "table",
    "table_grid": "table",
    "recursive": "text",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPORT_DIR = _PROJECT_ROOT / "data" / "ingest_reports"


@dataclass(slots=True)
class IngestRequest:
    path: str
    kb_id: str
    agent_scope: list[str]
    tenant_id: str = "default"
    doc_id: str | None = None
    doc_name: str = ""
    version: int = 1
    allow_partial: bool = False
    dry_run: bool = False
    keep_last: int = 1
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    report_dir: str | Path | None = None


@dataclass(slots=True)
class IngestResult:
    ok: bool
    stage: str
    doc_id: str
    file_type: str = ""
    profile: str = ""
    parents: int = 0
    chunks: int = 0
    points: int = 0
    version: int = 1
    coverage_ok: bool = True
    truncated_vectors: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==================== 质检闸门 ====================


def _gate(bundle: ChunkBundle, request: IngestRequest, result: IngestResult) -> bool:
    """入库前质检：不通过则直接拒绝，避免把残片写进向量库。"""
    report = bundle.report
    if report is None:
        result.error = "切分未返回质检报告"
        result.warnings.append(result.error)
        return False

    result.warnings.extend(report.warnings)

    if report.unit_count <= 0:
        result.error = "未解析出任何内容单元（文件为空、损坏或格式不支持）"
        result.warnings.append(result.error)
        return False

    if not report.coverage_ok or report.gaps:
        result.error = f"切片存在空洞，覆盖率校验失败（{len(report.gaps)} 处）"
        result.warnings.extend(report.gaps[:10])
        return False

    if not bundle.chunks:
        result.error = "切分未产出任何切片"
        return False

    hard_ratio = report.hard_cut_count / max(1, report.chunk_count)
    if hard_ratio > HARD_CUT_REJECT:
        message = f"句内硬切占比 {hard_ratio:.1%} 超过 {HARD_CUT_REJECT:.0%}"
        if request.allow_partial:
            result.warnings.append(message + "（allow_partial 已放行）")
        else:
            result.error = message + "，请调大 max_tokens 或改进切分策略"
            return False
    elif hard_ratio > HARD_CUT_WARN:
        result.warnings.append(f"句内硬切占比 {hard_ratio:.1%}，建议检查切分参数")

    bad_ratio = len(report.bad_units) / max(1, report.unit_count)
    if bad_ratio > BAD_UNIT_REJECT:
        message = f"坏单元占比 {bad_ratio:.1%} 超过 {BAD_UNIT_REJECT:.0%}"
        if request.allow_partial:
            result.warnings.append(message + "（allow_partial 已放行）")
        else:
            result.error = message + "，请检查解析器或改用 allow_partial"
            return False

    if report.oversized_count:
        result.warnings.append(f"{report.oversized_count} 片超出 token 预算 20%")

    return True


# ==================== 装配 payload ====================


def _build_records(
    bundle: ChunkBundle,
    request: IngestRequest,
    doc_id: str,
    doc_name: str,
    vectors: list[list[float]],
    result: IngestResult,
) -> list[KbChunkPoint]:
    parent_map = {parent.parent_id: parent for parent in bundle.parents}
    records: list[KbChunkPoint] = []

    for chunk, vector in zip(bundle.chunks, vectors):
        parent = parent_map.get(chunk.parent_id)
        parent_text = parent.text if parent else ""
        parent_truncated = len(parent_text) > MAX_PARENT_CHARS
        if parent_truncated:
            parent_text = parent_text[:MAX_PARENT_CHARS] + "…"
            result.warnings.append(f"{chunk.chunk_id}: parent_text 超过 32KB，已截断")

        strategy = str(chunk.payload.get("sys_strategy", ""))
        category = chunk.payload.get("sys_category") or {}
        biz_category = str(category.get("二级类别编码") or "")
        payload: dict[str, Any] = {
            "text": chunk.content,
            "parent_id": chunk.parent_id,
            "kb_id": request.kb_id,
            "agent_scope": list(request.agent_scope),
            "tenant_id": request.tenant_id,
            "doc_id": doc_id,
            "doc_name": doc_name,
            "strategy": strategy,
            "chunk_type": CHUNK_TYPE_BY_STRATEGY.get(strategy, "text"),
            "breadcrumb": list(chunk.payload.get("breadcrumb", [])),
            "question": str(chunk.payload.get("question", "") or ""),
            "answer": str(chunk.payload.get("answer", "") or ""),
            "biz_category": biz_category,
            "biz_category_name": str(category.get("二级类别名称") or ""),
            "biz_subcategory": str(category.get("细分类型") or ""),
            "sys_content_hash": str(chunk.payload.get("sys_content_hash", "")),
            "sys_strategy_version": chunk.payload.get("sys_strategy_version", 1),
            "parent_text": parent_text,
            "parent_truncated": parent_truncated,
            "part": chunk.part,
            "total": chunk.total,
            "loc": chunk.payload.get("sys_loc", {}) or {},
            "source_path": str(Path(request.path).resolve()),
        }
        # 业务扩展字段只能补位，不能覆盖上面已经确定的定位/隔离字段
        for key, value in request.extra_metadata.items():
            payload.setdefault(key, value)

        records.append(
            KbChunkPoint(
                chunk_id=chunk.chunk_id,
                kb_id=request.kb_id,
                vector=vector,
                payload=payload,
                index_version=request.version,
                status="staging",
            )
        )
    return records


# ==================== 版本控制 ====================


def _check_version(store: KbStore, doc_id: str, kb_id: str, version: int) -> None:
    """同一文档同一库内：已生效版本不可重复写，也不允许回退版本号。"""
    active = set(store.versions(doc_id, kb_id=kb_id, status="active"))
    archived = set(store.versions(doc_id, kb_id=kb_id, status="archived"))
    staging = set(store.versions(doc_id, kb_id=kb_id, status="staging"))
    all_versions = active | archived | staging

    if version in active or version in archived:
        raise ValueError(f"文档 `{doc_id}` 的版本 v{version} 已生效，请提升 version 后重试")
    if all_versions and version < max(all_versions):
        raise ValueError(
            f"版本号回退：v{version} < 已存在的 v{max(all_versions)}，请使用更大的 version"
        )
    if version in staging:
        # 上一次入库失败留下的残片，清掉后允许同版本重跑
        store.delete_version(doc_id, version, kb_id)


def _archive_old_versions(store: KbStore, doc_id: str, kb_id: str, version: int) -> None:
    for old in store.versions(doc_id, kb_id=kb_id, status="active"):
        if old != version:
            store.set_status(doc_id, "archived", kb_id=kb_id, index_version=old)


def _cleanup_old_versions(
    store: KbStore, doc_id: str, kb_id: str, keep_last: int
) -> None:
    archived = store.versions(doc_id, kb_id=kb_id, status="archived")
    for old in sorted(archived, reverse=True)[max(0, keep_last):]:
        store.delete_version(doc_id, old, kb_id)


# ==================== 报告 ====================


def _chunk_type_counts(bundle: ChunkBundle) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in bundle.chunks:
        strategy = str(chunk.payload.get("sys_strategy", ""))
        chunk_type = CHUNK_TYPE_BY_STRATEGY.get(strategy, "text")
        counts[chunk_type] = counts.get(chunk_type, 0) + 1
    return counts


def _write_report(
    request: IngestRequest,
    result: IngestResult,
    bundle: ChunkBundle,
    doc_name: str,
    elapsed: float,
) -> str:
    report_dir = Path(request.report_dir) if request.report_dir else _DEFAULT_REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".dryrun.json" if request.dry_run else ".json"
    path = report_dir / f"{result.doc_id}{suffix}"
    result.report_path = str(path)

    payload = {
        "request": {
            "path": request.path,
            "kb_id": request.kb_id,
            "agent_scope": list(request.agent_scope),
            "tenant_id": request.tenant_id,
            "doc_id": result.doc_id,
            "doc_name": doc_name,
            "version": request.version,
            "allow_partial": request.allow_partial,
            "dry_run": request.dry_run,
        },
        "result": result.to_dict(),
        "chunk_report": bundle.report.summary() if bundle.report else {},
        "chunk_types": _chunk_type_counts(bundle),
        "parents": len(bundle.parents),
        "chunks": len(bundle.chunks),
        "elapsed_seconds": round(elapsed, 3),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


# ==================== 主流程 ====================


def ingest_file(
    request: IngestRequest,
    *,
    store: KbStore | None = None,
    embedder: Embedder | None = None,
) -> IngestResult:
    started = time.perf_counter()
    doc_id = request.doc_id or Path(request.path).stem
    doc_name = request.doc_name or Path(request.path).name
    result = IngestResult(
        ok=False, stage="chunk", doc_id=doc_id, version=request.version
    )

    if not request.kb_id:
        return _fail(result, "validate", "kb_id 不能为空")
    if not request.agent_scope:
        return _fail(result, "validate", "agent_scope 不能为空")
    if request.version < 1:
        return _fail(result, "validate", "version 必须 >= 1")

    source = Path(request.path)
    if not source.exists():
        return _fail(result, "validate", f"文件不存在：{source}")

    # 1. 切分
    try:
        bundle = chunk_file(source, doc_id=doc_id, extra_metadata=request.extra_metadata)
    except Exception as exc:
        return _fail(result, "chunk", f"切分失败：{type(exc).__name__}: {exc}")

    report = bundle.report
    if report is not None:
        result.file_type = report.file_type
        result.profile = report.profile
        result.coverage_ok = report.coverage_ok
    result.parents = len(bundle.parents)
    result.chunks = len(bundle.chunks)

    # 2. 质检闸门
    if not _gate(bundle, request, result):
        result.stage = "gate"
        result.report_path = _write_report(request, result, bundle, doc_name, time.perf_counter() - started)
        return result

    if request.dry_run:
        result.ok = True
        result.stage = "dry_run"
        result.report_path = _write_report(request, result, bundle, doc_name, time.perf_counter() - started)
        log.info(
            f"[dry-run] {doc_name}: {result.parents} 父块 / {result.chunks} 切片，"
            f"覆盖率 {'OK' if result.coverage_ok else 'FAIL'}"
        )
        return result

    def fail(stage: str, error: str) -> IngestResult:
        """失败也落盘报告，方便排查卡在哪一步。"""
        _fail(result, stage, error)
        result.report_path = _write_report(
            request, result, bundle, doc_name, time.perf_counter() - started
        )
        return result

    store = store or get_kb_store()
    embedder = embedder or get_embedder()

    try:
        store.ensure_collection()
        _check_version(store, doc_id, request.kb_id, request.version)
    except Exception as exc:
        return fail("validate", f"{type(exc).__name__}: {exc}")

    # 3. 向量化（只嵌切片正文：FAQ 即问题，章节即子块）
    try:
        encoded = embedder.encode([chunk.content for chunk in bundle.chunks])
    except EmbeddingError as exc:
        return fail("embed", f"向量化失败：{exc}")
    except Exception as exc:
        return fail("embed", f"向量化异常：{type(exc).__name__}: {exc}")

    result.truncated_vectors = sum(1 for flag in encoded.truncated if flag)
    if result.truncated_vectors:
        result.warnings.append(
            f"{result.truncated_vectors} 个切片超出模型上下文，已按 token 截断后向量化"
        )

    if len(encoded.vectors) != len(bundle.chunks):
        return fail("embed", "向量数量与切片数量不一致")

    records = _build_records(bundle, request, doc_id, doc_name, encoded.vectors, result)

    # 4. 写入 staging
    try:
        store.delete_version(doc_id, request.version, request.kb_id)
        result.points = store.upsert(records)
        result.stage = "upsert"
    except Exception as exc:
        return fail("upsert", f"写入失败（已停留在 staging）：{type(exc).__name__}: {exc}")

    # 5. 校验条数，防止部分写入后误切 active
    try:
        written = store.count(
            kb_id=request.kb_id,
            doc_id=doc_id,
            status="staging",
            index_version=request.version,
        )
    except Exception as exc:
        return fail("verify", f"条数校验失败：{type(exc).__name__}: {exc}")

    if written != result.points:
        return fail(
            "verify",
            f"staging 校验失败：期望 {result.points} 条，实际 {written} 条（未切换 active）",
        )

    # 6. 切换 active，再清理旧版本
    try:
        # 先让新版本 active，再归档旧版本：任何时刻至少保留一个 active，避免检索空窗
        store.set_status(doc_id, "active", kb_id=request.kb_id, index_version=request.version)
        _archive_old_versions(store, doc_id, request.kb_id, request.version)
        _cleanup_old_versions(store, doc_id, request.kb_id, request.keep_last)
        result.stage = "activate"
    except Exception as exc:
        return fail("activate", f"切换 active 失败（新版本仍在 staging）：{type(exc).__name__}: {exc}")

    result.ok = True
    result.stage = "done"
    result.report_path = _write_report(request, result, bundle, doc_name, time.perf_counter() - started)
    log.info(
        f"入库完成 `{doc_name}` -> kb=`{request.kb_id}` v{request.version}，"
        f"{result.points} 点，{time.perf_counter() - started:.1f}s"
    )
    return result


def _fail(result: IngestResult, stage: str, error: str) -> IngestResult:
    result.ok = False
    result.stage = stage
    result.error = error
    if error not in result.warnings:
        result.warnings.append(error)
    log.error(f"入库失败 [{stage}] {result.doc_id}: {error}")
    return result


# ==================== CLI ====================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="知识库文件入库（切分 + 向量化 + 元数据）")
    parser.add_argument("path", help="待入库文件路径")
    parser.add_argument("--kb", required=True, help="知识库 ID，例如 kb_faq_v1")
    parser.add_argument(
        "--agent",
        action="append",
        required=True,
        help="可见该库的 Agent ID，可重复传入",
    )
    parser.add_argument("--tenant", default="default", help="租户 ID")
    parser.add_argument("--doc-id", default=None, help="文档 ID，默认取文件名")
    parser.add_argument("--doc-name", default="", help="展示名，默认取文件名")
    parser.add_argument("--version", type=int, default=1, help="文档版本号")
    parser.add_argument("--keep-last", type=int, default=1, help="保留的历史版本数")
    parser.add_argument("--allow-partial", action="store_true", help="放行硬切/坏单元阈值")
    parser.add_argument("--dry-run", action="store_true", help="只切分与质检，不写向量库")
    parser.add_argument("--report-dir", default=None, help="报告目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    request = IngestRequest(
        path=args.path,
        kb_id=args.kb,
        agent_scope=args.agent,
        tenant_id=args.tenant,
        doc_id=args.doc_id,
        doc_name=args.doc_name,
        version=args.version,
        allow_partial=args.allow_partial,
        dry_run=args.dry_run,
        keep_last=args.keep_last,
        report_dir=args.report_dir,
    )
    result = ingest_file(request)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
