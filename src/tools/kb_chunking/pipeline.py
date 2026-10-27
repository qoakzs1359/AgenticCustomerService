"""
入库管道：探测 → 解析 → 单元级路由 → 切分 → 覆盖率校验 → Parent/Chunk。

两条硬保证：
1. slice 型切片必须无空洞覆盖父块区间（gaps 为空）；
2. 每个切片都挂 parent_id，检索命中后可展开完整原文。
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .chunkers import (
    DEFAULT_ROUTES,
    Route,
    chunk_recursive,
    get_chunker,
    resolve_route,
    strategy_params,
)
from .detect import detect_file_type
from .models import Chunk, ChunkBundle, IngestReport, Parent, Piece, Unit
from .parsers import ParseError, parse_document

STRATEGY_VERSION = 1


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _validate_pieces(unit: Unit, pieces: list[Piece], report: IngestReport) -> None:
    slices = [p for p in pieces if p.slice_mode == "slice"]
    if not slices:
        return
    ordered = sorted(slices, key=lambda p: p.body_start)
    cursor = 0
    for p in ordered:
        if p.body_start < 0 or p.body_end > len(unit.text):
            report.gaps.append(f"{unit.unit_id}: 区间越界 [{p.body_start},{p.body_end})")
            continue
        if p.body_start > cursor and unit.text[cursor : p.body_start].strip():
            report.gaps.append(f"{unit.unit_id}: 空洞 [{cursor},{p.body_start})")
        cursor = max(cursor, p.body_end)
    if cursor < len(unit.text) and unit.text[cursor:].strip():
        report.gaps.append(f"{unit.unit_id}: 尾部未覆盖 [{cursor},{len(unit.text)})")


def _make_chunks(
    unit: Unit,
    pieces: list[Piece],
    route: Route,
    params: dict,
    file_type: str,
    extra_metadata: dict[str, Any],
    report: IngestReport,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    total = len(pieces)
    max_tokens = int(params.get("max_tokens", 480))

    for idx, piece in enumerate(pieces):
        if not piece.content.strip():
            report.warnings.append(f"{unit.unit_id}: 第 {idx + 1} 片为空，已跳过")
            continue
        if piece.tokens > max_tokens * 1.2:
            report.oversized_count += 1
        if piece.boundary == "char":
            report.hard_cut_count += 1

        payload: dict[str, Any] = {}
        payload.update(extra_metadata)
        payload.update(piece.payload)
        payload.update(
            {
                "sys_doc_id": unit.meta.get("doc_id") or unit.unit_id.split("#")[0],
                "sys_parent_id": unit.unit_id,
                "sys_file_type": file_type,
                "sys_strategy": route.strategy,
                "sys_strategy_version": params.get("strategy_version", STRATEGY_VERSION),
                "sys_side": unit.kind,
                "sys_unit_kind": unit.kind,
                "sys_part": idx,
                "sys_total": total,
                "sys_boundary": piece.boundary,
                "sys_overlap_chars": piece.overlap_chars,
                "sys_body_start": piece.body_start,
                "sys_body_end": piece.body_end,
                "sys_content_hash": _sha1(piece.content),
                "sys_parent_hash": _sha1(unit.text),
                "sys_loc": unit.loc,
                "sys_category": dict(unit.meta.get("category") or {}),
                "breadcrumb": list(unit.breadcrumb),
            }
        )
        chunks.append(
            Chunk(
                chunk_id=f"{unit.unit_id}#c{idx + 1}",
                parent_id=unit.unit_id,
                content=piece.content,
                payload=payload,
                body_start=piece.body_start,
                body_end=piece.body_end,
                part=idx,
                total=total,
                boundary=piece.boundary,
                overlap_chars=piece.overlap_chars,
                tokens=piece.tokens,
            )
        )
    return chunks


def chunk_file(
    path: str | Path,
    doc_id: str | None = None,
    routes: list[Route] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    title: str = "",
) -> ChunkBundle:
    """解析并切分单个文件，返回 Parent/Chunk 与质检报告。"""
    p = Path(path)
    doc_id = doc_id or p.stem
    info = detect_file_type(p)
    report = IngestReport(
        file=str(p),
        doc_id=doc_id,
        file_type=info.file_type,
        profile=info.profile,
        confidence=info.confidence,
    )
    report.warnings.extend(info.conflicts)
    if info.confidence < 0.6:
        report.warnings.append(f"类型判定置信度较低（{info.confidence:.2f}），已按内容兜底")

    bundle = ChunkBundle(report=report)
    try:
        doc = parse_document(p, info, doc_id, title)
    except ParseError as exc:
        report.warnings.append(str(exc))
        return bundle
    except Exception as exc:  # 损坏 / 加密 / 第三方库异常
        report.warnings.append(f"解析失败：{type(exc).__name__}: {exc}")
        return bundle

    report.warnings.extend(doc.diagnostics)
    report.unit_count = len(doc.units)
    extra_metadata = extra_metadata or {}

    for unit in doc.units:
        unit.meta.setdefault("doc_id", doc_id)
        profile = unit.profile if unit.profile and unit.profile != "generic" else doc.profile
        route = resolve_route(doc.file_type, profile, routes or DEFAULT_ROUTES)
        params = strategy_params(route.strategy, route.params)

        try:
            pieces = get_chunker(route.strategy)(unit, params)
        except Exception as exc:
            report.bad_units.append({"unit": unit.unit_id, "strategy": route.strategy, "error": str(exc)})
            pieces = chunk_recursive(unit, strategy_params("recursive", None))

        if not pieces:
            report.bad_units.append({"unit": unit.unit_id, "strategy": route.strategy, "error": "无切片产出"})
            continue

        _validate_pieces(unit, pieces, report)
        if len(pieces) > 1:
            report.split_unit_count += 1

        parent = Parent(
            parent_id=unit.unit_id,
            text=unit.text,
            breadcrumb=unit.breadcrumb,
            loc=unit.loc,
            kind=unit.kind,
            atomic=unit.atomic,
            meta={"profile": unit.profile, "route": route.strategy},
        )
        report.parent_count += 1

        chunks = _make_chunks(unit, pieces, route, params, doc.file_type, extra_metadata, report)
        bundle.parents.append(parent)
        bundle.chunks.extend(chunks)
        report.chunk_count += len(chunks)

    report.coverage_ok = not report.gaps
    return bundle


def preview(path: str | Path, doc_id: str | None = None, routes: list[Route] | None = None,
            extra_metadata: dict[str, Any] | None = None, content_limit: int = 600) -> dict[str, Any]:
    """Dry-run：输出判定结果、切片边界与覆盖率，用于入库前确认切分效果。"""
    bundle = chunk_file(path, doc_id=doc_id, routes=routes, extra_metadata=extra_metadata)
    info = detect_file_type(path)
    report = bundle.report

    parents = []
    for parent in bundle.parents:
        text = parent.text if len(parent.text) <= content_limit else parent.text[:content_limit] + "…"
        parents.append(
            {
                "parent_id": parent.parent_id,
                "kind": parent.kind,
                "breadcrumb": list(parent.breadcrumb),
                "loc": parent.loc,
                "chars": len(parent.text),
                "text": text,
                "truncated": len(parent.text) > content_limit,
            }
        )

    chunks = []
    for chunk in bundle.chunks:
        content = chunk.content if len(chunk.content) <= content_limit else chunk.content[:content_limit] + "…"
        chunks.append(
            {
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "part": chunk.part + 1,
                "total": chunk.total,
                "boundary": chunk.boundary,
                "tokens": chunk.tokens,
                "body": [chunk.body_start, chunk.body_end],
                "overlap_chars": chunk.overlap_chars,
                "content": content,
                "payload": chunk.payload,
            }
        )

    return {
        "detection": {
            "file_type": info.file_type,
            "profile": info.profile,
            "confidence": round(info.confidence, 2),
            "evidence": info.evidence,
            "conflicts": info.conflicts,
        },
        "report": report.summary() if report else {},
        "parents": parents,
        "chunks": chunks,
    }


def bundle_to_dict(bundle: ChunkBundle) -> dict[str, Any]:
    return {
        "report": asdict(bundle.report) if bundle.report else {},
        "parents": [asdict(p) for p in bundle.parents],
        "chunks": [asdict(c) for c in bundle.chunks],
    }
