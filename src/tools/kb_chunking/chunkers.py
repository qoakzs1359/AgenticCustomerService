"""
切分策略实现与注册表。

策略只接收 Unit + params，返回 list[Piece]：
- qa_pair      一问一答一行，问题进向量，答案进 parent/payload
- table_row    结构化表格的一行，超长按列组拆但保留行主键
- section      按章节/段落切，带面包屑，原子块不切
- recursive    无结构文本兜底，按段落→句子→分句降级
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .models import Piece, Route, Unit
from .splitter import estimate_tokens, split_text

ChunkFn = Callable[[Unit, dict], list[Piece]]
_CHUNKERS: dict[str, ChunkFn] = {}

DEFAULT_PARAMS: dict[str, dict] = {
    "qa_pair": {
        "max_synonyms_per_chunk": 8,
        "answer_in_embedding": False,
        "answer_head_chars": 120,
        "max_answer_chars": 1500,
    },
    "table_row": {
        "max_tokens": 480,
        "max_columns_per_chunk": 8,
        "key_columns": [],
    },
    "section": {
        "max_tokens": 480,
        "min_tokens": 120,
        "overlap_tokens": 60,
        "breadcrumb_in_content": True,
        "breadcrumb_sep": " > ",
    },
    "recursive": {
        "max_tokens": 480,
        "min_tokens": 80,
        "overlap_tokens": 72,
        "title_in_content": True,
    },
    "table_grid": {
        "max_tokens": 420,
        "repeat_header": True,
        "breadcrumb_in_content": True,
        "breadcrumb_sep": " > ",
    },
}


def register(name: str) -> Callable[[ChunkFn], ChunkFn]:
    def wrap(fn: ChunkFn) -> ChunkFn:
        _CHUNKERS[name] = fn
        return fn

    return wrap


def get_chunker(name: str) -> ChunkFn:
    if name not in _CHUNKERS:
        raise KeyError(f"未注册的切分策略: {name}，可用: {sorted(_CHUNKERS)}")
    return _CHUNKERS[name]


def strategy_params(strategy: str, route_params: dict | None) -> dict:
    base_key = "section" if strategy in ("markdown_heading", "docx_section", "pdf_layout", "html_semantic") else strategy
    params = dict(DEFAULT_PARAMS.get(base_key, {}))
    params.update(route_params or {})
    return params


# ==================== 策略实现 ====================


@register("qa_pair")
def chunk_qa_pair(unit: Unit, params: dict) -> list[Piece]:
    q = str(unit.meta.get("question", "")).strip()
    answer = str(unit.meta.get("answer", ""))
    synonyms = [s for s in unit.meta.get("synonyms", []) if str(s).strip()]
    max_syn = max(1, int(params.get("max_synonyms_per_chunk", 8)))

    if not q:
        return split_text(unit.text, unit.atoms, params.get("max_tokens", 480), 0, params.get("overlap_tokens", 0))

    groups = [synonyms[i : i + max_syn] for i in range(0, len(synonyms), max_syn)] or [[]]
    truncated = len(answer) > int(params.get("max_answer_chars", 1500))

    pieces: list[Piece] = []
    for group in groups:
        lines = [q, *group]
        if params.get("answer_in_embedding") and answer:
            lines.append(answer[: int(params.get("answer_head_chars", 120))])
        content = "\n".join(lines)
        pieces.append(
            Piece(
                content=content,
                body_start=0,
                body_end=len(unit.text),
                boundary="unit",
                slice_mode="whole",
                payload={
                    "question": q,
                    "synonyms": group,
                    "answer": answer,
                    "answer_truncated": truncated,
                },
                tokens=estimate_tokens(content),
            )
        )
    return pieces


@register("table_row")
def chunk_table_row(unit: Unit, params: dict) -> list[Piece]:
    cells: dict[str, str] = dict(unit.meta.get("cells", {}))
    columns: list[str] = list(unit.meta.get("columns", [])) or list(cells)
    max_tokens = int(params.get("max_tokens", 480))
    key_columns = params.get("key_columns") or columns[:2]
    max_cols = max(1, int(params.get("max_columns_per_chunk", 8)))

    def render(cols: list[str]) -> str:
        return "\n".join(f"{c}: {cells.get(c, '')}" for c in cols if str(cells.get(c, "")).strip())

    full = render(columns)
    if estimate_tokens(full) <= max_tokens:
        return [
            Piece(
                content=full or unit.text,
                body_start=0,
                body_end=len(unit.text),
                boundary="unit",
                slice_mode="whole",
                payload={"cells": cells, "columns": columns},
                tokens=estimate_tokens(full),
            )
        ]

    pieces: list[Piece] = []
    rest = [c for c in columns if c not in key_columns]
    for i in range(0, len(rest), max_cols):
        group = [c for c in key_columns if c in cells] + rest[i : i + max_cols]
        content = render(group)
        pieces.append(
            Piece(
                content=content,
                body_start=0,
                body_end=len(unit.text),
                boundary="columns",
                slice_mode="whole",
                payload={"cells": {c: cells.get(c, "") for c in group}, "columns": group},
                tokens=estimate_tokens(content),
            )
        )
    return pieces


def _section_piece(unit: Unit, params: dict, prefix: str) -> list[Piece]:
    prefix = prefix.strip()
    overhead = estimate_tokens(prefix) + (1 if prefix else 0)
    budget = max(60, int(params.get("max_tokens", 480)) - overhead)
    pieces = split_text(
        unit.text,
        unit.atoms,
        max_tokens=budget,
        min_tokens=int(params.get("min_tokens", 120)),
        overlap_tokens=int(params.get("overlap_tokens", 60)),
    )
    if prefix:
        for p in pieces:
            p.content = f"{prefix}\n{p.content}"
            p.tokens += overhead
    return pieces


@register("section")
def chunk_section(unit: Unit, params: dict) -> list[Piece]:
    prefix = ""
    if params.get("breadcrumb_in_content", True) and unit.breadcrumb:
        prefix = str(params.get("breadcrumb_sep", " > ")).join(unit.breadcrumb)
    return _section_piece(unit, params, prefix)


@register("recursive")
def chunk_recursive(unit: Unit, params: dict) -> list[Piece]:
    prefix = ""
    if params.get("title_in_content", True):
        title = unit.meta.get("doc_title") or (unit.breadcrumb[-1] if unit.breadcrumb else "")
        prefix = str(title or "")
    return _section_piece(unit, params, prefix)


@register("table_grid")
def chunk_table_grid(unit: Unit, params: dict) -> list[Piece]:
    """表格整表不切；超预算按行分组，每组重复表头，保证每片可独立召回。"""
    rows: list[list[str]] = [list(map(str, r)) for r in unit.meta.get("rows", [])]
    columns: list[str] = [str(c) for c in unit.meta.get("columns", [])]
    max_tokens = int(params.get("max_tokens", 420))
    prefix = ""
    if params.get("breadcrumb_in_content", True) and unit.breadcrumb:
        prefix = str(params.get("breadcrumb_sep", " > ")).join(unit.breadcrumb)
    header = " | ".join(columns)
    overhead = estimate_tokens(prefix) + estimate_tokens(header) + 2

    if not rows:
        content = "\n".join(x for x in (prefix, unit.text) if x)
        return [Piece(content=content, body_start=0, body_end=len(unit.text), boundary="unit",
                      slice_mode="whole", tokens=estimate_tokens(content))]

    groups: list[list[list[str]]] = []
    starts: list[int] = []
    cur: list[list[str]] = []
    cur_tokens = overhead
    cur_start = 0
    for idx, row in enumerate(rows):
        rt = estimate_tokens(" | ".join(row))
        if cur and cur_tokens + rt > max_tokens:
            groups.append(cur)
            starts.append(cur_start)
            cur, cur_tokens, cur_start = [], overhead, idx
        cur.append(row)
        cur_tokens += rt
    if cur:
        groups.append(cur)
        starts.append(cur_start)

    pieces: list[Piece] = []
    for i, group in enumerate(groups):
        body_lines = [" | ".join(r) for r in group]
        parts = [x for x in (prefix, header, *body_lines) if x]
        content = "\n".join(parts)
        pieces.append(
            Piece(
                content=content,
                body_start=0,
                body_end=len(unit.text),
                boundary="rows",
                slice_mode="whole",
                payload={
                    "columns": columns,
                    "rows": group,
                    "row_start": starts[i],
                    "table_part": i,
                },
                tokens=estimate_tokens(content),
            )
        )
    return pieces


# ==================== 路由 ====================


DEFAULT_ROUTES: list[Route] = [
    Route(5, "table_grid", "*", "table_block"),
    Route(10, "qa_pair", "csv", "qa_table"),
    Route(11, "qa_pair", "xlsx", "qa_table"),
    Route(20, "table_row", "csv", "table"),
    Route(21, "table_row", "xlsx", "table"),
    Route(30, "section", "md", "*", {"breadcrumb_in_content": True}),
    Route(31, "section", "html", "*", {"breadcrumb_in_content": True}),
    Route(40, "section", "pdf", "layout_doc", {"max_tokens": 420, "min_tokens": 100, "overlap_tokens": 60}),
    Route(41, "section", "docx", "*", {"breadcrumb_in_content": True}),
    Route(50, "recursive", "txt", "*", {"title_in_content": True}),
    Route(51, "qa_pair", "json", "qa_table"),
    Route(52, "table_row", "json", "table"),
    Route(53, "recursive", "json", "*", {"title_in_content": False}),
    Route(999, "recursive", "*", "*", {"title_in_content": False}),
]


def load_routes(path: str | Path) -> list[Route]:
    """从 JSON 文件加载路由，便于运行时调整而不改代码。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    routes = []
    for item in raw:
        routes.append(
            Route(
                priority=int(item["priority"]),
                strategy=str(item["strategy"]),
                file_type=str(item.get("file_type", "*")),
                profile=str(item.get("profile", "*")),
                params=dict(item.get("params", {})),
            )
        )
    return routes


def resolve_route(file_type: str, profile: str, routes: list[Route] | None = None) -> Route:
    """匹配顺序：类型+形态都精确 > 形态精确 > 类型精确 > 通配；同级按 priority。"""
    candidates = routes or DEFAULT_ROUTES
    matched = [
        r
        for r in candidates
        if r.file_type in (file_type, "*") and r.profile in (profile, "*")
    ]
    if not matched:
        return Route(999, "recursive")

    def score(r: Route) -> tuple[int, int]:
        specificity = (2 if r.file_type == file_type else 0) + (2 if r.profile == profile else 0)
        return (-specificity, r.priority)

    return sorted(matched, key=score)[0]
