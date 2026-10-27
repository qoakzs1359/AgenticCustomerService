"""
检索侧：把命中的子块按父块归并，还原完整内容。

与向量库解耦，入参是检索结果的通用字典，便于接 Qdrant / 其他后端。
"""
from __future__ import annotations

from typing import Any, Iterable

from .models import Parent
from .splitter import estimate_tokens


def expand_hits(
    hits: Iterable[dict[str, Any]],
    parents: dict[str, Parent] | Iterable[Parent],
    top_k: int = 5,
    budget_tokens: int = 1200,
) -> list[dict[str, Any]]:
    """按 parent 归并命中：取每个父块的最高分，返回父块全文（按预算截断）。

    hits 每项至少包含 parent_id 与 score，可选 chunk_id / content。
    """
    lookup: dict[str, Parent] = {}
    if isinstance(parents, dict):
        lookup = dict(parents)
    else:
        lookup = {p.parent_id: p for p in parents}

    best: dict[str, dict[str, Any]] = {}
    matched: dict[str, list[dict[str, Any]]] = {}
    for hit in hits:
        pid = str(hit.get("parent_id") or (hit.get("payload") or {}).get("sys_parent_id") or "")
        if not pid:
            continue
        score = float(hit.get("score", 0.0))
        matched.setdefault(pid, []).append(hit)
        if pid not in best or score > float(best[pid].get("score", 0.0)):
            best[pid] = hit

    ordered = sorted(best.values(), key=lambda h: float(h.get("score", 0.0)), reverse=True)
    out: list[dict[str, Any]] = []
    for hit in ordered[:top_k]:
        pid = str(hit.get("parent_id") or (hit.get("payload") or {}).get("sys_parent_id") or "")
        parent = lookup.get(pid)
        if parent is None:
            out.append(
                {
                    "parent_id": pid,
                    "score": float(hit.get("score", 0.0)),
                    "text": hit.get("content", ""),
                    "breadcrumb": list((hit.get("payload") or {}).get("breadcrumb", [])),
                    "matched_chunks": [h.get("chunk_id") for h in matched.get(pid, [])],
                    "expanded": False,
                    "truncated": False,
                }
            )
            continue

        text = parent.text
        truncated = estimate_tokens(text) > budget_tokens
        if truncated:
            # 按字符比例近似截断，保证不超出预算太多
            ratio = budget_tokens / max(estimate_tokens(text), 1)
            text = text[: max(1, int(len(text) * ratio))]

        out.append(
            {
                "parent_id": pid,
                "score": float(hit.get("score", 0.0)),
                "text": text,
                "breadcrumb": list(parent.breadcrumb),
                "loc": parent.loc,
                "kind": parent.kind,
                "matched_chunks": [h.get("chunk_id") for h in matched.get(pid, [])],
                "expanded": True,
                "truncated": truncated,
            }
        )
    return out
