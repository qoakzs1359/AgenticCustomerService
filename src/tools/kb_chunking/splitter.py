"""
切分工具：token 估算、原子块保护、无空洞切片、句边界对齐的 overlap。

约定：
- 所有切点优先落在原子块 / 段落 / 句子边界，句内硬切是最后手段；
- 同一单元内相邻切片可以重叠，但绝不允许出现空洞；
- 原子块宁可超预算也不切开。
"""
from __future__ import annotations

import re

from .models import Atom, Piece

_TERMINAL = "。！？!?；;"
_CLAUSE = "，、,:："
_CLOSERS = "”’）)】」』\"'"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    """轻量 token 估算：CJK 按 1，英文按词，其余按 0.35。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin_words = len(_TOKEN_RE.findall(text))
    latin_chars = sum(len(w) for w in _TOKEN_RE.findall(text))
    others = len(re.sub(r"\s", "", text)) - cjk - latin_chars
    return max(1, int(cjk + latin_words * 1.3 + max(others, 0) * 0.35))


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _char_spans(text: str, start: int, end: int, breaks: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cur = start
    for i in range(start, end):
        if text[i] in breaks:
            j = i + 1
            while j < end and text[j] in _CLOSERS:
                j += 1
            spans.append((cur, j))
            cur = j
    if cur < end:
        spans.append((cur, end))
    return [(s, e) for s, e in (trim_span(text, s, e) for s, e in spans) if e > s]


def _hard_spans(text: str, start: int, end: int, max_tokens: int) -> list[tuple[int, int]]:
    total = estimate_tokens(text[start:end])
    if total <= max_tokens:
        return [(start, end)]
    step = max(1, int((end - start) * max_tokens / total))
    spans = []
    cur = start
    while cur < end:
        nxt = min(cur + step, end)
        s, e = trim_span(text, cur, nxt)
        if e > s:
            spans.append((s, e))
        cur = nxt
    return spans


def _expand(text: str, atom: Atom, max_tokens: int) -> list[Atom]:
    """把超预算的非原子块逐级降级：段落 → 句子 → 分句 → 字符。"""
    seg = text[atom.start : atom.end]
    if atom.atomic or estimate_tokens(seg) <= max_tokens:
        return [atom]

    out: list[Atom] = []
    for s, e in _char_spans(text, atom.start, atom.end, _TERMINAL + "\n"):
        if estimate_tokens(text[s:e]) <= max_tokens:
            out.append(Atom(s, e, "sentence", False, atom.kind))
            continue
        for cs, ce in _char_spans(text, s, e, _CLAUSE):
            if estimate_tokens(text[cs:ce]) <= max_tokens:
                out.append(Atom(cs, ce, "clause", False, atom.kind))
            else:
                for hs, he in _hard_spans(text, cs, ce, max_tokens):
                    out.append(Atom(hs, he, "char", False, atom.kind))
    return out


def _prepare(text: str, atoms: list[Atom], max_tokens: int) -> list[Atom]:
    if not atoms:
        s, e = trim_span(text, 0, len(text))
        atoms = [Atom(s, e, "paragraph", False)]
    out: list[Atom] = []
    for a in atoms:
        if a.end <= a.start:
            continue
        out.extend(_expand(text, a, max_tokens))
    return out


def _overlap_start(text: str, end: int, budget: int, atoms: list[Atom]) -> int:
    """从 end 往前取不超过 budget token 的完整句子作为重叠起点。"""
    if budget <= 0 or end <= 0:
        return end
    i = end
    while i > 0 and estimate_tokens(text[i:end]) <= budget:
        i -= 1
    i = min(end, i + 1)
    # 对齐到句子边界
    while i < end and text[i] not in _TERMINAL and text[i] != "\n":
        i += 1
    if i < end and (text[i] in _TERMINAL or text[i] == "\n"):
        i += 1
    while i < end and text[i] in _CLOSERS:
        i += 1
    # 不落在原子块内部
    for a in atoms:
        if a.atomic and a.start < i < a.end:
            i = a.end
            break
    return i if i < end else end


def split_text(
    text: str,
    atoms: list[Atom] | None = None,
    max_tokens: int = 480,
    min_tokens: int = 0,
    overlap_tokens: int = 0,
) -> list[Piece]:
    """按原子块切分，返回带区间与重叠信息的切片。"""
    if not text or not text.strip():
        return []

    prepared = _prepare(text, list(atoms or []), max_tokens)

    # ① 贪心装箱：原子块独占，其余按顺序填满
    groups: list[list[Atom]] = []
    cur: list[Atom] = []
    cur_tokens = 0
    for a in prepared:
        at = estimate_tokens(text[a.start : a.end])
        if a.atomic:
            if cur:
                groups.append(cur)
                cur, cur_tokens = [], 0
            groups.append([a])
            continue
        if cur and cur_tokens + at > max_tokens:
            groups.append(cur)
            cur, cur_tokens = [], 0
        cur.append(a)
        cur_tokens += at
    if cur:
        groups.append(cur)

    spans: list[list[int | str]] = []  # [start, end, boundary, overlap]
    for g in groups:
        s, e = g[0].start, g[-1].end
        boundary = "atomic" if all(a.atomic for a in g) else g[0].boundary
        spans.append([s, e, boundary, 0])

    # ② 句边界重叠
    for idx in range(1, len(spans)):
        prev_end = int(spans[idx - 1][1])
        cur_start, cur_end = int(spans[idx][0]), int(spans[idx][1])
        if cur_start < prev_end:
            continue  # 合并产生的重叠，保持原样
        s = _overlap_start(text, prev_end, overlap_tokens, prepared)
        if s >= cur_start:
            continue
        ts, te = trim_span(text, s, cur_end)
        if te <= ts or estimate_tokens(text[ts:te]) > max_tokens * 1.2:
            continue
        spans[idx][0] = ts
        spans[idx][3] = prev_end - ts

    # ③ 过短的块向后合并到前一块（允许轻微超过预算）
    i = 1
    while i < len(spans):
        ts, te = int(spans[i][0]), int(spans[i][1])
        if estimate_tokens(text[ts:te]) < min_tokens:
            ps, pe = int(spans[i - 1][0]), int(spans[i - 1][1])
            if estimate_tokens(text[ps:te]) <= max_tokens * 1.25:
                spans[i - 1][1] = te
                spans.pop(i)
                continue
        i += 1
    if len(spans) > 1:
        ts, te = int(spans[0][0]), int(spans[0][1])
        if estimate_tokens(text[ts:te]) < min_tokens:
            n_end = int(spans[1][1])
            if estimate_tokens(text[ts:n_end]) <= max_tokens * 1.25:
                spans[1][0], spans[1][3] = ts, 0
                spans.pop(0)

    pieces: list[Piece] = []
    for s, e, boundary, overlap in spans:
        s, e = trim_span(text, int(s), int(e))
        if e <= s:
            continue
        body = text[s:e]
        pieces.append(
            Piece(
                content=body,
                body_start=s,
                body_end=e,
                boundary=str(boundary),
                slice_mode="slice",
                overlap_chars=int(overlap),
                tokens=estimate_tokens(body),
            )
        )
    return pieces
