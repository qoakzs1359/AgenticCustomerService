"""
解析器：把各类文件解析成 NormalizedDoc（Unit 列表）。

原则：
- 解析器不决定切分策略，只负责还原结构、标记原子块；
- 表格单独成 Unit（kind=table, profile=table_block），避免章节切分破坏表格；
- 跨页 / 跨块的连续段落先合并，再交给切分器。
"""
from __future__ import annotations

import csv
import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from .models import Atom, Block, NormalizedDoc, Unit
from .splitter import estimate_tokens

_QA_Q_KEYS = ("标准问", "问题", "问句", "用户问", "query", "question")
_QA_A_KEYS = ("标准回", "标准答", "答案", "回答", "answer", "reply", "回复")
_QA_SYN_KEYS = ("同义", "相似", "问法", "样例问")
_CATEGORY_KEYS = ("一级类别", "二级类别", "类别", "分类", "意图", "编码")
_SUMMARY_SHEETS = ("汇总", "说明", "目录", "cover", "summary", "readme")


class ParseError(RuntimeError):
    pass


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(raw).best()
        if best is not None:
            return str(best)
    except Exception:
        pass
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat(sep=" ") if hasattr(value, "hour") else value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _normalize_columns(header: Iterable[Any]) -> list[str]:
    cols: list[str] = []
    seen: dict[str, int] = {}
    for i, h in enumerate(header):
        name = _cell(h) or f"col{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        cols.append(name)
    return cols


def _match_col(columns: list[str], keys: tuple[str, ...]) -> int | None:
    for i, c in enumerate(columns):
        low = c.lower()
        if any(k.lower() in low for k in keys):
            return i
    return None


def _qa_profile(columns: list[str]) -> str:
    q = _match_col(columns, _QA_Q_KEYS + _QA_SYN_KEYS)
    a = _match_col(columns, _QA_A_KEYS)
    return "qa_table" if q is not None and a is not None else "table"


def _split_synonyms(value: str) -> list[str]:
    parts = re.split(r"[;；|\n]+", value or "")
    return [p.strip() for p in parts if p.strip()]


def _category_of(cells: dict[str, str], columns: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in columns:
        if any(k in c for k in _CATEGORY_KEYS) and cells.get(c):
            out[c] = cells[c]
    return out


# ==================== 通用构建 ====================


class _SectionBuilder:
    def __init__(self, unit_id: str, breadcrumb: tuple[str, ...], profile: str, loc: dict):
        self.unit_id = unit_id
        self.breadcrumb = breadcrumb
        self.profile = profile
        self.loc = dict(loc)
        self.parts: list[str] = []
        self.atoms: list[Atom] = []
        self.length = 0

    def add(self, text: str, boundary: str = "paragraph", atomic: bool = False, kind: str = "paragraph") -> None:
        text = text.strip("\n")
        if not text.strip():
            return
        if self.parts:
            self.parts.append("\n\n")
            self.length += 2
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        self.atoms.append(Atom(start, self.length, "atomic" if atomic else boundary, atomic, kind))

    def build(self) -> Unit | None:
        if not self.parts:
            return None
        return Unit(
            unit_id=self.unit_id,
            kind="section",
            text="".join(self.parts),
            profile=self.profile,
            breadcrumb=self.breadcrumb,
            loc=self.loc,
            atoms=self.atoms,
        )


def _render_table(rows: list[list[str]], columns: list[str]) -> str:
    width = max([len(columns)] + [len(r) for r in rows]) if rows or columns else 0
    cols = columns + [f"col{i + 1}" for i in range(len(columns), width)]
    lines = [" | ".join(cols)] if any(cols) else []
    for r in rows:
        padded = r + [""] * (width - len(r))
        lines.append(" | ".join(padded))
    return "\n".join(lines)


def _sections_from_blocks(
    blocks: list[Block],
    doc_id: str,
    profile: str,
    doc_title: str,
) -> list[Unit]:
    units: list[Unit] = []
    path: list[tuple[int, str]] = []
    sec: _SectionBuilder | None = None
    seq = 0
    table_seq = 0
    title_root = (doc_title,) if doc_title else ()

    def breadcrumb() -> tuple[str, ...]:
        titles = tuple(t for _, t in path)
        if title_root and (not titles or titles[0] != doc_title):
            return title_root + titles
        return titles or title_root

    def flush() -> None:
        nonlocal sec
        if sec is not None:
            unit = sec.build()
            if unit:
                units.append(unit)
        sec = None

    for b in blocks:
        if b.type == "heading":
            flush()
            while path and path[-1][0] >= b.level:
                path.pop()
            path.append((b.level, b.text.strip()))
            seq += 1
            sec = _SectionBuilder(f"{doc_id}#s{seq}", breadcrumb(), profile, b.loc)
            continue

        if b.type == "table":
            table_seq += 1
            rows = b.rows or []
            columns = b.columns or []
            units.append(
                Unit(
                    unit_id=f"{doc_id}#t{table_seq}",
                    kind="table",
                    text=_render_table(rows, columns),
                    profile="table_block",
                    breadcrumb=breadcrumb(),
                    loc=b.loc,
                    atomic=True,
                    meta={"rows": rows, "columns": columns, "kind": "table"},
                )
            )
            continue

        if sec is None:
            seq += 1
            sec = _SectionBuilder(f"{doc_id}#s{seq}", breadcrumb(), profile, b.loc)
        sec.add(b.text, atomic=(b.type == "code"), kind=b.type)
        if b.loc.get("page"):
            sec.loc.setdefault("page_start", b.loc["page"])
            sec.loc["page_end"] = b.loc["page"]

    flush()
    return units


# ==================== 表格类 ====================


def _tabular_units(
    rows: list[list[str]],
    columns: list[str],
    doc_id: str,
    profile: str,
    title: str,
    loc_base: dict,
    diagnostics: list[str],
    row_offset: int = 0,
) -> list[Unit]:
    units: list[Unit] = []
    bad_rows: list[int] = []

    q_idx = _match_col(columns, _QA_Q_KEYS)
    syn_idx = _match_col(columns, _QA_SYN_KEYS)
    a_idx = _match_col(columns, _QA_A_KEYS)
    if q_idx is None and syn_idx is not None and a_idx is not None:
        q_idx, syn_idx = syn_idx, None

    if profile == "qa_table" and (q_idx is None or a_idx is None):
        diagnostics.append("问答表缺少问题列或答案列，降级为普通表格切分")
        profile = "table"

    for r_i, row in enumerate(rows):
        row_no = row_offset + r_i + 1
        padded = list(row) + [""] * (len(columns) - len(row))
        cells = {c: _cell(v) for c, v in zip(columns, padded)}
        if not any(cells.values()):
            continue
        loc = dict(loc_base)
        loc["row"] = row_no

        if profile == "qa_table":
            q_raw = cells.get(columns[q_idx], "").strip() if q_idx is not None else ""
            a = cells.get(columns[a_idx], "").strip()
            syns: list[str] = []
            if syn_idx is not None and syn_idx != q_idx:
                syns = _split_synonyms(cells.get(columns[syn_idx], ""))
            if q_idx is not None and syn_idx == q_idx:
                # 该列本身就是多问句列：首条作主问句，其余作同义问句
                parts = _split_synonyms(q_raw)
                q = parts[0] if parts else ""
                syns = parts[1:]
            else:
                q = q_raw
            if not q and syns:
                q, syns = syns[0], syns[1:]
            if not q or not a:
                bad_rows.append(row_no)
                continue
            category = _category_of(cells, columns)
            text = "\n".join([q, *syns, a])
            units.append(
                Unit(
                    unit_id=f"{doc_id}#r{row_no}",
                    kind="qa_row",
                    text=text,
                    profile="qa_table",
                    breadcrumb=tuple(category.values()),
                    loc=loc,
                    atomic=True,
                    meta={
                        "question": q,
                        "answer": a,
                        "synonyms": syns,
                        "category": category,
                        "cells": cells,
                        "columns": columns,
                    },
                )
            )
        else:
            text = "\n".join(f"{c}: {cells[c]}" for c in columns if cells[c])
            units.append(
                Unit(
                    unit_id=f"{doc_id}#r{row_no}",
                    kind="record",
                    text=text,
                    profile="table",
                    breadcrumb=(title,) if title else (),
                    loc=loc,
                    atomic=True,
                    meta={"cells": cells, "columns": columns},
                )
            )

    if bad_rows:
        sample = ",".join(str(r) for r in bad_rows[:10])
        diagnostics.append(f"跳过 {len(bad_rows)} 行（缺少问题或答案）：行 {sample}")
    return units


def _header_and_rows(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    rows = [r for r in rows if any(_cell(c) for c in r)]
    if not rows:
        return [], []
    # 前 10 行里找真正的表头：至少两个非空单元格，且含文字
    for i in range(min(len(rows), 10)):
        cells = [_cell(c) for c in rows[i]]
        nonempty = [c for c in cells if c]
        if len(nonempty) >= 2 and any(re.search(r"[A-Za-z\u4e00-\u9fff]", c) for c in nonempty):
            return _normalize_columns(cells), rows[i + 1 :]
    width = max(len(r) for r in rows)
    return _normalize_columns([f"col{i + 1}" for i in range(width)]), rows


def parse_csv(path: Path, info: Any, doc_id: str, title: str) -> NormalizedDoc:
    text = _read_text(path)
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    raw_rows = list(csv.reader(io.StringIO(text), dialect))
    rows = [[_cell(c) for c in r] for r in raw_rows]
    columns, data = _header_and_rows(rows)
    profile = info.profile if info.profile in ("qa_table", "table") else _qa_profile(columns)
    diagnostics: list[str] = []
    units = _tabular_units(data, columns, doc_id, profile, title, {}, diagnostics)
    return NormalizedDoc(doc_id, "csv", profile, title, units, diagnostics)


def parse_xlsx(path: Path, info: Any, doc_id: str, title: str) -> NormalizedDoc:
    try:
        from openpyxl import load_workbook
    except Exception as exc:
        raise ParseError("缺少 openpyxl，无法解析 Excel；请先 pip install openpyxl") from exc

    wb = load_workbook(path, data_only=True)
    diagnostics: list[str] = []
    units: list[Unit] = []
    try:
        for ws in wb.worksheets:
            if getattr(ws, "sheet_state", "visible") != "visible":
                diagnostics.append(f"跳过隐藏工作表：{ws.title}")
                continue
            if ws.title.strip().lower() in _SUMMARY_SHEETS:
                diagnostics.append(f"跳过汇总类工作表：{ws.title}")
                continue

            matrix: list[list[str]] = [[_cell(v) for v in row] for row in ws.iter_rows(values_only=True)]
            for rng in ws.merged_cells.ranges:
                value = matrix[rng.min_row - 1][rng.min_col - 1] if rng.min_row - 1 < len(matrix) else ""
                for r in range(rng.min_row - 1, min(rng.max_row, len(matrix))):
                    for c in range(rng.min_col - 1, min(rng.max_col, len(matrix[r]))):
                        if not matrix[r][c]:
                            matrix[r][c] = value

            columns, data = _header_and_rows(matrix)
            if not data and not columns:
                continue
            profile = _qa_profile(columns)
            header_row_no = len(matrix) - len(data)
            sheet_units = _tabular_units(
                data,
                columns,
                f"{doc_id}#{ws.title}",
                profile,
                f"{title} / {ws.title}",
                {"sheet": ws.title},
                diagnostics,
                row_offset=header_row_no,
            )
            units.extend(sheet_units)
    finally:
        wb.close()

    profile = "qa_table" if any(u.profile == "qa_table" for u in units) else "table"
    return NormalizedDoc(doc_id, "xlsx", profile, title, units, diagnostics)


# ==================== Markdown / 文本 ====================


def _parse_md_blocks(text: str) -> list[Block]:
    lines = text.splitlines()
    blocks: list[Block] = []
    i = 0
    para: list[str] = []
    para_line = 0

    def flush_para() -> None:
        nonlocal para
        if para:
            body = "\n".join(para).strip()
            if body:
                blocks.append(Block("paragraph", body, loc={"line": para_line}))
            para = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_para()
            start_line = i + 1
            buf = [line]
            i += 1
            while i < len(lines):
                buf.append(lines[i])
                if lines[i].strip().startswith("```"):
                    break
                i += 1
            blocks.append(Block("code", "\n".join(buf), loc={"line": start_line}))
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_para()
            blocks.append(Block("heading", heading.group(2).strip(), level=len(heading.group(1)), loc={"line": i + 1}))
            i += 1
            continue

        if stripped.startswith("|") and stripped.count("|") >= 2:
            flush_para()
            start_line = i + 1
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows: list[list[str]] = []
            for tl in table_lines:
                cells = [c.strip() for c in tl.strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells):
                    continue
                rows.append(cells)
            columns = rows[0] if rows else []
            blocks.append(
                Block("table", _render_table(rows[1:], columns), loc={"line": start_line}, columns=columns, rows=rows[1:])
            )
            continue

        if re.match(r"^\s*([-*+]|\d+\.)\s+\S", line):
            flush_para()
            start_line = i + 1
            items = []
            while i < len(lines) and re.match(r"^\s*([-*+]|\d+\.)\s+\S", lines[i]):
                items.append(lines[i].strip())
                i += 1
            blocks.append(Block("list", "\n".join(items), loc={"line": start_line}))
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        if not para:
            para_line = i + 1
        para.append(line)
        i += 1

    flush_para()
    return blocks


def _front_matter_title(text: str) -> str:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not m:
        return ""
    for line in m.group(1).splitlines():
        if line.strip().lower().startswith("title:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return ""


def parse_markdown(path: Path, info: Any, doc_id: str, title: str) -> NormalizedDoc:
    text = _read_text(path)
    title = _front_matter_title(text) or title
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.S)
    blocks = _parse_md_blocks(text)
    units = _sections_from_blocks(blocks, doc_id, "heading_doc", title)
    return NormalizedDoc(doc_id, "md", "heading_doc", title, units)


def parse_text(path: Path, info: Any, doc_id: str, title: str) -> NormalizedDoc:
    text = _read_text(path)
    if info.file_type in ("md",) or info.profile == "heading_doc":
        return parse_markdown(path, info, doc_id, title)

    spans = []
    last = 0
    for m in re.finditer(r"\n\s*\n", text):
        spans.append((last, m.start()))
        last = m.end()
    spans.append((last, len(text)))

    units: list[Unit] = []
    for idx, (s, e) in enumerate(spans):
        body = text[s:e].strip()
        if not body:
            continue
        units.append(
            Unit(
                unit_id=f"{doc_id}#p{idx + 1}",
                kind="paragraph",
                text=body,
                profile="plain_doc",
                breadcrumb=(title,) if title else (),
                loc={"paragraph": idx + 1},
                atoms=[Atom(0, len(body), "paragraph", False, "paragraph")],
                meta={"doc_title": title},
            )
        )
    return NormalizedDoc(doc_id, "txt" if info.file_type == "txt" else info.file_type, "plain_doc", title, units)


def parse_json(path: Path, info: Any, doc_id: str, title: str) -> NormalizedDoc:
    text = _read_text(path)
    items: list[dict] | None = None
    data: Any = None
    try:
        data = json.loads(text)
    except Exception:
        # JSONL：逐行解析，常见于问答对导出
        rows: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                rows = []
                break
            if not isinstance(obj, dict):
                rows = []
                break
            rows.append(obj)
        items = rows or None
    else:
        if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
            items = data

    if items:
        columns: list[str] = []
        for item in items:
            for k in item:
                if k not in columns:
                    columns.append(str(k))
        matrix = [[_cell(item.get(c, "")) for c in columns] for item in items]
        diagnostics: list[str] = []
        units = _tabular_units(matrix, columns, doc_id, _qa_profile(columns), title, {}, diagnostics)
        return NormalizedDoc(doc_id, info.file_type, _qa_profile(columns), title, units, diagnostics)

    if data is None:
        return parse_text(path, info, doc_id, title)

    body = json.dumps(data, ensure_ascii=False, indent=2)
    units = [
        Unit(
            unit_id=f"{doc_id}#p1",
            kind="paragraph",
            text=body,
            profile="plain_doc",
            breadcrumb=(title,) if title else (),
            atoms=[Atom(0, len(body), "paragraph", False, "paragraph")],
            meta={"doc_title": title},
        )
    ]
    return NormalizedDoc(doc_id, info.file_type, "plain_doc", title, units)


# ==================== HTML ====================


class _HtmlCollector(HTMLParser):
    _SKIP = {"script", "style", "head", "nav", "footer", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._skip_depth = 0
        self._heading: int | None = None
        self._buf: list[str] = []
        self._buf_type: str | None = None
        self._table_rows: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush()
            self._heading = int(tag[1])
            self._buf_type = "heading"
        elif tag in {"p", "li", "blockquote"}:
            self._flush()
            self._buf_type = "paragraph"
        elif tag == "pre":
            self._flush()
            self._buf_type = "code"
        elif tag == "table":
            self._flush()
            self._table_rows = []
        elif tag == "tr" and self._table_rows is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._table_rows is not None and self._row is not None:
            if any(self._row):
                self._table_rows.append(self._row)
            self._row = None
        elif tag == "table" and self._table_rows is not None:
            rows = self._table_rows
            columns = rows[0] if rows else []
            self.blocks.append(Block("table", _render_table(rows[1:], columns), columns=columns, rows=rows[1:]))
            self._table_rows = None
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._buf_type == "heading":
            self._flush()
            self._heading = None
        elif tag in {"p", "li", "blockquote", "pre"}:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._cell is not None:
            self._cell.append(text + " ")
        elif self._buf_type:
            self._buf.append(text)

    def _flush(self) -> None:
        if not self._buf_type or not self._buf:
            self._buf = []
            return
        text = " ".join(self._buf).strip()
        self._buf = []
        if text:
            if self._buf_type == "heading":
                self.blocks.append(Block("heading", text, level=self._heading or 2))
            else:
                self.blocks.append(Block(self._buf_type, text))
        self._buf_type = None

    def close(self) -> None:
        self._flush()
        super().close()


def parse_html(path: Path, info: Any, doc_id: str, title: str) -> NormalizedDoc:
    text = _read_text(path)
    collector = _HtmlCollector()
    collector.feed(text)
    collector.close()
    blocks: list[Block] = []
    for b in collector.blocks:
        if b.type == "heading" and not b.level:
            # 收尾阶段补回标题级别
            b.level = 2
        blocks.append(b)
    units = _sections_from_blocks(blocks, doc_id, "heading_doc", title)
    return NormalizedDoc(doc_id, "html", "heading_doc", title, units)


# ==================== PDF ====================


def _pdf_blocks(path: Path, diagnostics: list[str]) -> list[Block]:
    try:
        try:
            import pymupdf as fitz
        except Exception:
            import fitz
    except Exception as exc:
        raise ParseError("缺少 PyMuPDF，无法解析 PDF；请先 pip install PyMuPDF") from exc

    doc = fitz.open(path)
    pages_blocks: list[list[dict]] = []
    sizes: list[float] = []
    try:
        for pno, page in enumerate(doc):
            width = page.rect.width
            raw_blocks: list[dict] = []
            tables: list[tuple[Any, list[list[str]]]] = []
            try:
                found = page.find_tables()
                for table in getattr(found, "tables", []):
                    rows = [[_cell(c) for c in row] for row in table.extract()]
                    if rows:
                        tables.append((table.bbox, rows))
            except Exception:
                pass

            data = page.get_text("dict")
            for b in data.get("blocks", []):
                if b.get("type") != 0:
                    continue
                lines = []
                block_sizes = []
                for line in b.get("lines", []):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                    if line_text:
                        lines.append(line_text)
                        block_sizes.extend(span.get("size", 10) for span in line.get("spans", []))
                text = "\n".join(lines).strip()
                if not text:
                    continue
                bbox = b.get("bbox", (0, 0, 0, 0))
                if any(_rect_overlap(bbox, tb) > 0.5 for tb, _ in tables):
                    continue
                raw_blocks.append({"text": text, "bbox": bbox, "size": max(block_sizes or [10]), "page": pno + 1})
                sizes.extend(block_sizes)

            for bbox, rows in tables:
                raw_blocks.append({"text": "", "bbox": bbox, "size": 0, "page": pno + 1, "table": rows})

            if not raw_blocks:
                continue

            left = [b for b in raw_blocks if b["bbox"][0] < width * 0.5 and (b["bbox"][2] - b["bbox"][0]) < width * 0.7]
            right = [b for b in raw_blocks if b["bbox"][0] >= width * 0.5 and (b["bbox"][2] - b["bbox"][0]) < width * 0.7]
            two_col = len(left) >= 2 and len(right) >= 2 and (len(left) + len(right)) >= 0.6 * len(raw_blocks)
            ordered = (
                sorted(left, key=lambda x: x["bbox"][1]) + sorted(right, key=lambda x: x["bbox"][1])
                if two_col
                else sorted(raw_blocks, key=lambda x: (x["bbox"][1], x["bbox"][0]))
            )
            for b in ordered:
                b["page_width"] = width
            pages_blocks.append(ordered)
    finally:
        doc.close()

    if not pages_blocks or sum(len(p) for p in pages_blocks) == 0:
        diagnostics.append("PDF 无可用文本层（疑似扫描件），未产出切片；需启用 OCR 后再入库")
        return []

    # 页眉页脚：跨页重复的首尾文本
    freq: dict[str, int] = {}
    for blocks in pages_blocks:
        texts = [b["text"] for b in blocks if b["text"] and len(b["text"]) <= 80]
        for t in {texts[0] if texts else "", texts[-1] if texts else ""}:
            if t:
                freq[t] = freq.get(t, 0) + 1
    total_pages = len(pages_blocks)
    repeated = {t for t, n in freq.items() if total_pages >= 3 and n >= total_pages * 0.5}

    sorted_sizes = sorted(sizes)
    body_size = sorted_sizes[len(sorted_sizes) // 2] if sorted_sizes else 10

    out: list[Block] = []
    prev_para: dict | None = None
    for blocks in pages_blocks:
        for b in blocks:
            if "table" in b:
                out.append(Block("table", _render_table(b["table"][1:], b["table"][0] if b["table"] else []),
                                 columns=b["table"][0] if b["table"] else [],
                                 rows=b["table"][1:], loc={"page": b["page"]}))
                prev_para = None
                continue
            text = b["text"].strip()
            if text in repeated:
                continue
            is_heading = b["size"] > body_size * 1.15 and len(text) <= 60 and "\n" not in text
            if is_heading:
                out.append(Block("heading", text, level=1, loc={"page": b["page"]}))
                prev_para = None
                continue
            # 跨页/跨块连续段落合并：上一段没有终止标点
            if prev_para is not None and not re.search(r"[。！？!?；;：:）)」”]$", prev_para["text"]):
                prev_para["text"] = prev_para["text"].rstrip() + text.lstrip()
                prev_para = None
                continue
            block = Block("paragraph", text, loc={"page": b["page"]})
            out.append(block)
            prev_para = block

    return out


def _rect_overlap(a: Any, b: Any) -> float:
    try:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0.0, min(ay1, by1) - max(ay0, by0))
        area = max(0.0, (ax1 - ax0) * (ay1 - ay0))
        return (ix * iy / area) if area else 0.0
    except Exception:
        return 0.0


def parse_pdf(path: Path, info: Any, doc_id: str, title: str) -> NormalizedDoc:
    diagnostics: list[str] = []
    blocks = _pdf_blocks(path, diagnostics)
    units = _sections_from_blocks(blocks, doc_id, "layout_doc", title)
    return NormalizedDoc(doc_id, "pdf", "layout_doc", title, units, diagnostics)


# ==================== DOCX ====================


def parse_docx(path: Path, info: Any, doc_id: str, title: str) -> NormalizedDoc:
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except Exception as exc:
        raise ParseError("缺少 python-docx，无法解析 Word；请先 pip install python-docx") from exc

    document = Document(path)
    blocks: list[Block] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            low = style.lower()
            if low.startswith("heading") or "标题" in style:
                digits = re.findall(r"\d+", style)
                level = int(digits[0]) if digits else 1
                blocks.append(Block("heading", text, level=level))
            elif "list" in low or "列表" in style:
                blocks.append(Block("list", text))
            else:
                blocks.append(Block("paragraph", text))
        elif tag == "tbl":
            table = Table(child, document)
            rows: list[list[str]] = []
            for row in table.rows:
                rows.append([_cell(c.text) for c in row.cells])
            if rows:
                blocks.append(Block("table", _render_table(rows[1:], rows[0]), columns=rows[0], rows=rows[1:]))

    units = _sections_from_blocks(blocks, doc_id, "heading_doc", title)
    return NormalizedDoc(doc_id, "docx", "heading_doc", title, units)


# ==================== 分发 ====================


def parse_document(path: str | Path, info: Any, doc_id: str, title: str = "") -> NormalizedDoc:
    p = Path(path)
    title = title or p.stem
    ft = info.file_type

    if ft in ("csv",):
        return parse_csv(p, info, doc_id, title)
    if ft in ("xlsx",):
        return parse_xlsx(p, info, doc_id, title)
    if ft in ("md",):
        return parse_markdown(p, info, doc_id, title)
    if ft in ("txt",):
        return parse_text(p, info, doc_id, title)
    if ft in ("html",):
        return parse_html(p, info, doc_id, title)
    if ft in ("json", "jsonl"):
        return parse_json(p, info, doc_id, title)
    if ft == "pdf":
        return parse_pdf(p, info, doc_id, title)
    if ft == "docx":
        return parse_docx(p, info, doc_id, title)
    if ft in ("doc", "xls", "pptx", "zip", "ole", "bin"):
        raise ParseError(f"暂不支持的格式：{ft}（请转换为 docx / xlsx / pdf 后重试）")

    return parse_text(p, info, doc_id, title)
