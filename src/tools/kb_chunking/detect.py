"""
文件类型基础判定：文件头魔数 + 扩展名 + 文本结构嗅探。

判定顺序：
1. 魔数判容器（PDF / ZIP / OLE2 / HTML / 文本）；
2. ZIP 内查 OOXML 路径，区分 docx / xlsx / pptx；
3. 文本类做结构打分（csv / md / json / html / txt）；
4. 扩展名与内容冲突时以内容为准，并记录 conflict。
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path

from .models import FileTypeInfo

_EXT_MAP = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".docm": "docx",
    ".doc": "doc",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".xls": "xls",
    ".pptx": "pptx",
    ".csv": "csv",
    ".tsv": "csv",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
    ".text": "txt",
    ".htm": "html",
    ".html": "html",
    ".json": "json",
    ".jsonl": "json",
}

_QA_QUESTION_HEADERS = ("问题", "问句", "标准问", "用户问", "query", "question")
_QA_ANSWER_HEADERS = ("答案", "回答", "标准回", "标准答", "回复", "answer", "reply", "response")

_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _ext(path: Path) -> str:
    return _EXT_MAP.get(path.suffix.lower(), "")


def _decode_head(head: bytes) -> str:
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(head).best()
        if best is not None:
            return str(best)
    except Exception:
        pass
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return head.decode(enc)
        except UnicodeDecodeError:
            continue
    return head.decode("utf-8", errors="replace")


def _looks_binary(head: bytes) -> bool:
    if not head:
        return False
    return b"\x00" in head[:4096]


def _csv_stats(text: str) -> tuple[bool, float]:
    """返回 (是否像 csv, 置信度)，用分隔符与列数一致性判断。"""
    lines = [ln for ln in text.splitlines()[:30] if ln.strip()]
    if len(lines) < 2:
        return False, 0.0
    for delim in (",", "\t", ";"):
        counts = [len(next(csv.reader([ln], delimiter=delim), [])) for ln in lines]
        if not counts or min(counts) < 2:
            continue
        avg = sum(counts) / len(counts)
        stable = sum(1 for c in counts if c == counts[0]) / len(counts)
        if avg >= 2 and stable >= 0.8:
            return True, 0.9 if stable >= 0.95 else 0.75
    return False, 0.0


def _md_stats(text: str) -> tuple[bool, float]:
    headings = len(re.findall(r"(?m)^#{1,6}\s+\S", text))
    fences = len(re.findall(r"(?m)^```", text))
    links = len(re.findall(r"\[[^\]]+\]\([^)]+\)", text))
    bullet = len(re.findall(r"(?m)^\s*[-*+]\s+\S", text))
    score = 0.0
    if headings >= 2:
        score += 0.6
    elif headings == 1:
        score += 0.35
    if fences >= 2:
        score += 0.2
    if links >= 2:
        score += 0.1
    if bullet >= 3:
        score += 0.1
    return score >= 0.5, min(score, 0.95)


def _html_stats(text: str) -> tuple[bool, float]:
    head = text[:4096].lower()
    tags = len(re.findall(r"<[a-z][a-z0-9]*\b", head))
    if re.search(r"<!doctype html|<html[\s>]|<body[\s>]", head) or tags >= 10:
        return True, 0.95 if tags >= 10 else 0.7
    return False, 0.0


def _json_stats(text: str) -> tuple[bool, float]:
    stripped = text.strip()
    if not stripped:
        return False, 0.0
    try:
        json.loads(stripped)
        return True, 0.95
    except Exception:
        pass
    lines = [ln for ln in stripped.splitlines()[:20] if ln.strip()]
    if lines and all(ln.lstrip().startswith("{") for ln in lines):
        try:
            for ln in lines:
                json.loads(ln)
            return True, 0.85
        except Exception:
            return False, 0.0
    return False, 0.0


def _qa_profile(header: list[str]) -> str:
    lowered = [h.strip().lower() for h in header]
    has_q = any(any(k in h for k in _QA_QUESTION_HEADERS) for h in lowered)
    has_a = any(any(k in h for k in _QA_ANSWER_HEADERS) for h in lowered)
    return "qa_table" if has_q and has_a else "table"


def _json_profile(text: str) -> str:
    """从 JSON/JSONL 的键名判断是否为问答结构。"""
    try:
        data = json.loads(text)
    except Exception:
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        try:
            data = json.loads(first)
        except Exception:
            return "table"
    keys: list[str] = []
    if isinstance(data, list) and data and isinstance(data[0], dict):
        keys = [str(k) for k in data[0].keys()]
    elif isinstance(data, dict):
        keys = [str(k) for k in data.keys()]
    return _qa_profile(keys) if keys else "table"


def _csv_header(text: str) -> list[str]:
    try:
        sample = text[:8192]
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    return [c.strip() for c in rows[0]] if rows else []


def _pdf_profile(path: Path) -> tuple[str, list[str], float]:
    """采样 PDF 文本层，区分文本型与扫描型。"""
    try:
        import pymupdf as fitz  # noqa: F401
    except Exception:
        try:
            import fitz
        except Exception:
            return "layout_doc", ["PyMuPDF 未安装，无法采样 PDF 文本层"], 0.5

    try:
        doc = fitz.open(path)
    except Exception as exc:  # 损坏或加密
        return "layout_doc", [f"PDF 打开失败: {exc}"], 0.3

    try:
        pages = min(len(doc), 3)
        chars = 0
        for i in range(pages):
            chars += len((doc[i].get_text("text") or "").strip())
        avg = chars / max(pages, 1)
        if avg < 50:
            return "scanned_pdf", [f"前 {pages} 页平均文本 {avg:.0f} 字，疑似扫描件"], 0.75
        return "layout_doc", [f"前 {pages} 页平均文本 {avg:.0f} 字，含文本层"], 0.95
    finally:
        doc.close()


def _zip_subtype(path: Path) -> tuple[str, list[str]]:
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
    except Exception as exc:
        return "zip", [f"ZIP 读取失败: {exc}"]
    if "word/document.xml" in names:
        return "docx", ["ZIP 内含 word/document.xml"]
    if "xl/workbook.xml" in names:
        return "xlsx", ["ZIP 内含 xl/workbook.xml"]
    if "ppt/presentation.xml" in names:
        return "pptx", ["ZIP 内含 ppt/presentation.xml"]
    return "zip", ["ZIP 容器但非 OOXML 文档"]


def detect_file_type(path: str | Path) -> FileTypeInfo:
    p = Path(path)
    ext_type = _ext(p)
    head = p.read_bytes()[:8192]
    evidence: list[str] = []
    conflicts: list[str] = []

    # ① 容器判定
    if head.startswith(b"%PDF"):
        evidence.append("魔数 %PDF")
        profile, notes, conf = _pdf_profile(p)
        evidence.extend(notes)
        if ext_type and ext_type != "pdf":
            conflicts.append(f"扩展名 .{p.suffix.lstrip('.')} 与内容 PDF 不一致")
            conf = min(conf, 0.85)
        return FileTypeInfo("pdf", profile, conf, evidence, conflicts)

    if head.startswith(b"PK\x03\x04"):
        sub, notes = _zip_subtype(p)
        evidence.extend(notes)
        if sub in ("docx", "xlsx", "pptx"):
            if ext_type and ext_type != sub:
                conflicts.append(f"扩展名 .{p.suffix.lstrip('.')} 与容器 {sub} 不一致")
            profile = _default_profile(sub, p) if sub == "xlsx" else _profile_for(sub)
            return FileTypeInfo(sub, profile, 0.99 if not conflicts else 0.85, evidence, conflicts)
        return FileTypeInfo(sub, "generic", 0.4, evidence, conflicts)

    if head.startswith(_OLE_MAGIC):
        evidence.append("魔数 OLE2")
        ext = p.suffix.lower()
        if ext in (".xls", ".xlsx", ".xlsm"):
            return FileTypeInfo("xls", "table", 0.9, evidence, conflicts)
        if ext in (".doc", ".docx", ".docm"):
            return FileTypeInfo("doc", "plain_doc", 0.9, evidence, conflicts)
        return FileTypeInfo("ole", "generic", 0.4, evidence, conflicts)

    if _looks_binary(head):
        evidence.append("内容含空字节，判定为未知二进制")
        return FileTypeInfo("bin", "unsupported", 0.2, evidence, conflicts)

    # ② 文本类：结构打分
    text = _decode_head(head)
    is_html, html_conf = _html_stats(text)
    is_json, json_conf = _json_stats(text)
    is_csv, csv_conf = _csv_stats(text)
    is_md, md_conf = _md_stats(text)

    candidates: list[tuple[str, float, str]] = []
    if is_json:
        candidates.append(("json", json_conf, "内容可解析为 JSON"))
    if is_html:
        candidates.append(("html", html_conf, "HTML 标签密度高"))
    if is_csv:
        candidates.append(("csv", csv_conf, "分隔符与列数稳定"))
    if is_md:
        candidates.append(("md", md_conf, "Markdown 标题/围栏特征"))
    if not candidates:
        candidates.append(("txt", 0.6, "纯文本，无明显结构标记"))

    # 扩展名只作为加权，不决定结论
    ext_bonus = 0.1
    for i, (t, conf, why) in enumerate(candidates):
        if t == ext_type:
            candidates[i] = (t, min(conf + ext_bonus, 0.99), why + "；与扩展名一致")

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_type, best_conf, best_why = candidates[0]
    evidence.append(best_why)

    if ext_type and ext_type != best_type:
        conflicts.append(f"扩展名 .{p.suffix.lstrip('.')} 与内容判定 {best_type} 不一致，以内容为准")
        best_conf = min(best_conf, 0.8)

    if best_type == "csv":
        header = _csv_header(text)
        profile = _qa_profile(header) if header else "table"
        evidence.append(f"表头形态判定 profile={profile}")
    elif best_type == "md":
        profile = "heading_doc"
    elif best_type == "html":
        profile = "heading_doc"
    elif best_type == "json":
        profile = _json_profile(text)
        evidence.append(f"键名形态判定 profile={profile}")
    else:
        profile = "plain_doc"

    return FileTypeInfo(best_type, profile, best_conf, evidence, conflicts)


def _profile_for(file_type: str) -> str:
    if file_type == "docx":
        return "heading_doc"
    if file_type == "pptx":
        return "heading_doc"
    return "generic"


def _default_profile(file_type: str, path: Path) -> str:
    if file_type != "xlsx":
        return _profile_for(file_type)
    try:
        from openpyxl import load_workbook

        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]
        header: list[str] = []
        for row in ws.iter_rows(max_row=5, values_only=True):
            vals = [str(v).strip() for v in row if v is not None]
            if vals:
                header = vals
                break
        wb.close()
        return _qa_profile(header) if header else "table"
    except Exception:
        return "table"
