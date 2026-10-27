"""
知识库切分的数据模型。

设计要点：
- 解析器只产出 Block / Atom / Unit，不关心怎么切；
- 切分器只产出 Piece，不关心从哪来；
- 管道负责把 Piece 包装成 Parent + Chunk，并校验覆盖率。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ==================== 解析层 ====================


@dataclass(slots=True)
class Block:
    """解析出的结构化块。"""

    type: str  # heading / paragraph / table / code / list / qa_row / raw
    text: str = ""
    level: int = 0
    loc: dict[str, Any] = field(default_factory=dict)  # page / sheet / row / bbox
    columns: list[str] = field(default_factory=list)  # 表格表头
    rows: list[list[str]] = field(default_factory=list)  # 表格数据
    cells: dict[str, str] = field(default_factory=dict)  # 问答/记录行
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Atom:
    """切分的最小寻址单元，start/end 指向 Unit.text。"""

    start: int
    end: int
    boundary: str = "paragraph"  # paragraph / sentence / clause / char / atomic
    atomic: bool = False  # 原子块：表格 / 代码块 / 问答行，不允许被切开
    kind: str = "paragraph"


@dataclass(slots=True)
class Unit:
    """一条可独立路由的完整内容单元：一行问答、一个表格、一个章节。"""

    unit_id: str
    kind: str  # qa_row / record / table / section / paragraph
    text: str
    profile: str = "generic"  # qa_table / table / heading_doc / layout_doc / plain_doc
    breadcrumb: tuple[str, ...] = ()
    loc: dict[str, Any] = field(default_factory=dict)
    atoms: list[Atom] = field(default_factory=list)
    atomic: bool = False  # 整个单元不可拆
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedDoc:
    doc_id: str
    file_type: str
    profile: str
    title: str
    units: list[Unit] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


# ==================== 类型探测 ====================


@dataclass(slots=True)
class FileTypeInfo:
    file_type: str  # pdf / docx / doc / xlsx / xls / pptx / csv / md / txt / html / json
    profile: str  # qa_table / table / heading_doc / layout_doc / scanned_pdf / plain_doc
    confidence: float
    evidence: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


# ==================== 切分层 ====================


@dataclass(slots=True)
class Route:
    """一条路由规则：文件类型 + 内容形态 → 策略 + 参数。"""

    priority: int
    strategy: str
    file_type: str = "*"
    profile: str = "*"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Piece:
    """切分器输出的一片。

    slice_mode="slice" 表示 content 由 Unit.text 的 [body_start, body_end) 片段派生，
    需要参与无空洞覆盖校验；"whole" 表示一片即可代表整个单元，不参与区间校验。
    """

    content: str
    body_start: int = 0
    body_end: int = 0
    boundary: str = "unit"
    slice_mode: str = "slice"
    overlap_chars: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    tokens: int = 0


@dataclass(slots=True)
class Parent:
    """完整内容单元，检索命中子块后可据此还原全文。"""

    parent_id: str
    text: str
    breadcrumb: tuple[str, ...]
    loc: dict[str, Any]
    kind: str
    atomic: bool
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    parent_id: str
    content: str
    payload: dict[str, Any]
    body_start: int
    body_end: int
    part: int
    total: int
    boundary: str
    overlap_chars: int
    tokens: int


@dataclass(slots=True)
class IngestReport:
    file: str
    doc_id: str
    file_type: str
    profile: str
    confidence: float
    unit_count: int = 0
    parent_count: int = 0
    chunk_count: int = 0
    split_unit_count: int = 0  # 被切成多片的单元数
    oversized_count: int = 0  # 单块超过预算
    hard_cut_count: int = 0  # 句内硬切
    gaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    bad_units: list[dict[str, Any]] = field(default_factory=list)
    coverage_ok: bool = True

    def summary(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "doc_id": self.doc_id,
            "file_type": self.file_type,
            "profile": self.profile,
            "confidence": round(self.confidence, 2),
            "units": self.unit_count,
            "parents": self.parent_count,
            "chunks": self.chunk_count,
            "split_units": self.split_unit_count,
            "oversized": self.oversized_count,
            "hard_cuts": self.hard_cut_count,
            "gap_count": len(self.gaps),
            "gaps": self.gaps,
            "coverage_ok": self.coverage_ok,
            "warnings": self.warnings,
            "bad_units": self.bad_units,
        }


@dataclass(slots=True)
class ChunkBundle:
    parents: list[Parent] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    report: IngestReport | None = None
