"""
知识库切分：按文件类型与内容形态动态选择切分策略。

对外主入口：
    detect_file_type(path) -> FileTypeInfo
    chunk_file(path, ...)  -> ChunkBundle（Parent + Chunk + 质检报告）
    preview(path, ...)     -> dict（dry-run 预览）
"""
from __future__ import annotations

from .chunkers import DEFAULT_ROUTES, Route, load_routes, resolve_route
from .detect import detect_file_type
from .models import (
    Atom,
    Block,
    Chunk,
    ChunkBundle,
    FileTypeInfo,
    IngestReport,
    NormalizedDoc,
    Parent,
    Piece,
    Unit,
)
from .pipeline import bundle_to_dict, chunk_file, preview
from .retrieval import expand_hits
from .splitter import estimate_tokens

__all__ = [
    "Atom",
    "Block",
    "Chunk",
    "ChunkBundle",
    "DEFAULT_ROUTES",
    "FileTypeInfo",
    "IngestReport",
    "NormalizedDoc",
    "Parent",
    "Piece",
    "Route",
    "Unit",
    "bundle_to_dict",
    "chunk_file",
    "detect_file_type",
    "estimate_tokens",
    "expand_hits",
    "load_routes",
    "preview",
    "resolve_route",
]
