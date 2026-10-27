"""命令行 dry-run：查看类型判定、切片边界与覆盖率报告。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunkers import load_routes
from .pipeline import bundle_to_dict, chunk_file, preview


def _print_preview(data: dict) -> None:
    det = data["detection"]
    rep = data["report"]
    print(f"文件类型: {det['file_type']}  profile={det['profile']}  置信度={det['confidence']}")
    for e in det["evidence"]:
        print(f"  证据: {e}")
    for c in det["conflicts"]:
        print(f"  冲突: {c}")

    print(
        f"\n单元={rep.get('units')} 父块={rep.get('parents')} 切片={rep.get('chunks')} "
        f"被切单元={rep.get('split_units')} 超预算={rep.get('oversized')} 硬切={rep.get('hard_cuts')}"
    )
    print(f"覆盖率校验: {'通过（无空洞）' if rep.get('coverage_ok') else '失败'}，空洞 {rep.get('gap_count', 0)} 处")
    for gap in rep.get("gaps", []):
        print(f"  空洞: {gap}")
    for w in rep.get("warnings", []):
        print(f"  提示: {w}")
    for b in rep.get("bad_units", []):
        print(f"  失败单元: {b}")

    print("\n切片明细:")
    for c in data["chunks"]:
        print(
            f"[{c['chunk_id']}] {c['part']}/{c['total']} boundary={c['boundary']} "
            f"tokens={c['tokens']} body={c['body']} overlap={c['overlap_chars']}"
        )
        text = c["content"].replace("\n", " / ")
        print(f"    {text}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="知识库切分 dry-run")
    parser.add_argument("path", help="待检查的文件")
    parser.add_argument("--title", default="", help="文档标题（默认取文件名）")
    parser.add_argument("--doc-id", default="", help="文档 ID（默认取文件名）")
    parser.add_argument("--routes", default="", help="路由配置 JSON 路径")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON（含 Parent/Chunk）")
    parser.add_argument("--out", default="", help="把 JSON 写入文件")
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"文件不存在: {path}")
        return 2

    routes = load_routes(args.routes) if args.routes else None
    doc_id = args.doc_id or path.stem

    if args.json:
        bundle = chunk_file(path, doc_id=doc_id, routes=routes, title=args.title)
        data = bundle_to_dict(bundle)
    else:
        data = preview(path, doc_id=doc_id, routes=routes)
        _print_preview(data)

    if args.out:
        Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入: {args.out}")
    elif args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
