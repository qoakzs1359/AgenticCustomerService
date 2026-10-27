"""
知识库检索评测：构建留出集 + 网格评测。

评测口径：
- 命中：返回的任一 context 的 biz_category 等于期望的二级类别编码；
- OOD：期望无命中，返回任何 context 都记一次误命中；
- 同时统计 hit@1 / hit@3 / MRR / no_hit / p50 / p95。

用法：
    # 1. 构建评测集（从细分问句与已入库问句去重）
    python -m src.rag.kb_eval build \
        --subdivision data/subdivision_sample_questions.csv \
        --intent data/intent_rag_kb.csv \
        --out data/eval/kb_eval_queries.jsonl

    # 2. 网格评测（前缀 × 阈值 × chunk_type）
    python -m src.rag.kb_eval run \
        --suite data/eval/kb_eval_queries.jsonl \
        --kb kb_faq_v1 --agent-role intent_agent --grid
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.logging import log
from src.rag.kb_retriever import (
    Retriever,
    RetrieveRequest,
    RetrieveResult,
    get_reranker,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_SUITE = _DATA_DIR / "eval" / "kb_eval_queries.jsonl"
DEFAULT_REPORT_DIR = _DATA_DIR / "eval" / "reports"

_NORM_RE = re.compile(r"[\s\W_]+", re.UNICODE)
_SPLIT_RE = re.compile(r"[;；]")
_GRID_THRESHOLDS = [None, 0.40, 0.45, 0.50, 0.55]

# 与知识库无关的域外问题：用于测误命中，不要放可能被合理命中的同域问题
OOD_QUERIES = [
    "今天北京天气怎么样",
    "帮我写一份辞职信",
    "股票明天会涨吗",
    "推荐一部科幻电影",
    "怎么学习 Python",
    "你叫什么名字",
    "讲个笑话",
    "1 加 1 等于几",
    "明天会下雨吗",
    "帮我翻译这句话",
    "附近有什么好吃的",
    "我的账号密码忘了怎么办",
    "我要投诉你们公司",
    "你们老板是谁",
    "如何申请信用卡",
    "推荐一款手机",
    "飞机票怎么退",
    "你在干什么",
    "帮我订一张电影票",
    "上海到北京怎么走",
]


# ==================== 数据结构 ====================


@dataclass(slots=True)
class EvalQuery:
    id: str
    query: str
    expected_category: str
    expected_category_name: str = ""
    type: str = "heldout"  # heldout / ood
    source: str = ""
    expected_kb: str = ""


@dataclass(slots=True)
class EvalConfig:
    use_query_instruction: bool = True
    score_threshold: float | None = None
    use_chunk_type: bool = True
    use_rerank: bool = False
    top_k: int = 10
    rerank_top_n: int = 5
    min_candidates: int = 5

    def label(self) -> str:
        prefix = "on" if self.use_query_instruction else "off"
        threshold = "none" if self.score_threshold is None else f"{self.score_threshold:.2f}"
        chunk_type = "on" if self.use_chunk_type else "off"
        rerank = "on" if self.use_rerank else "off"
        return f"prefix={prefix} thr={threshold} type={chunk_type} rerank={rerank}"


@dataclass(slots=True)
class EvalReport:
    config: EvalConfig
    total_queries: int
    in_domain: int
    hit_at_1: float
    hit_at_3: float
    mrr: float
    ood_false_hit_rate: float
    no_hit_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    per_category: dict[str, float]
    gates: dict[str, bool]
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["config_label"] = self.config.label()
        data["passed"] = all(self.gates.values())
        return data


# ==================== 构建评测集 ====================


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            with open(path, encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "unknown", b"", 0, 1, f"无法识别 {path} 的编码: {last_error}"
    )


def _normalize(text: str) -> str:
    return _NORM_RE.sub("", (text or "").lower())


def _trigrams(text: str) -> set[str]:
    if len(text) < 3:
        return {text} if text else set()
    return {text[i : i + 3] for i in range(len(text) - 2)}


def _jaccard(a: str, b: str) -> float:
    ga, gb = _trigrams(a), _trigrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _is_duplicate(text: str, existing: list[str], threshold: float = 0.9) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return True
    for item in existing:
        if _jaccard(normalized, item) >= threshold:
            return True
    return False


def build_suite(
    subdivision_path: str | Path,
    intent_path: str | Path,
    out_path: str | Path,
    *,
    per_category: int = 5,
    ood_count: int = 20,
    seed: int = 42,
    kb_id: str = "kb_faq_v1",
) -> dict[str, Any]:
    """按二级类别分层抽样，剔除与已入库问句高度相似的样本。"""
    intent_rows = _read_csv(intent_path)
    ingested: list[str] = []
    for row in intent_rows:
        for question in _SPLIT_RE.split(row.get("样例问句", "")):
            normalized = _normalize(question)
            if normalized:
                ingested.append(normalized)

    grouped: dict[str, list[dict[str, str]]] = {}
    names: dict[str, str] = {}
    for row in _read_csv(subdivision_path):
        code = (row.get("二级类别编码") or "").strip()
        if not code:
            continue
        grouped.setdefault(code, []).append(row)
        names.setdefault(code, (row.get("二级类别名称") or "").strip())

    rng = random.Random(seed)
    queries: list[EvalQuery] = []
    stats: dict[str, dict[str, int]] = {}

    for code in sorted(grouped):
        rows = grouped[code][:]
        rng.shuffle(rows)
        picked: list[EvalQuery] = []
        for row in rows:
            question = (row.get("样例问句") or "").strip()
            if not question or _is_duplicate(question, ingested):
                continue
            ingested.append(_normalize(question))
            picked.append(
                EvalQuery(
                    id=f"{code}-{len(picked) + 1:03d}",
                    query=question,
                    expected_category=code,
                    expected_category_name=names.get(code, ""),
                    type="heldout",
                    source=Path(subdivision_path).name,
                    expected_kb=kb_id,
                )
            )
            if len(picked) >= per_category:
                break
        queries.extend(picked)
        stats[code] = {"rows": len(rows), "evaluated": len(picked)}

    ood_pool = list(OOD_QUERIES)
    rng.shuffle(ood_pool)
    for idx, query in enumerate(ood_pool[: max(0, ood_count)], 1):
        queries.append(
            EvalQuery(
                id=f"OOD-{idx:03d}",
                query=query,
                expected_category="",
                type="ood",
                source="ood",
                expected_kb=kb_id,
            )
        )

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in queries:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    summary = {
        "out": str(path),
        "total": len(queries),
        "heldout": sum(1 for q in queries if q.type == "heldout"),
        "ood": sum(1 for q in queries if q.type == "ood"),
        "categories": stats,
    }
    log.info(
        f"评测集已生成：{summary['heldout']} 条留出 + {summary['ood']} 条域外 -> {path}"
    )
    return summary


def load_suite(path: str | Path) -> list[EvalQuery]:
    items: list[EvalQuery] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(EvalQuery(**json.loads(line)))
    return items


# ==================== 评测 ====================


def _latency_percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[index]


def _rank_of(result: RetrieveResult, expected: str) -> int | None:
    for index, ctx in enumerate(result.contexts, 1):
        if ctx.citation.biz_category == expected:
            return index
    return None


def _mrr(ranks: list[int | None]) -> float:
    total = 0.0
    for rank in ranks:
        if rank:
            total += 1.0 / rank
    return total / len(ranks) if ranks else 0.0


def evaluate(
    suite: list[EvalQuery],
    config: EvalConfig,
    *,
    kb_id: str,
    agent_role: str,
    retriever: Retriever,
) -> EvalReport:
    ranks: list[int | None] = []
    latencies: list[float] = []
    per_category_ranks: dict[str, list[int | None]] = {}
    failures: list[dict[str, Any]] = []

    in_domain = 0
    ood_total = 0
    ood_false_hits = 0
    no_hits = 0

    for item in suite:
        expected_kb = item.expected_kb or kb_id
        request = RetrieveRequest(
            query=item.query,
            agent_role=agent_role,
            kb_ids=[expected_kb],
            chunk_types=["faq"] if config.use_chunk_type else None,
            query_instruction=None if config.use_query_instruction else "",
            top_k=config.top_k,
            rerank_top_n=config.rerank_top_n,
            min_candidates=config.min_candidates,
            enable_rerank=config.use_rerank,
            score_threshold=config.score_threshold if not config.use_rerank else None,
        )
        started = time.perf_counter()
        try:
            result = retriever.retrieve(request)
        except Exception as exc:  # 单条失败不影响整体评测
            latency = (time.perf_counter() - started) * 1000
            latencies.append(latency)
            failures.append(
                {
                    "id": item.id,
                    "query": item.query,
                    "expected": item.expected_category,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if item.type == "heldout":
                in_domain += 1
                ranks.append(None)
                per_category_ranks.setdefault(item.expected_category, []).append(None)
            continue

        latency = float(result.diagnostics.get("timings_ms", {}).get("total", 0.0))
        if latency <= 0:
            latency = (time.perf_counter() - started) * 1000
        latencies.append(latency)

        if item.type == "ood":
            ood_total += 1
            if result.contexts:
                ood_false_hits += 1
                failures.append(
                    {
                        "id": item.id,
                        "query": item.query,
                        "expected": "(ood)",
                        "got": [
                            ctx.citation.biz_category or ctx.citation.doc_name
                            for ctx in result.contexts[:3]
                        ],
                    }
                )
            continue

        in_domain += 1
        rank = _rank_of(result, item.expected_category)
        ranks.append(rank)
        per_category_ranks.setdefault(item.expected_category, []).append(rank)
        if result.no_hit:
            no_hits += 1
        if rank is None or rank > 3:
            failures.append(
                {
                    "id": item.id,
                    "query": item.query,
                    "expected": item.expected_category,
                    "got": [
                        ctx.citation.biz_category or ctx.citation.doc_name
                        for ctx in result.contexts[:3]
                    ],
                }
            )

    hit_at_1 = sum(1 for r in ranks if r == 1) / len(ranks) if ranks else 0.0
    hit_at_3 = sum(1 for r in ranks if r and r <= 3) / len(ranks) if ranks else 0.0
    per_category = {
        code: (
            sum(1 for rank in values if rank and rank <= 3) / len(values)
            if values
            else 0.0
        )
        for code, values in per_category_ranks.items()
    }
    ood_false_hit_rate = (
        ood_false_hits / ood_total if ood_total else 0.0
    )
    no_hit_rate = no_hits / in_domain if in_domain else 0.0

    gates = {
        "hit_at_3": hit_at_3 >= 0.90,
        "hit_at_1": hit_at_1 >= 0.70,
        "category_floor": all(value >= 0.80 for value in per_category.values()),
        "ood_false_hit": ood_false_hit_rate <= 0.10,
    }

    return EvalReport(
        config=config,
        total_queries=len(suite),
        in_domain=in_domain,
        hit_at_1=round(hit_at_1, 4),
        hit_at_3=round(hit_at_3, 4),
        mrr=round(_mrr(ranks), 4),
        ood_false_hit_rate=round(ood_false_hit_rate, 4),
        no_hit_rate=round(no_hit_rate, 4),
        latency_p50_ms=round(_latency_percentile(latencies, 50), 1),
        latency_p95_ms=round(_latency_percentile(latencies, 95), 1),
        per_category={k: round(v, 4) for k, v in sorted(per_category.items())},
        gates=gates,
        failures=failures[:50],
    )


def grid_configs(enable_rerank: bool = False) -> list[EvalConfig]:
    configs: list[EvalConfig] = []
    for prefix in (True, False):
        for threshold in _GRID_THRESHOLDS:
            for chunk_type in (True, False):
                configs.append(
                    EvalConfig(
                        use_query_instruction=prefix,
                        score_threshold=threshold,
                        use_chunk_type=chunk_type,
                        use_rerank=enable_rerank,
                    )
                )
    return configs


def _print_table(reports: list[EvalReport]) -> None:
    header = (
        f"{'prefix':<6} {'thr':<5} {'type':<5} {'hit@1':>6} "
        f"{'hit@3':>6} {'mrr':>6} {'ood':>6} {'no_hit':>7} {'p95ms':>7}  passed"
    )
    print(header)
    print("-" * len(header))
    for report in reports:
        cfg = report.config
        print(
            f"{'on' if cfg.use_query_instruction else 'off':<6} "
            f"{'none' if cfg.score_threshold is None else f'{cfg.score_threshold:.2f}':<5} "
            f"{'on' if cfg.use_chunk_type else 'off':<5} "
            f"{report.hit_at_1:>6.2f} {report.hit_at_3:>6.2f} {report.mrr:>6.3f} "
            f"{report.ood_false_hit_rate:>6.2f} {report.no_hit_rate:>7.2f} "
            f"{report.latency_p95_ms:>7.1f}  "
            f"{'YES' if all(report.gates.values()) else 'no'}"
        )


def _best_report(reports: list[EvalReport]) -> EvalReport:
    passed = [r for r in reports if all(r.gates.values())]
    pool = passed or reports
    return sorted(
        pool,
        key=lambda r: (r.hit_at_3, r.hit_at_1, r.mrr, -r.ood_false_hit_rate),
        reverse=True,
    )[0]


def _write_report(
    suite_path: str,
    kb_id: str,
    agent_role: str,
    reports: list[EvalReport],
    out_dir: str | Path | None,
) -> str:
    directory = Path(out_dir) if out_dir else DEFAULT_REPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"{stamp}.json"
    payload = {
        "suite": suite_path,
        "kb_id": kb_id,
        "agent_role": agent_role,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": [report.to_dict() for report in reports],
        "best": _best_report(reports).to_dict(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


# ==================== CLI ====================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="知识库检索评测")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="构建评测集")
    build.add_argument(
        "--subdivision", default=str(_DATA_DIR / "subdivision_sample_questions.csv")
    )
    build.add_argument("--intent", default=str(_DATA_DIR / "intent_rag_kb.csv"))
    build.add_argument("--out", default=str(DEFAULT_SUITE))
    build.add_argument("--per-category", type=int, default=5)
    build.add_argument("--ood", type=int, default=20)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--kb", default="kb_faq_v1")

    run = sub.add_parser("run", help="执行评测")
    run.add_argument("--suite", default=str(DEFAULT_SUITE))
    run.add_argument("--kb", default="kb_faq_v1")
    run.add_argument("--agent-role", default="intent_agent")
    run.add_argument("--grid", action="store_true", help="跑前缀/阈值/chunk_type 网格")
    run.add_argument("--rerank", action="store_true", help="启用 CrossEncoder 重排")
    run.add_argument("--threshold", type=float, default=None)
    run.add_argument("--no-prefix", action="store_true")
    run.add_argument("--no-chunk-type", action="store_true")
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--rerank-top-n", type=int, default=5)
    run.add_argument("--report-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "build":
        summary = build_suite(
            args.subdivision,
            args.intent,
            args.out,
            per_category=args.per_category,
            ood_count=args.ood,
            seed=args.seed,
            kb_id=args.kb,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    suite = load_suite(args.suite)
    retriever = Retriever(reranker=get_reranker() if args.rerank else None)
    configs = (
        grid_configs(enable_rerank=args.rerank)
        if args.grid
        else [
            EvalConfig(
                use_query_instruction=not args.no_prefix,
                score_threshold=args.threshold,
                use_chunk_type=not args.no_chunk_type,
                use_rerank=args.rerank,
                top_k=args.top_k,
                rerank_top_n=args.rerank_top_n,
            )
        ]
    )

    reports: list[EvalReport] = []
    for index, config in enumerate(configs, 1):
        log.info(f"评测 {index}/{len(configs)}：{config.label()}")
        reports.append(
            evaluate(
                suite,
                config,
                kb_id=args.kb,
                agent_role=args.agent_role,
                retriever=retriever,
            )
        )

    _print_table(reports)
    best = _best_report(reports)
    print(f"\n最佳配置：{best.config.label()} | passed={all(best.gates.values())}")
    print(json.dumps({k: v for k, v in best.to_dict().items() if k != 'failures'},
                     ensure_ascii=False, indent=2))

    report_path = _write_report(
        args.suite, args.kb, args.agent_role, reports, args.report_dir
    )
    print(f"\n报告：{report_path}")
    return 0 if all(best.gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
