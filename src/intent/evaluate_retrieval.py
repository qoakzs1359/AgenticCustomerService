"""
RAG 检索效果批量评估 —— 用 knowledge_base.csv 自带的标签当测试集

评估指标（全部基于「向量检索层」，和 LLM 判定置信度无关）：
  1. Hit@K         正确 L2 标签是否出现在检索 top K 候选里（K=1/3/5）
  2. MRR           正确 L2 标签排名倒数均值（没找到算 0）
  3. 平均 top_score  所有 query 的最佳向量 score 均值（检索置信度核心指标）
  4. 覆盖率         有检索命中的 query 占比（no_hit=False）
  5. 平均候选数     每次检索返回的 context 数量均值

用法：
    python -m src.intent.evaluate_retrieval [--limit 50]  [query数量上限，默认全量190条]
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

from src.intent.evidence import get_evidence_builder

# 知识库 CSV 路径（相对项目根目录）
CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base.csv"


def load_test_set(limit: int | None = None) -> list[tuple[str, str, str]]:
    """加载测试集：(query, correct_l2, l1_name)"""
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row.get("问题", "").strip()
            l2 = row.get("二级类别编码", "").strip().upper()
            l1 = row.get("一级类别", "").strip()
            if query and l2:
                rows.append((query, l2, l1))
            if limit and len(rows) >= limit:
                break
    return rows


def evaluate(test_set: list[tuple[str, str, str]]) -> dict:
    """跑批量检索评估，返回指标字典"""
    builder = get_evidence_builder()
    
    hit_at = {1: 0, 3: 0, 5: 0}
    mrr_sum = 0.0
    total_queries = 0
    total_top_score = 0.0
    total_candidates = 0
    no_hit_count = 0
    # 按 L2 分组统计，方便看哪些 L2 检索效果差
    per_l2_stats = defaultdict(lambda: {"total": 0, "hit@1": 0, "hit@3": 0, "miss_score": []})
    
    start = time.perf_counter()
    
    for query, correct_l2, l1_name in test_set:
        total_queries += 1
        result = builder.retrieve_raw(query)
        
        # 覆盖率
        if result.no_hit or not result.contexts:
            no_hit_count += 1
            per_l2_stats[correct_l2]["total"] += 1
            per_l2_stats[correct_l2]["miss_score"].append(0.0)
            continue
        
        # 收集检索到的 L2 标签（按排序顺序）
        retrieved_l2s = []
        for ctx in result.contexts:
            l2 = str(ctx.citation.biz_category_name or "").strip().upper()
            if l2:
                retrieved_l2s.append((l2, ctx.score))
        
        # 平均 top_score
        total_top_score += result.diagnostics.get("best_score", 0.0)
        total_candidates += len(result.contexts)
        
        # 按 L2 统计
        stats = per_l2_stats[correct_l2]
        stats["total"] += 1
        
        # Hit@K 和 MRR
        hit = False
        for rank, (retrieved_l2, score) in enumerate(retrieved_l2s, 1):
            if retrieved_l2 == correct_l2:
                # MRR：第 rank 位命中，倒数 1/rank
                mrr_sum += 1.0 / rank
                for k in (1, 3, 5):
                    if rank <= k:
                        hit_at[k] += 1
                        if k == 1:
                            stats["hit@1"] += 1
                        if k == 3:
                            stats["hit@3"] += 1
                hit = True
                break
        if not hit:
            # 没找到：MRR 贡献 0，记录最高分方便排查
            best_score = retrieved_l2s[0][1] if retrieved_l2s else 0.0
            per_l2_stats[correct_l2]["miss_score"].append(best_score)
    
    elapsed = time.perf_counter() - start
    
    # 计算整体指标
    coverage = (total_queries - no_hit_count) / total_queries if total_queries else 0.0
    metrics = {
        "total_queries": total_queries,
        "no_hit_count": no_hit_count,
        "coverage": round(coverage, 4),
        "hit@1": round(hit_at[1] / total_queries, 4) if total_queries else 0.0,
        "hit@3": round(hit_at[3] / total_queries, 4) if total_queries else 0.0,
        "hit@5": round(hit_at[5] / total_queries, 4) if total_queries else 0.0,
        "mrr": round(mrr_sum / total_queries, 4) if total_queries else 0.0,
        "avg_top_score": round(total_top_score / (total_queries - no_hit_count), 4) if (total_queries - no_hit_count) else 0.0,
        "avg_candidates": round(total_candidates / total_queries, 2),
        "elapsed_sec": round(elapsed, 2),
        "qps": round(total_queries / elapsed, 2) if elapsed > 0 else 0.0,
    }
    metrics["per_l2"] = {
        l2: {
            "total": s["total"],
            "hit@1_rate": round(s["hit@1"] / s["total"], 3) if s["total"] else 0.0,
            "hit@3_rate": round(s["hit@3"] / s["total"], 3) if s["total"] else 0.0,
            "miss_avg_top_score": round(sum(s["miss_score"]) / len(s["miss_score"]), 3) if s["miss_score"] else 0.0,
        }
        for l2, s in per_l2_stats.items()
    }
    return metrics


def print_report(metrics: dict) -> None:
    """打印格式化的评估报告"""
    print("\n" + "=" * 60)
    print("📊 RAG 检索效果批量评估报告（向量检索层）")
    print("=" * 60)
    
    m = metrics
    print(f"\n📈 整体指标")
    print(f"   测试 query 数: {m['total_queries']}")
    print(f"   无命中数:       {m['no_hit_count']}")
    print(f"   覆盖率:         {m['coverage']*100:.2f}%")
    print(f"   Hit@1:          {m['hit@1']*100:.2f}%  ← 最核心：Top1 就是正确标签")
    print(f"   Hit@3:          {m['hit@3']*100:.2f}%  ← Top3 有正确标签（LLM 可修正）")
    print(f"   Hit@5:          {m['hit@5']*100:.2f}%")
    print(f"   MRR:            {m['mrr']:.4f}          ← 正确标签排名倒数均值")
    print(f"   平均 top_score:  {m['avg_top_score']:.4f}       ← 检索置信度核心（余弦相似度）")
    print(f"   平均候选数:      {m['avg_candidates']:.2f}")
    print(f"   耗时/QPS:       {m['elapsed_sec']}s / {m['qps']} qps")
    
    print(f"\n🔍 按 L2 细分（top 10 最差）")
    per_l2 = m["per_l2"]
    sorted_l2s = sorted(per_l2.items(), key=lambda x: x[1]["hit@1_rate"])
    worst = sorted_l2s[:10]
    for l2, s in worst:
        marker = "⚠️" if s["hit@1_rate"] < 0.5 else "🟡"
        print(f"   {marker} {l2}: Hit@1={s['hit@1_rate']*100:.0f}%  Hit@3={s['hit@3_rate']*100:.0f}%  miss_avg_top={s['miss_avg_top_score']:.3f}  (共{s['total']}条)")
    
    print("\n💡 解读建议")
    print("   - 平均 top_score 越接近 1，检索置信度越高；一般 0.6+ 算可用，0.8+ 很稳")
    print("   - Hit@1 < 50% 的 L2：LLM 很难救，优先优化 embedding / 入库分块 / 提示词")
    print("   - Hit@1 低但 Hit@3 高：检索能召回但排序有问题，可加 rerank")
    print("   - 无命中多：检查知识库是否入库、chunk_type 是否匹配、embedding API 是否正常")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索效果批量评估")
    parser.add_argument("--limit", type=int, default=None, help="query 数量上限，默认全量 190 条")
    parser.add_argument("--query", type=str, default=None, help="单独评估一条 query（跳过批量）")
    args = parser.parse_args()
    
    if args.query:
        # 单 query 快速查看
        builder = get_evidence_builder()
        raw = builder.retrieve_raw(args.query)
        print(f"\n🔍 单 query 检索结果: {args.query}")
        print(f"   best_score: {raw.diagnostics.get('best_score', 'N/A')}")
        for i, ctx in enumerate(raw.contexts[:6], 1):
            l2 = ctx.citation.biz_category_name or "unknown"
            print(f"   [{i}] score={ctx.score:.4f}  L2={l2.upper()}")
        return
    
    # 批量评估
    test_set = load_test_set(args.limit)
    print(f"加载测试集: {len(test_set)} 条 query （来自 knowledge_base.csv）")
    metrics = evaluate(test_set)
    print_report(metrics)


if __name__ == "__main__":
    main()
