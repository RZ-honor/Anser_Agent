"""
评测集构建入口（改进.md §二：分层评测集）

用法：
    D:\\MINICONDA\\envs\\Agent\\python.exe eval\\build_eval_set.py --max 30
    D:\\MINICONDA\\envs\\Agent\\python.exe eval\\build_eval_set.py --max 10 --domain insurance

输出：eval/eval_set_v1.jsonl，每行一道题（含金标 chunk/文档/锚点，供五层指标计算）
"""
import argparse
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.question_gen import (  # noqa: E402
    REFUSED, build_doc_full_texts, build_pairs_cache, gen_multi_hop,
    gen_single_hop, gen_table_numeric, gen_temporal, gen_tf,
    gen_unanswerable, load_domain_systems, quality_check,
)

# 分层配比（改进.md §二建议）：单跳30% / 表格数值25% / 跨文档20% / 时效10% / 无答案陷阱15%
LAYER_RATIO = {
    "single_hop": 0.30,
    "table_numeric": 0.25,
    "multi_hop": 0.20,
    "temporal": 0.10,
    "unanswerable": 0.15,
}
# 各层生成时的超额倍数（质检淘汰后补足目标量）
OVERSAMPLE = 3
DOMAINS = ["financial_contracts", "financial_reports", "insurance",
           "regulatory", "research"]


def build_dataset(max_total: int, seed: int, target_domain: str | None) -> list[dict]:
    """按分层配比生成题目，质检通过后均衡采样到目标数量"""
    rng = random.Random(seed)
    print("加载领域索引缓存（cache/domain_indexes_v2.pkl）...")
    systems = load_domain_systems()
    for d, s in systems.items():
        print(f"  {d}: {len(s.chunk_data)} chunks")
    full_texts = build_doc_full_texts(systems)
    print(f"全文语料就绪: {len(full_texts)} 份文档")

    domains = [target_domain] if target_domain else DOMAINS
    # 预计算各领域 chunk 的指标数值对缓存（生成器共用，避免重复跑正则）
    print("预计算指标数值对缓存...")
    pairs_cache_by_domain = {d: build_pairs_cache(systems[d]) for d in domains}
    # 时效题只依赖 financial_reports 领域
    layer_funcs = {
        "single_hop": lambda d, n: gen_single_hop(systems[d], d, rng, n,
                                                  pairs_cache_by_domain[d]),
        "table_numeric": lambda d, n: gen_table_numeric(systems[d], d, rng, n,
                                                        pairs_cache_by_domain[d]),
        "multi_hop": lambda d, n: gen_multi_hop(systems, d, rng, n,
                                                pairs_cache_by_domain[d]),
        "temporal": lambda d, n: gen_temporal(systems, rng, n,
                                              pairs_cache_by_domain[d]) if d == "financial_reports" else [],
        "unanswerable": lambda d, n: gen_unanswerable(systems[d], d, rng, n, full_texts,
                                                      pairs_cache_by_domain[d]),
    }

    # 逐层生成（跨领域汇总），质检淘汰并统计
    layer_pool: dict[str, list[dict]] = {k: [] for k in LAYER_RATIO}
    fail_stats: dict[str, int] = {k: 0 for k in LAYER_RATIO}
    per_layer_target = max(1, int(max_total * OVERSAMPLE))
    for layer, fn in layer_funcs.items():
        need = per_layer_target if layer != "temporal" else max(4, per_layer_target // len(domains))
        got = 0
        for d in domains:
            items = fn(d, max(1, need // len(domains) + 2))
            for it in items:
                ok, reason = quality_check(it, systems, full_texts)
                if ok:
                    layer_pool[layer].append(it)
                    got += 1
                else:
                    fail_stats[layer] += 1
            if got >= need:
                break
        print(f"[{layer}] 质检通过 {len(layer_pool[layer])} 题，淘汰 {fail_stats[layer]} 题")

    # 按配比采样到 max_total：各层目标数 = max_total * 比例，不足则全取
    final: list[dict] = []
    for layer, ratio in LAYER_RATIO.items():
        target = int(max_total * ratio)
        pool = layer_pool[layer]
        rng.shuffle(pool)
        final.extend(pool[:target])
        print(f"[{layer}] 目标 {target}，实际取 {min(target, len(pool))}")

    # 统一编号并打乱顺序（避免按层聚集）
    rng.shuffle(final)
    for i, it in enumerate(final):
        it["qid"] = f"eval_{i + 1:04d}"
    return final


def print_layer_stats(items: list[dict]):
    """打印分层/领域/题型统计，供人工核验构建质量"""
    from collections import Counter
    print("\n===== 评测集统计 =====")
    print("分层:", dict(Counter(i["layer"] for i in items)))
    print("领域:", dict(Counter(i["domain"] for i in items)))
    print("题型:", dict(Counter(i["answer_format"] for i in items)))
    print("难度:", dict(Counter(i["difficulty"] for i in items)))
    gold_dist = Counter(i["gold_answer"] for i in items)
    print("金标分布:", dict(gold_dist))


def main():
    parser = argparse.ArgumentParser(description="自动构建分层评测集")
    parser.add_argument("--max", type=int, default=300, help="目标题目总数")
    parser.add_argument("--domain", type=str, default=None, help="限定领域（默认全部5领域）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()

    items = build_dataset(args.max, args.seed, args.domain)
    print_layer_stats(items)

    out = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      f"eval_set_v1{'_' + args.domain if args.domain else ''}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\n已写出 {len(items)} 题 -> {out}")


if __name__ == "__main__":
    main()
