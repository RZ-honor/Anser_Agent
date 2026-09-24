"""v2 综合评测集构建入口（capability_framework.md §三：100 题四题型）

用法：
    D:\\MINICONDA\\envs\\Agent\\python.exe eval\\build_eval_set_v2.py

输出：eval/eval_set_v2_100.jsonl，100 题 = 单选40（含陷阱6） + 多选20 + 逻辑推理20 + 数值计算20
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.question_gen import (  # noqa: E402
    build_doc_full_texts, build_pairs_cache, gen_single_hop, gen_table_numeric,
    gen_unanswerable, load_domain_systems,
)
from eval.question_gen_v2 import (  # noqa: E402
    gen_calc, gen_logical_compare, gen_logical_rule, gen_multi_select,
    quality_check_v2,
)

DOMAINS = ["financial_contracts", "financial_reports", "insurance",
           "regulatory", "research"]
# 规则推理题优先领域（条款类），比较推理题优先领域（数据类）
RULE_DOMAINS = ["insurance", "regulatory", "financial_contracts"]
COMPARE_DOMAINS = ["financial_reports", "research"]
OVERSAMPLE = 3

# 单选题指标挖空噪声过滤：回退指标片段以下列词结尾或含聚合词时，问题语义不通顺
_MCQ_METRIC_NOISE = ("主要包括", "以及其他", "其中", "之和", "合计")
_MCQ_METRIC_NOISE_TAIL = ("约", "包括", "超过", "近", "达", "计", "占", "的")


def _mcq_clean(it: dict) -> bool:
    """单选题指标名干净性过滤（探测生成阶段的片段回退噪声）"""
    import re
    m = re.search(r"根据文档(?:表格)?，(.+?)(?:的数值为|为多少)", it["question"])
    if not m:
        return True
    met = m.group(1)
    met = met.split("的", 1)[-1] if "的" in met[:6] else met  # 去掉实体前缀再判
    if any(w in met for w in _MCQ_METRIC_NOISE):
        return False
    return not met.endswith(tuple(_MCQ_METRIC_NOISE_TAIL))


def _sample(pool: list, n: int, rng: random.Random) -> list:
    rng.shuffle(pool)
    return pool[:n]


def build_dataset(total: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    print("加载领域索引缓存（cache/domain_indexes_v2.pkl）...")
    systems = load_domain_systems()
    for d, s in systems.items():
        print(f"  {d}: {len(s.chunk_data)} chunks")
    full_texts = build_doc_full_texts(systems)
    print(f"全文语料就绪: {len(full_texts)} 份文档")
    print("预计算指标数值对缓存...")
    pairs = {d: build_pairs_cache(systems[d]) for d in DOMAINS}

    # ---- 单选 40：17 单跳 + 17 表格 + 6 陷阱（各领域轮转，保证覆盖均衡）----
    per_domain = len(DOMAINS)
    pool_single_hop, pool_table, pool_trap = [], [], []
    for d in DOMAINS:
        pool_single_hop += [it for it in gen_single_hop(systems[d], d, rng, 40, pairs[d])
                            if quality_check_v2(it, systems, full_texts)[0] and _mcq_clean(it)]
        pool_table += [it for it in gen_table_numeric(systems[d], d, rng, 40, pairs[d])
                       if quality_check_v2(it, systems, full_texts)[0] and _mcq_clean(it)]
        pool_trap += [it for it in gen_unanswerable(systems[d], d, rng, 3, full_texts, pairs[d])
                      if quality_check_v2(it, systems, full_texts)[0]]
    mcq = (_sample(pool_single_hop, 17, rng) + _sample(pool_table, 17, rng)
           + _sample(pool_trap, 6, rng))

    # ---- 多选 20：各领域轮转 ----
    pool_multi = []
    for d in DOMAINS:
        pool_multi += [it for it in gen_multi_select(systems[d], d, rng, 15, pairs[d], full_texts)
                       if quality_check_v2(it, systems, full_texts)[0]]
    multi = _sample(pool_multi, 20, rng)

    # ---- 逻辑推理 20：10 规则（条款域）+ 10 比较（数据域）----
    pool_rule, pool_compare = [], []
    for d in RULE_DOMAINS:
        pool_rule += [it for it in gen_logical_rule(systems[d], d, rng, 10, pairs[d])
                      if quality_check_v2(it, systems, full_texts)[0]]
    for d in COMPARE_DOMAINS:
        pool_compare += [it for it in gen_logical_compare(systems[d], d, rng, 15, pairs[d])
                         if quality_check_v2(it, systems, full_texts)[0]]
    logical = _sample(pool_rule, 10, rng) + _sample(pool_compare, 10, rng)

    # ---- 数值计算 20：数据域优先，条款域兜底 ----
    pool_calc = []
    for d in COMPARE_DOMAINS + DOMAINS:
        pool_calc += [it for it in gen_calc(systems[d], d, rng, 15, pairs[d])
                      if quality_check_v2(it, systems, full_texts)[0]]
    calc = _sample(pool_calc, 20, rng)

    final = mcq + multi + logical + calc
    rng.shuffle(final)
    for i, it in enumerate(final):
        it["qid"] = f"cap_{i + 1:04d}"
        it.setdefault("capabilities", ["answer_accuracy", "factual_consistency",
                                       "retrieval_relevance"])
    return final


def print_stats(items: list[dict]):
    print("\n===== 评测集统计 =====")
    print("题型:", dict(Counter(i["answer_format"] for i in items)))
    print("领域:", dict(Counter(i["domain"] for i in items)))
    print("难度:", dict(Counter(i["difficulty"] for i in items)))
    gold_dist = Counter(i["gold_answer"] for i in items)
    print("金标分布(前10):", dict(gold_dist.most_common(10)))


def main():
    parser = argparse.ArgumentParser(description="构建 v2 综合 100 题评测集")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    items = build_dataset(100, args.seed)
    print_stats(items)
    out = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "eval_set_v2_100.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\n已写出 {len(items)} 题 -> {out}")


if __name__ == "__main__":
    main()
