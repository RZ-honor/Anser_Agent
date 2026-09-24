"""
评测指标体系（改进.md §二 验证指标）

五层指标：
1. 分层准确率（single_hop / table_numeric / multi_hop / temporal / unanswerable）
2. 无答案陷阱：拒答率 + 拒答正确率（金标 REFUSED 且模型也拒答才算对）
3. 检索层：Recall@10 / MRR@10（金标锚定 chunk 级）
4. 分领域准确率
5. Token 效率（均/总 token）
"""
import json
from collections import defaultdict

# 评测集金标：无答案陷阱题的金标答案
REFUSED = "REFUSED"

# 拒答判定：模型输出含这些短语视为"拒答"（与 citation_manager 口径一致）
_REFUSAL_PHRASES = ["文档中未提及", "文档未提及", "文档中没有", "未找到相关证据",
                    "无法从文档", "证据不足"]


def extract_answer_letter(text: str, valid_options: list) -> str:
    """从模型输出提取答案字母（基线模式共用）；无字母且含拒答短语返回 REFUSED"""
    if not text:
        return ""
    upper = text.upper()
    letters = [c for c in "ABCD" if c in upper and c in valid_options]
    if letters:
        return "".join(sorted(set(letters)))
    if any(p in text for p in _REFUSAL_PHRASES):
        return REFUSED
    return ""


def is_refusal_output(text: str) -> bool:
    """模型输出是否为拒答（配合 answer_source=refused_guess 使用）"""
    return bool(text) and any(p in text for p in _REFUSAL_PHRASES)


def retrieval_metrics(rank_lists: dict, meta: dict = None, k: int = 10) -> dict:
    """检索层指标

    Args:
        rank_lists: {qid: {"ranked": [chunk_id...], "gold": [chunk_id...]}}
        meta: 可选，{qid: {"layer": str, "domain": str}}，提供时输出分层/分领域 Recall
        k: 截断深度（默认10，检索诊断时可传8与生产top-k对齐）

    Returns:
        {"recall@10": float, "mrr@10": float, "hit_detail": {qid: rank},
         "layers": {layer: {...}}, "domains": {domain: {...}}}
    """
    recall_sum, mrr_sum = 0.0, 0.0
    n = 0
    hit_detail = {}
    layer_stats = defaultdict(lambda: {"n": 0, "hit": 0, "rr_sum": 0.0})
    domain_stats = defaultdict(lambda: {"n": 0, "hit": 0, "rr_sum": 0.0})
    for qid, item in rank_lists.items():
        ranked = item["ranked"][:k]
        gold = set(item["gold"])
        n += 1
        # 记录该题所属层/领域的统计桶（meta 缺失时跳过分层统计）
        meta_info = (meta or {}).get(qid) or {}
        buckets = []
        if meta_info.get("layer"):
            buckets.append(layer_stats[meta_info["layer"]])
        if meta_info.get("domain"):
            buckets.append(domain_stats[meta_info["domain"]])
        for b in buckets:
            b["n"] += 1
        if gold & set(ranked):
            recall_sum += 1
            # 第一个命中的金标 chunk 排名（1-based）
            rank = min(i + 1 for i, cid in enumerate(ranked) if cid in gold)
            mrr_sum += 1.0 / rank
            hit_detail[qid] = rank
            for b in buckets:
                b["hit"] += 1
                b["rr_sum"] += 1.0 / rank
        else:
            hit_detail[qid] = 0
    def _agg(stats: dict) -> dict:
        return {name: {"recall": s["hit"] / s["n"] if s["n"] else 0.0,
                       "mrr": s["rr_sum"] / s["n"] if s["n"] else 0.0,
                       "n": s["n"], "hit": s["hit"]}
                for name, s in stats.items()}
    return {
        f"recall@{k}": recall_sum / n if n else 0.0,
        f"mrr@{k}": mrr_sum / n if n else 0.0,
        "hit_detail": hit_detail,
        "layers": _agg(layer_stats),
        "domains": _agg(domain_stats),
    }


def answer_accuracy(eval_items: list, answers: dict) -> dict:
    """答题准确率（分层 / 分领域 / 拒答统计）

    Args:
        eval_items: 评测集条目列表（含 qid/layer/domain/gold_answer）
        answers: {qid: {"answer": str, "refused": bool, "tokens": int}}

    Returns:
        汇总指标字典
    """
    total = len(eval_items)
    correct = 0
    layer_stats = defaultdict(lambda: {"n": 0, "correct": 0})
    domain_stats = defaultdict(lambda: {"n": 0, "correct": 0})
    # 无答案陷阱专 statistics
    trap_total = trap_refused = trap_correct = 0
    # 误拒统计：有金标答案的题模型却拒答
    false_refusal = 0
    tokens_total = 0

    for item in eval_items:
        qid = item["qid"]
        gold = item["gold_answer"]
        layer = item["layer"]
        pred = (answers.get(qid) or {}).get("answer", "")
        pred_norm = (pred or "").replace(" ", "").upper()
        gold_norm = gold.replace(" ", "").upper()
        tokens_total += (answers.get(qid) or {}).get("tokens", 0)

        layer_stats[layer]["n"] += 1
        domain_stats[item["domain"]]["n"] += 1

        if gold == REFUSED:
            trap_total += 1
            # 拒答判定：答案标记为 REFUSED 或模型声明拒答
            if pred_norm == REFUSED or (answers.get(qid) or {}).get("refused"):
                trap_refused += 1
                layer_stats[layer]["correct"] += 1
                domain_stats[item["domain"]]["correct"] += 1
                correct += 1
                trap_correct += 1
        else:
            if pred_norm == REFUSED or (answers.get(qid) or {}).get("refused"):
                false_refusal += 1
            elif pred_norm == gold_norm:
                correct += 1
                layer_stats[layer]["correct"] += 1
                domain_stats[item["domain"]]["correct"] += 1

    return {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "layers": {k: {"accuracy": v["correct"] / v["n"] if v["n"] else 0.0,
                       "n": v["n"], "correct": v["correct"]}
                   for k, v in layer_stats.items()},
        "domains": {k: {"accuracy": v["correct"] / v["n"] if v["n"] else 0.0,
                        "n": v["n"]}
                    for k, v in domain_stats.items()},
        "unanswerable": {
            "trap_total": trap_total,
            "refused_rate": trap_refused / trap_total if trap_total else 0.0,
            "trap_correct": trap_correct,
        },
        "false_refusal": false_refusal,
        "avg_tokens": tokens_total / total if total else 0,
        "total_tokens": tokens_total,
    }


def format_report(mode: str, metrics: dict, retrieval: dict = None) -> str:
    """格式化 markdown 评测报告"""
    lines = [f"# 评测报告（{mode}）", ""]
    lines.append(f"- 总题数: {metrics['total']}")
    lines.append(f"- 总体准确率: **{metrics['accuracy']:.1%}**")
    lines.append(f"- 平均 token/题: {metrics['avg_tokens']:.0f}（总计 {metrics['total_tokens']:,}）")
    lines.append("")
    lines.append("## 分层准确率")
    lines.append("| 层 | 准确率 | 对/总 |")
    lines.append("|---|---|---|")
    for layer, s in metrics["layers"].items():
        lines.append(f"| {layer} | {s['accuracy']:.1%} | {s['correct']}/{s['n']} |")
    lines.append("")
    lines.append("## 分领域准确率")
    lines.append("| 领域 | 准确率 | 题数 |")
    lines.append("|---|---|---|")
    for d, s in metrics["domains"].items():
        lines.append(f"| {d} | {s['accuracy']:.1%} | {s['n']} |")
    ua = metrics["unanswerable"]
    lines.append("")
    lines.append(f"## 无答案陷阱：拒答率 {ua['refused_rate']:.1%}（{ua['trap_correct']}/{ua['trap_total']}），"
                 f"误拒（有答案却拒答）{metrics['false_refusal']} 题")
    if retrieval:
        lines.append("")
        lines.append(f"## 检索层：Recall@10 = **{retrieval['recall@10']:.1%}**，"
                     f"MRR@10 = **{retrieval['mrr@10']:.3f}**")
    return "\n".join(lines)
