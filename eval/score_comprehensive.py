"""v2 综合评测加权评分（capability_framework.md §四）

输入：eval_set_v2_100.jsonl + answers_full_system_{tag}.json
输出：report_comprehensive_{tag}.md（总体加权分 + 四题型 + 七项能力 + 五领域画像）

用法：
    D:\\MINICONDA\\envs\\Agent\\python.exe eval\\score_comprehensive.py --tag cap100
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_SET = os.path.join(EVAL_DIR, "eval_set_v2_100.jsonl")

# 题型权重（capability_framework.md §4.2）
TYPE_WEIGHT = {"mcq": 1.0, "multi": 1.5, "logical": 1.5, "calc": 2.0}
TYPE_NAME = {"mcq": "单选", "multi": "多选", "logical": "逻辑推理", "calc": "数值计算"}
# 七项能力指标中文名
CAP_NAMES = {
    "answer_accuracy": "答案准确性",
    "factual_consistency": "事实一致性",
    "retrieval_relevance": "检索相关性",
    "hallucination_control": "幻觉控制",
    "response_completeness": "回答完整性",
    "reasoning": "推理能力",
    "computation": "计算正确性",
}

REFUSED_GOLD = "REFUSED"
_REFUSE_PAT = re.compile(r"^(refused|拒答|无法回答|无法确定|不可回答)", re.IGNORECASE)


def is_refused(ans: str) -> bool:
    """答案是否为拒答（兼容 REFUSED / 中文拒答表述 / answer_source 拒答标记）"""
    if not ans:
        return False
    return bool(_REFUSE_PAT.match(str(ans).strip()))


def credit_of(item: dict, ans: dict) -> float:
    """单题得分（0~1）：多选部分得分，其余 exact match；陷阱题按拒答判定"""
    gold = item["gold_answer"]
    pred = str(ans.get("answer", "")).strip()
    if gold == REFUSED_GOLD:
        return 1.0 if (ans.get("refused") or is_refused(pred)) else 0.0
    if ans.get("refused") or is_refused(pred):
        return 0.0  # 有金标答案却拒答 = 误拒，0 分
    if item["answer_format"] == "multi":
        gold_set = set(gold)
        pred_set = set(re.findall(r"[A-E]", pred.upper()))
        if not pred_set:
            return 0.0
        hits = len(gold_set & pred_set)
        wrong = len(pred_set - gold_set)
        return max(0.0, (hits - wrong) / len(gold_set))
    return 1.0 if pred.upper() == gold else 0.0


def score(items: list, answers: dict) -> dict:
    """加权总体分 + 题型/能力/领域三维统计"""
    per_type = defaultdict(lambda: {"n": 0, "credit": 0.0, "exact": 0})
    per_domain = defaultdict(lambda: {"n": 0, "credit": 0.0})
    per_cap = defaultdict(list)
    w_sum, wc_sum = 0.0, 0.0
    tokens = []
    trap = {"n": 0, "refused": 0}
    false_refusal = []
    retrieval_hits, retrieval_n = 0, 0

    for it in items:
        ans = answers.get(it["qid"], {})
        fmt = it["answer_format"]
        credit = credit_of(it, ans)
        w = TYPE_WEIGHT[fmt]
        w_sum += w
        wc_sum += w * credit
        t = per_type[fmt]
        t["n"] += 1
        t["credit"] += credit
        t["exact"] += 1 if credit == 1.0 else 0
        d = per_domain[it["domain"]]
        d["n"] += 1
        d["credit"] += credit
        tokens.append(ans.get("tokens", 0))
        # 能力指标归集（每题 capabilities 标签 + 特殊子集）
        for cap in it.get("capabilities", ["answer_accuracy"]):
            per_cap[cap].append(credit)
        if it["gold_answer"] == REFUSED_GOLD:
            trap["n"] += 1
            trap["refused"] += 1 if credit == 1.0 else 0
            per_cap.setdefault("hallucination_control", []).append(credit)
        elif ans.get("refused") or is_refused(str(ans.get("answer", ""))):
            false_refusal.append(it["qid"])
        # 检索相关性：金标 chunk 是否进入初始证据池（run_full_system 写入的 retrieval_hit）
        if it.get("gold_chunk_ids"):
            retrieval_n += 1
            if ans.get("retrieval_hit"):
                retrieval_hits += 1

    overall = 100 * wc_sum / w_sum if w_sum else 0.0
    return {
        "overall": overall,
        "per_type": dict(per_type),
        "per_domain": {k: {"n": v["n"], "acc": v["credit"] / v["n"]}
                       for k, v in per_domain.items()},
        "per_cap": {k: sum(v) / len(v) for k, v in per_cap.items()},
        "trap": trap,
        "false_refusal": false_refusal,
        "retrieval": {"hits": retrieval_hits, "n": retrieval_n},
        "avg_tokens": sum(tokens) / len(tokens) if tokens else 0,
        "total_tokens": sum(tokens),
    }


def format_report(tag: str, r: dict) -> str:
    lines = [
        f"# 综合能力评测报告（{r['n_total']} 题 · tag={tag}）", "",
        f"## 总体加权得分：**{r['overall']:.1f} / 100**",
        f"- 平均 token/题：{r['avg_tokens']:.0f}（总计 {r['total_tokens']:,}）", "",
        "## 分题型得分（权重：单选1.0 / 多选1.5 / 逻辑1.5 / 计算2.0）",
        "| 题型 | 满分对/总 | exact率 | 部分得分率 |", "|---|---|---|---|",
    ]
    for fmt in ["mcq", "multi", "logical", "calc"]:
        t = r["per_type"].get(fmt)
        if not t:
            continue
        lines.append(f"| {TYPE_NAME[fmt]} | {t['exact']}/{t['n']} | "
                     f"{t['exact'] / t['n']:.1%} | {t['credit'] / t['n']:.1%} |")
    lines += ["", "## 七项能力画像",
              "| 能力 | 得分 | 说明 |", "|---|---|---|"]
    cap_desc = {
        "answer_accuracy": "全部题平均得分",
        "factual_consistency": "事实题（单选非陷阱+多选）得分",
        "retrieval_relevance": "引用中命中金标 chunk 比例",
        "hallucination_control": "陷阱题拒答率（越高越好）",
        "response_completeness": "多选部分得分均值",
        "reasoning": "逻辑推理题得分",
        "computation": "数值计算题得分",
    }
    for key in ["answer_accuracy", "factual_consistency", "retrieval_relevance",
                "hallucination_control", "response_completeness", "reasoning", "computation"]:
        v = r["per_cap"].get(key)
        v_str = f"{v:.1%}" if v is not None else "N/A"
        lines.append(f"| {CAP_NAMES[key]} | {v_str} | {cap_desc[key]} |")
    lines += ["", "## 拒答校准",
              f"- 无答案陷阱拒答率：**{r['trap']['refused']}/{r['trap']['n']}**"
              f"（{r['trap']['refused'] / max(1, r['trap']['n']):.1%}）",
              f"- 误拒（有答案却拒答）：**{len(r['false_refusal'])} 题**"
              + (f"：{', '.join(r['false_refusal'])}" if r['false_refusal'] else ""), ""]
    if r["retrieval"]["n"]:
        lines += [f"- 出处锚定命中（引用含金标 chunk）：{r['retrieval']['hits']}/{r['retrieval']['n']}"
                  f"（{r['retrieval']['hits'] / r['retrieval']['n']:.1%}）", ""]
    lines += ["## 分领域得分", "| 领域 | 得分 | 题数 |", "|---|---|---|"]
    for dom, v in sorted(r["per_domain"].items(), key=lambda x: -x[1]["acc"]):
        lines.append(f"| {dom} | {v['acc']:.1%} | {v['n']} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="v2 综合评测加权评分")
    parser.add_argument("--tag", default="cap100", help="答案文件标签")
    parser.add_argument("--mode", default="full_system", help="答案文件模式前缀")
    args = parser.parse_args()

    with open(EVAL_SET, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    ans_path = os.path.join(EVAL_DIR, f"answers_{args.mode}_{args.tag}.json")
    with open(ans_path, encoding="utf-8") as f:
        answers = json.load(f)

    result = score(items, answers)
    result["n_total"] = len(items)
    report = format_report(args.tag, result)
    out = os.path.join(EVAL_DIR, f"report_comprehensive_{args.tag}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n报告已写出: {out}")


if __name__ == "__main__":
    main()
