"""提取 B 榜题目概要（qid/类型/问题/选项），便于整体规划推理"""
import json

summary = []
with open("b_questions_with_evidence.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        opts = r.get("options", {})
        opt_str = ""
        if opts:
            opt_str = " | ".join(f"{k}:{v[:30]}" for k, v in sorted(opts.items()))
        summary.append({
            "qid": r["qid"],
            "domain": r.get("domain", ""),
            "type": r.get("type", ""),
            "question": r["question"][:80],
            "options": opt_str,
        })

# 按领域+类型分组打印
from collections import defaultdict
by_domain = defaultdict(list)
for s in summary:
    by_domain[s["domain"]].append(s)

print(f"总题数: {len(summary)}\n")
for domain, items in by_domain.items():
    print(f"=== {domain} ({len(items)}题) ===")
    for s in items:
        print(f"  {s['qid']} [{s['type']}] {s['question']}")
        if s["options"]:
            print(f"      选项: {s['options']}")
    print()

# 统计题型
type_counts = defaultdict(int)
for s in summary:
    type_counts[s["type"]] += 1
print("题型分布:", dict(type_counts))
