# -*- coding: utf-8 -*-
"""按领域切分 test_v2 证据文件，便于 agent 并行推理"""
import os
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_test_v2", "evidence_all.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_test_v2", "by_domain")

DOMAINS = ["financial_contracts", "financial_reports", "insurance", "regulatory", "research"]

os.makedirs(OUT_DIR, exist_ok=True)

with open(INPUT, encoding='utf-8') as f:
    cases = json.load(f)

by_domain = {d: [] for d in DOMAINS}
for c in cases:
    d = c.get('domain')
    if d in by_domain:
        by_domain[d].append(c)

for d, items in by_domain.items():
    out_path = os.path.join(OUT_DIR, f"{d}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"{d}: {len(items)} 题 -> {out_path}")
