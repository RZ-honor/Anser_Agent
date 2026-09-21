# -*- coding: utf-8 -*-
"""将 qwen_answers 格式归一化为 {qid: answer} 字典格式"""
import os
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
A_DIR = os.path.join(BASE_DIR, "qwen_answers")

DOMAINS = ["financial_contracts", "financial_reports", "insurance", "regulatory", "research"]

for domain in DOMAINS:
    path = os.path.join(A_DIR, f"answers_{domain}.json")
    if not os.path.isfile(path):
        print(f"[跳过] {path} 不存在")
        continue

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    normalized = {}
    for qid, val in data.items():
        if isinstance(val, str):
            normalized[qid] = val
        elif isinstance(val, dict) and "answer" in val:
            normalized[qid] = val["answer"]
        else:
            print(f"[警告] {qid} 未知格式: {type(val)}")

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    print(f"{domain}: 归一化为 {len(normalized)} 题 -> {path}")
