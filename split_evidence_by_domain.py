"""
按领域切分 50 题证据，输出 5 个独立文件供 agent 推理
"""
import os
import json

INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_test", "evidence_all.json")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_test", "by_domain")

DOMAINS = ["financial_contracts", "financial_reports", "insurance", "regulatory", "research"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE, encoding='utf-8') as f:
        all_cases = json.load(f)

    for domain in DOMAINS:
        cases = [c for c in all_cases if c['domain'] == domain]
        out_path = os.path.join(OUTPUT_DIR, f"{domain}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)
        print(f"  {domain}: {len(cases)} 题 -> {out_path}")


if __name__ == '__main__':
    main()
