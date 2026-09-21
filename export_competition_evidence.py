# -*- coding: utf-8 -*-
"""
为100道比赛题导出完整证据（BM25 检索 top-20）到 JSON 文件
使用当前窗口的 Qwen3.7 模型完成推理
"""
import os
import sys
import json
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domain_retrieval import DomainRetrievalSystem

# 比赛题目录
QUESTIONS_DIR = r"D:\PROJECT\tianci\public_dataset_a\public_dataset_upload\questions\group_a"

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "competition_evidence")


def load_competition_questions():
    """加载100道比赛题"""
    all_questions = []
    for fname in sorted(os.listdir(QUESTIONS_DIR)):
        if fname.endswith("_questions.json"):
            with open(os.path.join(QUESTIONS_DIR, fname), encoding='utf-8') as f:
                data = json.load(f)
            for q in data:
                q['_source'] = 'competition'
                all_questions.append(q)
    return all_questions


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("加载索引...")
    domain_systems = {}
    for domain in ["financial_contracts", "financial_reports", "insurance", "regulatory", "research"]:
        print(f"  构建 {domain} 索引...")
        sys_obj = DomainRetrievalSystem(domain)
        sys_obj.build_from_json_dir()
        domain_systems[domain] = sys_obj

    print("\n加载比赛题...")
    questions = load_competition_questions()
    print(f"共 {len(questions)} 道比赛题")

    # 按领域分组
    by_domain = defaultdict(list)
    for q in questions:
        by_domain[q["domain"]].append(q)

    # 为每个领域导出证据
    for domain, qs in by_domain.items():
        print(f"\n{'=' * 60}")
        print(f"导出 {domain} 领域 {len(qs)} 题的证据...")

        system = domain_systems[domain]
        evidence_cases = []

        for i, q in enumerate(qs):
            qid = q["qid"]
            try:
                results = system.search(
                    q["question"], q["options"],
                    doc_ids=q.get("doc_ids"),
                    top_k=20, debug=True,
                )
                evidence = [
                    {
                        "rank": idx + 1,
                        "chunk_id": r["chunk_id"],
                        "doc_id": r["doc_id"],
                        "score": round(r["score"], 2),
                        "score_breakdown": r.get("score_breakdown", {}),
                        "text": r["text"][:1200],
                    }
                    for idx, r in enumerate(results)
                ]
            except Exception as e:
                print(f"  [警告] {qid} 检索失败: {e}")
                evidence = []

            case = {
                "qid": qid,
                "domain": domain,
                "source": "competition",
                "question": q["question"],
                "options": q["options"],
                "answer_format": q["answer_format"],
                "doc_ids": q.get("doc_ids", []),
                "evidence_count": len(evidence),
                "evidence": evidence,
            }
            evidence_cases.append(case)
            print(f"  [{i+1}/{len(qs)}] {qid} 召回 {len(evidence)} chunks")

        # 保存
        out_path = os.path.join(OUTPUT_DIR, f"{domain}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(evidence_cases, f, ensure_ascii=False, indent=2)
        print(f"  保存: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")

    print("\n" + "=" * 60)
    print("证据导出完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    for domain in by_domain.keys():
        path = os.path.join(OUTPUT_DIR, f"{domain}.json")
        print(f"  {domain}: {os.path.getsize(path)/1024:.1f} KB")


if __name__ == "__main__":
    main()
