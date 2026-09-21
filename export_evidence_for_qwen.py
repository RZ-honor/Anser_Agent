# -*- coding: utf-8 -*-
"""
导出 100 道题的完整证据（BM25 检索 top-20）到 JSON 文件

目的：
- 为 Task agent 提供完整证据，替代已耗尽配额的 ModelScope API
- 使用当前窗口的 Qwen3.7 模型完成推理
"""
import os
import sys
import json
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domain_retrieval import DomainRetrievalSystem

# 题集目录
DEBUG_Q_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_questions")
TEST_V2_Q_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_questions", "test_v2")

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_evidence")


def load_all_questions():
    """加载调试题集 + test_v2 题集（共 100 题）"""
    all_questions = []
    for q_dir, label in [(DEBUG_Q_DIR, "debug"), (TEST_V2_Q_DIR, "test_v2")]:
        if not os.path.isdir(q_dir):
            continue
        for f in sorted(os.listdir(q_dir)):
            if f.endswith('.json'):
                with open(os.path.join(q_dir, f), encoding='utf-8') as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    for q in data:
                        q['_source'] = label
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

    print("\n加载题集...")
    questions = load_all_questions()
    print(f"共 {len(questions)} 题")

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
                        "text": r["text"][:1000],
                    }
                    for idx, r in enumerate(results)
                ]
            except Exception as e:
                print(f"  [警告] {qid} 检索失败: {e}")
                evidence = []
            
            case = {
                "qid": qid,
                "domain": domain,
                "source": q.get("_source", "unknown"),
                "question": q["question"],
                "options": q["options"],
                "answer_format": q["answer_format"],
                "correct_answer": q.get("answer", ""),
                "difficulty": q.get("difficulty", ""),
                "test_point": q.get("test_point", ""),
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
