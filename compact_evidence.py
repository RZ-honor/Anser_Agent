# -*- coding: utf-8 -*-
"""精简证据文件，每题只保留top-5证据，方便模型推理"""
import os
import json

EVIDENCE_DIR = "competition_evidence"
OUTPUT_DIR = "competition_evidence_compact"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(EVIDENCE_DIR):
    if not fname.endswith('.json'):
        continue
    with open(os.path.join(EVIDENCE_DIR, fname), encoding='utf-8') as f:
        cases = json.load(f)

    compact_cases = []
    for case in cases:
        # 只保留top-3证据，且文本截断到500字
        top3 = case["evidence"][:3]
        for e in top3:
            e["text"] = e["text"][:500]
        compact_case = {
            "qid": case["qid"],
            "question": case["question"],
            "options": case["options"],
            "answer_format": case["answer_format"],
            "doc_ids": case["doc_ids"],
            "evidence_top3": [
                {"rank": e["rank"], "chunk_id": e["chunk_id"], "doc_id": e["doc_id"], "text": e["text"]}
                for e in top3
            ],
        }
        compact_cases.append(compact_case)

    out_path = os.path.join(OUTPUT_DIR, fname)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(compact_cases, f, ensure_ascii=False, indent=2)
    print(f"{fname}: {os.path.getsize(out_path)/1024:.1f} KB")

print("精简完成")
