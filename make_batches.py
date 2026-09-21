"""生成精简批次文件，供推理模型分批作答

每题保留：qid / 领域 / 题型 / 问题 / 选项 / 最相关2条证据(各≤300字) / 选项命中标记
"""
import json
import re

BATCH_SIZE = 15  # 每批15题
out_idx = 1
batch = []

def compact(record):
    """精简单条记录"""
    qid = record["qid"]
    domain = record.get("domain", "")
    qtype = record.get("type", "")
    question = record.get("question", "")
    options = record.get("options", {})
    evidence = record.get("evidence", [])
    option_evidence = record.get("option_evidence", {})

    # 选项级命中标记（每个选项是否有直接证据命中）
    opt_hits = {}
    for opt_key, opt_ev_list in option_evidence.items():
        opt_hits[opt_key] = len(opt_ev_list)

    # 取最相关2条证据（按 match_count 降序，截断300字）
    sorted_ev = sorted(evidence, key=lambda x: -x.get("match_count", 0))[:2]
    ev_texts = []
    for ev in sorted_ev:
        text = ev.get("text", "").replace("\n", " ")[:300]
        ev_texts.append(text)

    return {
        "qid": qid,
        "domain": domain,
        "type": qtype,
        "question": question,
        "options": options,
        "evidence": ev_texts,
        "opt_hits": opt_hits,
        "option_evidence": {
            k: [e.get("text", "").replace("\n", " ")[:250]
                for e in v[:2]]
            for k, v in option_evidence.items()
        },
    }


with open("b_questions_with_evidence.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        batch.append(compact(r))
        if len(batch) >= BATCH_SIZE:
            with open(f"b_batch_{out_idx:02d}.json", "w", encoding="utf-8") as fout:
                json.dump(batch, fout, ensure_ascii=False, indent=2)
            print(f"b_batch_{out_idx:02d}.json: {len(batch)}题")
            out_idx += 1
            batch = []

if batch:
    with open(f"b_batch_{out_idx:02d}.json", "w", encoding="utf-8") as fout:
        json.dump(batch, fout, ensure_ascii=False, indent=2)
    print(f"b_batch_{out_idx:02d}.json: {len(batch)}题")
