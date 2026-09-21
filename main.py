"""
主程序 - 端到端流程：解析 → 建图 → 检索 → 问答 → 输出
无embedding版，纯关键词/实体匹配
"""
import os
import json
import time
import csv

from config import (
    DATASET_DIR, QUESTIONS_DIR, RAW_DOCS_DIR, OUTPUT_DIR, CACHE_DIR,
    DASHSCOPE_API_KEY, DOMAINS, QA_AUDIT_MODE,
)
from graph_builder import build_graph, save_graph, load_graph
from graph_retriever import graph_retrieve
from qa_engine import QAEngine

QA_CACHE_PATH = os.path.join(CACHE_DIR, "qa_cache.json")
RETRIEVAL_CACHE_PATH = os.path.join(CACHE_DIR, "retrieval_cache.json")


def _load_json_cache(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_json_cache(cache: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def load_questions(questions_dir: str) -> list[dict]:
    """加载所有问题"""
    questions = []
    for fname in sorted(os.listdir(questions_dir)):
        if not fname.endswith("_questions.json"):
            continue
        fpath = os.path.join(questions_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        questions.extend(data)
        print(f"  加载 {fname}: {len(data)} 题")
    return questions


def _parse_and_chunk(raw_dir: str, parser_type: str, max_pages: int) -> dict:
    """统一的文档解析 + 分块入口"""
    if parser_type == "model":
        print("2. [MinerU 2.5 Pro VLM] vLLM 异步批量解析文档...")
        from vlm_parser import parse_all_documents_vlm, chunk_vlm_result, shutdown_vlm
        from config import VLM_DPI
        try:
            all_docs_raw = parse_all_documents_vlm(raw_dir, dpi=VLM_DPI)
        finally:
            shutdown_vlm()
        chunk_fn = chunk_vlm_result
    elif parser_type == "scnet":
        print("2. [SCNET OCR] 云端文档解析...")
        from scnet_ocr import parse_all_documents_scnet, chunk_with_tables
        all_docs_raw = parse_all_documents_scnet(raw_dir)
        chunk_fn = chunk_with_tables
    elif parser_type == "qwen-ocr":
        print(f"2. [Qwen-OCR] 解析文档 (max_pages={max_pages or '全部'})...")
        from qwen_ocr import parse_all_documents_qwen_ocr, chunk_qwen_ocr_result
        all_docs_raw = parse_all_documents_qwen_ocr(raw_dir, max_pages_per_doc=max_pages)
        chunk_fn = chunk_qwen_ocr_result
    else:
        print(f"[错误] 未知解析器: {parser_type}，请使用 model/scnet/qwen-ocr")
        return {}

    # 统一后处理：分块
    all_docs = {}
    for domain, domain_docs in all_docs_raw.items():
        all_docs[domain] = {}
        for doc_id, doc_data in domain_docs.items():
            if "chunks" not in doc_data:
                doc_data["chunks"] = chunk_fn(doc_data)
            all_docs[domain][doc_id] = doc_data
    return all_docs


def run_pipeline(
    use_cache: bool = True,
    max_questions: int = 0,
    parser_type: str = "model",
    max_pages: int = 0,
    audit_mode: bool = None,
) -> str:
    """
    运行完整流程

    Args:
        use_cache: 是否使用缓存的图
        max_questions: 最大处理题数（0=全部）
        parser_type: PDF解析器类型
        max_pages: 每个文档最大页数
        audit_mode: 是否启用审计模式（输出详细推理和选项判断）

    Returns:
        正式模式返回 answer.csv 路径；审计模式返回 answer_audit.json 路径
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # resolve audit_mode from config default when not explicitly set
    if audit_mode is None:
        audit_mode = QA_AUDIT_MODE

    # 路径检测
    print(f"数据集目录: {DATASET_DIR} ({'存在' if os.path.isdir(DATASET_DIR) else '不存在'})")
    print(f"问题目录:   {QUESTIONS_DIR} ({'存在' if os.path.isdir(QUESTIONS_DIR) else '不存在'})")
    print(f"文档目录:   {RAW_DOCS_DIR} ({'存在' if os.path.isdir(RAW_DOCS_DIR) else '不存在'})")
    if not os.path.isdir(QUESTIONS_DIR):
        print(f"\n[错误] 问题目录不存在: {QUESTIONS_DIR}")
        print(f"请设置环境变量 DATASET_DIR 指向数据集根目录，或检查目录结构")
        print(f"  例: DATASET_DIR=/path/to/dataset python main.py --parser model")
        return ""
    if not os.path.isdir(RAW_DOCS_DIR):
        print(f"\n[错误] 文档目录不存在: {RAW_DOCS_DIR}")
        return ""

    start_time = time.time()

    # ====== 1. 加载问题 ======
    print("=" * 50)
    print("1. 加载问题...")
    questions = load_questions(QUESTIONS_DIR)
    if max_questions > 0:
        questions = questions[:max_questions]
    print(f"   共 {len(questions)} 题")

    # ====== 2. 解析文档 & 构建图 ======
    print("=" * 50)
    graph_path = os.path.join(CACHE_DIR, "graph.json")

    if use_cache and os.path.exists(graph_path):
        print("2. 加载缓存图...")
        G = load_graph(graph_path)
    else:
        all_docs = _parse_and_chunk(RAW_DOCS_DIR, parser_type, max_pages)
        if not all_docs:
            return ""

        print("\n构建图...")
        G = build_graph(all_docs)
        save_graph(G, graph_path)

    # ====== 3. 初始化问答引擎 ======
    print("=" * 50)
    print("3. 初始化问答引擎...")
    qa_engine = QAEngine()

    # ====== 4. 逐题处理 ======
    print("=" * 50)
    print("4. 开始答题...")
    answers = []
    answer_trace = []

    # 加载 QA / 检索缓存
    qa_cache = _load_json_cache(QA_CACHE_PATH)
    retrieval_cache = _load_json_cache(RETRIEVAL_CACHE_PATH)
    qa_cache_hits = 0
    retrieval_cache_hits = 0

    for i, q in enumerate(questions):
        qid = q["qid"]
        domain = q["domain"]
        question = q["question"]
        options = q["options"]
        answer_format = q["answer_format"]
        doc_ids = q.get("doc_ids")

        print(f"\n[{i+1}/{len(questions)}] {qid} ({domain})")

        # 4a. 检索结果缓存
        if qid in retrieval_cache:
            evidence = retrieval_cache[qid]
            retrieval_cache_hits += 1
            print(f"   [缓存命中] 检索: {len(evidence)} 条证据")
        else:
            evidence = graph_retrieve(
                question, options, G,
                doc_ids=doc_ids,
                qid=qid,
            )
            retrieval_cache[qid] = evidence
            _save_json_cache(retrieval_cache, RETRIEVAL_CACHE_PATH)
            print(f"   检索到 {len(evidence)} 条证据")
        for ev in evidence[:2]:
            print(f"   - [{ev['score']:.2f}] {ev['doc_id']}: {ev['text'][:60]}...")

        # 4b. QA 结果缓存
        if qid in qa_cache and not audit_mode:
            cached = qa_cache[qid]
            result = cached
            qa_cache_hits += 1
            print(f"   [缓存命中] 答案: {result['answer']}  Token: {result['total_tokens']}")
        else:
            result = qa_engine.answer_question(
                question, options, answer_format, evidence, doc_ids,
                domain=domain, audit_mode=audit_mode,
            )
            qa_cache[qid] = result
            _save_json_cache(qa_cache, QA_CACHE_PATH)
            print(f"   答案: {result['answer']}  Token: {result['total_tokens']}")

        answers.append({
            "qid": qid,
            "answer": result["answer"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        })

        # 收集证据追溯信息
        trace_entry = {
            "qid": qid,
            "answer": result["answer"],
            "domain": domain,
            "answer_format": answer_format,
            "reasoning": result.get("reasoning", ""),
            "evidence_retrieval": [
                {
                    "chunk_id": e.get("chunk_id", ""),
                    "doc_id": e.get("doc_id", ""),
                    "page": e.get("page", 0),
                    "score": round(e.get("score", 0), 2),
                    "quoted_clause": e.get("text", "")[:300],
                }
                for e in evidence
            ],
        }
        if "option_judgements" in result:
            trace_entry["option_judgements"] = result["option_judgements"]
        answer_trace.append(trace_entry)

    # ====== 5. 输出结果 ======
    stats = qa_engine.get_total_stats()

    if audit_mode:
        # 审计模式：输出 JSON（含推理过程、选项判断、证据追溯）
        print("=" * 50)
        print("5. [审计模式] 生成 answer_audit.json...")

        audit_path = os.path.join(OUTPUT_DIR, "answer_audit.json")
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(answer_trace, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        print(f"\n完成！耗时: {elapsed:.1f}s")
        print(f"审计输出: {audit_path}")
        print(f"总 Token: {stats['total_tokens']}")
        print(f"缓存命中: 检索 {retrieval_cache_hits}/{len(questions)}, QA {qa_cache_hits}/{len(questions)}")
        return audit_path
    else:
        # 正式模式：只输出 CSV
        print("=" * 50)
        print("5. 生成 answer.csv...")

        csv_path = os.path.join(OUTPUT_DIR, "answer.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])

            writer.writerow(["summary", "", stats["prompt_tokens"],
                             stats["completion_tokens"], stats["total_tokens"]])

            for ans in answers:
                writer.writerow([
                    ans["qid"], ans["answer"],
                    ans["prompt_tokens"], ans["completion_tokens"], ans["total_tokens"],
                ])

        elapsed = time.time() - start_time
        print(f"\n完成！耗时: {elapsed:.1f}s")
        print(f"输出: {csv_path}")
        print(f"总 Token: {stats['total_tokens']}")
        print(f"缓存命中: 检索 {retrieval_cache_hits}/{len(questions)}, QA {qa_cache_hits}/{len(questions)}")
        return csv_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AFAC2025 金融长文档问答 Agent")
    parser.add_argument("--no-cache", action="store_true", help="不使用缓存")
    parser.add_argument("--parser", choices=["model", "scnet", "qwen-ocr"], default="model",
                        help="PDF解析器：model(默认,MinerU 2.5 Pro), scnet, qwen-ocr")
    parser.add_argument("--max-questions", type=int, default=0, help="最大处理题数")
    parser.add_argument("--max-pages", type=int, default=0, help="每个文档最大页数(0=全部)")
    parser.add_argument("--audit", action="store_true", help="启用审计模式（输出详细推理和选项判断）")
    args = parser.parse_args()

    run_pipeline(
        use_cache=not args.no_cache,
        max_questions=args.max_questions,
        parser_type=args.parser,
        max_pages=args.max_pages,
        audit_mode=args.audit,
    )
