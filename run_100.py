"""
100题顺序执行 - 使用模型驱动检索（多轮工具调用 + CoT 推理）

修复点（P1+P4）：
1. 索引构建改用 DomainRetrievalSystem（领域差异化参数真正生效）
   - 原先 NewRetrievalSystem 用硬编码加分权重（2.5/1.0/1.5）
   - 现在 DomainRetrievalSystem 重写加分方法，使用领域配置值（15-25 量级）
   - doc_id 直接来自 JSON 文件名，与题目 doc_ids 完全一致（无 txt_ 前缀）
2. 答题路径从 answer_question_single_call 切换到 answer_question（多轮工具调用）
   - 模型可主动调用 search_number/search_phrase/search_keyword 等工具
   - max_rounds=3，每轮可调多个工具，更接近真实 CoT 推理
"""
import hashlib
import json
import os
import pickle
import sys
import time
import csv
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 改用 DomainRetrievalSystem：让领域专属 BM25 参数和加分权重真正生效
from domain_retrieval import DomainRetrievalSystem
from model_driven_retrieval import ModelDrivenRetriever
from config import CACHE_DIR

INDEX_CACHE_PATH = os.path.join(CACHE_DIR, "domain_indexes_v2.pkl")  # v2 标记新索引结构
QUESTIONS_DIR = "D:/PROJECT/tianci/public_dataset_a/public_dataset_upload/questions/group_a"


def _compute_index_hash():
    """计算索引内容哈希，用于判断是否需要重建"""
    preparsed_dir = "D:/PROJECT/tianci/1"
    h = hashlib.md5()
    # 哈希包含所有预解析 JSON 文件名 + 修改时间
    for f in sorted(os.listdir(preparsed_dir)):
        if f.endswith('.json'):
            h.update(f.encode())
            h.update(str(os.path.getmtime(os.path.join(preparsed_dir, f))).encode())
    # 哈希包含领域配置版本（避免配置变更后仍用旧缓存）
    h.update(b"domain_retrieval_v2_with_optimizers")
    return h.hexdigest()


def load_or_build_indexes(force_rebuild=False):
    """加载或构建领域专属索引（使用 DomainRetrievalSystem）

    修复点：
    - 用 DomainRetrievalSystem 替代 NewRetrievalSystem
    - doc_id 直接来自 JSON 文件名（无 txt_ 前缀），与题目 doc_ids 一致
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    current_hash = _compute_index_hash()

    # 优先尝试加载缓存
    if not force_rebuild and os.path.exists(INDEX_CACHE_PATH):
        try:
            with open(INDEX_CACHE_PATH, 'rb') as f:
                cache = pickle.load(f)
            if cache.get('hash') == current_hash:
                print(f"从缓存加载领域专属索引: {INDEX_CACHE_PATH}")
                domain_systems = cache['domain_systems']
                for domain, system in domain_systems.items():
                    print(f"  {domain}: {len(system.chunk_data)} chunks")
                return domain_systems
        except Exception as e:
            print(f"缓存加载失败: {e}")

    # 重建：为 5 个领域分别构建 DomainRetrievalSystem 索引
    print("构建领域专属索引（DomainRetrievalSystem）...")
    domains = ["financial_contracts", "financial_reports", "insurance",
               "regulatory", "research"]
    domain_systems = {}
    for domain in domains:
        print(f"\n{'='*60}\n构建 {domain} 领域索引")
        system = DomainRetrievalSystem(domain)
        system.build_from_json_dir()
        domain_systems[domain] = system

    # 持久化缓存（含哈希校验）
    try:
        with open(INDEX_CACHE_PATH, 'wb') as f:
            pickle.dump({'hash': current_hash, 'domain_systems': domain_systems}, f)
        print(f"\n索引已缓存到: {INDEX_CACHE_PATH}")
    except Exception as e:
        print(f"\n缓存保存失败: {e}")

    return domain_systems


def load_all_questions():
    questions = []
    for f in sorted(os.listdir(QUESTIONS_DIR)):
        if f.endswith('.json'):
            with open(os.path.join(QUESTIONS_DIR, f), encoding='utf-8') as fh:
                data = json.load(fh)
            if isinstance(data, list):
                questions.extend(data)
    return questions


def process_one(q, retriever_by_domain, domain_systems):
    """单题处理：使用多轮工具调用 + CoT 推理

    修复点：
    - 从 answer_question_single_call 切换到 answer_question
    - 模型可主动调用 3 轮工具（每轮多工具并行）
    - 初始证据 top_k 从 20 调整为 15（避免初始上下文过长挤占推理 token）
    """
    qid = q["qid"]
    domain = q["domain"]
    doc_ids = q.get("doc_ids", [])
    answer_format = q.get("answer_format", "multi")

    retriever = retriever_by_domain.get(domain)
    if not retriever:
        return {"qid": qid, "domain": domain, "answer": "A", "error": "no_retriever", "tokens": 0}

    try:
        # 获取初始证据（用 DomainRetrievalSystem 的领域专属检索）
        system = domain_systems.get(domain)
        initial_evidence = []
        if system:
            # top_k=15：平衡初始上下文与推理 token 预算
            initial_evidence = system.search(q["question"], q["options"], doc_ids=doc_ids, top_k=15)

        # 多轮工具调用 + CoT 推理（max_rounds=None：按题型从 config.MAX_ROUNDS_BY_FORMAT 取）
        result = retriever.answer_question(
            question=q["question"],
            options=q["options"],
            answer_format=answer_format,
            initial_evidence=initial_evidence,
            domain=domain,
            doc_ids=doc_ids,
        )
        return {
            "qid": qid, "domain": domain,
            "answer": result.get("answer", "A"),
            "tokens": result.get("total_tokens", 0),
            "tool_calls": len(result.get("tool_calls_log", [])),
            "reasoning": result.get("reasoning", "")[:300],
            "answer_source": result.get("answer_source", "model"),
            "citations": result.get("citations", {}),
            "refused": result.get("refused", False),
        }
    except Exception as e:
        return {"qid": qid, "domain": domain, "answer": "A", "error": str(e), "tokens": 0}


def main():
    # 多轮工具调用预计每题 4-8K tokens，100 题约需 400-800K
    # 评分公式 TokenScore = max(0, (5M - TotalTokens) / 5M)
    # 800K 对应 TokenScore=0.84，仍可保留较高分数
    TOKEN_BUDGET = 1_500_000  # 额度限制：超过此值停止 API 调用

    print("加载索引...")
    domain_systems = load_or_build_indexes()

    retriever_by_domain = {}
    for domain, system in domain_systems.items():
        retriever_by_domain[domain] = ModelDrivenRetriever(
            retrieval_system=system, mode="offline_competition"
        )

    questions = load_all_questions()
    print(f"共 {len(questions)} 题，顺序执行（多轮工具调用 + CoT 推理）")
    print(f"Token 预算: {TOKEN_BUDGET:,}\n")

    results = []
    total_tokens_used = 0
    budget_exceeded = False
    t0 = time.time()

    for i, q in enumerate(questions):
        # 检查额度限制
        if budget_exceeded:
            r = {"qid": q["qid"], "domain": q["domain"], "answer": "A", "tokens": 0, "tool_calls": 0, "reasoning": "", "error": "budget_exceeded"}
            results.append(r)
            print(f"  [{i+1}/{len(questions)}] ⏭️ {r['qid']} [{r['domain']}] → A (额度已耗尽，跳过)")
            continue

        r = process_one(q, retriever_by_domain, domain_systems)
        results.append(r)

        # 累计 token 使用量
        tokens_this_question = r.get("tokens", 0)
        total_tokens_used += tokens_this_question

        status = "✅" if not r.get("error") else "❌"
        source = r.get("answer_source", "model")[:6]
        print(f"  [{i+1}/{len(questions)}] {status} {r['qid']} [{r['domain']}] → {r['answer']} "
              f"(tok={tokens_this_question}, 工具={r.get('tool_calls', 0)}, 来源={source}, 累计={total_tokens_used:,})")

        # 检查是否超过额度限制
        if total_tokens_used >= TOKEN_BUDGET:
            budget_exceeded = True
            print(f"\n⚠️ Token 使用量已达 {total_tokens_used:,}，超过预算 {TOKEN_BUDGET:,}，后续题目将跳过 API 调用")

    elapsed = time.time() - t0
    total_tokens = sum(r.get("tokens", 0) for r in results)
    errors = sum(1 for r in results if r.get("error"))

    print(f"\n{'='*60}")
    print(f"完成！总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"总题数: {len(results)}, 错误: {errors}, tokens: {total_tokens}")

    domain_stats = defaultdict(lambda: {"count": 0, "errors": 0})
    for r in results:
        domain_stats[r["domain"]]["count"] += 1
        if r.get("error"):
            domain_stats[r["domain"]]["errors"] += 1

    print(f"\n领域统计:")
    for d, s in sorted(domain_stats.items()):
        print(f"  {d}: {s['count']}题, {s['errors']}个错误")

    os.makedirs("output", exist_ok=True)
    with open("output/run_100_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open("output/answer.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        w.writerow(["summary", "", "0", "0", str(total_tokens)])
        for r in results:
            ans = r["answer"]
            if len(ans) > 1 and ans.isalpha():
                ans = "".join(sorted(ans))
            w.writerow([r["qid"], ans, "0", "0", str(r.get("tokens", 0))])

    print(f"\n结果已保存: output/run_100_results.json")
    print(f"答案已保存: output/answer.csv")


if __name__ == "__main__":
    main()
