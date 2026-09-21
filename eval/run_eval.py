"""
评测运行器（改进.md §二 四基线对比）

四种基线模式（证明"符号检索+工具循环优于朴素向量方案"的叙事）：
1. bare_llm          LLM 裸问：无证据直接回答
2. naive_vector      朴素向量 RAG：Chroma 语义 top-8 + 单轮 LLM
3. symbolic_single_turn  符号检索单轮：BM25+数值+短语 top-8 + 单轮 LLM
4. full_system       完整系统：BM25 初始证据 + 多轮工具调用 + 引用校验

另含 retrieval 模式：纯检索评测（Recall@10 / MRR@10），不需要问答 API。

用法：
  python eval/run_eval.py --mode retrieval
  python eval/run_eval.py --mode full_system --max 50
"""
import os
import sys
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from config import DASHSCOPE_API_KEY, QWEN_MODEL, QWEN_BASE_URL, QWEN_EXTRA_BODY
from eval.metrics import (retrieval_metrics, answer_accuracy, format_report,
                          extract_answer_letter)

EVAL_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v1.jsonl")
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
VALID_OPTIONS = ["A", "B", "C", "D"]


def load_eval_set(max_items: int = None) -> list:
    items = []
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items[:max_items] if max_items else items


def _chat(messages: list, max_tokens: int = 800, tools=None, tool_choice=None):
    """轻量 LLM 调用（基线模式共用；经 llm_api 适配层支持 Responses API）"""
    from openai import OpenAI
    from llm_api import call_llm
    from config import USE_RESPONSES_API
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=QWEN_BASE_URL)
    return call_llm(client, QWEN_MODEL, messages, tools=tools, tool_choice=tool_choice,
                    temperature=0.1, max_tokens=max_tokens,
                    extra_body=QWEN_EXTRA_BODY, use_responses_api=USE_RESPONSES_API)


def _opts_dict(q: dict) -> dict:
    """评测集 options 为列表（[opt1, opt2,...]），转成检索/问答接口要求的 {A:..,B:..}"""
    opts = q["options"]
    if isinstance(opts, dict):
        return opts
    return dict(zip(VALID_OPTIONS, opts))


def _build_question_text(q: dict) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(_opts_dict(q).items()))
    return f"{q['question']}\n\n{opts}\n\n请用一行回答：答案：X（X为选项字母）"


# ============ 模式 1：LLM 裸问 ============

def run_bare_llm(items: list) -> dict:
    answers = {}
    for i, q in enumerate(items):
        try:
            resp = _chat([{"role": "user", "content": _build_question_text(q)}])
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            ans = extract_answer_letter(content, VALID_OPTIONS)
            answers[q["qid"]] = {"answer": ans, "tokens": usage.prompt_tokens + usage.completion_tokens}
        except Exception as e:
            print(f"  [错误] {q['qid']}: {str(e)[:80]}")
            answers[q["qid"]] = {"answer": "", "tokens": 0}
        print(f"  [{i+1}/{len(items)}] {q['qid']} -> {answers[q['qid']]['answer']}")
    return answers


# ============ 模式 2：朴素向量 RAG ============

def run_naive_vector(items: list) -> dict:
    from vector_store import get_vector_store
    store = get_vector_store()
    answers = {}
    for i, q in enumerate(items):
        try:
            hits = store.query(q["question"], domain=q["domain"],
                               doc_ids=q.get("gold_doc_ids"), top_k=8)
            evidence = "\n\n".join(f"[{j+1}] {h['text'][:300]}" for j, h in enumerate(hits))
            prompt = f"## 参考文档片段\n{evidence}\n\n## 问题\n{_build_question_text(q)}"
            resp = _chat([{"role": "user", "content": prompt}])
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            ans = extract_answer_letter(content, VALID_OPTIONS)
            answers[q["qid"]] = {"answer": ans, "tokens": usage.prompt_tokens + usage.completion_tokens}
        except Exception as e:
            print(f"  [错误] {q['qid']}: {str(e)[:80]}")
            answers[q["qid"]] = {"answer": "", "tokens": 0}
        print(f"  [{i+1}/{len(items)}] {q['qid']} -> {answers[q['qid']]['answer']}")
    return answers


# ============ 模式 3：符号检索单轮 ============

def run_symbolic_single_turn(items: list, domain_systems: dict) -> dict:
    answers = {}
    for i, q in enumerate(items):
        try:
            system = domain_systems[q["domain"]]
            evs = system.search(q["question"], _opts_dict(q),
                                doc_ids=q.get("gold_doc_ids"), top_k=8)
            evidence = "\n\n".join(f"[{j+1}] {e['text'][:300]}" for j, e in enumerate(evs))
            prompt = f"## 参考文档片段\n{evidence}\n\n## 问题\n{_build_question_text(q)}"
            resp = _chat([{"role": "user", "content": prompt}])
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            ans = extract_answer_letter(content, VALID_OPTIONS)
            answers[q["qid"]] = {"answer": ans, "tokens": usage.prompt_tokens + usage.completion_tokens}
        except Exception as e:
            print(f"  [错误] {q['qid']}: {str(e)[:80]}")
            answers[q["qid"]] = {"answer": "", "tokens": 0}
        print(f"  [{i+1}/{len(items)}] {q['qid']} -> {answers[q['qid']]['answer']}")
    return answers


# ============ 模式 4：完整系统 ============

def run_full_system(items: list, domain_systems: dict) -> dict:
    from model_driven_retrieval import ModelDrivenRetriever
    retrievers = {d: ModelDrivenRetriever(retrieval_system=s, mode="offline_competition")
                  for d, s in domain_systems.items()}
    answers = {}
    for i, q in enumerate(items):
        try:
            system = domain_systems[q["domain"]]
            evs = system.search(q["question"], _opts_dict(q),
                                doc_ids=q.get("gold_doc_ids"), top_k=15)
            result = retrievers[q["domain"]].answer_question(
                question=q["question"], options=_opts_dict(q),
                answer_format=q["answer_format"], initial_evidence=evs,
                domain=q["domain"], doc_ids=q.get("gold_doc_ids"))
            answers[q["qid"]] = {
                "answer": result.get("answer", ""),
                "refused": result.get("refused", False),
                "tokens": result.get("total_tokens", 0),
                # 引用校验结果（金融场景追溯性指标：有效引用数/幻觉引用数/证据覆盖率）
                "citations": result.get("citations"),
                # 答案来源（model/refused_guess/validate_guess 等，诊断用）
                "answer_source": result.get("answer_source", ""),
            }
        except Exception as e:
            print(f"  [错误] {q['qid']}: {str(e)[:80]}")
            answers[q["qid"]] = {"answer": "", "tokens": 0}
        print(f"  [{i+1}/{len(items)}] {q['qid']} -> {answers[q['qid']]['answer']} "
              f"(tok={answers[q['qid']]['tokens']})")
    return answers


# ============ 纯检索评测（无 LLM） ============

def run_retrieval(items: list, domain_systems: dict, retriever: str = "symbolic") -> dict:
    """纯检索评测：rank_lists 收集后统一计算 Recall@10 / MRR@10（含分层统计）

    retriever="symbolic" 用符号检索（BM25+加分项）；"vector" 用 Chroma 向量检索
    """
    from vector_store import get_vector_store
    store = get_vector_store() if retriever == "vector" else None
    rank_lists, meta = {}, {}
    for i, q in enumerate(items):
        if store is not None:
            hits = store.query(q["question"], domain=q["domain"],
                               doc_ids=q.get("gold_doc_ids"), top_k=10)
            ranked = [h["chunk_id"] for h in hits]
        else:
            system = domain_systems[q["domain"]]
            evs = system.search(q["question"], _opts_dict(q),
                                doc_ids=q.get("gold_doc_ids"), top_k=10)
            ranked = [e["chunk_id"] for e in evs]
        rank_lists[q["qid"]] = {
            "ranked": ranked,
            "gold": q.get("gold_chunk_ids", []),
        }
        meta[q["qid"]] = {"layer": q.get("layer", ""), "domain": q.get("domain", "")}
        if (i + 1) % 20 == 0:
            print(f"  检索进度 {i+1}/{len(items)}")
    return retrieval_metrics(rank_lists, meta)


# ============ 主入口 ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="retrieval",
                        choices=["retrieval", "bare_llm", "naive_vector",
                                 "symbolic_single_turn", "full_system"])
    parser.add_argument("--max", type=int, default=None, help="限制评测题数")
    parser.add_argument("--retry-failed", action="store_true",
                        help="只重跑上次 tokens=0 或空答案的题目，合并结果重新出报告")
    parser.add_argument("--tag", default="v1",
                        help="输出文件标签（如 cit），避免小样本评测覆盖正式报告")
    parser.add_argument("--layer", default=None,
                        help="只评测指定层（如 unanswerable），配合 --tag 使用")
    args = parser.parse_args()

    items = load_eval_set(args.max)
    if args.layer:
        items = [q for q in items if q.get("layer") == args.layer]
    print(f"评测集: {len(items)} 题，模式: {args.mode}"
          + (f"，层: {args.layer}" if args.layer else ""))

    # 重试模式：加载上次结果，筛出失败题（API 断连导致 tok=0 的 fallback 答案）
    prev_answers = None
    if args.retry_failed:
        ans_path = os.path.join(REPORT_DIR, f"answers_{args.mode}_v1.json")
        with open(ans_path, encoding="utf-8") as f:
            prev_answers = json.load(f)
        failed_ids = {qid for qid, a in prev_answers.items()
                      if a.get("tokens", 0) == 0 or not a.get("answer")}
        items = [q for q in items if q["qid"] in failed_ids]
        print(f"重试模式: 待重跑失败题 {len(items)} 道 -> {sorted(failed_ids)}")
        if not items:
            print("无失败题，无需重跑")
            return

    t0 = time.time()
    answers, retrieval = None, None

    if args.mode == "retrieval":
        from run_100 import load_or_build_indexes
        domain_systems = load_or_build_indexes()
        # 两种检索器都评（均无需 LLM），输出统一对照报告
        print("== 符号检索 ==")
        sym = run_retrieval(items, domain_systems, retriever="symbolic")
        print(f"Recall@10 = {sym['recall@10']:.1%}, MRR@10 = {sym['mrr@10']:.3f}")
        print("== 向量检索 ==")
        vec = run_retrieval(items, domain_systems, retriever="vector")
        print(f"Recall@10 = {vec['recall@10']:.1%}, MRR@10 = {vec['mrr@10']:.3f}")
        report = [f"# 检索评测报告（{len(items)} 题，金标锚定 chunk 级）", "",
                  f"## 符号检索（BM25 + 数值/短语/实体加分 + 表格保底 + 向量兜底）",
                  f"- Recall@10: **{sym['recall@10']:.1%}**",
                  f"- MRR@10: **{sym['mrr@10']:.3f}**",
                  "",
                  "## 向量检索对照（Chroma + qwen3.7-text-embedding-flash）",
                  f"- Recall@10: {vec['recall@10']:.1%}",
                  f"- MRR@10: {vec['mrr@10']:.3f}",
                  "",
                  "## 分层 Recall@10 对照",
                  "| 层 | 符号检索 | 向量检索 | 命中差异 |",
                  "|---|---|---|---|"]
        for layer, s in sym["layers"].items():
            v = vec["layers"].get(layer) or {"recall": 0.0, "n": s["n"]}
            diff = s["recall"] - v["recall"]
            report.append(f"| {layer}（n={s['n']}） | {s['recall']:.1%} | {v['recall']:.1%} | {diff:+.1%} |")
        report += ["", "## 分领域 Recall@10 对照",
                   "| 领域 | 符号检索 | 向量检索 | 命中差异 |",
                   "|---|---|---|---|"]
        for dom in sym["domains"]:
            s = sym["domains"][dom]
            v = vec["domains"].get(dom, {"recall": 0.0, "n": s["n"]})
            diff = s["recall"] - v["recall"]
            report.append(f"| {dom}（n={s['n']}） | {s['recall']:.1%} | {v['recall']:.1%} | {diff:+.1%} |")
        out = os.path.join(REPORT_DIR, f"report_{args.mode}_{args.tag}.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"报告已写出: {out}")
        return

    from run_100 import load_or_build_indexes
    domain_systems = load_or_build_indexes()

    if args.mode == "bare_llm":
        answers = run_bare_llm(items)
    elif args.mode == "naive_vector":
        answers = run_naive_vector(items)
    elif args.mode == "symbolic_single_turn":
        answers = run_symbolic_single_turn(items, domain_systems)
    elif args.mode == "full_system":
        answers = run_full_system(items, domain_systems)

    # 重试模式：新结果覆盖旧结果后整体重算（失败的 tok=0 条目被真实答案替换）
    if prev_answers is not None:
        answers = {**prev_answers, **answers}

    # 准确率只统计有答案记录的题目（重试模式下 items 是失败题子集，
    # 且上次可能只跑了前 N 题——分母必须与 answers 的键集合一致，避免缺失题被算错）
    full_items = load_eval_set()
    answered_ids = set(answers.keys())
    full_items = [q for q in full_items if q["qid"] in answered_ids] if prev_answers is not None \
        else items
    metrics = answer_accuracy(full_items, answers)
    print(f"\n准确率: {metrics['accuracy']:.1%} | 平均 token: {metrics['avg_tokens']:.0f}")

    out = os.path.join(REPORT_DIR, f"report_{args.mode}_{args.tag}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(format_report(args.mode, metrics, retrieval))
    with open(os.path.join(REPORT_DIR, f"answers_{args.mode}_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    print(f"报告已写出: {out}（耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
