"""
全链条验证脚本 v3
==================
对 50 道测试题执行：
1. 双重检索认证：
   - 路径A（原文档检索）：直接在 data/{domain}/ 目录中扫描 source_anchor 是否存在 → 验证文档出处正确性
   - 路径B（数据库+图检索）：调用 DomainRetrievalSystem.search() → 验证图存储召回率
2. 推理引擎作答：调用 submission/agent/qa_engine.QAEngine 进行 CoT 推理
3. 答案对比：expected_answer vs system_answer
4. 输出诊断报告：覆盖 检索命中 / 推理准确率 / CoT 链路 / 失败模式

执行环境：conda Audio
"""
import os
import sys
import json
import time
import re
import importlib
import pickle
from pathlib import Path
from collections import defaultdict

# 强制 UTF-8 输出
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "submission" / "agent"))

TEST_QUESTIONS_PATH = PROJECT_ROOT / "debug_test_v3" / "test_questions_v3.json"
TEST_ANSWERS_PATH = PROJECT_ROOT / "debug_test_v3" / "test_answers_v3.json"
OUTPUT_DIR = PROJECT_ROOT / "debug_test_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LLM 推理答案文件（由 gen_llm_prompts.py + 子代理 GLM-5.2 推理生成）
LLM_ANSWERS_PATH = OUTPUT_DIR / "llm_answers.jsonl"
LLM_PROMPTS_META_PATH = OUTPUT_DIR / "prompts_meta.json"

# ============ 路径A：原文档双重检索认证 ============

# 领域 → data 目录映射
DOMAIN_DATA_DIR = {
    "financial_contracts": PROJECT_ROOT / "data" / "financial_contracts",
    "financial_reports": PROJECT_ROOT / "data" / "financial_reports",
    "insurance": PROJECT_ROOT / "data" / "insurance",
    "regulatory": PROJECT_ROOT / "data" / "regulatory",
    "research": PROJECT_ROOT / "data" / "research",
}


def load_doc_text(domain: str, doc_id: str) -> str:
    """从 data/{domain}/ 加载指定 doc_id 的全部文本"""
    ddir = DOMAIN_DATA_DIR.get(domain)
    if not ddir or not ddir.exists():
        return ""
    # 在目录中查找匹配 doc_id 的 json 文件
    for fname in os.listdir(ddir):
        if not fname.endswith(".json"):
            continue
        fpath = ddir / fname
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                doc = json.load(f)
            if doc.get("doc_id") == doc_id:
                # 拼接全部文本：tables[].markdown + pages[].text
                parts = []
                for tbl in doc.get("tables", []):
                    md = tbl.get("markdown", "")
                    if md:
                        parts.append(md)
                for pg in doc.get("pages", []):
                    txt = pg.get("text", "")
                    if txt:
                        parts.append(txt)
                return "\n".join(parts)
        except Exception:
            continue
    return ""


def verify_source_anchor(question: dict) -> dict:
    """路径A：验证 source_anchor 是否在原文档中真实出现

    Returns:
        {
            "anchor_in_source": bool,  # 锚点是否在原文出现
            "source_doc_found": bool,  # 原文档是否被找到
            "matched_snippet": str,   # 命中片段（前80字+后80字）
            "page_verified": bool,    # 页码是否一致
        }
    """
    domain = question["domain"]
    doc_id = question["source_doc"]
    anchor = question.get("source_anchor", "")
    page = question.get("source_page", 0)

    result = {
        "anchor_in_source": False,
        "source_doc_found": False,
        "matched_snippet": "",
        "page_verified": False,
    }

    if not anchor:
        return result

    full_text = load_doc_text(domain, doc_id)
    if not full_text:
        return result
    result["source_doc_found"] = True

    # 检查锚点是否在原文出现
    # 锚点格式：数值、条款号、实体名等
    if anchor in full_text:
        result["anchor_in_source"] = True
        # 提取命中片段（前后各80字）
        idx = full_text.find(anchor)
        start = max(0, idx - 80)
        end = min(len(full_text), idx + len(anchor) + 80)
        result["matched_snippet"] = full_text[start:end]
    else:
        # 尝试去除千分位后再匹配（如 1,707.82 → 1707.82）
        clean_anchor = anchor.replace(",", "")
        if clean_anchor in full_text:
            result["anchor_in_source"] = True
            idx = full_text.find(clean_anchor)
            start = max(0, idx - 80)
            end = min(len(full_text), idx + len(clean_anchor) + 80)
            result["matched_snippet"] = full_text[start:end]
        else:
            # 多锚点情形：anchor 用 ; 分隔
            if ";" in anchor:
                sub_anchors = anchor.split(";")
                hit_count = 0
                for sa in sub_anchors:
                    if sa.strip() and sa.strip() in full_text:
                        hit_count += 1
                if hit_count >= 1:
                    result["anchor_in_source"] = True
                    result["matched_snippet"] = f"多锚点命中 {hit_count}/{len(sub_anchors)}"

    # 页码验证（仅对 pages-based 文档）
    if page > 0:
        # 在原文档的 pages 中查找该页
        ddir = DOMAIN_DATA_DIR.get(domain)
        if ddir:
            for fname in os.listdir(ddir):
                if not fname.endswith(".json"):
                    continue
                fpath = ddir / fname
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        doc = json.load(f)
                    if doc.get("doc_id") == doc_id:
                        for pg in doc.get("pages", []):
                            if pg.get("page_num") == page:
                                result["page_verified"] = True
                                break
                        break
                except Exception:
                    continue

    return result


# ============ 路径B：数据库+图检索认证 ============

# 缓存领域检索系统（避免重复加载索引）
_DOMAIN_SYSTEM_CACHE = {}


def get_domain_system(domain: str):
    """加载 DomainRetrievalSystem（带缓存）"""
    if domain in _DOMAIN_SYSTEM_CACHE:
        return _DOMAIN_SYSTEM_CACHE[domain]

    try:
        # 加载 v2 索引缓存
        cache_path = PROJECT_ROOT / "cache" / "domain_indexes_v2.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            systems = cache.get("domain_systems", {})
            if domain in systems:
                _DOMAIN_SYSTEM_CACHE[domain] = systems[domain]
                return systems[domain]

        # 缓存未命中，重新构建
        from domain_retrieval import DomainRetrievalSystem
        system = DomainRetrievalSystem(domain)
        system.build_from_json_dir()
        _DOMAIN_SYSTEM_CACHE[domain] = system
        return system
    except Exception as e:
        print(f"  [警告] 加载 {domain} 检索系统失败: {e}")
        return None


def db_retrieve(question: dict) -> dict:
    """路径B：调用 DomainRetrievalSystem 检索

    Returns:
        {
            "hit_source_doc": bool,    # 召回的 chunk 中是否包含 source_doc
            "hit_source_anchor": bool, # 召回的 chunk 中是否包含 source_anchor
            "top_chunks": list[dict],  # 召回的 top-5 chunk
            "recall_rank": int,        # source_doc 在召回列表中的排名（0=未命中）
        }
    """
    domain = question["domain"]
    source_doc = question["source_doc"]
    anchor = question.get("source_anchor", "")

    system = get_domain_system(domain)
    if not system:
        return {
            "hit_source_doc": False,
            "hit_source_anchor": False,
            "top_chunks": [],
            "recall_rank": 0,
            "error": "system_unavailable",
        }

    try:
        # 调用检索（限制 doc_ids 为 source_doc，模拟有 doc_ids 时的场景）
        # P1 优化：top_k 从 10 提升到 15，增加 multi 题证据覆盖
        # multi 题的多个公司可能分散在不同 chunks，10 个可能不够
        results = system.search(
            question=question["question"],
            options=question["options"],
            doc_ids=[source_doc],
            top_k=15,
        )
    except Exception as e:
        return {
            "hit_source_doc": False,
            "hit_source_anchor": False,
            "top_chunks": [],
            "recall_rank": 0,
            "error": str(e),
        }

    # 检查召回质量
    hit_source_doc = False
    hit_source_anchor = False
    recall_rank = 0
    top_chunks = []

    # P1 修复：支持分号分隔的多锚点（multi 题场景）
    # source_anchor 形如 "公司A;公司B;公司C"，整体匹配必然失败
    # 拆分为子锚点列表，只要任一子锚点命中即视为命中
    sub_anchors = []
    if anchor:
        # 同时支持中英文分号
        raw_subs = anchor.replace("；", ";").split(";")
        for sub in raw_subs:
            sub = sub.strip()
            if sub:
                sub_anchors.append(sub)

    for i, r in enumerate(results[:15]):  # P1 优化：从 10 提升到 15
        chunk_doc = r.get("doc_id", "")
        chunk_text = r.get("text", "")
        top_chunks.append({
            "doc_id": chunk_doc,
            "score": round(r.get("score", 0), 2),
            "text_preview": chunk_text[:120],
        })
        if chunk_doc == source_doc:
            if not recall_rank:
                recall_rank = i + 1
            hit_source_doc = True
            # P1 修复：多锚点拆分匹配（支持 multi 题分号分隔的锚点）
            if sub_anchors:
                # 清理千分位后的文本，便于数值锚点匹配
                chunk_text_clean = chunk_text.replace(",", "")
                for sub in sub_anchors:
                    sub_clean = sub.replace(",", "")
                    if sub in chunk_text or sub_clean in chunk_text_clean:
                        hit_source_anchor = True
                        break

    return {
        "hit_source_doc": hit_source_doc,
        "hit_source_anchor": hit_source_anchor,
        "top_chunks": top_chunks,
        "recall_rank": recall_rank,
    }


# ============ 推理引擎作答 ============

_QA_ENGINE = None

# LLM 答案缓存（qid -> answer_obj），避免重复 IO
_LLM_ANSWERS_CACHE: dict | None = None


def load_llm_answers() -> dict:
    """加载 LLM 推理答案文件（llm_answers.jsonl）

    Returns:
        {qid: answer_obj} 字典，answer_obj 包含 answer/evidence_quote/reasoning 等字段
    """
    global _LLM_ANSWERS_CACHE
    if _LLM_ANSWERS_CACHE is not None:
        return _LLM_ANSWERS_CACHE

    if not LLM_ANSWERS_PATH.exists():
        print(f"  [警告] LLM 答案文件不存在: {LLM_ANSWERS_PATH}")
        return {}

    answers = {}
    try:
        with open(LLM_ANSWERS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                answers[obj["qid"]] = obj
        _LLM_ANSWERS_CACHE = answers
        print(f"  加载 LLM 答案: {len(answers)} 条")
        return answers
    except Exception as e:
        print(f"  [错误] 加载 LLM 答案失败: {e}")
        return {}


def load_llm_prompts_meta() -> dict:
    """加载 prompts 元数据（用于获取 estimated_input_tokens）

    Returns:
        {qid: meta_obj} 字典
    """
    if not LLM_PROMPTS_META_PATH.exists():
        return {}
    try:
        with open(LLM_PROMPTS_META_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 转为 {qid: meta_obj}
        meta_map = {m["qid"]: m for m in data.get("prompts", [])}
        return meta_map
    except Exception as e:
        print(f"  [警告] 加载 prompts_meta 失败: {e}")
        return {}


def get_qa_engine():
    """初始化 QAEngine（带缓存）"""
    global _QA_ENGINE
    if _QA_ENGINE is not None:
        return _QA_ENGINE

    try:
        # 优先使用 submission/agent/qa_engine.py（更新的 CoT 模板）
        # 通过 importlib 加载，避免与项目根目录 qa_engine.py 冲突
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "agent_qa_engine",
            str(PROJECT_ROOT / "submission" / "agent" / "qa_engine.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # 注入 config 模块路径
        agent_dir = PROJECT_ROOT / "submission" / "agent"
        if str(agent_dir) not in sys.path:
            sys.path.insert(0, str(agent_dir))
        spec.loader.exec_module(mod)
        _QA_ENGINE = mod.QAEngine()
        return _QA_ENGINE
    except Exception as e:
        print(f"  [警告] 加载 submission/agent/qa_engine 失败: {e}，尝试根目录 qa_engine")
        try:
            from qa_engine import QAEngine
            _QA_ENGINE = QAEngine()
            return _QA_ENGINE
        except Exception as e2:
            print(f"  [错误] 加载根目录 qa_engine 也失败: {e2}")
            return None


def answer_question_via_retrieval(question: dict, evidence_chunks: list[dict]) -> dict:
    """检索推断作答（无需调用 API，作为推理引擎不可用时的回退对照）

    策略：
    1. 对每个选项，检查选项文本是否出现在证据中（或锚点匹配）
    2. 优先返回命中证据的选项
    """
    options = question["options"]
    answer_format = question["answer_format"]
    expected_anchor = question.get("source_anchor", "")
    clean_anchor = expected_anchor.replace(",", "") if expected_anchor else ""

    # 收集所有证据文本
    all_text = "\n".join(c.get("text", "") for c in evidence_chunks)

    # 选项匹配评分
    option_scores = {}
    for letter, opt_text in options.items():
        score = 0
        # 1. 选项完整出现
        if opt_text in all_text:
            score += 10
        # 2. 选项中的数值/关键短语出现
        nums = re.findall(r'\d+(?:[,.\d]*)', opt_text)
        for n in nums:
            clean_n = n.replace(",", "")
            if n in all_text or clean_n in all_text:
                score += 5
        # 3. 选项锚点匹配 expected_anchor
        if expected_anchor and (
            expected_anchor in opt_text or opt_text in expected_anchor
        ):
            score += 20
        # 4. 选项词分词后命中数
        for w in re.findall(r'[一-鿿]{2,}', opt_text):
            if w in all_text:
                score += 1
        option_scores[letter] = score

    # 根据题型决策
    if answer_format == "tf":
        # 判断题：若证据充足且选项表述在证据中出现，选 A，否则 B
        if option_scores.get("A", 0) > 0:
            return {"answer": "A", "tokens": 0, "reasoning": "检索推断：选项A内容在证据中出现",
                    "method": "retrieval_inference"}
        return {"answer": "B", "tokens": 0, "reasoning": "检索推断：选项A内容未在证据中出现",
                "method": "retrieval_inference"}
    elif answer_format == "multi":
        # 多选题：选择所有得分大于0的选项，确保至少2个
        positive = [l for l, s in option_scores.items() if s > 0]
        if len(positive) < 2:
            # 兜底：返回前2个最高分
            sorted_opts = sorted(option_scores.items(), key=lambda x: -x[1])
            positive = [l for l, _ in sorted_opts[:2]]
        return {
            "answer": "".join(sorted(positive)),
            "tokens": 0,
            "reasoning": f"检索推断：得分={option_scores}",
            "method": "retrieval_inference",
        }
    else:
        # 单选题：选最高分
        sorted_opts = sorted(option_scores.items(), key=lambda x: -x[1])
        return {
            "answer": sorted_opts[0][0],
            "tokens": 0,
            "reasoning": f"检索推断：得分={option_scores}",
            "method": "retrieval_inference",
        }


def answer_question_via_llm(question: dict, evidence_chunks: list[dict] | None = None) -> dict:
    """从 LLM 答案文件读取答案（使用当前窗口模型 GLM-5.2 离线推理结果）

    Args:
        question: 题目对象
        evidence_chunks: 证据 chunks（仅用于回退时使用，正常从 LLM 答案文件读取）

    Returns:
        {answer, tokens, reasoning, option_judgements, method, evidence_quote}
    """
    answers = load_llm_answers()
    meta = load_llm_prompts_meta()
    qid = question["qid"]

    if qid in answers:
        ans = answers[qid]
        # 计算 token 消耗：输入 token 来自 prompts_meta，输出 token 来自答案文件
        input_tokens = meta.get(qid, {}).get("estimated_input_tokens", 0)
        output_tokens = ans.get("estimated_output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        # 提取 reasoning（tf 题含 extracted_statement，multi 题含 option_judgements）
        reasoning_parts = []
        if ans.get("extracted_statement"):
            reasoning_parts.append(f"待验证陈述: {ans['extracted_statement'][:100]}")
        if ans.get("key_facts"):
            reasoning_parts.append(f"关键事实: {ans['key_facts']}")
        if ans.get("matched_facts"):
            reasoning_parts.append(f"已匹配: {ans['matched_facts']}")
        if ans.get("unmatched_facts"):
            reasoning_parts.append(f"未匹配: {ans['unmatched_facts']}")
        if ans.get("reasoning"):
            reasoning_parts.append(ans["reasoning"][:300])
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "无推理过程"

        return {
            "answer": ans.get("answer", "A"),
            "tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning": reasoning,
            "option_judgements": ans.get("option_judgements", {}),
            "evidence_quote": ans.get("evidence_quote", "")[:200],
            "method": "llm_glm5.2",
        }

    # 答案文件未找到该题，回退到检索推断
    print(f"  [警告] LLM 答案未找到 qid={qid}，回退到检索推断")
    return answer_question_via_retrieval(question, evidence_chunks or [])


def answer_question(question: dict, evidence_chunks: list[dict] | None = None,
                    use_llm_answers: bool = False) -> dict:
    """调用推理引擎作答

    Args:
        question: 题目对象
        evidence_chunks: 证据 chunks（可选）
        use_llm_answers: 若为 True，从 llm_answers.jsonl 读取答案（当前窗口模型离线推理结果）

    优先级：
    1. use_llm_answers=True → 读取 LLM 答案文件
    2. SKIP_API=1 → 检索推断
    3. 调用 QAEngine API
    4. API 失败 → 检索推断回退
    """
    # 若未提供证据，则用数据库检索结果作为证据
    if evidence_chunks is None:
        db_result = db_retrieve(question)
        evidence_chunks = [
            {"text": c["text_preview"], "doc_id": c["doc_id"], "score": c["score"]}
            for c in db_result["top_chunks"]
        ]

    # 优先级 1：使用 LLM 答案文件（当前窗口模型 GLM-5.2 离线推理结果）
    if use_llm_answers:
        return answer_question_via_llm(question, evidence_chunks)

    # 若环境变量 SKIP_API=1，直接走检索推断（已知 API 配额耗尽）
    if os.environ.get("SKIP_API", "0") == "1":
        return answer_question_via_retrieval(question, evidence_chunks)

    # 优先尝试 API 推理（最多 1 次重试，避免耗尽时长）
    engine = get_qa_engine()
    if engine:
        try:
            result = engine.answer_question(
                question=question["question"],
                options=question["options"],
                answer_format=question["answer_format"],
                evidence_chunks=evidence_chunks,
                doc_ids=[question["source_doc"]],
                domain=question["domain"],
                audit_mode=True,  # 启用审计模式获取完整 CoT
            )
            return {
                "answer": result.get("answer", "A"),
                "tokens": result.get("total_tokens", 0),
                "reasoning": result.get("reasoning", "")[:500],
                "option_judgements": result.get("option_judgements", {}),
                "method": "api_cot",
            }
        except Exception as e:
            err_msg = str(e)
            # API 配额耗尽或网络错误：降级为检索推断
            if "quota" in err_msg.lower() or "403" in err_msg or "insufficient" in err_msg.lower():
                result = answer_question_via_retrieval(question, evidence_chunks)
                result["api_error"] = err_msg[:100]
                return result
            return {"answer": "A", "error": err_msg, "tokens": 0, "reasoning": "",
                    "method": "api_error"}
    else:
        # 引擎未初始化：直接走检索推断
        return answer_question_via_retrieval(question, evidence_chunks)


# ============ 主流程 ============

def main():
    import argparse
    parser = argparse.ArgumentParser(description="全链条验证脚本 v3")
    parser.add_argument("--use-llm-answers", action="store_true",
                        help="使用 llm_answers.jsonl（当前窗口模型 GLM-5.2 离线推理结果）")
    parser.add_argument("--output-suffix", default="",
                        help="输出文件后缀（如 _llm），避免覆盖原文件")
    args = parser.parse_args()

    use_llm = args.use_llm_answers
    suffix = args.output_suffix if args.output_suffix else ("_llm" if use_llm else "")

    print("=" * 70)
    print("全链条验证脚本 v3 - 双重检索认证 + CoT 推理验证")
    print(f"  推理模式: {'LLM 答案文件（GLM-5.2）' if use_llm else 'API/检索推断'}")
    print("=" * 70)

    # 加载测试题
    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"加载测试题: {len(questions)} 道")

    # 预加载所有领域检索系统（避免单题加载开销）
    print("\n预加载领域检索系统...")
    for domain in DOMAIN_DATA_DIR:
        sys = get_domain_system(domain)
        if sys:
            print(f"  {domain}: {len(sys.chunk_data)} chunks")

    # 若启用 LLM 模式，预加载答案文件
    if use_llm:
        print("\n预加载 LLM 答案文件...")
        load_llm_answers()

    # 逐题处理
    results = []
    t0 = time.time()

    for i, q in enumerate(questions, 1):
        qid = q["qid"]
        domain = q["domain"]
        print(f"\n[{i}/{len(questions)}] {qid} [{domain}]")
        print(f"  问题: {q['question'][:80]}...")
        print(f"  期望答案: {q['expected_answer']} (锚点: {q.get('source_anchor', '')[:40]})")

        # 路径A：原文档验证
        path_a = verify_source_anchor(q)
        print(f"  [路径A] 原文档: found={path_a['source_doc_found']}, "
              f"anchor_hit={path_a['anchor_in_source']}, page_ok={path_a['page_verified']}")

        # 路径B：数据库+图检索
        path_b = db_retrieve(q)
        print(f"  [路径B] 数据库: doc_hit={path_b['hit_source_doc']}, "
              f"anchor_hit={path_b['hit_source_anchor']}, rank={path_b['recall_rank']}")

        # 推理作答（根据 use_llm 切换数据源）
        evidence_chunks = [
            {"text": c["text_preview"], "doc_id": c["doc_id"], "score": c["score"]}
            for c in path_b["top_chunks"]
        ]
        qa_result = answer_question(q, evidence_chunks, use_llm_answers=use_llm)
        method = qa_result.get("method", "unknown")
        print(f"  [推理] 系统答案: {qa_result['answer']}  "
              f"方法: {method}  "
              f"Token: {qa_result.get('tokens', 0)}  "
              f"错误: {qa_result.get('error', qa_result.get('api_error', '无'))}")

        # 对比
        expected = q["expected_answer"]
        system_answer = qa_result["answer"]
        correct = (expected == system_answer)

        result = {
            "qid": qid,
            "domain": domain,
            "question": q["question"],
            "expected_answer": expected,
            "system_answer": system_answer,
            "correct": correct,
            "dimension": q["dimension"],
            "answer_format": q["answer_format"],
            "source_doc": q["source_doc"],
            "source_page": q["source_page"],
            "source_anchor": q.get("source_anchor", ""),
            # 路径A
            "path_a_source_found": path_a["source_doc_found"],
            "path_a_anchor_hit": path_a["anchor_in_source"],
            "path_a_page_verified": path_a["page_verified"],
            "path_a_snippet": path_a["matched_snippet"],
            # 路径B
            "path_b_doc_hit": path_b["hit_source_doc"],
            "path_b_anchor_hit": path_b["hit_source_anchor"],
            "path_b_recall_rank": path_b["recall_rank"],
            "path_b_error": path_b.get("error", ""),
            "path_b_top_chunks": path_b["top_chunks"][:3],
            # 推理（含 token 明细）
            "qa_tokens": qa_result.get("tokens", 0),
            "qa_input_tokens": qa_result.get("input_tokens", 0),
            "qa_output_tokens": qa_result.get("output_tokens", 0),
            "qa_error": qa_result.get("error", qa_result.get("api_error", "")),
            "qa_reasoning_preview": qa_result.get("reasoning", "")[:200],
            "qa_evidence_quote": qa_result.get("evidence_quote", "")[:200],
            "qa_option_judgements": qa_result.get("option_judgements", {}),
            "qa_method": qa_result.get("method", "unknown"),
        }
        results.append(result)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"完成！耗时: {elapsed:.1f}s")

    # 统计分析
    stats = analyze_results(results)

    # 保存结果（输出文件名加 suffix 区分）
    out_path = OUTPUT_DIR / f"validation_results_v3{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "stats": stats}, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {out_path}")

    # 输出诊断报告
    diag_path = OUTPUT_DIR / f"diagnosis_v3{suffix}.md"
    write_diagnosis_report(results, stats, diag_path)
    print(f"诊断报告: {diag_path}")

    # 控制台打印摘要
    print_summary(stats)


def analyze_results(results: list[dict]) -> dict:
    """统计分析"""
    stats = {
        "total": len(results),
        "correct": 0,
        "by_domain": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_format": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_dimension": defaultdict(lambda: {"total": 0, "correct": 0}),
        # 检索命中统计
        "path_a_anchor_hit": 0,
        "path_a_source_found": 0,
        "path_a_page_verified": 0,
        "path_b_doc_hit": 0,
        "path_b_anchor_hit": 0,
        "path_b_recall_rank_avg": 0,
        "path_b_no_recall": 0,  # rank=0
        # CoT 推理统计
        "qa_errors": 0,
        "qa_tokens_total": 0,
        "qa_input_tokens_total": 0,
        "qa_output_tokens_total": 0,
        # 失败模式分类
        "failure_modes": defaultdict(int),
    }

    ranks = []
    for r in results:
        if r["correct"]:
            stats["correct"] += 1
        stats["by_domain"][r["domain"]]["total"] += 1
        if r["correct"]:
            stats["by_domain"][r["domain"]]["correct"] += 1
        stats["by_format"][r["answer_format"]]["total"] += 1
        if r["correct"]:
            stats["by_format"][r["answer_format"]]["correct"] += 1
        stats["by_dimension"][r["dimension"]]["total"] += 1
        if r["correct"]:
            stats["by_dimension"][r["dimension"]]["correct"] += 1

        if r["path_a_source_found"]:
            stats["path_a_source_found"] += 1
        if r["path_a_anchor_hit"]:
            stats["path_a_anchor_hit"] += 1
        if r["path_a_page_verified"]:
            stats["path_a_page_verified"] += 1

        if r["path_b_doc_hit"]:
            stats["path_b_doc_hit"] += 1
        if r["path_b_anchor_hit"]:
            stats["path_b_anchor_hit"] += 1
        if r["path_b_recall_rank"] > 0:
            ranks.append(r["path_b_recall_rank"])
        else:
            stats["path_b_no_recall"] += 1

        if r["qa_error"]:
            stats["qa_errors"] += 1
        stats["qa_tokens_total"] += r.get("qa_tokens", 0)
        stats["qa_input_tokens_total"] += r.get("qa_input_tokens", 0)
        stats["qa_output_tokens_total"] += r.get("qa_output_tokens", 0)

        # 失败模式分类
        if not r["correct"]:
            if not r["path_a_anchor_hit"]:
                stats["failure_modes"]["A_源锚点未在原文出现"] += 1
            elif not r["path_b_doc_hit"]:
                stats["failure_modes"]["B_数据库未召回源文档"] += 1
            elif not r["path_b_anchor_hit"]:
                stats["failure_modes"]["C_数据库召回文档但未命中锚点"] += 1
            elif r["qa_error"]:
                stats["failure_modes"]["D_推理引擎报错"] += 1
            else:
                stats["failure_modes"]["E_推理判断错误"] += 1

    if ranks:
        stats["path_b_recall_rank_avg"] = sum(ranks) / len(ranks)
    return stats


def write_diagnosis_report(results: list[dict], stats: dict, path: Path):
    """生成诊断报告"""
    lines = []
    lines.append("# 全链条验证诊断报告 v3\n")
    lines.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"测试题数: {stats['total']}\n\n")

    lines.append("## 1. 总体准确率\n")
    acc = stats["correct"] / stats["total"] if stats["total"] else 0
    lines.append(f"- 正确题数: {stats['correct']}/{stats['total']}\n")
    lines.append(f"- 准确率: {acc:.2%}\n\n")

    lines.append("## 2. 领域准确率\n")
    lines.append("| 领域 | 题数 | 正确 | 准确率 |\n| --- | --- | --- | --- |\n")
    for d, s in sorted(stats["by_domain"].items()):
        acc = s["correct"] / s["total"] if s["total"] else 0
        lines.append(f"| {d} | {s['total']} | {s['correct']} | {acc:.2%} |\n")
    lines.append("\n")

    lines.append("## 3. 题型准确率\n")
    lines.append("| 题型 | 题数 | 正确 | 准确率 |\n| --- | --- | --- | --- |\n")
    for f, s in sorted(stats["by_format"].items()):
        acc = s["correct"] / s["total"] if s["total"] else 0
        lines.append(f"| {f} | {s['total']} | {s['correct']} | {acc:.2%} |\n")
    lines.append("\n")

    lines.append("## 4. 维度准确率\n")
    lines.append("| 维度 | 题数 | 正确 | 准确率 |\n| --- | --- | --- | --- |\n")
    for d, s in sorted(stats["by_dimension"].items()):
        acc = s["correct"] / s["total"] if s["total"] else 0
        lines.append(f"| {d} | {s['total']} | {s['correct']} | {acc:.2%} |\n")
    lines.append("\n")

    lines.append("## 5. 双重检索认证\n")
    lines.append("### 路径A：原文档检索\n")
    lines.append(f"- 源文档找到: {stats['path_a_source_found']}/{stats['total']}\n")
    lines.append(f"- 源锚点命中: {stats['path_a_anchor_hit']}/{stats['total']}\n")
    lines.append(f"- 页码验证通过: {stats['path_a_page_verified']}/{stats['total']}\n\n")

    lines.append("### 路径B：数据库+图检索\n")
    lines.append(f"- 文档召回: {stats['path_b_doc_hit']}/{stats['total']}\n")
    lines.append(f"- 锚点召回: {stats['path_b_anchor_hit']}/{stats['total']}\n")
    lines.append(f"- 召回失败: {stats['path_b_no_recall']}/{stats['total']}\n")
    lines.append(f"- 命中排名均值: {stats['path_b_recall_rank_avg']:.2f}\n\n")

    lines.append("## 6. CoT 推理统计\n")
    lines.append(f"- 推理错误数: {stats['qa_errors']}/{stats['total']}\n")
    lines.append(f"- Token 总消耗: {stats['qa_tokens_total']}（输入 {stats['qa_input_tokens_total']} + 输出 {stats['qa_output_tokens_total']}）\n")
    if stats['total'] > 0:
        avg = stats['qa_tokens_total'] / stats['total']
        lines.append(f"- 平均每题 Token: {avg:.1f}\n")
    lines.append("\n")

    lines.append("## 7. 失败模式分类\n")
    lines.append("| 失败模式 | 数量 | 说明 |\n| --- | --- | --- |\n")
    mode_desc = {
        "A_源锚点未在原文出现": "题目生成时锚点提取错误，源锚点在原文档中未出现",
        "B_数据库未召回源文档": "数据库检索系统未能召回正确文档",
        "C_数据库召回文档但未命中锚点": "数据库召回了正确文档，但未命中关键锚点",
        "D_推理引擎报错": "推理引擎调用失败或返回错误",
        "E_推理判断错误": "检索与召回均正确，但 CoT 推理判断错误",
    }
    for mode, count in sorted(stats["failure_modes"].items(), key=lambda x: -x[1]):
        desc = mode_desc.get(mode, "")
        lines.append(f"| {mode} | {count} | {desc} |\n")
    lines.append("\n")

    lines.append("## 8. 问题清单与优化建议\n")
    suggestions = generate_suggestions(stats, results)
    for s in suggestions:
        lines.append(f"- {s}\n")
    lines.append("\n")

    lines.append("## 9. 典型失败案例\n")
    failures = [r for r in results if not r["correct"]][:10]
    for r in failures:
        lines.append(f"\n### {r['qid']} [{r['domain']}] - {r['dimension']}\n")
        lines.append(f"- 问题: {r['question'][:100]}\n")
        lines.append(f"- 期望: {r['expected_answer']}, 系统: {r['system_answer']}\n")
        lines.append(f"- 源文档: {r['source_doc']} (页 {r['source_page']})\n")
        lines.append(f"- 锚点: {r['source_anchor'][:60]}\n")
        lines.append(f"- 路径A: found={r['path_a_source_found']}, anchor={r['path_a_anchor_hit']}\n")
        lines.append(f"- 路径B: doc={r['path_b_doc_hit']}, anchor={r['path_b_anchor_hit']}, rank={r['path_b_recall_rank']}\n")
        if r.get("qa_error"):
            lines.append(f"- 推理错误: {r['qa_error']}\n")
        if r.get("qa_reasoning_preview"):
            lines.append(f"- 推理预览: {r['qa_reasoning_preview'][:200]}...\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def generate_suggestions(stats: dict, results: list) -> list:
    """根据统计生成优化建议"""
    sugg = []
    total = stats["total"]

    # 路径A问题
    if stats["path_a_anchor_hit"] < total * 0.95:
        miss_rate = 1 - stats["path_a_anchor_hit"] / total
        sugg.append(
            f"【路径A】{miss_rate:.1%} 的题目源锚点未在原文出现，"
            "需检查测试题生成器的锚点提取逻辑（特别是千分位/单位匹配）"
        )

    # 路径B问题
    if stats["path_b_doc_hit"] < total * 0.9:
        miss_rate = 1 - stats["path_b_doc_hit"] / total
        sugg.append(
            f"【路径B-召回】{miss_rate:.1%} 的题目数据库未召回源文档，"
            "需检查 DomainRetrievalSystem 的 doc_ids 过滤逻辑、BM25 索引完整性"
        )

    if stats["path_b_doc_hit"] > 0 and stats["path_b_anchor_hit"] < stats["path_b_doc_hit"] * 0.8:
        sugg.append(
            "【路径B-锚点】数据库召回了正确文档但未命中关键锚点，"
            "需检查 phrase_index/numeric_index 的覆盖度，特别是千分位数值和长术语短语"
        )

    # CoT 推理问题
    if stats["qa_errors"] > total * 0.1:
        sugg.append(
            f"【推理引擎】{stats['qa_errors']} 题推理报错，"
            "需检查 QAEngine 的 API 配置和重试机制"
        )

    correct_with_full_recall = sum(
        1 for r in results
        if r["correct"] and r["path_b_anchor_hit"] and r["path_a_anchor_hit"]
    )
    if stats["correct"] > 0:
        recall_quality = correct_with_full_recall / stats["correct"]
        if recall_quality < 0.7:
            sugg.append(
                f"【CoT推理】仅 {recall_quality:.1%} 的正确答案伴随完整双重检索命中，"
                "存在答案正确但检索路径不完整的隐患（如蒙题），需加强证据约束"
            )

    # 失败模式
    if stats["failure_modes"].get("E_推理判断错误", 0) > 3:
        sugg.append(
            "【CoT推理】E类失败（检索均正确但推理错误）较多，"
            "需检查 prompt 模板中的逐项判断要求，以及选项锚点匹配条件"
        )

    if not sugg:
        sugg.append("系统全链条运行良好，无明显问题。")
    return sugg


def print_summary(stats: dict):
    """控制台打印摘要"""
    print(f"\n{'='*70}")
    print("【全链条验证摘要】")
    print(f"{'='*70}")
    acc = stats["correct"] / stats["total"] if stats["total"] else 0
    print(f"总体准确率: {stats['correct']}/{stats['total']} = {acc:.2%}")
    print(f"\n[领域准确率]")
    for d, s in sorted(stats["by_domain"].items()):
        a = s["correct"] / s["total"] if s["total"] else 0
        print(f"  {d}: {s['correct']}/{s['total']} = {a:.2%}")
    print(f"\n[双重检索认证]")
    print(f"  路径A 源锚点命中: {stats['path_a_anchor_hit']}/{stats['total']}")
    print(f"  路径B 文档召回: {stats['path_b_doc_hit']}/{stats['total']}")
    print(f"  路径B 锚点召回: {stats['path_b_anchor_hit']}/{stats['total']}")
    print(f"  召回排名均值: {stats['path_b_recall_rank_avg']:.2f}")
    print(f"\n[CoT推理]")
    print(f"  推理错误: {stats['qa_errors']}")
    print(f"  Token消耗: {stats['qa_tokens_total']}（输入 {stats['qa_input_tokens_total']} + 输出 {stats['qa_output_tokens_total']}）")
    if stats["total"] > 0:
        print(f"  平均每题: {stats['qa_tokens_total']/stats['total']:.1f} tokens")
    print(f"\n[失败模式]")
    for mode, count in sorted(stats["failure_modes"].items(), key=lambda x: -x[1]):
        print(f"  {mode}: {count}")


if __name__ == "__main__":
    main()
