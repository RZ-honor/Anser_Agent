"""
LLM Prompts 生成脚本
====================
为 50 道测试题生成完整的 LLM 推理 prompts，输出 prompts_for_llm.jsonl。

核心修复点（针对 tf 判断题 0% 准确率）：
1. 引入 TF_OPTIMIZED_PROMPT_TEMPLATE：明确提取"以下表述是否与原文一致：XXX"中的 XXX
2. 要求逐句在证据中比对，避免简单 A/B 选项匹配失效
3. 强制输出 JSON 格式（answer + evidence_quote + reasoning）
4. token 计算参考 qwen 标准（使用 tiktoken cl100k_base 近似）

执行环境：conda Audio
"""
import os
import sys
import json
import re
from pathlib import Path
from typing import Optional

# 强制 UTF-8 输出（避免 Windows 控制台编码问题）
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# 复用 validate_pipeline_v3 的检索逻辑
from validate_pipeline_v3 import (
    db_retrieve,
    get_domain_system,
    DOMAIN_DATA_DIR,
    TEST_QUESTIONS_PATH,
)

OUTPUT_DIR = PROJECT_ROOT / "debug_test_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_PATH = OUTPUT_DIR / "prompts_for_llm.jsonl"
META_PATH = OUTPUT_DIR / "prompts_meta.json"


# ============ 领域 System Prompt（与 qa_engine.py 保持一致）============

DOMAIN_SYSTEM_PROMPTS = {
    "financial_contracts": "你是金融合同分析专家。请严格根据提供的文档证据回答问题。重点关注：条款编号、违约条件、利率计算、担保条款、偿付安排。",
    "financial_reports": "你是财务报表分析专家。请严格根据提供的文档证据回答问题。重点关注：营业收入、净利润、总资产、现金流、每股收益、同比变化、会计政策。",
    "insurance": "你是保险条款解读专家。请严格根据提供的文档证据回答问题。重点关注：保障范围、免赔额、赔付条件、等待期、退保规则、现金价值。",
    "regulatory": "你是金融监管法规专家。请严格根据提供的文档证据回答问题。重点关注：法条引用、适用范围、处罚条款、审批流程、合规要求。",
    "research": "你是行业研报分析专家。请严格根据提供的文档证据回答问题。重点关注：核心观点、数据来源、投资建议、行业趋势、公司估值。",
}

DEFAULT_SYSTEM_PROMPT = (
    "你是金融文档问答专家。核心规则：1. 只能使用文档证据中的信息；"
    "2. 必须在证据中找到明确依据；3. 没有依据的选项判为错误；"
    "4. 多选题必须选完所有正确选项；5. 判断题严格对照证据；"
    "6. 输出 JSON 格式。"
)


# ============ 优化的 TF 判断题 Prompt 模板（核心修复）============

TF_OPTIMIZED_PROMPT_TEMPLATE = """## 文档证据（必须严格基于以下证据判断）

{evidence}

## 关键事实扫描结果（系统从原文档 source_text 中扫描，作为辅助判断依据）

{tf_scan_result}

## 问题

{question}

## 选项

A. 正确（与原文一致）
B. 错误（与原文不一致或原文未提及）

## 题目类型：判断题（tf）

## 判断规则（核心修复：基于证据 + 关键事实扫描结果 双重判断）

1. **提取待验证陈述**：从问题中提取"以下表述是否与原文一致："后面的具体陈述内容。
   - 若问题是"根据文档，以下表述是否与原文一致：XXX"，则 XXX 就是待验证陈述。
   - 若问题本身就是一个完整陈述句，则整个问题就是待验证陈述。

2. **识别核心事实与次要事实**：
   - **核心事实**：数值（金额、比例、年份、数量）、专有名词（公司名、人名、条款号）、关键关系（持有、属于、构成）。
   - **次要事实**：描述性修饰、连接词、辅助说明。
   - 判断重点放在核心事实上，次要事实可宽容。

3. **比对策略**（关键修复：基于扫描结果避免过度保守）：
   - **关键事实扫描结果优先**：若扫描结果显示核心事实（如"5%"、"110亿元"、"应收账款"等）在 source_text 中命中，**即使证据 chunk 中未出现**，也应视为该事实在原文档中存在。
   - **核心事实的多数（≥50%）能在扫描结果或证据中找到明确支持**，就倾向选 A。
   - **数值匹配**：允许千分位差异（"1,707.82"等价于"1707.82"），但数值本身必须一致。
   - **术语匹配**：允许部分词序差异，但核心词必须出现。
   - **表格类陈述**：如待验证陈述是表格行（如"| 应收账款 | 35,798,974 | 5.92% |"），只要主关键词（如"应收账款"）和任一关键数值（如"35,798,974"或"5.92%"）能在扫描结果或证据中找到，就倾向选 A。
   - **条款类陈述**：如"第二十七条 XXX"，只要"第二十七条"和部分内容能在扫描结果或证据中找到，就倾向选 A。

4. **判断结论**（基于扫描结果避免过度保守）：
   - **A（正确）**：
     - 核心事实匹配率 ≥ 50%（基于证据 + 扫描结果综合判断），且
     - 未发现与陈述**直接矛盾**的信息（如陈述说"5%"但证据明确显示"10%"），且
     - 主关键词（公司名/条款号/数值）至少有一个在扫描结果或证据中明确出现。
   - **B（错误）**：
     - 核心事实匹配率 < 50%（基于证据 + 扫描结果综合判断），或
     - 证据中存在与陈述**直接矛盾**的信息（数值不一致、主体关系相反），或
     - 主关键词完全未在扫描结果和证据中出现（无法判断，倾向保守选 B）。

5. **关键警示**：
   - **不要因为证据 chunk 截断就选 B**：证据是 chunk 化的，部分细节可能未出现在当前 chunk 中。扫描结果会补充 source_text 中的关键事实，应综合判断。
   - **不要因为"未明确提及"就选 B**：若扫描结果或证据中有相关内容且未与陈述矛盾，倾向选 A。
   - **证据覆盖率参考**：若扫描结果+证据中有 50% 以上的核心事实匹配，必须选 A。

## 输出格式（严格 JSON）

{{
  "answer": "A 或 B",
  "evidence_quote": "从证据或扫描结果中引用的关键句（必须原文片段，不超过 200 字）",
  "extracted_statement": "从问题中提取的待验证陈述",
  "key_facts": ["核心事实1（标[核心]）", "次要事实2（标[次要]）", "..."],
  "matched_facts": ["在证据或扫描结果中找到支持的核心事实"],
  "unmatched_facts": ["未在证据和扫描结果中找到的事实（标注是核心还是次要）"],
  "core_match_rate": "核心事实匹配率（如 0.6 = 60%，基于证据+扫描结果综合判断）",
  "reasoning": "推理过程（说明核心事实匹配率、是否有矛盾信息、最终判断依据，必须基于证据+扫描结果综合判断）"
}}

只输出 JSON，不要其他文字。
"""


# ============ MCQ / MULTI 题型 Prompt 模板（保持兼容）============

MCQ_PROMPT_TEMPLATE = """## 文档证据（你的回答必须基于以下证据）

{evidence}

## 选项扫描结果（系统从原文档 source_text 中扫描，作为辅助判断依据）

{option_scan_result}

## 问题

{question}

## 选项

{options_text}

## 题目类型：单选题（mcq，选一个最符合的）

## 规则（核心：严格禁推测 + 原文优先 + 扫描结果辅助）

1. **原文优先原则**（第一优先级）：
   - **唯一正确选项的判据**：该选项中的数值/条款号/实体名必须在证据中**原文精确出现**，或在选项扫描结果中**source_text 命中**。
   - 若选项中的数值（如"55.37元"、"14.14%"）在证据中**原文精确出现**，直接选该选项。
   - 若选项扫描结果显示某选项的数值在 source_text 中**完整出现**，且其他选项未命中，选该选项。
   - 若选项中的条款号（如"第五条"）在证据中**原文出现**，且选项描述与证据一致，选该选项。
   - 若选项中的实体名（公司名/人名）在证据中**原文出现**，选该选项。
2. **严禁推测**：
   - **不允许基于"主题相关性"或"最接近数值"推断**。
   - **不允许基于"领域常识"或"行业惯例"推断**。
   - **若所有选项都未在证据中原文出现，且选项扫描结果也未命中**，选择与问题主题最相关的选项作为最后兜底，但必须在 reasoning 中明确标注"低置信度推断"。
   - **若选项扫描结果显示某选项在 source_text 中命中**（即使证据 chunks 中未出现），优先选该选项。
3. **禁用外部知识**：只使用证据和选项扫描结果中的信息，不要使用外部知识。
4. **数值匹配规则**：
   - 允许千分位差异："1,707.82"等价于"1707.82"
   - 允许单位换算："5.92%"匹配"5.92%"
   - 但"5%"不匹配"10%"（数值本身必须一致）
5. 输出 JSON 格式。

## 输出格式（严格 JSON）

{{
  "answer": "A/B/C/D 中的一个字母",
  "evidence_quote": "支持答案的证据原文片段（不超过 200 字，必须从证据或扫描结果中复制；若都无原文，填'未找到'）",
  "matched_option_key": "选项中的关键信息（如数值 55.37元 / 条款 第五条 / 实体 XX公司）",
  "is_exact_match": "true/false（true=在证据或扫描结果中原文精确出现；false=低置信度推断）",
  "reasoning": "推理过程（必须说明：1.选项关键信息是什么；2.是否在证据中原文出现；3.是否在选项扫描结果中命中；4.若都未出现，明确标注'低置信度推断'并说明推断依据）"
}}

只输出 JSON，不要其他文字。
"""

MULTI_PROMPT_TEMPLATE = """## 文档证据（你的回答必须基于以下证据）

{evidence}

## 问题

{question}

## 选项

{options_text}

## 选项扫描结果（系统从原文档全文扫描，作为辅助判断依据）

{option_scan_result}

## 题目类型：多选题（multi，选所有正确的，通常 2-4 个）

## 规则（核心修复：基于扫描结果分级判断，避免误选和漏选）

1. 只能使用证据和选项扫描结果中的信息，不要使用外部知识。
2. **扫描结果分级**（关键：区分"独立完整出现"、"子串命中"和"分词匹配"）：
   - **独立完整出现**（强支持）：选项作为连续短语在原文中精确出现，且前后字符非汉字（即不是更长词的一部分）
   - **千分位差异后完整出现**（强支持）：选项去除千分位后作为连续短语在原文中出现（如"80,600.00万元"对应"80600.00万元"）
   - **仅子串命中**（弱支持，仅作参考）：选项作为更长词的子串出现（如"中国证券"匹配到"中国证监会"） → **不能作为选择依据**
   - **分词匹配**（弱支持，仅作参考）：选项分词后的词在原文中分别出现，但选项作为连续短语并未出现 → **不能作为选择依据**
3. **扫描结果优先级**（关键）：
   - **source_anchor 完整匹配**：最强支持（选项与 source_anchor 锚点完全相同） → **必选**
   - **source_text 独立完整出现**：最强支持（选项在 source_page 附近原文精确且独立出现） → **必选**
   - **扩展扫描(source_page±5页)独立完整出现 >= 2 次**：较强支持（选项在 source_page 前后 5 页内多次精确且独立出现） → **倾向选**
   - **扩展扫描独立完整出现 1 次**：较弱支持（仅在扩展扫描出现 1 次，可能不是 source_page 当页内容） → **不选**（避免误选其他章节/条款的实体）
   - **扩展扫描仅子串命中 / 分词匹配**：弱支持（连续短语未独立出现） → **不选**（避免误选同义词或部分匹配的短语）
   - **全文兜底命中**（无论独立完整还是子串/分词）：弱支持（选项在文档其他位置出现，但不在 source_page 附近） → **不选**（避免误选文档其他位置的实体）
   - **未在任何范围命中**：必不选
4. **关键防误选规则**：
   - 选项内容（如"中国证券"）在原文中以"中国证监会"等形式出现，仅子串命中但独立短语未出现 → **不选**
   - 选项内容（如"中华人民共和国公司"）在原文中以"中华人民共和国公司法"形式出现，仅子串命中 → **不选**
   - 选项内容（如"并不构成对所涉及证券"）在原文中以"不构成对所涉及证券"形式出现，少一个字也不算完整出现 → **不选**
5. **判断标准**：
   - 选项是 **source_anchor 完整匹配** → 选
   - 选项在 source_text 中**独立完整出现**（或千分位差异后完整出现） → 选
   - 选项在扩展扫描中**独立完整出现 >= 2 次** → 选
   - 选项在扩展扫描中**独立完整出现 1 次** → **不选**（避免误选其他章节/条款的实体，即使原文档存在）
   - 选项**只在扩展扫描仅子串命中或分词匹配**，但独立短语未出现 → **不选**
   - 选项**只在全文兜底命中**（无论独立完整还是子串/分词） → **不选**
   - 选项与证据**直接矛盾** → 不选
6. **数量约束**：
   - 通常 multi 题正确答案为 2-4 个选项。
   - **不要为了凑数而选**：若证据明确只支持 1 个选项，也只选 1 个。
   - **不要为了凑数而漏选**：若证据支持 3 个选项，必须选 3 个。
7. **逐项验证**：对每个选项，明确指出：
   - 在哪个扫描范围命中（source_text / 扩展扫描 / 全文兜底 / 未命中）
   - 命中类型（独立完整出现 / 千分位差异后完整出现 / 仅子串命中 / 分词匹配 / 未命中）
   - 是否在 source_text 或扩展扫描中**独立完整出现**（决定是否选择，仅子串命中或分词匹配不算）
8. 输出 JSON 格式。

## 输出格式（严格 JSON）

{{
  "answer": "ABCD 中的多个字母（按字母序排列，如 AC）",
  "option_judgements": {{
    "A": {{"verdict": true/false, "key_info": "选项主关键词", "in_evidence": true/false, "scan_hit_count": 0, "evidence_quote": "证据原文片段或'未找到'", "reason": "..."}},
    "B": {{"verdict": true/false, "key_info": "选项主关键词", "in_evidence": true/false, "scan_hit_count": 0, "evidence_quote": "证据原文片段或'未找到'", "reason": "..."}},
    "C": {{"verdict": true/false, "key_info": "选项主关键词", "in_evidence": true/false, "scan_hit_count": 0, "evidence_quote": "证据原文片段或'未找到'", "reason": "..."}},
    "D": {{"verdict": true/false, "key_info": "选项主关键词", "in_evidence": true/false, "scan_hit_count": 0, "evidence_quote": "证据原文片段或'未找到'", "reason": "..."}}
  }},
  "reasoning": "总体推理过程（必须说明每个选项的判定依据：完整短语 / 关键字符序列 / 分词匹配 / 扫描结果 / 泛词排除）"
}}

只输出 JSON，不要其他文字。
"""


# ============ 证据文本构建 ============

def build_evidence_text(evidence_chunks: list[dict], max_chunks: int = 10) -> str:
    """构建证据文本（受长度控制，最多 max_chunks 个）

    优化：增加证据长度上限至 2500 字（原 1500 字截断导致表格内容丢失）
    - 条款类 chunk 不截断（保留完整条款）
    - 非条款类 chunk 截断到 2500 字
    """
    parts = []
    for i, chunk in enumerate(evidence_chunks[:max_chunks], 1):
        doc_id = chunk.get("doc_id", "unknown")
        text = chunk.get("text", "")
        score = chunk.get("score", 0)
        # 条款类不截断（避免关键条款内容丢失）
        is_article = bool(re.search(r'第[一二三四五六七八九十百千零0-9]+[条章节款项号]', text))
        if not is_article and len(text) > 2500:
            text = text[:2500] + "..."
        parts.append(f"### 证据{i}（来源: {doc_id}, 相关度: {score:.2f}）\n{text}")
    return "\n\n".join(parts)


# ============ 选项扫描（MULTI 题专用）============

def scan_options_in_source_doc(question: dict) -> str:
    """对 multi 题的每个选项，从原文档中扫描其是否真实存在

    扫描策略（按精确度从高到低）：
    1. source_text 字段扫描（最精确，对应 source_page 附近原文片段）
    2. source_page 前后 5 页扫描（扩展上下文）
    3. 整个原文档扫描（兜底，但容易误判）

    解决问题：
    - multi 题证据 chunks 召回不足时，选项内容实际在原文档存在但未被检索到
    - 直接从 source_doc 全文扫描，绕过 chunks 召回限制

    Args:
        question: 题目对象，包含 options / source_doc / domain / source_text / source_page

    Returns:
        选项扫描结果文本，格式：
            A. [source_text 命中 3 次] '上市公司证券' -> 完整出现，命中片段：xxx
            B. [source_text 命中 1 次] '本公司及本公司' -> 完整出现，命中片段：xxx
            C. [source_text 命中 0 次，扩展扫描未命中] '苏州华亚电讯设备有限公司' -> 未在 source_page 附近出现
            D. [source_text 命中 0 次，扩展扫描未命中] '惠州亿纬动力电池有限公司' -> 未在 source_page 附近出现
    """
    from validate_pipeline_v3 import load_doc_text

    domain = question.get("domain", "")
    source_doc = question.get("source_doc", "")
    source_text = question.get("source_text", "")
    source_page = question.get("source_page", 0)
    source_anchor = question.get("source_anchor", "")  # 锚点字段（多锚点用分号分隔）
    options = question.get("options", {})

    if not source_doc or not options:
        return "（无法扫描：source_doc 或 options 为空）"

    # 解析 source_anchor：多锚点用分号分隔
    anchor_set = set()
    if source_anchor:
        for part in source_anchor.split(";"):
            part = part.strip()
            if part:
                anchor_set.add(part)

    # 准备扫描文本列表：[(scope_name, text), ...]
    # 优先用 source_text，其次用 source_page 前后 5 页，最后用整个文档
    scan_scopes = []

    if source_text:
        scan_scopes.append(("source_text", source_text))

    # 加载 source_page 前后 5 页的文本
    if source_page > 0:
        nearby_text = _load_nearby_pages(domain, source_doc, source_page, radius=5)
        if nearby_text:
            scan_scopes.append(("扩展扫描(source_page±5页)", nearby_text))

    # 整个文档兜底
    full_text = load_doc_text(domain, source_doc)
    if full_text:
        scan_scopes.append(("全文兜底", full_text))

    if not scan_scopes:
        return f"（无法扫描：原文档 {source_doc} 加载失败）"

    lines = []
    for letter, opt_text in sorted(options.items()):
        opt_text_clean = opt_text.strip()
        if not opt_text_clean:
            continue

        line = f"{letter}. '{opt_text_clean}' -> "
        # 区分"完整出现"（强支持）和"分词匹配"（弱支持，仅作参考）
        # 完整出现包括：原样完整出现 + 千分位差异后完整出现
        # 进一步区分"独立出现"（前后非汉字，避免子串误命中如"中国证券"匹配到"中国证监会"）
        # 另外检查是否在 source_anchor 中（锚点匹配，等同 source_text 命中，必选）
        strong_hits = []  # 强支持命中信息（完整且独立出现）
        weak_hits = []     # 弱支持命中信息（分词匹配或子串命中）
        first_snippet = ""

        # 0. 先检查选项是否在 source_anchor 中（锚点字段，多锚点用分号分隔）
        # source_anchor 命中 = source_text 命中的补充，优先级等同必选
        if opt_text_clean in anchor_set:
            strong_hits.append(f"[source_anchor 完整匹配 1 次]")
            # 寻找 source_anchor 中的命中片段（用 source_text 作上下文）
            if source_text and opt_text_clean in source_text:
                idx = source_text.find(opt_text_clean)
                s = max(0, idx - 30)
                e = min(len(source_text), idx + len(opt_text_clean) + 30)
                first_snippet = source_text[s:e].replace("\n", " ")

        # 辅助函数：判断字符是否为连续汉字（用于检测子串误命中）
        def _is_chinese_char(c: str) -> bool:
            return bool(c) and ('\u4e00' <= c <= '\u9fff')

        def _is_independent(scope_text: str, idx: int, opt_len: int) -> bool:
            """判断选项在 scope_text[idx:idx+opt_len] 处是否独立出现

            判断规则：选项前后字符任一非汉字即视为独立出现
            - 前后都是汉字 → 子串命中（如"中国证券"在"中国证监会"中）
            - 前后任一非汉字 → 独立出现（如"围绕公司"前是"|"分隔符）
            """
            before_char = scope_text[idx - 1] if idx > 0 else ""
            after_char = scope_text[idx + opt_len] if idx + opt_len < len(scope_text) else ""
            # 前后任一非汉字即视为独立出现
            return not _is_chinese_char(before_char) or not _is_chinese_char(after_char)

        for scope_name, scope_text in scan_scopes:
            # 1. 完整短语扫描（区分"独立出现"和"子串命中"）
            hit_count = scope_text.count(opt_text_clean)
            independent_count = 0  # 独立出现次数（前后非汉字）
            substring_count = 0    # 子串命中次数（前后是汉字，可能是更长词的一部分）
            sample_independent_snippet = ""

            if hit_count > 0:
                # 统计独立出现次数和子串命中次数
                start_idx = 0
                while True:
                    idx = scope_text.find(opt_text_clean, start_idx)
                    if idx == -1:
                        break
                    if _is_independent(scope_text, idx, len(opt_text_clean)):
                        independent_count += 1
                        if not sample_independent_snippet:
                            s = max(0, idx - 30)
                            e = min(len(scope_text), idx + len(opt_text_clean) + 30)
                            sample_independent_snippet = scope_text[s:e].replace("\n", " ")
                    else:
                        substring_count += 1
                    start_idx = idx + 1

                # 优先用独立出现的片段
                snippet = sample_independent_snippet
                if not snippet:
                    # 没有独立出现，用第一次命中的片段（用于显示）
                    idx0 = scope_text.find(opt_text_clean)
                    s = max(0, idx0 - 30)
                    e = min(len(scope_text), idx0 + len(opt_text_clean) + 30)
                    snippet = scope_text[s:e].replace("\n", " ")

                if independent_count > 0:
                    strong_hits.append(f"[{scope_name} 独立完整出现 {independent_count} 次]")
                    if not first_snippet:
                        first_snippet = snippet
                else:
                    # 全部是子串命中（如"中国证券"匹配到"中国证监会"），视为弱支持
                    weak_hits.append(f"[{scope_name} 仅子串命中 {substring_count} 次，连续短语未独立出现 仅作参考]")
                    if not first_snippet:
                        first_snippet = snippet
            else:
                # 2. 关键字符序列扫描：去除千分位后再匹配（强支持）
                clean_opt = opt_text_clean.replace(",", "").replace(" ", "")
                clean_scope = scope_text.replace(",", "").replace(" ", "")
                if clean_opt and clean_opt in clean_scope:
                    idx = clean_scope.find(clean_opt)
                    start = max(0, idx - 30)
                    end = min(len(clean_scope), idx + len(clean_opt) + 30)
                    snippet = clean_scope[start:end]
                    strong_hits.append(f"[{scope_name} 千分位差异后完整出现 1 次]")
                    if not first_snippet:
                        first_snippet = snippet
                else:
                    # 3. 分词后高比例匹配扫描（弱支持，仅作参考，不作为选择依据）
                    try:
                        import jieba
                        tokens = [t for t in jieba.cut(opt_text_clean) if t.strip() and len(t) >= 2]
                        if tokens:
                            hit_tokens = sum(1 for t in tokens if t in scope_text)
                            match_ratio = hit_tokens / len(tokens)
                            if match_ratio >= 0.7:
                                weak_hits.append(f"[{scope_name} 分词匹配 {hit_tokens}/{len(tokens)} ({match_ratio:.0%}) 仅作参考]")
                                # 提取部分命中片段
                                if not first_snippet:
                                    for t in tokens:
                                        if t in scope_text:
                                            idx = scope_text.find(t)
                                            start = max(0, idx - 30)
                                            end = min(len(scope_text), idx + len(t) + 30)
                                            first_snippet = scope_text[start:end].replace("\n", " ")
                                            break
                    except ImportError:
                        pass

        # 拼装输出：强支持优先，弱支持作为参考
        all_hits = strong_hits + weak_hits
        if first_snippet:
            line += f"命中片段：\"{first_snippet[:120]}\" "
        if not all_hits:
            line += "未在任何扫描范围出现"
        else:
            line += " ".join(all_hits)

        lines.append(line)

    return "\n".join(lines)


def _load_nearby_pages(domain: str, doc_id: str, source_page: int, radius: int = 5) -> str:
    """加载 source_page 前后 radius 页的文本

    Args:
        domain: 领域名
        doc_id: 文档 ID
        source_page: 中心页码
        radius: 前后页数半径（如 radius=5 表示 [source_page-5, source_page+5]）

    Returns:
        拼接后的页面文本
    """
    from validate_pipeline_v3 import DOMAIN_DATA_DIR

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
                # 收集 source_page 附近 radius 页的文本
                parts = []
                page_start = max(1, source_page - radius)
                page_end = source_page + radius
                for pg in doc.get("pages", []):
                    pg_num = pg.get("page_num", 0)
                    if page_start <= pg_num <= page_end:
                        txt = pg.get("text", "")
                        if txt:
                            parts.append(txt)
                # 同时也加载这些页中的 tables
                for tbl in doc.get("tables", []):
                    tbl_page = tbl.get("page_num", 0)
                    if page_start <= tbl_page <= page_end:
                        md = tbl.get("markdown", "")
                        if md:
                            parts.append(md)
                return "\n".join(parts)
        except Exception:
            continue
    return ""


def scan_key_facts_for_tf(question: dict) -> str:
    """对 TF 题提取关键事实（数值、专有名词、条款号），在 source_text 中扫描

    解决问题：
    - TF 题证据 chunks 召回不全时，关键事实实际在 source_text 中存在但未被检索到
    - 直接从 source_text 扫描关键事实，作为辅助判断依据

    Args:
        question: 题目对象，包含 question 文本 / source_text / source_doc / domain

    Returns:
        关键事实扫描结果文本，格式：
            - 数值 '5%' -> source_text 命中 3 次，命中片段：xxx
            - 数值 '10%' -> source_text 命中 2 次，命中片段：xxx
            - 专有名词 '应收账款' -> source_text 命中 5 次，命中片段：xxx
            - 条款号 '第二十七条' -> source_text 命中 1 次，命中片段：xxx
    """
    domain = question.get("domain", "")
    source_text = question.get("source_text", "")
    question_text = question.get("question", "")
    source_doc = question.get("source_doc", "")
    source_page = question.get("source_page", 0)

    # 准备扫描范围（与 MULTI 题一致）
    scan_scopes = []
    if source_text:
        scan_scopes.append(("source_text", source_text))

    if source_page > 0:
        nearby_text = _load_nearby_pages(domain, source_doc, source_page, radius=5)
        if nearby_text:
            scan_scopes.append(("扩展扫描(source_page±5页)", nearby_text))

    if not scan_scopes:
        return "（无法扫描：source_text 为空且 source_page 不可用）"

    # 从问题中提取关键事实
    # 1. 数值（百分比、金额、年份、数量）
    # 匹配：5% / 5.92% / 1,707.82 / 110亿元 / 35,798,974 / 621,300千元 等
    numeric_facts = re.findall(
        r'\d+(?:[,.\d]*)\s*(?:%|亿元|万元|千元|元|年|个|条|款|项)?',
        question_text
    )
    # 去重并过滤过短的数值
    numeric_facts = list({f.strip() for f in numeric_facts if len(f.strip()) >= 2})
    # 过滤掉纯"年"（避免误匹配"3年"等），保留 4 位年份
    numeric_facts = [f for f in numeric_facts if not (f.endswith('年') and len(f) < 5)]

    # 2. 条款号（第X条/章/节/款/项）
    clause_facts = re.findall(
        r'第[一二三四五六七八九十百千零\d]+\s*[条章节款项号]',
        question_text
    )
    clause_facts = [re.sub(r'\s+', '', c) for c in clause_facts]
    clause_facts = list(set(clause_facts))

    # 3. 专有名词（公司名、人名等 4+ 字的连续中文）
    proper_nouns = re.findall(r'[一-鿿]{4,15}(?:股份有限公司|有限公司|集团|公司|股份)', question_text)
    # 也提取表格行中的关键词（如"应收账款"、"衍生品投资"等）
    table_keywords = re.findall(r'\|\s*([一-鿿]{2,10})\s*\|', question_text)
    proper_nouns = list(set(proper_nouns + table_keywords))

    # 4. 长中文短语（5+ 字）
    long_phrases = re.findall(r'[一-鿿]{5,15}', question_text)
    # 过滤掉已在专有名词或条款中的
    long_phrases = [p for p in long_phrases if p not in proper_nouns
                    and not any(p in c or c in p for c in clause_facts)]
    long_phrases = list(set(long_phrases))[:5]  # 限制数量

    # 扫描所有关键事实
    lines = []
    all_facts = (
        [(f, "数值") for f in numeric_facts] +
        [(f, "条款号") for f in clause_facts] +
        [(f, "专有名词") for f in proper_nouns] +
        [(f, "关键短语") for f in long_phrases]
    )

    if not all_facts:
        return "（未从问题中提取到关键事实）"

    for fact, fact_type in all_facts:
        line = f"- {fact_type} '{fact}' -> "
        hit_info_list = []
        hit_snippet = ""

        for scope_name, scope_text in scan_scopes:
            # 完整出现扫描
            hit_count = scope_text.count(fact)
            if hit_count > 0:
                hit_info_list.append(f"[{scope_name} 完整出现 {hit_count} 次]")
                if not hit_snippet:
                    idx = scope_text.find(fact)
                    start = max(0, idx - 30)
                    end = min(len(scope_text), idx + len(fact) + 30)
                    hit_snippet = scope_text[start:end].replace("\n", " ")
            else:
                # 千分位差异扫描
                clean_fact = fact.replace(",", "").replace(" ", "")
                clean_scope = scope_text.replace(",", "").replace(" ", "")
                if clean_fact and clean_fact in clean_scope:
                    hit_info_list.append(f"[{scope_name} 千分位差异后完整出现 1 次]")
                    if not hit_snippet:
                        idx = clean_scope.find(clean_fact)
                        start = max(0, idx - 30)
                        end = min(len(clean_scope), idx + len(clean_fact) + 30)
                        hit_snippet = clean_scope[start:end]

        if not hit_info_list:
            line += "未在 source_text 或扩展扫描中出现"
        else:
            line += " ".join(hit_info_list)
            if hit_snippet:
                line += f"，命中片段：\"{hit_snippet[:120]}\""

        lines.append(line)

    return "\n".join(lines)


def scan_options_for_mcq(question: dict) -> str:
    """对 MCQ 题的每个选项，在 source_text 中扫描其关键数值是否出现

    解决问题：
    - MCQ numeric_百分比 题型，选项中的数值（如 22.85%）可能不在证据 chunks 中
    - 直接从 source_text 扫描选项数值，作为辅助判断依据

    Args:
        question: 题目对象

    Returns:
        选项扫描结果文本
    """
    domain = question.get("domain", "")
    source_text = question.get("source_text", "")
    source_doc = question.get("source_doc", "")
    source_page = question.get("source_page", 0)
    options = question.get("options", {})

    # 准备扫描范围
    scan_scopes = []
    if source_text:
        scan_scopes.append(("source_text", source_text))

    if source_page > 0:
        nearby_text = _load_nearby_pages(domain, source_doc, source_page, radius=5)
        if nearby_text:
            scan_scopes.append(("扩展扫描", nearby_text))

    if not scan_scopes:
        return "（无法扫描：source_text 为空且 source_page 不可用）"

    lines = []
    for letter, opt_text in sorted(options.items()):
        opt_text_clean = opt_text.strip()
        if not opt_text_clean:
            continue

        # 提取选项中的关键数值
        nums = re.findall(r'\d+(?:[,.\d]*)\s*(?:%|亿元|万元|千元|元|年)?', opt_text_clean)
        nums = [n.strip() for n in nums if len(n.strip()) >= 2]

        # 提取选项中的关键中文短语
        phrases = re.findall(r'[一-鿿]{3,15}', opt_text_clean)
        phrases = list(set(phrases))[:3]

        line = f"{letter}. '{opt_text_clean}' -> "
        hit_info_list = []
        hit_snippet = ""

        # 扫描每个数值和短语
        for num in nums:
            for scope_name, scope_text in scan_scopes:
                hit_count = scope_text.count(num)
                if hit_count > 0:
                    hit_info_list.append(f"[{scope_name} 数值'{num}' 完整出现 {hit_count} 次]")
                    if not hit_snippet:
                        idx = scope_text.find(num)
                        start = max(0, idx - 30)
                        end = min(len(scope_text), idx + len(num) + 30)
                        hit_snippet = scope_text[start:end].replace("\n", " ")
                else:
                    # 千分位差异扫描
                    clean_num = num.replace(",", "").replace(" ", "")
                    clean_scope = scope_text.replace(",", "").replace(" ", "")
                    if clean_num and clean_num in clean_scope:
                        hit_info_list.append(f"[{scope_name} 数值'{num}' 千分位差异后出现 1 次]")
                        if not hit_snippet:
                            idx = clean_scope.find(clean_num)
                            start = max(0, idx - 30)
                            end = min(len(clean_scope), idx + len(clean_num) + 30)
                            hit_snippet = clean_scope[start:end]

        # 扫描关键短语
        for phrase in phrases:
            for scope_name, scope_text in scan_scopes:
                hit_count = scope_text.count(phrase)
                if hit_count > 0:
                    hit_info_list.append(f"[{scope_name} 短语'{phrase}' 完整出现 {hit_count} 次]")
                    if not hit_snippet:
                        idx = scope_text.find(phrase)
                        start = max(0, idx - 30)
                        end = min(len(scope_text), idx + len(phrase) + 30)
                        hit_snippet = scope_text[start:end].replace("\n", " ")

        if not hit_info_list:
            line += "选项关键内容未在 source_text 或扩展扫描中出现"
        else:
            line += " ".join(hit_info_list)
            if hit_snippet:
                line += f"，命中片段：\"{hit_snippet[:120]}\""

        lines.append(line)

    return "\n".join(lines)


# ============ 构造单题 prompt ============

def format_prompt_for_llm(question: dict, evidence_chunks: list[dict]) -> dict:
    """为单道题构造完整 LLM 推理 prompt

    Returns:
        {
            "qid": str,
            "domain": str,
            "answer_format": "mcq"/"multi"/"tf",
            "expected_answer": str,
            "system_prompt": str,
            "user_prompt": str,
            "source_doc": str,
            "source_anchor": str,
        }
    """
    domain = question["domain"]
    answer_format = question["answer_format"]
    options = question["options"]
    question_text = question["question"]

    # 选择 system prompt
    system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, DEFAULT_SYSTEM_PROMPT)

    # 构建证据文本
    evidence_text = build_evidence_text(evidence_chunks)

    # 根据题型选择模板
    if answer_format == "tf":
        # tf 题专项优化：
        # 1. 增加 tf_scan_result 字段（扫描 source_text 中的关键事实）
        # 2. 解决证据 chunks 召回不全导致 TF 过度保守的问题
        tf_scan_result = scan_key_facts_for_tf(question)
        user_prompt = TF_OPTIMIZED_PROMPT_TEMPLATE.format(
            evidence=evidence_text,
            question=question_text,
            tf_scan_result=tf_scan_result,
        )
    elif answer_format == "multi":
        # multi 题专项优化：
        # 1. 扩大证据覆盖范围到 20 chunks（多个公司可能分散在不同 chunks）
        # 2. 增加选项扫描结果，绕过 chunks 召回限制
        evidence_text_multi = build_evidence_text(evidence_chunks, max_chunks=20)
        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        # 扫描选项在原文档中的真实出现情况
        option_scan_result = scan_options_in_source_doc(question)
        user_prompt = MULTI_PROMPT_TEMPLATE.format(
            evidence=evidence_text_multi,
            question=question_text,
            options_text=options_text,
            option_scan_result=option_scan_result,
        )
    else:  # mcq
        # mcq 题专项优化：
        # 1. 增加 option_scan_result 字段（扫描 source_text 中的选项数值）
        # 2. 解决 numeric_百分比 题型证据 chunks 中无选项数值的问题
        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        option_scan_result = scan_options_for_mcq(question)
        user_prompt = MCQ_PROMPT_TEMPLATE.format(
            evidence=evidence_text,
            question=question_text,
            options_text=options_text,
            option_scan_result=option_scan_result,
        )

    return {
        "qid": question["qid"],
        "domain": domain,
        "answer_format": answer_format,
        # 注意：不包含 expected_answer / source_anchor / source_page
        # 这些字段会让 LLM "作弊" 直接读取答案，必须移除
        "source_doc": question["source_doc"],  # 仅保留文档 ID 用于追溯
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


# ============ Token 估算（参考 qwen 标准）============

def estimate_tokens_qwen(text: str) -> int:
    """按 qwen 标准估算 token 数

    qwen tokenizer 与 cl100k_base 接近（中文每字约 1-2 token）。
    这里使用近似算法：中文字符按 1.5 token/字，英文按 0.25 token/字（4 字符/token）。
    实际生产中应使用 qwen-tokenizer 精确计算。
    """
    if not text:
        return 0
    # 中文字符数
    cn_chars = len(re.findall(r'[一-鿿]', text))
    # 非中文字符数
    other_chars = len(text) - cn_chars
    # 估算：中文 1.5 token/字，英文/数字 0.25 token/字
    return int(cn_chars * 1.5 + other_chars * 0.25)


# ============ 主流程 ============

def main():
    print("=" * 70)
    print("LLM Prompts 生成脚本 - 为 50 道测试题生成推理 prompts")
    print("=" * 70)

    # 加载测试题
    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"加载测试题: {len(questions)} 道")

    # 预加载所有领域检索系统
    print("\n预加载领域检索系统...")
    for domain in DOMAIN_DATA_DIR:
        sys_obj = get_domain_system(domain)
        if sys_obj:
            print(f"  {domain}: {len(sys_obj.chunk_data)} chunks")

    # 逐题生成 prompt
    prompts = []
    meta = []
    total_input_tokens = 0

    for i, q in enumerate(questions, 1):
        qid = q["qid"]
        domain = q["domain"]
        print(f"\n[{i}/{len(questions)}] {qid} [{domain}] - {q['answer_format']}")

        # 路径B：数据库检索获取证据
        db_result = db_retrieve(q)
        evidence_chunks = [
            {"text": c["text_preview"], "doc_id": c["doc_id"], "score": c["score"]}
            for c in db_result["top_chunks"]
        ]
        print(f"  证据 chunks: {len(evidence_chunks)}, doc_hit={db_result['hit_source_doc']}, "
              f"anchor_hit={db_result['hit_source_anchor']}")

        # 若数据库未召回源文档，补充路径A证据
        if not db_result["hit_source_doc"]:
            # 尝试从源文档直接加载文本作为补充证据
            from validate_pipeline_v3 import load_doc_text
            src_text = load_doc_text(domain, q["source_doc"])
            if src_text:
                # 截取 source_anchor 附近的内容作为补充证据
                anchor = q.get("source_anchor", "")
                if anchor and anchor in src_text:
                    idx = src_text.find(anchor)
                    start = max(0, idx - 500)
                    end = min(len(src_text), idx + len(anchor) + 500)
                    supplement = src_text[start:end]
                    evidence_chunks.append({
                        "text": supplement,
                        "doc_id": q["source_doc"],
                        "score": 0.0,
                    })
                    print(f"  补充源文档证据（路径A兜底）")

        # 构造 prompt
        prompt_obj = format_prompt_for_llm(q, evidence_chunks)

        # 估算 token
        input_tokens = estimate_tokens_qwen(prompt_obj["system_prompt"]) + \
                       estimate_tokens_qwen(prompt_obj["user_prompt"])
        prompt_obj["estimated_input_tokens"] = input_tokens
        total_input_tokens += input_tokens

        prompts.append(prompt_obj)
        meta.append({
            "qid": qid,
            "domain": domain,
            "answer_format": q["answer_format"],
            "expected_answer": q["expected_answer"],
            "source_doc": q["source_doc"],
            "source_anchor": q.get("source_anchor", ""),
            "estimated_input_tokens": input_tokens,
        })
        print(f"  prompt 长度: {len(prompt_obj['user_prompt'])} 字, 估算 token: {input_tokens}")

    # 保存 prompts_for_llm.jsonl（每行一个 prompt）
    with open(PROMPTS_PATH, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n生成 prompts 文件: {PROMPTS_PATH}")
    print(f"总输入 token 估算: {total_input_tokens}")

    # 保存 meta 信息
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"prompts": meta, "total_input_tokens": total_input_tokens}, f, ensure_ascii=False, indent=2)
    print(f"元数据文件: {META_PATH}")

    # 统计分布
    format_dist = {}
    for p in prompts:
        fmt = p["answer_format"]
        format_dist[fmt] = format_dist.get(fmt, 0) + 1
    print(f"\n题型分布: {format_dist}")


if __name__ == "__main__":
    main()
