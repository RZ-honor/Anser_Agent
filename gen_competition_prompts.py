"""比赛 100 题 prompts 生成器（跨文档对比专用）

与 debug_test_v3 的单文档题不同，比赛题是跨文档对比题：
- 每题涉及 2 个文档（doc_ids），选项常含"第一份文档"/"第二份文档"/"两份文档均"等表述
- 没有 source_doc/source_page/source_text/source_anchor 字段

本生成器策略：
1. 从 competition_evidence_compact/{domain}.json 加载 100 题（已含 evidence_top3 证据）
2. 对每题加载 doc_ids 中所有文档全文
3. 对每个选项：
   - 识别选项涉及的文档（基于"第一份"/"第二份"/"两份"等关键词）
   - 提取选项中的关键事实（数值、专有名词、条款号）
   - 在对应文档中扫描，区分独立完整出现/子串命中/分词匹配
4. 生成跨文档对比专用 prompt
"""
import json
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).parent
EVIDENCE_DIR = BASE / "competition_evidence_compact"
DATA_DIR = BASE / "data"
OUTPUT_PROMPTS = BASE / "debug_test_v3" / "prompts_for_competition.jsonl"

# ====== 5 个领域的数据目录 ======
DOMAIN_DATA_DIR = {
    "financial_contracts": DATA_DIR / "financial_contracts",
    "financial_reports": DATA_DIR / "financial_reports",
    "insurance": DATA_DIR / "insurance",
    "regulatory": DATA_DIR / "regulatory",
    "research": DATA_DIR / "research",
}

# 额外的监管法规文档目录（部分 csrc_*.json 在此）
EXTRA_REG_DIR = Path(r"D:\PROJECT\tianci\1")


# ====== 文档加载与缓存 ======
_DOC_TEXT_CACHE = {}  # {(domain, doc_id): full_text} 缓存避免重复加载


def _normalize_latex(text: str) -> str:
    """去除 LaTeX 转义符，统一数值与单位格式

    修复点：
    1. 去除 LaTeX 包裹符 \\( ... \\)
    2. 去除百分号转义符 \\%
    3. 去除其他常见 LaTeX 转义符（\\, 等）
    4. 统一数字与中文单位之间的空格（"10 亿元"→"10亿元"）

    注：text10 等文档大量使用 \\(43.24\\%) 形式包裹数值，不规范化会导致
    选项"资产负债率为 43.24%"扫描失败。
    """
    if not text:
        return ""
    # 去除 LaTeX 包裹符 \( ... \)
    text = re.sub(r'\\\(\s*', '', text)
    text = re.sub(r'\s*\\\)', '', text)
    # 去除百分号转义符 \%
    text = re.sub(r'\\%', '%', text)
    # 去除其他常见 LaTeX 转义符
    text = re.sub(r'\\,', '', text)
    text = re.sub(r'\\ ', ' ', text)
    # 统一数字与中文单位之间的空格（去除空格）
    text = re.sub(r'(\d)\s+(万元|亿元|元/股|元|股|张|年|月|日|倍|个|名|人|%)', r'\1\2', text)
    return text


def load_doc_text(domain: str, doc_id: str) -> str:
    """加载指定文档的全部文本（带缓存，并已 LaTeX 规范化）

    查找顺序：
    1. data/{domain}/{doc_id}.json
    2. D:/PROJECT/tianci/1/{doc_id}.json（额外监管法规目录）
    """
    cache_key = (domain, doc_id)
    if cache_key in _DOC_TEXT_CACHE:
        return _DOC_TEXT_CACHE[cache_key]

    # 在领域数据目录中查找
    ddir = DOMAIN_DATA_DIR.get(domain)
    candidates = []
    if ddir and ddir.exists():
        candidates.append(ddir / f"{doc_id}.json")
    # 额外目录（主要针对 regulatory 领域的 csrc_*.json）
    if EXTRA_REG_DIR.exists():
        candidates.append(EXTRA_REG_DIR / f"{doc_id}.json")

    for fpath in candidates:
        if not fpath.exists():
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                doc = json.load(f)
            # 校验 doc_id 一致（部分文件 doc_id 字段可能缺失）
            if doc.get("doc_id") and doc.get("doc_id") != doc_id:
                continue
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
            # LaTeX 规范化后再缓存，确保缓存的是规范化版本
            full_text = _normalize_latex("\n".join(parts))
            _DOC_TEXT_CACHE[cache_key] = full_text
            return full_text
        except Exception:
            continue

    _DOC_TEXT_CACHE[cache_key] = ""
    return ""


# ====== 选项关键事实提取 ======
def extract_key_facts(text: str) -> list:
    """从选项文本中提取关键事实

    提取策略：
    1. 数值（百分比、金额、年份、数量）
    2. 条款号（第X条/章/节/款/项）
    3. 信用评级（AAA, AA+, BBB- 等）
    4. 专有名词（公司名、人名等 4+ 字的连续中文，含"公司"/"集团"等后缀）
    5. 长中文短语（5+ 字）

    过滤：
    - 排除"第一份文档"/"第二份文档"/"两份文档"等指代词
    - 排除"本期债券"/"发行人"等通用词（长度<=4）
    """
    facts = []

    # 0. 先移除指代词和通用词，避免干扰专有名词提取
    # 例如"第一份文档明确指定国信证券股份有限公司为受托管理人"
    # 移除"第二份文档"后，才能正确提取"国信证券股份有限公司"
    text_clean = text
    clean_patterns = [
        r'第[一二两]份文档?', r'两份文档', r'两个文档', r'两份文件',
        r'第一份', r'第二份', r'下列', r'以下', r'上述', r'该等',
        r'募集说明书', r'本期债券', r'本期发行',
        r'明确指定', r'明确指出', r'明确标注', r'明确提到',
        # 移除连接性短语，避免贪婪匹配把整个句子都捕获为公司名
        r'发行人名称为', r'发行人', r'名称为', r'受托管理人',
        r'主承销商', r'簿记管理人', r'联席主承销商',
    ]
    for pat in clean_patterns:
        text_clean = re.sub(pat, '', text_clean)

    # 1. 数值（含百分比、金额、年份等）- 从原文提取（保留完整数值信息）
    numeric_facts = re.findall(r'\d+(?:[,.\d]*)\s*(?:%|亿元|万元|千元|元|年|个|条|款|项)?', text)
    facts.extend([f for f in numeric_facts if len(f) >= 2])

    # 2. 条款号（第X条/章/节/款/项/项）
    clause_facts = re.findall(r'第[一二三四五六七八九十百千零\d]+\s*[条章节款项号]', text)
    facts.extend(clause_facts)

    # 3. 信用评级（AAA, AA+, AA, AA-, A+, BBB+ 等字母+符号组合）
    rating_facts = re.findall(r'\bA{1,4}[+-]?\b|\bB{1,4}[+-]?\b', text)
    facts.extend(rating_facts)

    # 4. 专有名词（公司名、人名等 4+ 字的连续中文，含"公司"/"集团"等后缀）
    # 使用贪婪匹配（尽可能长地匹配），并从清理后的文本提取
    # 贪婪匹配确保"广晟控股集团有限公司"作为整体被提取，而不是只匹配"有限公司"
    proper_nouns = re.findall(r'[一-鿿]{2,20}(?:股份有限公司|有限公司|集团|公司|股份)', text_clean)
    # 过滤掉太短的通用词（如"有限公司"只有4字且太通用）
    proper_nouns = [p for p in proper_nouns if len(p) >= 6]
    # 去掉前导的"的"/"是"/"为"等连接字符（贪婪匹配可能把"的是广东省..."作为整体捕获）
    # 关键修复：循环去掉所有前导连接字符，而不是只去一个
    def _strip_leading_connectors(s: str) -> str:
        """循环去掉所有前导连接字符（的/为/是/和/与/及）"""
        while s and s[0] in '的为是和与及':
            s = s[1:]
        return s
    proper_nouns = [_strip_leading_connectors(p) for p in proper_nouns]
    proper_nouns = [p for p in proper_nouns if len(p) >= 6]
    facts.extend(proper_nouns)

    # 5. 长中文短语（5+ 字，排除上述已提取的）- 从清理后的文本提取
    long_phrases = re.findall(r'[一-鿿]{5,15}', text_clean)
    for p in long_phrases:
        if p not in facts and not any(p in f or f in p for f in facts):
            facts.append(p)

    # 过滤：排除剩余的通用词（如"本期债券"如果还残留）
    # 注意：大部分通用词已在 clean_patterns 中移除，这里仅作兜底
    filter_patterns = [
        r'本期债券', r'本期发行',
        r'文档', r'文件',
    ]
    filtered_facts = []
    for f in facts:
        if not f or len(f) < 2:
            continue
        skip = False
        for pat in filter_patterns:
            if re.search(pat, f):
                skip = True
                break
        if not skip:
            filtered_facts.append(f)

    # 去重并保持顺序
    seen = set()
    unique_facts = []
    for f in filtered_facts:
        if f not in seen:
            seen.add(f)
            unique_facts.append(f)
    return unique_facts


# ====== 选项扫描 ======
def _is_chinese_char(c: str) -> bool:
    """判断字符是否为汉字"""
    return bool(c) and ('\u4e00' <= c <= '\u9fff')


def _is_independent(scope_text: str, idx: int, opt_len: int) -> bool:
    """判断选项在 scope_text[idx:idx+opt_len] 处是否独立出现

    判断规则：选项前后字符任一非汉字即视为独立出现
    - 前后都是汉字 → 子串命中（如"中国证券"在"中国证监会"中）
    - 前后任一非汉字 → 独立出现
    """
    before_char = scope_text[idx - 1] if idx > 0 else ""
    after_char = scope_text[idx + opt_len] if idx + opt_len < len(scope_text) else ""
    return not _is_chinese_char(before_char) or not _is_chinese_char(after_char)


def scan_option_in_doc(option_text: str, doc_text: str, doc_label: str) -> dict:
    """扫描单个选项在单个文档中的出现情况

    Returns:
        {
            "doc_label": 文档标签,
            "independent_count": 独立完整出现次数,
            "substring_count": 子串命中次数,
            "sample_snippet": 命中片段,
            "key_facts_hits": 关键事实命中详情,
        }
    """
    result = {
        "doc_label": doc_label,
        "independent_count": 0,
        "substring_count": 0,
        "sample_snippet": "",
        "key_facts_hits": [],
    }

    if not option_text or not doc_text:
        return result

    # 选项也要 LaTeX 规范化（与文档保持一致）
    # 否则文档规范化后变成"10亿元"，但选项仍是"10 亿元"，不匹配
    option_text = _normalize_latex(option_text)

    # 1. 完整短语扫描
    hit_count = doc_text.count(option_text)
    if hit_count > 0:
        start_idx = 0
        for _ in range(hit_count):
            idx = doc_text.find(option_text, start_idx)
            if idx == -1:
                break
            if _is_independent(doc_text, idx, len(option_text)):
                result["independent_count"] += 1
                if not result["sample_snippet"]:
                    s = max(0, idx - 30)
                    e = min(len(doc_text), idx + len(option_text) + 30)
                    result["sample_snippet"] = doc_text[s:e].replace("\n", " ")
            else:
                result["substring_count"] += 1
            start_idx = idx + 1

    # 2. 千分位差异后完整扫描
    if result["independent_count"] == 0:
        clean_opt = option_text.replace(",", "").replace(" ", "")
        clean_doc = doc_text.replace(",", "").replace(" ", "")
        if clean_opt and clean_opt in clean_doc:
            idx = clean_doc.find(clean_opt)
            s = max(0, idx - 30)
            e = min(len(clean_doc), idx + len(clean_opt) + 30)
            result["independent_count"] = 1
            result["sample_snippet"] = clean_doc[s:e]

    # 3. 关键事实扫描（更细粒度）
    key_facts = extract_key_facts(option_text)
    for fact in key_facts:
        if not fact or len(fact) < 2:
            continue
        fact_count = doc_text.count(fact)
        if fact_count > 0:
            # 判断关键事实是否独立出现
            idx = doc_text.find(fact)
            is_indep = _is_independent(doc_text, idx, len(fact))
            result["key_facts_hits"].append({
                "fact": fact,
                "count": fact_count,
                "independent": is_indep,
            })

    return result


def identify_target_docs(option_text: str, doc_ids: list) -> list:
    """根据选项文本识别其涉及的目标文档

    规则：
    - 选项含"第一份文档"/"文档一"/"第一份" → 主要在 doc_ids[0] 中扫描
    - 选项含"第二份文档"/"文档二"/"第二份" → 主要在 doc_ids[1] 中扫描
    - 选项含"两份文档均"/"两份文档都"/"两家均" → 在所有文档中扫描
    - 没有明确指代 → 在所有文档中扫描

    Returns:
        [(doc_id, doc_label_in_option), ...]
        doc_label_in_option: "doc1" / "doc2" / "all"
    """
    if not doc_ids:
        return []

    # 检测选项中提到的文档
    has_doc1 = bool(re.search(r'第一份文档|文档\s*一|第一份| fc_text_001', option_text))
    has_doc2 = bool(re.search(r'第二份文档|文档\s*二|第二份| fc_text_002', option_text))
    has_both = bool(re.search(r'两份文档均|两份文档都|两份文件均|两家均|两份均|两份文档|两份文件|两家公司均|两个文档', option_text))

    targets = []
    if has_both or (not has_doc1 and not has_doc2):
        # 涉及所有文档
        for i, did in enumerate(doc_ids):
            targets.append((did, "all"))
    else:
        if has_doc1 and len(doc_ids) >= 1:
            targets.append((doc_ids[0], "doc1"))
        if has_doc2 and len(doc_ids) >= 2:
            targets.append((doc_ids[1], "doc2"))
    return targets


def _scan_tf_question(question_text: str, docs_text: dict, doc_ids: list) -> str:
    """TF 题专用扫描：对 question 提取关键事实并在文档中扫描

    Args:
        question_text: 问题陈述（即待验证的陈述）
        docs_text: {doc_id: doc_text} 字典
        doc_ids: 涉及的文档列表

    Returns:
        扫描结果文本
    """
    # 识别陈述涉及的目标文档
    targets = identify_target_docs(question_text, doc_ids)
    if not targets:
        targets = [(did, "all") for did in doc_ids]

    # 提取关键事实
    key_facts = extract_key_facts(question_text)

    lines = [f"待验证陈述: {question_text}", ""]
    lines.append(f"提取的关键事实: {key_facts[:10]}")
    lines.append("")
    lines.append("扫描结果:")

    # 检查陈述是否涉及"两份文档均"
    requires_both = bool(re.search(r'两份文档均|两份文档都|两份文件均|两家均|两份均|两个文档', question_text))

    for doc_id, doc_label in targets:
        doc_text = docs_text.get(doc_id, "")
        if not doc_text:
            lines.append(f"  [{doc_id} {doc_label}] 文档未加载")
            continue

        # 扫描每个关键事实
        hits = []
        for fact in key_facts:
            if not fact or len(fact) < 2:
                continue
            fact_count = doc_text.count(fact)
            if fact_count > 0:
                idx = doc_text.find(fact)
                is_indep = _is_independent(doc_text, idx, len(fact))
                hits.append({
                    "fact": fact,
                    "count": fact_count,
                    "independent": is_indep,
                })

        # 完整短语扫描（短陈述）
        phrase_count = doc_text.count(question_text) if len(question_text) <= 50 else 0

        if hits:
            indep_hits = [h for h in hits if h["independent"]]
            substr_hits = [h for h in hits if not h["independent"]]
            parts = []
            if phrase_count > 0:
                parts.append(f"陈述完整出现 {phrase_count} 次")
            if indep_hits:
                fact_details = [f"'{h['fact']}'×{h['count']}" for h in indep_hits[:5]]
                parts.append(f"关键事实独立命中 {len(indep_hits)} 项: " + ", ".join(fact_details))
            if substr_hits:
                fact_details = [f"'{h['fact']}'" for h in substr_hits[:3]]
                parts.append(f"关键事实子串命中 {len(substr_hits)} 项: " + ", ".join(fact_details))
            lines.append(f"  [{doc_id} {doc_label}] " + "、".join(parts))
        else:
            lines.append(f"  [{doc_id} {doc_label}] 无关键事实命中")

    # 跨文档判断提示
    if requires_both:
        lines.append("")
        lines.append("提示: 陈述含'两份文档均'，需在两个文档中都命中核心事实才倾向选 A")

    return "\n".join(lines)


def scan_options_for_competition(question: dict) -> str:
    """对比赛题的每个选项，在涉及文档中扫描

    Returns:
        扫描结果文本（按选项分组）
    """
    domain = question.get("domain", "")
    doc_ids = question.get("doc_ids", [])
    options = question.get("options", {})
    answer_format = question.get("answer_format", "")
    question_text = question.get("question", "")

    if not doc_ids or not options:
        return "（无 doc_ids 或 options，无法扫描）"

    # 加载所有涉及文档的全文
    docs_text = {}
    for did in doc_ids:
        docs_text[did] = load_doc_text(domain, did)

    # TF 题特殊处理：对 question 提取关键事实，而非 options
    # TF 题的 options 是"A. 正确"/"B. 错误"，无可提取的关键事实
    # 否则会导致 fc_a_003、reg_a_003、res_a_003 等 TF 题关键事实命中=0，A→B 翻转
    if answer_format == "tf":
        return _scan_tf_question(question_text, docs_text, doc_ids)

    # 普通 multi/mcq 题原有逻辑
    lines = []
    for letter, opt_text in sorted(options.items()):
        opt_text_clean = opt_text.strip()
        if not opt_text_clean:
            continue

        # 识别选项涉及的目标文档
        targets = identify_target_docs(opt_text_clean, doc_ids)

        line = f"{letter}. '{opt_text_clean}' -> "
        if not targets:
            line += "未识别到目标文档"
            lines.append(line)
            continue

        # 在每个目标文档中扫描
        doc_results = []
        for doc_id, doc_label in targets:
            doc_text = docs_text.get(doc_id, "")
            if not doc_text:
                doc_results.append(f"[{doc_id} 文档未加载]")
                continue
            r = scan_option_in_doc(opt_text_clean, doc_text, doc_label)
            # 详细输出：整个选项短语 + 关键事实命中
            parts = []
            if r["independent_count"] > 0:
                parts.append(f"选项独立完整出现 {r['independent_count']} 次")
            if r["substring_count"] > 0:
                parts.append(f"选项子串命中 {r['substring_count']} 次")

            # 关键事实命中详情（核心信息）
            indep_facts = [h for h in r["key_facts_hits"] if h["independent"]]
            substr_facts = [h for h in r["key_facts_hits"] if not h["independent"]]
            if indep_facts:
                # 列出每个独立命中的关键事实及其出现次数
                fact_details = [f"'{h['fact']}'×{h['count']}" for h in indep_facts[:5]]
                parts.append(f"关键事实独立命中 {len(indep_facts)} 项: " + ", ".join(fact_details))
            if substr_facts:
                fact_details = [f"'{h['fact']}'" for h in substr_facts[:3]]
                parts.append(f"关键事实子串命中 {len(substr_facts)} 项: " + ", ".join(fact_details))
            if not r["key_facts_hits"]:
                parts.append("无关键事实命中")

            summary = "、".join(parts) if parts else "未命中"
            doc_results.append(f"[{doc_id} {doc_label}] {summary}")
            # 附加命中片段
            if r["sample_snippet"]:
                doc_results.append(f"  片段：\"{r['sample_snippet'][:100]}\"")

        line += " ".join(doc_results)
        lines.append(line)

    return "\n".join(lines)


# ====== 证据构建 ======
def build_evidence_text(evidence_top3: list, max_chars_per_chunk: int = 800) -> str:
    """构建证据文本（来自 evidence_top3）

    Args:
        evidence_top3: 排序后的证据列表，每项含 rank, chunk_id, doc_id, text
        max_chars_per_chunk: 每个 chunk 最多保留的字符数（避免 prompt 过长）

    Returns:
        格式化的证据文本
    """
    if not evidence_top3:
        return "（无证据）"

    parts = []
    for ev in evidence_top3:
        rank = ev.get("rank", 0)
        chunk_id = ev.get("chunk_id", "")
        doc_id = ev.get("doc_id", "")
        text = ev.get("text", "")
        # 截断过长 chunk
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."
        parts.append(f"证据{rank} [{doc_id} / {chunk_id}]:\n{text}")
    return "\n\n".join(parts)


# ====== Prompt 模板 ======
MULTI_TEMPLATE = """## 文档证据（你的回答必须基于以下证据）

{evidence}

## 选项扫描结果（系统从原文档全文扫描，作为辅助判断依据）

{scan_result}

## 问题

{question}

## 选项

{options_text}

## 题目类型：多选题（multi，跨文档对比题，选所有正确的，通常 2-4 个）

## 规则（核心：基于扫描结果分级判断，避免误选和漏选）

1. 只能使用证据和选项扫描结果中的信息，不要使用外部知识。
2. **扫描结果分级**（关键）：
   - **独立完整出现**（强支持）：选项作为连续短语在文档中精确出现，且前后字符非汉字
   - **千分位差异后完整出现**（强支持）：选项去除千分位后作为连续短语在文档中出现
   - **仅子串命中**（弱支持，仅作参考）：选项作为更长词的子串出现 → **不能作为选择依据**
   - **关键事实独立命中**：选项中的关键数值/实体名在文档中独立出现 → 中等支持
3. **扫描结果优先级**（辅助判断，非唯一依据）：
   - 选项在目标文档中**独立完整出现**或**关键事实独立命中** → 倾向选
   - 选项**仅子串命中**或**未命中**，但证据中明确支持 → 仍可选
   - 选项与证据**直接矛盾**（数值不一致、关系相反） → 不选
   - **关键**：扫描覆盖率有限，未命中不等于陈述错误，应结合证据综合判断
4. **跨文档对比规则**：
   - 选项含"第一份文档" → 主要看 doc_ids[0] 的扫描结果
   - 选项含"第二份文档" → 主要看 doc_ids[1] 的扫描结果
   - 选项含"两份文档均"/"两份均" → 必须在两个文档中都命中才选
   - 选项含"高于"/"低于"/"超过" → 需对比两个文档中的数值
5. **关键防误选规则**：
   - 选项"中国证券"在原文中以"中国证监会"形式出现 → 子串命中 → 不选
   - 选项中的数值与证据中的数值**直接矛盾** → 不选
6. **数量约束**：multi 题通常 2-4 个正确选项。不要为凑数而多选，但也不要因扫描未命中就漏选有证据支持的选项。
7. **逐项验证**：对每个选项，明确指出：
   - 在哪个文档命中（doc1 / doc2 / both / 未命中）
   - 命中类型（独立完整 / 子串 / 关键事实独立 / 未命中）
8. 输出严格 JSON（单行，无 markdown 代码块）。
"""

MCQ_TEMPLATE = """## 文档证据（你的回答必须基于以下证据）

{evidence}

## 选项扫描结果（系统从原文档全文扫描，作为辅助判断依据）

{scan_result}

## 问题

{question}

## 选项

{options_text}

## 题目类型：单选题（mcq，跨文档对比题，选一个最符合的）

## 规则（核心：原文优先 + 扫描结果辅助）

1. **原文优先原则**：选项中的数值/实体名必须在证据中原文精确出现，或在选项扫描结果中独立完整出现。
2. **扫描结果优先级**：
   - 选项在目标文档中**独立完整出现**或**关键事实独立命中** → 优先选
   - 选项**仅子串命中**或**未命中** → 不选
3. **严禁推测**：不允许基于"主题相关性"或"最接近数值"推断。
4. **跨文档对比规则**：选项含"第一份"/"第二份"/"两份均" 时，按对应文档扫描结果判断。
5. 输出严格 JSON（单行，无 markdown 代码块）。
"""

TF_TEMPLATE = """## 文档证据（必须严格基于以下证据判断）

{evidence}

## 关键事实扫描结果（系统从原文档全文扫描，作为辅助判断依据）

{scan_result}

## 问题

{question}

## 选项

A. 正确（与原文一致）
B. 错误（与原文不一致或原文未提及）

## 题目类型：判断题（tf，跨文档对比题）

## 判断规则（核心：基于证据 + 关键事实扫描结果 双重判断）

1. **提取待验证陈述**：从问题中提取核心事实（数值、专有名词、条款号）。
2. **关键事实扫描结果优先**：若扫描结果显示核心事实在原文档中独立完整出现，**即使证据 chunk 中未出现**，也应视为该事实在原文档中存在。
3. **比对策略**（放宽标准，避免过度怀疑）：
   - **核心事实匹配率 ≥ 30%** 或 **关键事实在扫描中独立命中** → 倾向选 A
   - **证据中存在与陈述直接矛盾的信息**（数值不一致、关系相反） → 选 B
   - **主关键词完全未在扫描结果和证据中出现，且证据明确缺失** → 倾向选 B
   - **扫描未命中不等于陈述错误**：扫描覆盖率有限，只要证据未明确否定，不要轻易选 B
4. **跨文档判断规则**：
   - 陈述涉及"两份文档均..." → 必须在两个文档中都命中核心事实才选 A
   - 陈述仅涉及单一文档 → 在对应文档中判断
5. 输出严格 JSON（单行，无 markdown 代码块）。
"""


# ====== Prompt 格式化 ======
def format_prompt_for_competition(question: dict) -> dict:
    """为单道比赛题生成 prompt

    Returns:
        {
            "qid": ...,
            "domain": ...,
            "answer_format": ...,
            "doc_ids": ...,
            "user_prompt": 完整 prompt,
            "estimated_input_tokens": 估算的输入 token 数,
        }
    """
    qid = question["qid"]
    domain = question["domain"]
    answer_format = question.get("answer_format", "multi")
    doc_ids = question.get("doc_ids", [])
    question_text = question.get("question", "")
    options = question.get("options", {})
    evidence_top3 = question.get("evidence_top3", [])

    # 构建证据文本
    evidence_text = build_evidence_text(evidence_top3)

    # 构建选项文本
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))

    # 扫描选项
    scan_result = scan_options_for_competition(question)

    # 根据题型选择模板
    if answer_format == "tf":
        user_prompt = TF_TEMPLATE.format(
            evidence=evidence_text,
            scan_result=scan_result,
            question=question_text,
        )
    elif answer_format == "multi":
        user_prompt = MULTI_TEMPLATE.format(
            evidence=evidence_text,
            scan_result=scan_result,
            question=question_text,
            options_text=options_text,
        )
    else:  # mcq
        user_prompt = MCQ_TEMPLATE.format(
            evidence=evidence_text,
            scan_result=scan_result,
            question=question_text,
            options_text=options_text,
        )

    # 估算输入 token（qwen 标准：中文 1.5 token/字，英文 0.25 token/字）
    cn_chars = len(re.findall(r'[一-鿿]', user_prompt))
    en_chars = len(user_prompt) - cn_chars
    estimated_tokens = int(cn_chars * 1.5 + en_chars * 0.25)

    return {
        "qid": qid,
        "domain": domain,
        "answer_format": answer_format,
        "doc_ids": doc_ids,
        "user_prompt": user_prompt,
        "estimated_input_tokens": estimated_tokens,
    }


# ====== 主函数 ======
def main():
    """主函数：加载 100 题，生成 prompts，输出到 JSONL"""
    print("=" * 60)
    print("比赛 100 题 prompts 生成器（跨文档对比专用）")
    print("=" * 60)

    # 加载所有比赛题
    all_questions = []
    for domain_file in sorted(EVIDENCE_DIR.iterdir()):
        if not domain_file.name.endswith(".json"):
            continue
        domain = domain_file.stem  # financial_contracts 等
        try:
            with open(domain_file, "r", encoding="utf-8") as f:
                questions = json.load(f)
            # 题目本身可能没有 domain 字段，从文件名注入
            for q in questions:
                q.setdefault("domain", domain)
            print(f"  {domain}: {len(questions)} 题")
            all_questions.extend(questions)
        except Exception as e:
            print(f"  {domain}: 加载失败 {e}")

    print(f"\n共加载 {len(all_questions)} 题")

    # 统计题型分布
    format_counts = {}
    for q in all_questions:
        fmt = q.get("answer_format", "unknown")
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    print(f"题型分布: {format_counts}")

    # 生成 prompts
    print(f"\n生成 prompts...")
    prompts = []
    total_input_tokens = 0
    for i, q in enumerate(all_questions):
        p = format_prompt_for_competition(q)
        prompts.append(p)
        total_input_tokens += p["estimated_input_tokens"]
        if (i + 1) % 20 == 0:
            print(f"  已生成 {i+1}/{len(all_questions)} 题")

    # 输出
    OUTPUT_PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PROMPTS, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"完成！输出 {len(prompts)} 题 prompts 到 {OUTPUT_PROMPTS}")
    print(f"总输入 token 估算: {total_input_tokens:,}")
    print(f"平均每题: {total_input_tokens // len(prompts):,} tokens")


if __name__ == "__main__":
    main()
