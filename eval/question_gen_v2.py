"""v2 综合评测框架新增题型生成器（多选 / 逻辑推理 / 数值计算）

设计要点（与 question_gen.py 对齐）：
1. 金标锚定：全部从 chunk_data 出题，source_anchor 可在金标 chunk 文本中验证；
2. 数值计算题：金标是"计算值"，特意不在原文（原文单值作干扰项，探测 F7 运算失败）；
3. 多选题：3 正确 + 2 错误表述，正确表述锚点可验证，错误表述篡改值全文反向验证；
4. 逻辑推理题：规则方向/阈值篡改（F6）+ 同表数值比较推断。
"""
import re
from collections import defaultdict

from eval.question_gen import (
    KNOWN_ENTITIES, anchor_in_text, build_mcq_options, extract_metric_value_pairs,
    normalize_num,
)

# 计算题/比较题允许的"同单位后缀"（后缀不同则量纲不可比，跳过）
_SUFFIX_RE = re.compile(r"(亿|万|千|百)?(元|股|份|条|家|人|次|年|个月|天|％|%)?$")

# 金额类指标词：这类指标配百分比数值大概率是表格列错配（如"净资产 50%"）
_MONEY_CHARS = ("净利润", "收入", "总资产", "净资产", "货币资金", "负债", "费用",
                "募集资金", "发行总额", "存货", "借款", "保险金额", "保险费")


def _pair_sane(p: dict) -> bool:
    """指标-数值配对合理性检查：金额类指标不应配百分比，比率类指标应配百分比"""
    num, met = p["num"].strip(), p["metric"]
    is_pct = num.endswith("%") or num.endswith("％")
    if is_pct and any(mc in met for mc in _MONEY_CHARS):
        return False
    if not is_pct and ("收益率" in met or "渗透率" in met or "占有率" in met):
        return False
    return True


def _metric_clean(met: str) -> bool:
    """指标名干净性检查（无括号/标点/黑名单词，长度合理）"""
    if not met or len(met) < 2 or len(met) > 16:
        return False
    if re.search(r"[（(）)：:，,、%％]", met):
        return False
    return not any(bw in met for bw in ("截至", "已经", "目前", "此外", "同时",
                                        "由于", "因此", "如果", "根据", "按照"))


def _split_num(num: str) -> tuple:
    """拆分数值串为 (核心数字串, 单位后缀)，如 '9,227,990万元' -> ('9,227,990', '万元')"""
    m = _SUFFIX_RE.search(num)
    suffix = m.group(0) if m and m.start() > 0 else ""
    core = num[: len(num) - len(suffix)] if suffix else num
    return core, suffix


def _fmt_result(value: float, use_comma: bool, has_decimal: bool, suffix: str) -> str:
    """计算结果按输入风格格式化：千分位/小数位与原值一致，保留单位后缀"""
    if has_decimal:
        s = f"{value:.2f}"
    else:
        s = str(int(round(value)))
    if use_comma:
        if has_decimal:
            whole, frac = s.split(".")
        else:
            whole, frac = s, ""
        whole = f"{int(whole):,}"
        s = f"{whole}.{frac}" if frac else whole
    return s + suffix


# ============ 多选题生成器 ============

def gen_multi_select(system, domain: str, rng, max_items: int,
                     pairs_cache: dict, full_texts: dict) -> list[dict]:
    """多选题（5 选项选 2-3）：同一文档 3 条真实事实 + 2 条篡改事实

    真实表述：词表指标 + 实体（跨 chunk 选取），锚点可验证；
    错误表述：数值 ×2 篡改，且篡改值不在文档全文（防"碰巧正确"）。
    """
    items = []
    by_doc = defaultdict(list)
    for cid, c in system.chunk_data.items():
        if len(c.get("text", "")) > 60:
            vocab = [p for p in (pairs_cache.get(cid) or [])
                     if p.get("is_vocab")
                     and (not p.get("entity") or p.get("entity") in KNOWN_ENTITIES)
                     and _pair_sane(p)
                     and not re.search(r"[（(）)：:，,、]", p["metric"])]
            if vocab:
                by_doc[c["doc_id"]].append((cid, vocab))
    doc_ids = sorted(by_doc.keys())
    rng.shuffle(doc_ids)
    for doc_id in doc_ids:
        if len(items) >= max_items:
            break
        chunk_pairs = by_doc[doc_id][:]
        rng.shuffle(chunk_pairs)
        full_text = full_texts.get(doc_id, "")
        # 收集 5 条可用事实（3 真 + 2 篡改为假），须指标互异、chunk 互异
        # 实体放宽：无已知实体时省略实体前缀（与 gen_multi_hop 一致），仅要求词表指标
        facts, used_metrics, used_chunks = [], set(), set()
        for cid, vocab in chunk_pairs:
            for p in vocab:
                # 允许同 chunk 多指标（分块大、指标多时提高事实可用性），仅要求指标互异
                if p["metric"] in used_metrics:
                    continue
                facts.append((cid, p))
                used_metrics.add(p["metric"])
                used_chunks.add(cid)
            if len(facts) >= 5:
                break
        if len(facts) < 5:
            continue
        true_facts, fake_srcs = facts[:3], facts[3:]
        # 篡改第 4/5 条事实构造 2 个错误表述，篡改值均须不在全文
        fake_stmts = []
        for cid_f, p_f in fake_srcs:
            try:
                v = float(normalize_num(_split_num(p_f["num"])[0]))
            except ValueError:
                continue
            core_f, suffix_f = _split_num(p_f["num"])
            fake_num = f"{v * 2:.2f}".rstrip("0").rstrip(".") + suffix_f
            if anchor_in_text(fake_num, full_text):
                continue
            ent_pre = f"{p_f['entity']}的" if p_f["entity"] else ""
            fake_stmts.append(f"{ent_pre}{p_f['metric']}为{fake_num}")
        if len(fake_stmts) < 2:
            continue
        true_stmts = [f"{p['entity']}的{p['metric']}为{p['num']}" if p["entity"]
                      else f"{p['metric']}为{p['num']}" for _, p in true_facts]
        opts = true_stmts + fake_stmts[:2]
        rng.shuffle(opts)
        letters = ["A", "B", "C", "D", "E"]
        gold = "".join(sorted(letters[opts.index(s)] for s in true_stmts))
        items.append({
            "domain": domain, "answer_format": "multi",
            "question": "根据文档，以下哪些表述与文档相符？（多选，可能有两个及以上正确项）",
            "options": opts, "gold_answer": gold,
            "gold_doc_ids": [doc_id],
            "gold_chunk_ids": [cid for cid, _ in true_facts],
            "source_anchor": ";".join(p["num"] for _, p in true_facts),
            "layer": "single_hop", "difficulty": "hard",
            "capabilities": ["answer_accuracy", "factual_consistency", "response_completeness"],
            "generation_note": f"真表述锚定{len(true_facts)}chunk; 假表述篡改自 {[cid for cid, _ in fake_srcs]}",
        })
    return items


# ============ 逻辑推理题生成器 ============

# 条件规则连接词（探测 F6：规则方向/阈值误读）
_RULE_CONN = ["不得超过", "不超过", "不得低于", "不得少于", "不低于", "应当不低于", "高于", "低于"]
_RULE_RE = re.compile(
    r"([一-鿿（）()]{2,12})?(不超过|不得超过|不得低于|不得少于|不低于|应当不低于|高于|低于)"
    r"\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)(\s*(?:亿|万|千|百)?\s*(?:元|%|％|元/股|股)?)?"
)


def _flip_direction(conn: str) -> str:
    """规则方向翻转：不超过 <-> 不低于，用于构造方向性错误选项"""
    pairs = {"不超过": "不低于", "不得超过": "不低于", "不低于": "不超过",
             "不得低于": "不超过", "不得少于": "不超过", "应当不低于": "不超过",
             "高于": "低于", "低于": "高于"}
    return pairs.get(conn, conn)


def gen_logical_rule(system, domain: str, rng, max_items: int,
                     pairs_cache: dict) -> list[dict]:
    """条件规则推理题：从保险/监管/合同条款抽"主语 + 方向词 + 阈值"规则

    金标 = 原规则复述（锚点可验证）；干扰项 = 阈值篡改 / 方向翻转（探测 F6）。
    """
    items = []
    cand = [(cid, c) for cid, c in system.chunk_data.items()
            if c.get("chunk_type") == "text" and len(c.get("text", "")) > 60]
    rng.shuffle(cand)
    for cid, c in cand:
        if len(items) >= max_items:
            break
        sent_text = c["text"]
        for m in _RULE_RE.finditer(sent_text):
            subject, conn, num, suffix = m.group(1), m.group(2), m.group(3), m.group(4) or ""
            # 主语须为干净片段：仅排除强噪声词（允许含"的"，中文规则句普遍存在）
            if not subject or any(sw in subject for sw in
                                  ("作为", "如果", "但是", "其中", "根据", "以及", "并且", "或者")):
                continue
            subject = subject.rstrip("的，,、；;")
            # 剔除开头连接词与"拟以"类引导片段（如"且每次借款期限"/"公司拟以"）
            subject = re.sub(r"^[且并或及而但与，,\s]+", "", subject)
            if subject.endswith(("以", "拟", "按", "照", "对")) or "拟" in subject:
                continue
            if len(subject) < 2:
                continue
            try:
                v = float(normalize_num(num))
            except ValueError:
                continue
            if v < 1:
                continue
            gold_stmt = f"{subject}{conn}{num}{suffix.strip()}"
            # 干扰项 1：阈值篡改（×0.8，保留后缀）；须不在原文
            fake_num = _fmt_result(v * 0.8, "," in num, "." in num, suffix.strip())
            if anchor_in_text(fake_num, sent_text):
                continue
            d1 = f"{subject}{conn}{fake_num}"
            # 干扰项 2：方向翻转（阈值不变）
            d2 = f"{subject}{_flip_direction(conn)}{num}{suffix.strip()}"
            if d2 == gold_stmt:
                continue
            # 干扰项 3：方向翻转 + 阈值篡改
            d3 = f"{subject}{_flip_direction(conn)}{fake_num}"
            opts, gold_letter = build_mcq_options(gold_stmt, [d1, d2, d3], rng)
            if len(opts) < 4:
                continue
            items.append({
                "domain": domain, "answer_format": "logical",
                "question": f"根据文档条款，关于\"{subject}\"的规定，下列表述正确的是？",
                "options": opts, "gold_answer": gold_letter,
                "gold_doc_ids": [c["doc_id"]], "gold_chunk_ids": [cid],
                "source_anchor": f"{conn}{num}", "layer": "single_hop",
                "difficulty": "hard",
                "capabilities": ["answer_accuracy", "reasoning", "factual_consistency"],
                "generation_note": f"规则句: {subject}{conn}{num}{suffix.strip()}",
            })
            break  # 每 chunk 最多 1 题
    return items


def gen_logical_compare(system, domain: str, rng, max_items: int,
                        pairs_cache: dict) -> list[dict]:
    """数值比较推理题：同一表格 chunk 内两个同量纲指标，推断大小关系

    金标 = 大小关系表述；干扰项 = 关系翻转 / 数值互换（探测 F3+F6 组合）。
    """
    items = []
    cand = [(cid, c) for cid, c in system.chunk_data.items()
            if c.get("chunk_type") == "table" and len(c.get("text", "")) > 80]
    rng.shuffle(cand)
    for cid, c in cand:
        if len(items) >= max_items:
            break
        pairs = [p for p in (pairs_cache.get(cid) or [])
                 if _metric_clean(p["metric"]) and _pair_sane(p)]
        # 需至少 2 个指标数值对且数值可解析（指标名已过干净性检查）
        parsed = []
        for p in pairs:
            core, suffix = _split_num(p["num"])
            try:
                val = float(normalize_num(core))
            except ValueError:
                continue
            parsed.append((p, val, suffix))
        if len(parsed) < 2:
            continue
        (pa, va, sa), (pb, vb, sb) = parsed[0], parsed[1]
        # 量纲一致才可比（后缀相同，或均无后缀）
        if sa != sb:
            continue
        if va == vb:
            continue
        hi, lo = (pa, pb) if va > vb else (pb, pa)
        hi_v, lo_v = max(va, vb), min(va, vb)
        gold_stmt = f"{hi['metric']}（{hi['num']}）高于{lo['metric']}（{lo['num']}）"
        flipped = f"{lo['metric']}（{lo['num']}）高于{hi['metric']}（{hi['num']}）"
        swapped = f"{hi['metric']}（{_fmt_result(lo_v, ',' in _split_num(hi['num'])[0], '.' in _split_num(hi['num'])[0], sa)}）高于{lo['metric']}（{_fmt_result(hi_v, ',' in _split_num(lo['num'])[0], '.' in _split_num(lo['num'])[0], sa)}）"
        # 干扰项 3：对高值/低值做数值篡改（×2 / ×0.5），须不在原文，防碰巧正确
        hi_core, _ = _split_num(hi["num"])
        d3 = ""
        for factor in (2.0, 0.5):
            base = hi_v if factor == 2.0 else lo_v
            fake = _fmt_result(base * factor, "," in hi_core, "." in hi_core, sa)
            if not anchor_in_text(fake, c["text"]):
                target = hi if factor == 2.0 else lo
                other = lo if factor == 2.0 else hi
                d3 = f"{target['metric']}（{fake}）高于{other['metric']}（{other['num']}）"
                break
        if not d3:
            continue
        opts, gold_letter = build_mcq_options(gold_stmt, [flipped, swapped, d3], rng)
        if len(opts) < 4 or len(set(opts)) < 4:
            continue
        items.append({
            "domain": domain, "answer_format": "logical",
            "question": "根据文档表格数据，下列比较关系正确的是？",
            "options": opts, "gold_answer": gold_letter,
            "gold_doc_ids": [c["doc_id"]], "gold_chunk_ids": [cid],
            "source_anchor": f"{hi['num']};{lo['num']}", "layer": "table_numeric",
            "difficulty": "hard",
            "capabilities": ["answer_accuracy", "reasoning", "computation"],
            "generation_note": f"比较: {hi['metric']}({hi['num']}) vs {lo['metric']}({lo['num']})",
        })
    return items


# ============ 数值计算题生成器 ============

def gen_calc(system, domain: str, rng, max_items: int,
             pairs_cache: dict) -> list[dict]:
    """数值计算题（20%）：同一表格句内两个同量纲数值求和/求差

    金标 = 计算值（特意不在原文）；干扰项 = 两个原始值 + 错误运算结果（探测 F7）。
    """
    items = []
    cand = [(cid, c) for cid, c in system.chunk_data.items()
            if c.get("chunk_type") == "table" and len(c.get("text", "")) > 80]
    rng.shuffle(cand)
    for cid, c in cand:
        if len(items) >= max_items:
            break
        pairs = [p for p in (pairs_cache.get(cid) or [])
                 if _metric_clean(p["metric"]) and _pair_sane(p)]
        if len(pairs) < 2:
            continue
        made = 0
        used = set()
        for i in range(len(pairs)):
            if made >= 2 or len(items) >= max_items:
                break
            p1 = pairs[i]
            core1, suf1 = _split_num(p1["num"])
            try:
                v1 = float(normalize_num(core1))
            except ValueError:
                continue
            if v1 <= 0 or p1["metric"] in used:
                continue
            for j in range(i + 1, len(pairs)):
                p2 = pairs[j]
                core2, suf2 = _split_num(p2["num"])
                if suf1 != suf2 or p2["metric"] in used or p2["metric"] == p1["metric"]:
                    continue
                try:
                    v2 = float(normalize_num(core2))
                except ValueError:
                    continue
                if v2 <= 0 or v2 == v1:
                    continue
                # 单位量纲安全：两值数量级差超过 4 则可能跨表头错配，跳过
                if abs(v1 - v2) <= 0 or max(v1, v2) / min(v1, v2) > 10000:
                    continue
                use_comma = "," in core1 or "," in core2
                has_decimal = "." in core1 or "." in core2
                # 随机选择求和/求差（差取大减小，保证非负）
                if rng.random() < 0.5:
                    op_word, result = "与{}之和".format(p2["metric"]), v1 + v2
                else:
                    op_word = "与{}之差".format(p2["metric"]) if v1 > v2 else \
                        "与{}之差（大减小）".format(p2["metric"])
                    result = abs(v1 - v2)
                    op_word = f"与{p2['metric']}之差的绝对值"
                gold_num = _fmt_result(result, use_comma, has_decimal, suf1)
                # 金标（计算值）不得在原文出现（否则出现多金标）
                if anchor_in_text(gold_num, c["text"]):
                    continue
                d_wrong_op = _fmt_result(v1 + v2 if "差" in op_word else abs(v1 - v2),
                                         use_comma, has_decimal, suf1)
                if d_wrong_op == gold_num:
                    continue
                distractors = [p1["num"], p2["num"], d_wrong_op]
                # 干扰项去重（两个原值可能相同——已用 v1!=v2 排除）
                opts, gold_letter = build_mcq_options(gold_num, distractors, rng)
                if len(opts) < 4:
                    continue
                ent = p1["entity"] if p1["entity"] and p1["entity"] in KNOWN_ENTITIES else ""
                ent_part = f"{ent}的" if ent else ""
                question = f"根据文档表格，{ent_part}{p1['metric']}{op_word}为多少？（数值计算）"
                items.append({
                    "domain": domain, "answer_format": "calc",
                    "question": question, "options": opts, "gold_answer": gold_letter,
                    "gold_doc_ids": [c["doc_id"]], "gold_chunk_ids": [cid],
                    "source_anchor": f"{p1['num']};{p2['num']}", "layer": "table_numeric",
                    "difficulty": "hard",
                    "capabilities": ["answer_accuracy", "computation", "retrieval_relevance"],
                    "generation_note": f"计算: {p1['metric']}({p1['num']}) op {p2['metric']}({p2['num']}) = {gold_num}",
                })
                used.add(p1["metric"])
                used.add(p2["metric"])
                made += 1
                break
    return items


# ============ v2 质检 ============

def quality_check_v2(item: dict, systems: dict, full_texts: dict) -> tuple:
    """v2 题型质检：锚点命中金标 chunk / 多选选项数 5 / 计算题金标不在原文"""
    fmt = item["answer_format"]
    if item.get("layer") == "unanswerable":
        # 陷阱题：实体与指标在目标文档全文不共现（复用 v1 反向验证思路）
        doc_id = item["gold_doc_ids"][0]
        ft = full_texts.get(doc_id, "")
        m = re.search(r"根据文档，(.+?)的(.+?)数值是多少", item["question"])
        if m and ft and m.group(1) in ft and m.group(2) in ft:
            return False, "陷阱题实体与指标在原文共现"
        return True, "ok"
    if fmt == "multi" and len(item["options"]) != 5:
        return False, "多选选项数不足5"
    if fmt in ("mcq", "logical", "calc") and len(item["options"]) != 4:
        return False, "选项数量不足4"
    if fmt == "multi":
        # 真实表述锚点须命中对应金标 chunk
        anchors = item["source_anchor"].split(";")
        chunks = [systems[item["domain"]].chunk_data.get(cid) for cid in item["gold_chunk_ids"]]
        if len(anchors) != len(chunks):
            return False, "锚点与chunk数不一致"
        for a, ch in zip(anchors, chunks):
            if ch and anchor_in_text(a, ch["text"]):
                continue
            return False, "多选锚点未命中金标chunk"
        return True, "ok"
    if fmt == "calc":
        # 计算题：金标值不在原文（是计算值），两个原值锚点须命中
        gold_opt = item["options"][["A", "B", "C", "D"].index(item["gold_answer"])]
        chunks = [systems[item["domain"]].chunk_data.get(cid) for cid in item["gold_chunk_ids"]]
        anchors = item["source_anchor"].split(";")
        for a, ch in zip(anchors, chunks):
            if ch and anchor_in_text(a, ch["text"]):
                continue
            return False, "计算题原值锚点未命中"
        if any(anchor_in_text(gold_opt, ch["text"]) for ch in chunks if ch):
            return False, "计算题金标值在原文出现"
        return True, "ok"
    # mcq / logical：锚点须命中金标 chunk
    for cid in item["gold_chunk_ids"]:
        chunk = systems[item["domain"]].chunk_data.get(cid)
        if chunk and anchor_in_text(item["source_anchor"], chunk["text"]):
            return True, "ok"
    return False, "锚点未命中金标chunk"
