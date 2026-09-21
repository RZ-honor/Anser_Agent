"""
分层评测集题目生成器（求职作品集 · 改进.md §二 验证指标体系）

设计要点：
1. 金标锚定：从 cache/domain_indexes_v2.pkl 的 chunk_data（检索真实单元）出发出题，
   gold_chunk_id 与检索系统天然对齐，Recall@10/MRR@10 的计算完全可信；
2. 五个分层：单跳事实(30%) / 表格数值(25%) / 跨文档(20%) / 时效版本(10%) / 无答案陷阱(15%)；
3. 质检门禁：锚点必须命中原文（复用 verify_source_anchor 的千分位/多锚点思路）、
   干扰项必须不在原文（防止多金标）、陷阱题必须全文反向检索确认"不存在"。
"""
import json
import os
import random
import re
import pickle
from collections import defaultdict

# ============ 常量配置 ============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# 无答案陷阱题的金标答案标记（评测脚本据此计算拒答率）
REFUSED = "REFUSED"

# 领域 -> 数据目录
DOMAIN_DATA_DIR = {
    "financial_contracts": os.path.join(DATA_DIR, "financial_contracts"),
    "financial_reports": os.path.join(DATA_DIR, "financial_reports"),
    "insurance": os.path.join(DATA_DIR, "insurance"),
    "regulatory": os.path.join(DATA_DIR, "regulatory"),
    "research": os.path.join(DATA_DIR, "research"),
}

# 常见财务/金融指标词表（用于"数值前缀=指标名"的提取，覆盖 5 个领域高频指标）
METRIC_KEYWORDS = [
    "归属于母公司股东的净利润", "归属于上市公司股东的净利润", "归属于上市公司股东的净资产",
    "扣除非经常性损益的净利润", "经营活动产生的现金流量净额", "加权平均净资产收益率",
    "营业收入", "归母净利润", "净利润", "总资产", "净资产", "研发费用", "基本每股收益",
    "资产负债率", "毛利率", "货币资金", "存货", "短期借款", "长期借款", "应付账款",
    "保险金额", "保险费", "免赔额", "给付比例", "犹豫期", "等待期", "基本保额",
    "发行总额", "票面利率", "募集资金", "承销费用", "评级", "信用等级",
    "市场规模", "市场占有率", "同比增长", "年复合增长率", "出货量", "目标价", "渗透率",
]

# 双年报公司（时效版本题素材）：公司简称 -> 文档名片段
DUAL_REPORT_COMPANIES = {
    "比亚迪": ("annual_byd_2024_report", "annual_byd_2025_report"),
    "宁德时代": ("annual_catl_2024_report", "annual_catl_2025_report"),
    "美的集团": ("annual_midea_2024_report", "annual_midea_2025_report"),
    "中国建筑": ("annual_cscec_2024_report", "annual_cscec_2025_report"),
}

# 实体停用词：回退实体提取时剔除的噪声片段（防止"作为创新型科技"这类误匹配）
ENTITY_STOPWORDS = ["作为", "基于", "载有", "公司", "相关", "上述", "其中", "以及", "我们",
                    "根据", "扰乱", "但若", "也不得", "主要是", "应当", "收到", "签收"]

# 指标回退黑名单：常见副词/连接词/时间引导词，出现即判为噪声
METRIC_BLACKLIST = ["截至", "已经", "目前", "此外", "同时", "其中", "以及", "由于",
                    "因此", "如果", "根据", "按照", "关于", "对于", "本次", "该", "报告期"]

# 已知主体词表（data/ 覆盖的公司与发行人，优先于正则回退，保证实体干净）
KNOWN_ENTITIES = [
    "比亚迪股份有限公司", "比亚迪", "宁德时代新能源科技股份有限公司", "宁德时代",
    "美的集团股份有限公司", "美的集团", "美的", "中国建筑股份有限公司", "中国建筑",
    "中国移动通信集团有限公司", "中国移动", "招商银行股份有限公司", "招商银行",
    "广东省广晟控股集团有限公司", "深圳市投资控股有限公司",
]
# 指标词交替正则（按长度降序保证最长匹配），指标词必须紧邻数值（允许"为/约/是/达"连接词）
_METRIC_ALT = "|".join(sorted(METRIC_KEYWORDS, key=len, reverse=True))
_METRIC_TAIL_RE = re.compile(rf"(?:{_METRIC_ALT})\s*(?:共计|分别为?|约为?|是|达到?|不超过)?\s*(?:人民币|港币|美元|港元|欧元)?\s*$")

# 数值正则：支持千分位/小数/百分比/中文单位
NUM_PATTERN = re.compile(
    r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(\s*(?:亿|万|千|百)?\s*(?:元|股|份|条|家|人|次|年|个月|天)|%|亿元|万元|％)?"
)
# 实体正则：公司/机构名（与 domain_retrieval 实体索引正则对齐）
ENTITY_PATTERN = re.compile(r"[一-鿿]{2,15}(?:股份|集团|科技|证券|银行|保险|控股|公司)")


# ============ 工具函数 ============

def load_domain_systems():
    """加载领域索引缓存（只加载不重建），返回 {domain: system}"""
    cache_path = os.path.join(CACHE_DIR, "domain_indexes_v2.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    return cache.get("domain_systems", {})


def load_doc_json(domain: str, doc_prefix: str) -> dict:
    """按文档名前缀加载原始文档 JSON（用于全文反向验证）"""
    ddir = DOMAIN_DATA_DIR.get(domain)
    if not ddir:
        return {}
    for fname in os.listdir(ddir):
        if fname.startswith(doc_prefix) and fname.endswith(".json"):
            with open(os.path.join(ddir, fname), encoding="utf-8") as f:
                return json.load(f)
    return {}


def normalize_num(s: str) -> str:
    """数值归一化：去除千分位（锚点匹配时与 verify_source_anchor 思路一致）"""
    return s.replace(",", "")


def anchor_in_text(anchor: str, text: str) -> bool:
    """锚点是否命中文本：先精确匹配，失败后去千分位再匹配（多锚点用 ; 分隔）"""
    if not anchor or not text:
        return False
    if anchor in text:
        return True
    if normalize_num(anchor) in text:
        return True
    if ";" in anchor:
        subs = [a.strip() for a in anchor.split(";") if a.strip()]
        hits = sum(1 for sa in subs if sa in text or normalize_num(sa) in text)
        if hits >= max(1, len(subs) - 1):  # 允许一个子锚点缺失
            return True
    return False


def _extract_entity(sent: str) -> str:
    """实体提取：已知主体词表优先，正则回退并过滤停用词噪声"""
    # 优先：句中含已知主体名
    for ent in KNOWN_ENTITIES:
        if ent in sent:
            return ent
    m = ENTITY_PATTERN.search(sent)
    if not m:
        return ""
    ent = m.group()
    # 停用词过滤：剔除以噪声词开头/包含噪声词的片段
    for sw in ENTITY_STOPWORDS:
        if ent.startswith(sw) or sw in ent:
            return ""
    # 剔除以数字/中文数字开头的实体（如"二零二四年本集团"是日期+机构的拼接噪声）
    if re.match(r"^[\d一二三四五六七八九零百千万]", ent):
        return ""
    return ent


def extract_metric_value_pairs(text: str) -> list[dict]:
    """从文本中抽取 (指标名, 数值串, 所在句子) 三元组

    策略：
    - 表格 markdown（无换行、| 分隔单元格）：先把 | 和 --- 清洗为分隔符，按单元格切"句"
    - 普通文本：按 。；;\n 切句
    - 指标名：词表交替正则要求紧邻数值（尾部匹配）；回退片段须无停用词噪声
    """
    # 表格文本预处理：markdown 表格的指标与数值分属不同单元格（如 "| 净利润 | 7,771 |"），
    # 若按分号切句会把配对切断，故把 | 替换为空格、清除 --- 分隔行后整表作为一个"句子"
    if "|" in text:
        text = re.sub(r"[|]", " ", text)
        text = re.sub(r"\s*[-:]{3,}\s*", " ", text)
        text = re.sub(r"\s+", " ", text)
        sentences = [text]
    else:
        # 普通文本：按句子切分，避免跨句拼凑
        sentences = re.split(r"[。；;\n]", text)
    pairs = []
    for sent in sentences:
        for m in NUM_PATTERN.finditer(sent):
            num_str = m.group(1)
            # 过滤过短数值（如年份后两位、编号）
            if len(num_str.replace(".", "")) < 2:
                continue
            # 过滤纯年份数值（如"2025 年"），年份作挖空答案没有意义
            year_only = re.fullmatch(r"(19|20)\d{2}", num_str) and \
                (m.group(2) is None or "年" in m.group(2))
            if year_only:
                continue
            prefix = sent[: m.start()].strip()
            # 指标名：词表交替正则匹配"指标词(+连接词)紧邻数值"的尾部
            # 取最右匹配（finditer[-1]）：search 会返回最左匹配，导致
            # "犹豫期内...犹豫期为22" 截取出整段而非"犹豫期"
            metric = ""
            tms = list(_METRIC_TAIL_RE.finditer(prefix))
            tm = tms[-1] if tms else None
            if tm:
                metric = re.sub(r"\s*(?:共计|分别为?|约为?|是|达到?|不超过)+\s*$", "", tm.group())
            if not metric:
                # 回退：取数值前 2-12 字中文片段，须无停用词噪声且不以"的"结尾
                cm = re.search(r"([一-鿿（）()]{2,12})$", prefix)
                if cm and not any(sw in cm.group(1) for sw in ENTITY_STOPWORDS) \
                        and not cm.group(1).endswith("的"):
                    metric = cm.group(1)
            # 指标黑名单过滤 + 词表命中标记（is_vocab 供 multi_hop 等高质量要求层使用）
            if metric and not any(bw in metric for bw in METRIC_BLACKLIST):
                pairs.append({
                    "metric": metric,
                    "is_vocab": metric in METRIC_KEYWORDS,
                    "num": m.group(0).strip(),
                    "num_core": num_str,
                    "sentence": sent.strip(),
                    "entity": _extract_entity(sent),
                })
    return pairs


def make_distractors(gold_num: str, pool_nums: list[str], rng: random.Random,
                     n: int = 3) -> list[str]:
    """生成干扰项：优先取同文档其他真实数值，不足时做数值篡改

    质检约束：干扰项必须不等于金标（防多金标）
    """
    gold_clean = normalize_num(gold_num)
    distractors = []
    seen = {gold_clean}
    for cand in pool_nums:
        c = normalize_num(cand)
        if c not in seen and len(distractors) < n:
            distractors.append(cand)
            seen.add(c)
    # 不足时数值篡改补齐（×2 / +10% / 小数移位）
    variants = []
    try:
        v = float(gold_clean)
        variants = [f"{v * 2:.2f}".rstrip("0").rstrip("."), f"{v * 1.1:.2f}".rstrip("0").rstrip("."), f"{v / 10:.4f}".rstrip("0").rstrip(".")]
    except ValueError:
        pass
    for v in variants:
        if len(distractors) >= n:
            break
        if v not in seen:
            distractors.append(v)
            seen.add(v)
    return distractors[:n]


def build_mcq_options(gold: str, distractors: list[str], rng: random.Random):
    """组装 4 选 1 选项并返回 (options 列表, 金标字母)"""
    opts = [gold] + distractors[:3]
    rng.shuffle(opts)
    letters = ["A", "B", "C", "D"]
    gold_letter = letters[opts.index(gold)]
    return opts, gold_letter


def build_pairs_cache(system) -> dict:
    """预计算全部 chunk 的指标数值对缓存（避免生成阶段重复跑正则，全量构建提速关键）"""
    return {cid: extract_metric_value_pairs(c.get("text", ""))
            for cid, c in system.chunk_data.items()}


def build_domain_num_pool(pairs_cache: dict, exclude_chunk_ids: set = None) -> list[str]:
    """领域数值池：跨文档干扰项来源；可排除指定 chunk（防干扰项撞金标原文）"""
    exclude_chunk_ids = exclude_chunk_ids or set()
    pool = [p["num"] for cid, pairs in pairs_cache.items()
            if cid not in exclude_chunk_ids for p in pairs]
    rng_order = random.Random(7)
    rng_order.shuffle(pool)  # 打乱避免总是取到同一批干扰项
    return pool


# ============ 五层生成器 ============

def gen_single_hop(system, domain: str, rng: random.Random, max_items: int,
                   pairs_cache: dict = None) -> list[dict]:
    """单跳事实题（30%）：从文本 chunk 抽实体+数值事实，数值挖空成 4 选 1"""
    items = []
    if pairs_cache is None:
        pairs_cache = build_pairs_cache(system)
    cand_chunks = [(cid, c) for cid, c in system.chunk_data.items()
                   if c.get("chunk_type") == "text" and len(c["text"]) > 60]
    rng.shuffle(cand_chunks)
    for cid, c in cand_chunks:
        if len(items) >= max_items:
            break
        pairs = pairs_cache.get(cid) or []
        if not pairs:
            continue
        # 干扰项池：跨 chunk 取值（排除金标 chunk 自身数值，避免干扰项撞原文被质检淘汰）
        pool = build_domain_num_pool(pairs_cache, exclude_chunk_ids={cid})
        pair = pairs[0]
        distractors = make_distractors(pair["num"], pool, rng)
        if len(distractors) < 3:
            continue
        # 干扰项质检：不得出现在原文（防多金标）
        if any(anchor_in_text(d, c["text"]) for d in distractors):
            continue
        entity_part = f"{pair['entity']}的" if pair["entity"] else ""
        question = f"根据文档，{entity_part}{pair['metric']}的数值为（单位按原文）？"
        opts, gold_letter = build_mcq_options(pair["num"], distractors, rng)
        items.append({
            "domain": domain, "answer_format": "mcq",
            "question": question, "options": opts, "gold_answer": gold_letter,
            "gold_doc_ids": [c["doc_id"]], "gold_chunk_ids": [cid],
            "source_anchor": pair["num"], "layer": "single_hop",
            "difficulty": "easy", "generation_note": f"锚点句: {pair['sentence'][:60]}",
        })
    return items


def gen_table_numeric(system, domain: str, rng: random.Random, max_items: int,
                      pairs_cache: dict = None) -> list[dict]:
    """表格数值题（25%）：从表格 chunk 抽指标+数值"""
    items = []
    if pairs_cache is None:
        pairs_cache = build_pairs_cache(system)
    cand_chunks = [(cid, c) for cid, c in system.chunk_data.items()
                   if c.get("chunk_type") == "table" and len(c["text"]) > 80]
    rng.shuffle(cand_chunks)
    for cid, c in cand_chunks:
        if len(items) >= max_items:
            break
        pairs = pairs_cache.get(cid) or []
        # 表格 chunk 需至少 2 组指标数值对才可出题
        if len(pairs) < 2:
            continue
        pair = pairs[0]
        # 干扰项池：排除金标表格 chunk 自身数值，跨 chunk 取真实值
        pool = build_domain_num_pool(pairs_cache, exclude_chunk_ids={cid})
        distractors = make_distractors(pair["num"], pool, rng)
        if len(distractors) < 3:
            continue
        if any(anchor_in_text(d, c["text"]) for d in distractors):
            continue
        entity_part = f"{pair['entity']}的" if pair["entity"] else ""
        question = f"根据文档表格，{entity_part}{pair['metric']}为多少？"
        opts, gold_letter = build_mcq_options(pair["num"], distractors, rng)
        items.append({
            "domain": domain, "answer_format": "mcq",
            "question": question, "options": opts, "gold_answer": gold_letter,
            "gold_doc_ids": [c["doc_id"]], "gold_chunk_ids": [cid],
            "source_anchor": pair["num"], "layer": "table_numeric",
            "difficulty": "medium", "generation_note": f"表格锚点句: {pair['sentence'][:60]}",
        })
    return items


def gen_multi_hop(systems: dict, domain: str, rng: random.Random, max_items: int,
                  pairs_cache: dict = None) -> list[dict]:
    """跨文档关联题（20%）：同领域两个文档各抽一个事实，出实体归属题"""
    items = []
    if pairs_cache is None:
        pairs_cache = build_pairs_cache(systems[domain])
    domain_chunks = defaultdict(list)
    for cid, c in systems[domain].chunk_data.items():
        if c.get("chunk_type") == "text" and len(c["text"]) > 60:
            domain_chunks[c["doc_id"]].append((cid, c))
    doc_ids = sorted(domain_chunks.keys())
    if len(doc_ids) < 2:
        return items
    rng.shuffle(doc_ids)
    for i in range(0, len(doc_ids) - 1, 2):
        if len(items) >= max_items:
            break
        doc_a, doc_b = doc_ids[i], doc_ids[i + 1]
        # 跨文档题指标只接受词表命中（回退指标是句子片段，噪声率过高不出题）
        # 实体仅接受已知主体词表命中，其余置空（组句时省略实体前缀，避免"扰乱证券"类噪声）
        def _vocab_pairs(doc_id):
            return [dict(p, entity=p["entity"] if p["entity"] in KNOWN_ENTITIES else "")
                    for cid, _ in domain_chunks[doc_id]
                    for p in (pairs_cache.get(cid) or [])
                    if p.get("is_vocab") and not re.search(r"[（(）)：:，,、]", p["metric"])]
        pa, pb = _vocab_pairs(doc_a), _vocab_pairs(doc_b)
        if not pa or not pb:
            continue
        fb = pb[0]
        # 每 doc pair 最多出 2 题（不同 fa 事实），提升跨文档层产量
        made = 0
        for fa in pa:
            if made >= 2 or len(items) >= max_items:
                break
            # 两个事实数值必须不同，否则对比题无意义
            if normalize_num(fa["num"]) == normalize_num(fb["num"]):
                continue
            if not fa["metric"] or not fa["sentence"]:
                continue
            gold_sent = f"{fa['entity']}的{fa['metric']}为{fa['num']}" if fa["entity"] else f"{fa['metric']}为{fa['num']}"
            # 对比项/干扰项同样省略空实体（统一"指标为数值"句式，保证选项风格一致）
            wrong_sent = f"{fb['entity'] + '的' if fb['entity'] else ''}{fa['metric']}为{fb['num']}"
            # 干扰项补足：对金标数值做篡改，生成另外两个假表述（保留原数值单位后缀，防止凭单位排除干扰项）
            try:
                v = float(normalize_num(fa["num_core"]))
                suffix = normalize_num(fa["num"])[len(normalize_num(fa["num_core"])):]
                fakes = [f"{v * 2:.2f}".rstrip("0").rstrip(".") + suffix,
                         f"{v * 1.1:.2f}".rstrip("0").rstrip(".") + suffix]
            except ValueError:
                fakes = []
            pre = f"{fa['entity']}的" if fa["entity"] else ""
            extra = [f"{pre}{fa['metric']}为{fk}" for fk in fakes]
            question = f"根据文档，关于{fa['metric']}，以下哪个表述与文档一致？"
            opts, gold_letter = build_mcq_options(gold_sent, [wrong_sent] + extra, rng)
            if len(opts) < 4:
                continue
            items.append({
                "domain": domain, "answer_format": "mcq",
                "question": question, "options": opts, "gold_answer": gold_letter,
                "gold_doc_ids": [doc_a], "gold_chunk_ids": [domain_chunks[doc_a][0][0]],
                "source_anchor": fa["num"], "layer": "multi_hop",
                "difficulty": "hard",
                "generation_note": f"对比文档: {doc_a} vs {doc_b}",
            })
            made += 1
    return items


def gen_temporal(systems: dict, rng: random.Random, max_items: int,
                 pairs_cache: dict = None) -> list[dict]:
    """时效/版本题（10%）：4 家双年报公司，2025 年报相对 2024 年报的同指标对比

    质量约束：两年 pair 的指标词必须相同且来自词表尾部命中（回退噪声指标不出题）
    """
    items = []
    fr = systems.get("financial_reports")
    if fr is None:
        return items
    if pairs_cache is None:
        pairs_cache = build_pairs_cache(fr)
    for company, (doc24, doc25) in DUAL_REPORT_COMPANIES.items():
        if len(items) >= max_items:
            break
        # 各自遍历 chunk，收集"指标词表命中 + 公司相关"的 pair，按指标归组
        # 实体放宽：年报正文多用"本集团/本公司"自指，文档前缀已锁定公司归属
        SELF_REF = ("本集团", "本公司", "该公司", "公司")
        year_pairs = {}
        for year, prefix in (("2024", doc24), ("2025", doc25)):
            metric_map = defaultdict(list)
            for cid, c in fr.chunk_data.items():
                # 年报数值多在表格 chunk 中，不限定 chunk_type
                if not cid.startswith(prefix):
                    continue
                for p in (pairs_cache.get(cid) or []):
                    if p.get("is_vocab") and (
                            (p["entity"] and company in p["entity"])
                            or any(sr in p["sentence"] for sr in SELF_REF)):
                        metric_map[p["metric"]].append((cid, p))
                        break  # 每 chunk 只取一个 pair，避免堆在同一章
            year_pairs[year] = metric_map
        # 找两年共有的指标，取数值不同的出题（每公司最多 2 题，保持公司覆盖广度）
        made = 0
        common = set(year_pairs["2024"]) & set(year_pairs["2025"])
        for metric in sorted(common):
            if made >= 2 or len(items) >= max_items:
                break
            cid24, p24 = year_pairs["2024"][metric][0]
            cid25, p25 = year_pairs["2025"][metric][0]
            if normalize_num(p24["num"]) == normalize_num(p25["num"]):
                continue
            correct = f"{company}{p25['metric']}为{p25['num']}"
            wrong = f"{company}{p25['metric']}为{p24['num']}"
            # 干扰项补足：对 2025 数值做篡改，凑满 4 选项（保留原数值单位后缀）
            try:
                v = float(normalize_num(p25["num_core"]))
                suffix = normalize_num(p25["num"])[len(normalize_num(p25["num_core"])):]
                fakes = [f"{v * 2:.2f}".rstrip("0").rstrip(".") + suffix,
                         f"{v * 1.1:.2f}".rstrip("0").rstrip(".") + suffix]
            except ValueError:
                fakes = []
            extra = [f"{company}{p25['metric']}为{fk}" for fk in fakes]
            question = f"根据文档，{company}2025年度{p25['metric']}的表述，正确的是？"
            opts, gold_letter = build_mcq_options(correct, [wrong] + extra, rng)
            if len(opts) < 4:
                continue
            items.append({
                "domain": "financial_reports", "answer_format": "mcq",
                "question": question, "options": opts, "gold_answer": gold_letter,
                "gold_doc_ids": [doc25], "gold_chunk_ids": [cid25],
                "source_anchor": p25["num"], "layer": "temporal",
                "difficulty": "hard",
                "generation_note": f"双年报对比 {doc24}/{doc25}: {p25['sentence'][:50]}",
            })
            made += 1
    return items


def gen_unanswerable(system, domain: str, rng: random.Random, max_items: int,
                     full_texts: dict, pairs_cache: dict = None) -> list[dict]:
    """无答案陷阱题（15%）：问目标文档确实没有的信息，金标=REFUSED

    门禁：实体+指标关键词在目标文档全文中不共现，确保是真陷阱
    """
    items = []
    if pairs_cache is None:
        pairs_cache = build_pairs_cache(system)
    # 从同领域"其他文档"抽取真实事实作为陷阱素材
    by_doc = defaultdict(list)
    for cid, c in system.chunk_data.items():
        if c.get("chunk_type") == "text" and len(c["text"]) > 60:
            pairs = pairs_cache.get(cid) or []
            if pairs:
                by_doc[c["doc_id"]].append((cid, pairs))
    doc_ids = sorted(by_doc.keys())
    rng.shuffle(doc_ids)
    for target_doc in doc_ids:
        if len(items) >= max_items:
            break
        target_full = full_texts.get(target_doc, "")
        if not target_full:
            continue
        # 从其他文档找一条"目标文档不存在"的事实
        src_docs = [d for d in doc_ids if d != target_doc]
        rng.shuffle(src_docs)
        placed = False
        for src_doc in src_docs:
            for cid, pairs in by_doc[src_doc]:
                p = pairs[0]
                if not p["entity"]:
                    continue
                # 反向验证：实体与指标在目标文档中不共现
                if p["entity"] in target_full and p["metric"] in target_full:
                    continue
                # 双保险：数值也不能出现在目标文档
                if anchor_in_text(p["num"], target_full):
                    continue
                question = (f"根据文档，{p['entity']}的{p['metric']}数值是多少？"
                            f"（若文档未提及，请拒答）")
                # 干扰项池：从其他文档取真实数值（排除陷阱素材本身的金标值）
                pool = [x["num"] for d2 in src_docs[:10]
                        for _, ps in by_doc[d2][:5] for x in ps if x["num"] != p["num"]]
                distractors = make_distractors(p["num"], pool, rng)
                if len(distractors) < 3:
                    continue
                opts, _ = build_mcq_options(p["num"], distractors, rng)
                items.append({
                    "domain": domain, "answer_format": "mcq",
                    "question": question, "options": opts, "gold_answer": REFUSED,
                    "gold_doc_ids": [target_doc], "gold_chunk_ids": [],
                    "source_anchor": "", "layer": "unanswerable",
                    "difficulty": "medium",
                    "generation_note": f"陷阱素材来自 {src_doc}: {p['sentence'][:50]}",
                })
                placed = True
                break
            if placed:
                break
    return items


def gen_tf(system, domain: str, rng: random.Random, max_items: int,
           pairs_cache: dict = None) -> list[dict]:
    """判断题（补充题型，A=正确/B=错误）：True=原文事实句；False=数值篡改句"""
    items = []
    if pairs_cache is None:
        pairs_cache = build_pairs_cache(system)
    cand_chunks = [(cid, c) for cid, c in system.chunk_data.items()
                   if c.get("chunk_type") == "text" and len(c["text"]) > 60]
    rng.shuffle(cand_chunks)
    for cid, c in cand_chunks:
        if len(items) >= max_items:
            break
        pairs = pairs_cache.get(cid) or []
        if not pairs:
            continue
        p = pairs[0]
        true_claim = p["sentence"]
        # 构造 False 句：篡改数值（×2），并验证篡改值不在原文
        try:
            v = float(normalize_num(p["num_core"]))
            fake = f"{v * 2:.2f}".rstrip("0").rstrip(".")
        except ValueError:
            continue
        fake_full = p["sentence"].replace(p["num_core"], fake, 1)
        if fake_full == p["sentence"] or anchor_in_text(fake, c["text"]):
            continue
        # 随机决定金标是 True 还是 False
        is_true = rng.random() < 0.5
        claim = true_claim if is_true else fake_full
        items.append({
            "domain": domain, "answer_format": "tf",
            "question": f"根据文档判断以下说法的正误：\n{claim}",
            "options": ["A. 正确", "B. 错误"],
            "gold_answer": "A" if is_true else "B",
            "gold_doc_ids": [c["doc_id"]], "gold_chunk_ids": [cid],
            "source_anchor": p["num"], "layer": "single_hop",
            "difficulty": "easy",
            "generation_note": f"原句: {true_claim[:60]}",
        })
    return items


# ============ 质检 ============

def quality_check(item: dict, systems: dict, full_texts: dict) -> tuple[bool, str]:
    """题目前置质检：锚点可命中金标 chunk 文本 / 陷阱题反向验证 / 选项数量正确"""
    if item["answer_format"] == "mcq" and len(item["options"]) != 4:
        return False, "选项数量不足4"
    if item["layer"] == "unanswerable":
        # 陷阱题：金标为 REFUSED，要求题干关键词在目标文档不共现
        doc_id = item["gold_doc_ids"][0]
        ft = full_texts.get(doc_id, "")
        m = re.search(r"根据文档，(.+?)的(.+?)数值是多少", item["question"])
        if m and ft:
            ent, met = m.group(1), m.group(2)
            if ent in ft and met in ft:
                return False, "陷阱题实体与指标在原文共现"
        return True, "ok"
    # 非陷阱题：锚点必须命中金标 chunk 文本
    for cid in item["gold_chunk_ids"]:
        chunk = systems[item["domain"]].chunk_data.get(cid)
        if chunk and anchor_in_text(item["source_anchor"], chunk["text"]):
            return True, "ok"
    return False, "锚点未命中金标chunk"


def build_doc_full_texts(systems: dict) -> dict:
    """拼接各文档全部 chunk 文本，作为全文反向验证语料（避免重复读原始 JSON）"""
    full_texts = {}
    for domain, system in systems.items():
        by_doc = defaultdict(str)
        for cid, c in system.chunk_data.items():
            by_doc[c["doc_id"]] += c.get("text", "")
        for doc_id, txt in by_doc.items():
            full_texts[doc_id] = txt
    return full_texts
