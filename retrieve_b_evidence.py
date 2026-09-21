"""
B 榜题目证据检索脚本

功能：
  1. 加载 D:\\PROJECT\\tianci\\upload_b\\question_b 下所有 B 榜题目
  2. 使用项目内置检索工具（new_retrieval_system）对每题检索证据
  3. 输出 b_questions_with_evidence.jsonl，供推理模型分批作答

运行环境：conda Audio
"""
import os
import sys
import json
import re

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# 引入项目检索系统（new_retrieval_system 依赖 jieba、rank_bm25、networkx）
from new_retrieval_system import build_domain_indexes, NewRetrievalSystem
from pdf_parser import chunk_blocks
from supplement_search import QID_DOMAIN, resolve_doc_ids


def chunk_document(doc_data, domain_name=None):
    """对单个文档进行分块（复用 pdf_parser.chunk_blocks）

    Args:
        doc_data: 文档数据 dict
        domain_name: 领域名（保留参数兼容，本脚本不读取领域专属分块参数）

    Returns:
        [{"chunk_id", "text", "has_table", "table_html", "page", "doc_id"}]
    """
    doc_id = doc_data.get("doc_id", "")
    pages = doc_data.get("pages", [])
    page_list = [
        {
            "page_num": p.get("page_num", 0),
            "text": p.get("text", ""),
            "blocks": p.get("blocks", []),
        }
        for p in pages
    ]
    chunks = chunk_blocks(page_list, doc_id)
    # 兜底：若结构化分块失败，用简单分块
    if not chunks:
        from pdf_parser import split_into_chunks
        raw_chunks = split_into_chunks(doc_data.get("full_text", ""))
        chunks = [
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "has_table": False,
                "table_html": "",
                "page": 0,
                "doc_id": doc_id,
            }
            for c in raw_chunks
        ]
    return chunks

# ============ 数据库路径 ============
# 项目下预处理好的文档数据库（data/<domain>/*.json）
PROCESSED_DIR = os.path.join(BASE, "submission", "processed_data")
# 部分监管文档存放在 D:\\PROJECT\\tianci\\1
EXTRA_REGULATORY_DIR = r"D:\PROJECT\tianci\1"

# 题目目录（B 榜）
QUESTIONS_DIR = os.path.join(BASE, "question_b")

# 输出文件
OUTPUT_PATH = os.path.join(BASE, "b_questions_with_evidence.jsonl")


# ============ 数据库文档加载（复用 run_competition.py 的领域映射） ============
DOMAIN_FILES = {
    "insurance": [str(i) for i in range(1, 17)],  # 修正：1-16共16个保险产品文件
    "financial_contracts": [f"text{i:02d}" for i in range(1, 15)],  # 实际只有text01-text14
    "financial_reports": [
        "annual_byd_2024_report", "annual_byd_2025_report",
        "annual_catl_2024_report", "annual_catl_2025_report",
        "annual_chinamobile_2025_report", "annual_cmb_2025_report",
        "annual_cscec_2024_report", "annual_cscec_2025_report",
        "annual_midea_2024_report", "annual_midea_2025_report",
    ],
    "regulatory": [f"csrc_{i:04d}_att1" for i in [9, 23, 27, 35, 37, 38]] + [
        "csrc_0262", "csrc_0271", "csrc_0377",
        "strict_v3_008_中国人民银行令〔2025〕第12号（金融机构客户受益所有人识别管理办法）",
        "strict_v3_009_中国人民银行_国家金融监督管理总局_中国证券监督管理委员会令〔2025〕第11号（金融机构客户尽职调查和客户身份资料及交易记录保存管理办法）",
        "strict_v3_015_中国人民银行令〔2025〕第3号（中国人民银行业务领域数据安全管理办法）",
        "strict_v3_016_中国人民银行_国家金融监督管理总局令〔2025〕第2号（银行卡清算机构管理办法）",
        "strict_v3_017_中华人民共和国反洗钱法",
        "strict_v3_018_中国人民银行令〔2024〕第4号（非银行支付机构监督管理条例实施细则）",
    ],
    "research": [f"pack2_text{i:02d}" for i in range(1, 21)],
}


def load_all_docs():
    """加载所有领域的文档并分块"""
    all_docs = {}
    for domain, doc_ids in DOMAIN_FILES.items():
        all_docs[domain] = {}
        for doc_id in doc_ids:
            # 在预处理目录和额外监管目录中查找
            for d in [PROCESSED_DIR, EXTRA_REGULATORY_DIR]:
                candidate = os.path.join(d, f"{doc_id}.json")
                if os.path.isfile(candidate):
                    try:
                        with open(candidate, encoding="utf-8") as f:
                            all_docs[domain][doc_id] = json.load(f)
                    except Exception:
                        pass
                    break

    # 分块（复用 submission/script/build_all_graphs.py 的 chunk_document）
    for domain, domain_docs in all_docs.items():
        for doc_id, doc_data in domain_docs.items():
            if "chunks" not in doc_data:
                doc_data["chunks"] = chunk_document(doc_data, domain)

    return all_docs


# ============ 文本规范化（去除 LaTeX 转义符，与 run_competition.py 一致） ============
def normalize_text(text):
    """去除 LaTeX 转义符，统一数字与中文单位之间的空格"""
    if not text:
        return ""
    text = re.sub(r"\\\(\s*", "", text)
    text = re.sub(r"\s*\\\)", "", text)
    text = re.sub(r"\\%", "%", text)
    text = re.sub(r"\\,", "", text)
    text = re.sub(r"\\ ", " ", text)
    # 统一数字与中文单位之间的空格
    text = re.sub(
        r"(\d)\s+(万元|亿元|元/股|元|股|张|年|月|日|倍|个|名|人|%)",
        r"\1\2", text
    )
    return text


SEARCH_DIRS = [PROCESSED_DIR, EXTRA_REGULATORY_DIR]


def load_doc_text(doc_id):
    """加载单个文档全文（已规范化）"""
    for d in SEARCH_DIRS:
        path = os.path.join(d, f"{doc_id}.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                raw = "\n".join(p.get("text", "") for p in data.get("pages", []))
                return normalize_text(raw)
            except Exception:
                pass
    return ""


# ============ 直接文档关键词搜索（绕过索引 scope 限制） ============
def extract_keywords(question, options):
    """从问题和选项中提取搜索关键词"""
    keywords = []
    # 条款号
    keywords += re.findall(r"第[一二三四五六七八九十百千\d]+\s*[条章节款项]", question)
    # 法规名
    keywords += re.findall(r"《[^》]{3,80}》", question)
    # 数值
    all_text = question + " " + " ".join(str(v) for v in options.values())
    keywords += re.findall(r"\d+(?:\.\d+)?\s*(?:年|月|日|%|亿|万|元|倍|个|系数)", all_text)
    keywords += re.findall(r"\d{4}|\d+\.\d+%", all_text)
    # 中文实义词
    cn_words = re.findall(r"[\u4e00-\u9fa5]{2,}", all_text)
    stopwords = {"的", "了", "为", "在", "和", "与", "或", "及", "等", "中", "其", "该", "此",
                 "是", "不", "未", "无", "有", "对", "由", "从", "到", "向", "被", "把", "让",
                 "关于", "根据", "下列", "以下", "选项", "描述", "内容", "文档", "文件",
                 "其中", "本期", "发行", "公司", "正确", "错误", "成立", "符合", "是否",
                 "进行", "相关", "具体", "规定", "如下", "上述", "可以", "应当", "不得",
                 "如果", "由于", "因为", "所以", "但是", "并且", "以及", "对于", "通过",
                 "说法", "哪些", "包括", "不超过"}
    cn_words = [w for w in cn_words if w not in stopwords]
    keywords += cn_words

    # 去重，保留顺序
    seen = set()
    unique_kws = []
    for kw in keywords:
        if kw not in seen and len(kw) >= 2:
            seen.add(kw)
            unique_kws.append(kw)
    return unique_kws[:15]


def search_in_text(text, keywords, window=600):
    """在文本中搜索关键词，返回相关段落"""
    if not text or not keywords:
        return []

    positions = []
    for kw in keywords:
        if not kw or len(kw) < 2:
            continue
        start = 0
        while True:
            idx = text.find(kw, start)
            if idx < 0:
                break
            positions.append((idx, kw))
            start = idx + len(kw)

    if not positions:
        return []

    positions.sort()
    segments = []
    used_positions = set()

    for pos, kw in positions:
        if pos in used_positions:
            continue
        seg_start = max(0, pos - window // 3)
        seg_end = min(len(text), pos + window * 2 // 3)
        segment = text[seg_start:seg_end]

        for p in range(seg_start, seg_end):
            used_positions.add(p)

        matched_kws = [k for k in keywords if k and len(k) >= 2 and k in segment]
        segments.append({
            "text": segment,
            "matched_keywords": matched_kws,
            "match_count": len(matched_kws),
            "position": pos,
        })

    segments.sort(key=lambda x: (-x["match_count"], x["position"]))
    return segments


# ============ 加载 B 榜题目 ============
def _read_json_file(path):
    """读取 JSON 文件（兼容 UTF-8 BOM）"""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _read_jsonl_file(path):
    """读取 JSONL 文件（兼容 UTF-8 BOM）"""
    records = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_b_questions():
    """加载 question_b 目录下所有题目"""
    questions = []
    # financial_contracts_b_question.json（数组）
    fc_path = os.path.join(QUESTIONS_DIR, "financial_contracts_b_question.json")
    if os.path.isfile(fc_path):
        questions.extend(_read_json_file(fc_path))

    # insurance_b_questions.json（数组）
    ins_path = os.path.join(QUESTIONS_DIR, "insurance_b_questions.json")
    if os.path.isfile(ins_path):
        questions.extend(_read_json_file(ins_path))

    # financial_reports_b_questions.jsonl（每行一题）
    fr_path = os.path.join(QUESTIONS_DIR, "financial_reports_b_questions.jsonl")
    if os.path.isfile(fr_path):
        questions.extend(_read_jsonl_file(fr_path))

    # regulatory_b_questions.jsonl
    reg_path = os.path.join(QUESTIONS_DIR, "regulatory_b_questions.jsonl")
    if os.path.isfile(reg_path):
        questions.extend(_read_jsonl_file(reg_path))

    # research_b_question.jsonl
    res_path = os.path.join(QUESTIONS_DIR, "research_b_question.jsonl")
    if os.path.isfile(res_path):
        questions.extend(_read_jsonl_file(res_path))

    # 按 qid 排序
    questions.sort(key=lambda q: q["qid"])
    return questions


# ============ 主流程 ============
def main():
    print("加载 B 榜题目...")
    questions = load_b_questions()
    print(f"共 {len(questions)} 道题")

    # 按领域统计
    from collections import Counter
    domain_counts = Counter(q["domain"] for q in questions)
    print(f"领域分布: {dict(domain_counts)}")

    print("\n构建检索系统索引...")
    all_docs = load_all_docs()
    for domain, docs in all_docs.items():
        print(f"  {domain}: {len(docs)} 个文档")
    domain_systems = build_domain_indexes(all_docs)
    print("索引构建完成")

    # 逐题检索证据
    print("\n开始检索证据...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for i, q in enumerate(questions, 1):
            qid = q["qid"]
            domain = q.get("domain", "")
            question = q.get("question", "")
            options = q.get("options", {})
            qtype = q.get("type", "")
            doc_ids = q.get("doc_ids", [])  # B 榜通常不提供 doc_ids

            # 1. 系统检索（BM25 + 数值/短语/实体加分）
            sys_evidence = []
            retrieval_sys = domain_systems.get(domain)
            if retrieval_sys:
                top_k = 10 if qtype in ("多选题", "计算题") else 6
                sys_evidence = retrieval_sys.search(
                    question=question, options=options,
                    doc_ids=doc_ids if doc_ids else None,
                    top_k=top_k,
                )

            # 2. 直接文档关键词搜索（兜底，遍历该领域所有文档）
            keywords = extract_keywords(question, options)
            direct_evidence = []
            domain_docs = all_docs.get(domain, {})
            # 限制搜索文档数，避免耗时过长
            search_doc_ids = doc_ids if doc_ids else list(domain_docs.keys())
            for doc_id in search_doc_ids[:8]:
                text = load_doc_text(doc_id)
                if text:
                    segments = search_in_text(text, keywords, window=800)
                    for seg in segments:
                        seg["doc_id"] = doc_id
                        direct_evidence.append(seg)
            direct_evidence.sort(key=lambda x: (-x["match_count"], x.get("position", 0)))

            # 2.5 QID_DOMAIN兜底检索 - 针对特定题目在指定文档中搜索
            # 解决问题：表格chunk（如退保费用比例表、中期分红文本）在BM25中得分低
            fallback_evidence = []
            if qid in QID_DOMAIN:
                fallback_doc_id = QID_DOMAIN[qid]
                fallback_doc_ids = resolve_doc_ids(fallback_doc_id)
                # 补充搜索领域文档列表外的文档 + 加大window重搜领域内文档
                existing_search = set(search_doc_ids[:8])
                for doc_id in fallback_doc_ids:
                    text = load_doc_text(doc_id)
                    if not text:
                        continue
                    window = 800 if doc_id in existing_search else 1000
                    segments = search_in_text(text, keywords, window=window)
                    for seg in segments:
                        seg["doc_id"] = doc_id
                        seg["is_fallback"] = True
                        fallback_evidence.append(seg)
                fallback_evidence.sort(key=lambda x: (-x["match_count"], x.get("position", 0)))

            # 3. 多选题选项级证据（每个选项单独检索）
            option_evidence = {}
            if qtype == "多选题" and options:
                for opt_key, opt_text in sorted(options.items()):
                    opt_kws = extract_keywords(opt_text, {})
                    opt_ev = []
                    for doc_id in search_doc_ids[:8]:
                        text = load_doc_text(doc_id)
                        if text:
                            segments = search_in_text(text, opt_kws, window=500)
                            for seg in segments:
                                seg["doc_id"] = doc_id
                                opt_ev.append(seg)
                    opt_ev.sort(key=lambda x: (-x["match_count"], x.get("position", 0)))
                    option_evidence[opt_key] = opt_ev[:3]

            # 4. 合并证据（去重，取前 N 条）
            # 优先级：fallback（定向搜索最精准） > direct（领域内搜索） > sys（BM25系统）
            all_evidence = []
            seen_texts = set()
            # 优先fallback证据
            for ev in fallback_evidence[:4]:
                ev_text = ev.get("text", "")[:400]
                key = ev_text[:100]
                if key not in seen_texts:
                    seen_texts.add(key)
                    all_evidence.append({
                        "text": ev_text,
                        "doc_id": ev.get("doc_id", ""),
                        "match_count": ev.get("match_count", 0),
                        "matched_keywords": ev.get("matched_keywords", []),
                        "is_fallback": True,
                    })
            # 其次direct证据
            for ev in direct_evidence[:6]:
                ev_text = ev.get("text", "")[:400]
                key = ev_text[:100]
                if key not in seen_texts:
                    seen_texts.add(key)
                    all_evidence.append({
                        "text": ev_text,
                        "doc_id": ev.get("doc_id", ""),
                        "match_count": ev.get("match_count", 0),
                        "matched_keywords": ev.get("matched_keywords", []),
                    })
            # 补充系统检索证据
            for ev in sys_evidence[:4]:
                ev_text = ev.get("text", "")[:400]
                key = ev_text[:100]
                if key not in seen_texts:
                    seen_texts.add(key)
                    all_evidence.append({
                        "text": ev_text,
                        "doc_id": ev.get("doc_id", ""),
                        "match_count": 1,
                        "matched_keywords": [],
                    })

            # 5. 输出
            record = {
                "qid": qid,
                "domain": domain,
                "question": question,
                "type": qtype,
                "options": options,
                "evidence": all_evidence,
                "option_evidence": option_evidence,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if i % 10 == 0:
                print(f"  已检索 {i}/{len(questions)} 题")

    print(f"\n证据检索完成，输出到: {OUTPUT_PATH}")

    # 统计证据数量
    total_ev = 0
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            total_ev += len(r.get("evidence", []))
    print(f"平均每题证据: {total_ev / len(questions):.1f} 条")


if __name__ == "__main__":
    main()
