"""选项级证据全面检索脚本

功能：
  1. 加载 b_questions_with_evidence.jsonl 中所有100题
  2. 对每道选择题的每个选项，使用项目检索工具（new_retrieval_system + supplement_search）检索证据
  3. 对证据为空或不足的选项，使用 supplement_search 进行精确补充检索
  4. 输出 b_option_evidence_full.jsonl，包含每题每选项的证据

运行环境：conda Audio
"""
import os
import sys
import json
import re
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# 引入项目检索工具
from new_retrieval_system import build_domain_indexes, NewRetrievalSystem
from pdf_parser import chunk_blocks
from supplement_search import load_doc_text, search as supplement_search_fn, resolve_doc_ids, QID_DOMAIN

# ============ 数据库路径 ============
PROCESSED_DIR = os.path.join(BASE, "submission", "processed_data")
EXTRA_REGULATORY_DIR = r"D:\PROJECT\tianci\1"

# 题目目录
QUESTIONS_DIR = os.path.join(BASE, "question_b")

# 输出文件
OUTPUT_PATH = os.path.join(BASE, "b_option_evidence_full.jsonl")

# ============ 领域文档映射 ============
DOMAIN_FILES = {
    "insurance": [str(i) for i in range(1, 17)],
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
            for d in [PROCESSED_DIR, EXTRA_REGULATORY_DIR]:
                candidate = os.path.join(d, f"{doc_id}.json")
                if os.path.isfile(candidate):
                    try:
                        with open(candidate, encoding="utf-8") as f:
                            all_docs[domain][doc_id] = json.load(f)
                    except Exception:
                        pass
                    break
    # 分块
    for domain, domain_docs in all_docs.items():
        for doc_id, doc_data in domain_docs.items():
            if "chunks" not in doc_data:
                page_list = [
                    {"page_num": p.get("page_num", 0), "text": p.get("text", ""), "blocks": p.get("blocks", [])}
                    for p in doc_data.get("pages", [])
                ]
                chunks = chunk_blocks(page_list, doc_id)
                if not chunks:
                    from pdf_parser import split_into_chunks
                    raw_chunks = split_into_chunks(doc_data.get("full_text", ""))
                    chunks = [
                        {"chunk_id": c["chunk_id"], "text": c["text"], "has_table": False, "table_html": "", "page": 0, "doc_id": doc_id}
                        for c in raw_chunks
                    ]
                doc_data["chunks"] = chunks
    return all_docs


def normalize_text(text):
    """去除 LaTeX 转义符"""
    if not text:
        return ""
    text = re.sub(r"\\\(\s*", "", text)
    text = re.sub(r"\s*\\\)", "", text)
    text = re.sub(r"\\%", "%", text)
    text = re.sub(r"\\,", "", text)
    text = re.sub(r"\\ ", " ", text)
    text = re.sub(r"(\d)\s+(万元|亿元|元/股|元|股|张|年|月|日|倍|个|名|人|%)", r"\1\2", text)
    return text


def extract_option_keywords(option_text):
    """从选项文本中提取搜索关键词"""
    keywords = []
    # 数值
    keywords += re.findall(r"\d+(?:\.\d+)?\s*(?:年|月|日|%|亿|万|元|倍|个|系数|kwh|kWh)", option_text)
    keywords += re.findall(r"\d{4}|\d+\.\d+%", option_text)
    # 条款号
    keywords += re.findall(r"第[一二三四五六七八九十百千\d]+\s*[条章节款项]", option_text)
    # 法规名
    keywords += re.findall(r"《[^》]{3,80}》", option_text)
    # 中文实义词（2-6字）
    cn_words = re.findall(r"[\u4e00-\u9fa5]{2,6}", option_text)
    stopwords = {"的", "了", "为", "在", "和", "与", "或", "及", "等", "中", "其", "该", "此",
                 "是", "不", "未", "无", "有", "对", "由", "从", "到", "向", "被", "把", "让",
                 "关于", "根据", "下列", "以下", "选项", "描述", "内容", "文档", "文件",
                 "其中", "本期", "发行", "公司", "正确", "错误", "成立", "符合", "是否",
                 "进行", "相关", "具体", "规定", "如下", "上述", "可以", "应当", "不得",
                 "如果", "由于", "因为", "所以", "但是", "并且", "以及", "对于", "通过",
                 "说法", "哪些", "包括", "不超过", "不超过", "本次", "交易", "标的", "资产",
                 "发行人", "投资者", "持有人", "合同", "协议", "条款", "金额", "比例", "期限"}
    cn_words = [w for w in cn_words if w not in stopwords]
    keywords += cn_words
    # 去重
    seen = set()
    unique_kws = []
    for kw in keywords:
        if kw not in seen and len(kw) >= 2:
            seen.add(kw)
            unique_kws.append(kw)
    return unique_kws[:10]


def search_option_in_docs(option_text, domain_docs, domain_systems, domain, qid=None, top_n=3):
    """对单个选项在领域文档中搜索证据

    三级检索策略：
    1. BM25系统检索（top_k提升到8以覆盖排序靠后的关键chunk）
    2. 领域内文档直接关键词搜索（原有逻辑）
    3. QID_DOMAIN兜底检索（针对特定题目在指定文档中搜索，解决表格chunk排序低的问题）
    """
    # 方法1：使用项目检索系统（BM25）- top_k从3提升到8
    sys_evidence = []
    retrieval_sys = domain_systems.get(domain)
    if retrieval_sys:
        try:
            sys_evidence = retrieval_sys.search(
                question=option_text, options={},
                doc_ids=None, top_k=max(8, top_n * 2),
            )
        except Exception:
            pass

    # 方法2：领域内文档直接关键词搜索（supplement_search）
    keywords = extract_option_keywords(option_text)
    direct_evidence = []
    for doc_id, doc_data in domain_docs.items():
        text = "\n".join(p.get("text", "") for p in doc_data.get("pages", []))
        text = normalize_text(text)
        if not text or not keywords:
            continue
        segments = supplement_search_fn(text, keywords, window=400, topn=3)
        for cnt, pos, seg, matched in segments:
            direct_evidence.append({
                "doc_id": doc_id,
                "text": seg[:300],
                "match_count": cnt,
                "matched_keywords": matched,
            })

    # 方法3：QID_DOMAIN兜底检索 - 针对特定题目在指定文档中搜索
    # 解决问题：表格chunk（如退保费用比例表）在BM25中得分低，需定向搜索
    fallback_evidence = []
    if qid and qid in QID_DOMAIN:
        fallback_doc_id = QID_DOMAIN[qid]
        fallback_doc_ids = resolve_doc_ids(fallback_doc_id)
        # 过滤已在domain_docs中搜索过的文档，避免重复
        existing_doc_ids = set(domain_docs.keys())
        extra_doc_ids = [d for d in fallback_doc_ids if d not in existing_doc_ids]
        for doc_id in extra_doc_ids:
            text = load_doc_text(doc_id)
            text = normalize_text(text)
            if not text or not keywords:
                continue
            segments = supplement_search_fn(text, keywords, window=500, topn=3)
            for cnt, pos, seg, matched in segments:
                fallback_evidence.append({
                    "doc_id": doc_id,
                    "text": seg[:300],
                    "match_count": cnt,
                    "matched_keywords": matched,
                })
        # 对QID_DOMAIN中指定但在domain_docs中的文档，用更大的window重新搜索
        for doc_id in fallback_doc_ids:
            if doc_id in existing_doc_ids and doc_id in domain_docs:
                text = "\n".join(p.get("text", "") for p in domain_docs[doc_id].get("pages", []))
                text = normalize_text(text)
                if not text or not keywords:
                    continue
                segments = supplement_search_fn(text, keywords, window=600, topn=4)
                for cnt, pos, seg, matched in segments:
                    fallback_evidence.append({
                        "doc_id": doc_id,
                        "text": seg[:300],
                        "match_count": cnt,
                        "matched_keywords": matched,
                    })

    # 合并去重 - 优先fallback > direct > sys（fallback最精准）
    all_ev = []
    seen_texts = set()
    # 优先fallback证据（QID_DOMAIN定向搜索，最精准）
    for ev in sorted(fallback_evidence, key=lambda x: -x["match_count"])[:top_n]:
        key = ev["text"][:80]
        if key not in seen_texts:
            seen_texts.add(key)
            all_ev.append(ev)
    # 其次direct证据
    for ev in sorted(direct_evidence, key=lambda x: -x["match_count"])[:top_n]:
        key = ev["text"][:80]
        if key not in seen_texts:
            seen_texts.add(key)
            all_ev.append(ev)
    # 最后sys证据
    for ev in sys_evidence[:top_n]:
        ev_text = ev.get("text", "")[:300]
        key = ev_text[:80]
        if key not in seen_texts:
            seen_texts.add(key)
            all_ev.append({
                "doc_id": ev.get("doc_id", ""),
                "text": ev_text,
                "match_count": 1,
                "matched_keywords": [],
            })

    return all_ev


def _read_json_file(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _read_jsonl_file(path):
    records = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_b_questions():
    """加载所有B榜题目"""
    questions = []
    fc_path = os.path.join(QUESTIONS_DIR, "financial_contracts_b_question.json")
    if os.path.isfile(fc_path):
        questions.extend(_read_json_file(fc_path))
    ins_path = os.path.join(QUESTIONS_DIR, "insurance_b_questions.json")
    if os.path.isfile(ins_path):
        questions.extend(_read_json_file(ins_path))
    fr_path = os.path.join(QUESTIONS_DIR, "financial_reports_b_questions.jsonl")
    if os.path.isfile(fr_path):
        questions.extend(_read_jsonl_file(fr_path))
    reg_path = os.path.join(QUESTIONS_DIR, "regulatory_b_questions.jsonl")
    if os.path.isfile(reg_path):
        questions.extend(_read_jsonl_file(reg_path))
    res_path = os.path.join(QUESTIONS_DIR, "research_b_question.jsonl")
    if os.path.isfile(res_path):
        questions.extend(_read_jsonl_file(res_path))
    questions.sort(key=lambda q: q["qid"])
    return questions


def main():
    print("加载B榜题目...")
    questions = load_b_questions()
    print(f"共 {len(questions)} 道题")

    print("\n构建检索系统索引...")
    all_docs = load_all_docs()
    for domain, docs in all_docs.items():
        print(f"  {domain}: {len(docs)} 个文档")
    domain_systems = build_domain_indexes(all_docs)
    print("索引构建完成")

    print("\n开始选项级证据检索...")
    stats = defaultdict(int)  # 统计
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for i, q in enumerate(questions, 1):
            qid = q["qid"]
            domain = q.get("domain", "")
            question = q.get("question", "")
            options = q.get("options", {})
            qtype = q.get("type", "")

            record = {
                "qid": qid,
                "domain": domain,
                "type": qtype,
                "question": question,
                "options": options,
                "option_evidence": {},
            }

            # 计算题：检索问题关键词证据
            if qtype == "计算题":
                # 对计算题，检索问题本身
                domain_docs = all_docs.get(domain, {})
                ev_list = search_option_in_docs(question, domain_docs, domain_systems, domain, qid=qid, top_n=5)
                record["option_evidence"]["_calc_"] = ev_list
                stats["calc_with_evidence"] += 1
            else:
                # 选择题/判断题：对每个选项检索证据
                domain_docs = all_docs.get(domain, {})
                for opt_key, opt_text in sorted(options.items()):
                    ev_list = search_option_in_docs(opt_text, domain_docs, domain_systems, domain, qid=qid, top_n=3)
                    record["option_evidence"][opt_key] = ev_list
                    if ev_list:
                        stats["opt_with_evidence"] += 1
                    else:
                        stats["opt_without_evidence"] += 1
                        stats[f"no_ev_{domain}"] += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if i % 10 == 0:
                print(f"  已检索 {i}/{len(questions)} 题")

    print(f"\n选项级证据检索完成，输出到: {OUTPUT_PATH}")
    print(f"\n统计:")
    print(f"  计算题有证据: {stats['calc_with_evidence']}")
    print(f"  选项有证据: {stats['opt_with_evidence']}")
    print(f"  选项无证据: {stats['opt_without_evidence']}")
    if stats['opt_without_evidence'] > 0:
        print(f"  无证据分布:")
        for k, v in stats.items():
            if k.startswith("no_ev_"):
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
