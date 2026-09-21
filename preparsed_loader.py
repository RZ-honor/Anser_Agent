"""
预解析结果加载器 - 从 MinerU VLM 预解析的 JSON 文件加载文档
用于本地环境（无 GPU）直接使用云端解析结果
"""
import os
import json
import sys

from config import DATASET_DIR, QUESTIONS_DIR


# 预解析 JSON 文件目录（可通过环境变量覆盖）
PREPARSED_DIR = os.environ.get("PREPARSED_DIR", r"D:\PROJECT\tianci\1")


def _load_doc_json(json_path: str) -> dict:
    """加载单个预解析 JSON 文件，转换为 pipeline 标准格式"""
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    doc_id = data.get("doc_id", os.path.splitext(os.path.basename(json_path))[0])
    pages = data.get("pages", [])

    # 将 tables 按 page 索引
    tables_by_page = {}
    for t in data.get("tables", []):
        pg = t.get("page", 0)
        if pg not in tables_by_page:
            tables_by_page[pg] = []
        tables_by_page[pg].append(t)

    # 转换 pages 格式，将 tables 注入 blocks
    converted_pages = []
    for page in pages:
        page_num = page.get("page_num", 0)
        blocks = list(page.get("blocks", []))

        # 将该页的 tables 添加为 table blocks
        for tbl in tables_by_page.get(page_num, []):
            md = tbl.get("markdown", "")
            html = tbl.get("html", "")
            if md or html:
                blocks.append({
                    "type": "table",
                    "text": md,
                    "html": html,
                    "markdown": md,
                })

        converted_pages.append({
            "page_num": page_num,
            "blocks": blocks,
            "text": page.get("text", ""),
        })

    return {
        "doc_id": doc_id,
        "filename": data.get("filename", os.path.basename(json_path)),
        "path": data.get("path", json_path),
        "page_count": data.get("page_count", len(pages)),
        "pages": converted_pages,
        "full_text": data.get("full_text", ""),
    }


def _infer_domain(doc_id: str, filename: str) -> str:
    """根据 doc_id/filename 推断文档领域"""
    did = doc_id.lower()
    fn = filename.lower()

    if did.startswith("text") or did.startswith("fc_"):
        return "financial_contracts"
    if did.startswith("pack2_text"):
        return "research"
    if "annual_" in did or "report" in fn:
        return "financial_reports"
    if did.startswith(("ins_", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16")):
        # 保险文档：1-16.json 是保险条款
        return "insurance"
    if "csrc_" in did or "att" in did or "strict_v3" in did:
        return "regulatory"

    # 默认：根据 questions 文件中的 doc_ids 匹配
    return ""


def _get_doc_domain_mapping() -> dict:
    """从 questions 文件中读取 doc_id -> domain 映射"""
    mapping = {}
    # 优先使用实际题目目录（config.py的路径可能指向不存在的本地子目录）
    questions_dir = QUESTIONS_DIR
    if not os.path.isdir(questions_dir):
        # 回退到实际题目目录
        fallback = r"D:\PROJECT\tianci\public_dataset_a\public_dataset_upload\questions\group_a"
        if os.path.isdir(fallback):
            questions_dir = fallback
        else:
            return mapping

    for fname in os.listdir(questions_dir):
        if not fname.endswith('_questions.json'):
            continue
        fpath = os.path.join(questions_dir, fname)
        with open(fpath, encoding='utf-8') as f:
            questions = json.load(f)
        for q in questions:
            domain = q.get("domain", "")
            for did in q.get("doc_ids", []):
                mapping[did] = domain
    return mapping


def load_preparsed_documents(preparsed_dir: str = None) -> dict:
    """
    加载所有预解析 JSON 文件，按领域分组。

    Returns:
        {domain: {doc_id: {"filename", "path", "pages", "doc_id", ...}}}
    """
    preparsed_dir = preparsed_dir or PREPARSED_DIR
    if not os.path.isdir(preparsed_dir):
        print(f"[错误] 预解析目录不存在: {preparsed_dir}")
        return {}

    # 获取 doc_id -> domain 映射
    doc_domain_map = _get_doc_domain_mapping()

    all_docs = {}
    json_files = sorted(f for f in os.listdir(preparsed_dir) if f.endswith('.json'))

    print(f"加载 {len(json_files)} 个预解析文件...")

    for fname in json_files:
        json_path = os.path.join(preparsed_dir, fname)
        try:
            doc_data = _load_doc_json(json_path)
        except Exception as e:
            print(f"  [警告] 加载 {fname} 失败: {e}")
            continue

        doc_id = doc_data["doc_id"]

        # 确定领域
        domain = doc_domain_map.get(doc_id, "")
        if not domain:
            domain = _infer_domain(doc_id, doc_data["filename"])
        if not domain:
            domain = "unknown"

        if domain not in all_docs:
            all_docs[domain] = {}
        all_docs[domain][doc_id] = doc_data

    # 统计
    for domain, docs in all_docs.items():
        total_pages = sum(d.get("page_count", 0) for d in docs.values())
        print(f"  {domain}: {len(docs)} 文档, {total_pages} 页")

    return all_docs


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    all_docs = load_preparsed_documents()
    print(f"\n总计: {len(all_docs)} 领域")
    for domain, docs in all_docs.items():
        print(f"  {domain}: {len(docs)} 文档")
