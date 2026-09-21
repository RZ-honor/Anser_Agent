"""从年报文档批量提取关键财务数据，生成 fin 题计算所需数据表

提取指标：
  - 营业收入、归母净利润、经营活动现金流净额、基本每股收益
  - 资产负债率、流动比率、速动比率、研发费用
  - 现金分红（每10股）
  - 分地区营业收入（境外/境内）
"""
import os
import json
import re

BASE = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE, "submission", "processed_data")

# 年报文档映射
REPORT_FILES = {
    "byd_2024": "annual_byd_2024_report",
    "byd_2025": "annual_byd_2025_report",
    "catl_2024": "annual_catl_2024_report",
    "catl_2025": "annual_catl_2025_report",
    "midea_2024": "annual_midea_2024_report",
    "midea_2025": "annual_midea_2025_report",
    "cmb_2025": "annual_cmb_2025_report",
    "chinamobile_2025": "annual_chinamobile_2025_report",
    "cscec_2024": "annual_cscec_2024_report",
    "cscec_2025": "annual_cscec_2025_report",
}


def load_full_text(doc_id):
    """加载文档全文"""
    path = os.path.join(PROCESSED_DIR, f"{doc_id}.json")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    parts = []
    for p in data.get("pages", []):
        parts.append(p.get("text", ""))
    for c in data.get("chunks", []):
        parts.append(c.get("text", ""))
    return "\n".join(parts)


def search_context(text, keyword, window=400, topn=5):
    """搜索关键词上下文"""
    results = []
    start = 0
    while True:
        idx = text.find(keyword, start)
        if idx < 0:
            break
        seg_start = max(0, idx - window // 3)
        seg_end = min(len(text), idx + window * 2 // 3)
        results.append(text[seg_start:seg_end])
        start = idx + len(keyword)
        if len(results) >= topn:
            break
    return results


def extract_company_data(name, doc_id):
    """提取单个公司年报数据"""
    print(f"\n{'='*60}")
    print(f"公司: {name} ({doc_id})")
    print(f"{'='*60}")
    text = load_full_text(doc_id)
    if not text:
        print("  [未找到文档]")
        return

    # 关键指标搜索
    keywords = [
        "营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额",
        "基本每股收益", "资产负债率", "流动比率", "速动比率",
        "研发费用", "现金分红", "每10股", "境外",
    ]
    for kw in keywords:
        ctxs = search_context(text, kw, window=300, topn=3)
        if ctxs:
            print(f"\n--- {kw} ---")
            for i, c in enumerate(ctxs[:2]):
                # 清理换行便于阅读
                c = c.replace("\n", " ")[:280]
                print(f"  [{i}] {c}")


def main():
    for name, doc_id in REPORT_FILES.items():
        extract_company_data(name, doc_id)


if __name__ == "__main__":
    main()
