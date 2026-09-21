"""
多尺度符号索引 - 无 embedding 的离散符号召回基础
"""
import re
from collections import defaultdict
from typing import Optional

# ============ 符号提取 ============

METRIC_KEYWORDS = [
    '营业收入', '净利润', '总资产', '净资产', '负债率', '资产负债率',
    '现金流', '研发费用', '毛利率', '净利率', 'ROE', 'ROA',
    '每股收益', '市盈率', '市净率', '分红', '派息',
    '保险金', '保费', '赔付', '免赔额', '等待期',
    '利率', '费率', '注册资本', '资本充足率',
    '经营活动', '投资活动', '筹资活动',
    '归母净利润', '扣非净利润', '基本每股收益',
]

PERIOD_PATTERN = re.compile(
    r'(?:20\d{2}(?:年|年度)?|'
    r'第[一二三四]季度|'
    r'\d{4}[-/]\d{2}(?:[-/]\d{2})?|'
    r'(?:半年|年度)报告)'
)

NUMBER_PATTERN = re.compile(
    r'[\d,]+\.?\d*\s*(?:万元|亿元|元|万亿|万|百万|千万)'
)

PERCENTAGE_PATTERN = re.compile(r'[\d.]+\s*%')

CLAUSE_PATTERN = re.compile(
    r'(?:第[一二三四五六七八九十百千\d]+条|'
    r'第[（(][一二三四五六七八九十\d]+[)）]款|'
    r'条款\s*\d+|'
    r'第\s*\d+\s*条)'
)


def extract_keywords(text: str, min_len: int = 2, max_len: int = 12) -> list:
    """提取关键词（简单分词）"""
    # 在 CJK 与非 CJK 之间插入空格，再在标点处断开
    t = re.sub(r'([一-鿿])([0-9a-zA-Z])', r'\1 \2', text)
    t = re.sub(r'([0-9a-zA-Z])([一-鿿])', r'\1 \2', t)
    t = re.sub(r'[^一-鿿\w]', ' ', t)
    words = t.split()
    # 过滤纯数字和长度
    keywords = [w for w in words if min_len <= len(w) <= max_len and not w.isdigit()]
    return list(set(keywords))


def extract_metrics(text: str) -> list:
    """提取财务指标"""
    found = []
    for m in METRIC_KEYWORDS:
        if m in text:
            found.append(m)
    return found


def extract_periods(text: str) -> list:
    """提取时间/时期"""
    return PERIOD_PATTERN.findall(text)


def extract_numbers(text: str) -> list:
    """提取数值/金额"""
    return NUMBER_PATTERN.findall(text)


def extract_percentages(text: str) -> list:
    """提取百分比"""
    return PERCENTAGE_PATTERN.findall(text)


def extract_clauses(text: str) -> list:
    """提取条款号"""
    return CLAUSE_PATTERN.findall(text)


# ============ 索引构建 ============

def build_symbolic_index(G) -> dict:
    """
    在 NetworkX 图上构建多尺度符号索引。

    Args:
        G: NetworkX DiGraph，包含 chunk 节点

    Returns:
        dict: 多尺度符号索引
    """
    keyword_index = defaultdict(list)
    metric_index = defaultdict(list)
    period_index = defaultdict(list)
    number_index = defaultdict(list)
    table_index = defaultdict(list)
    clause_index = defaultdict(list)
    doc_profile_index = {}

    for node, attrs in G.nodes(data=True):
        if attrs.get("type") != "chunk":
            continue

        chunk_id = node
        doc_id = attrs.get("doc_id", "")
        text = attrs.get("text", "")
        page = attrs.get("page", 0)
        has_table = attrs.get("has_table", False)
        entities = attrs.get("entities", [])

        if not text:
            continue

        # 基础权重：表格 > 普通文本
        base_weight = 1.3 if has_table else 1.0

        # 关键词索引
        keywords = extract_keywords(text)
        for kw in keywords[:20]:  # 限制每个 chunk 的关键词数
            keyword_index[kw].append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "page": page,
                "weight": base_weight,
            })

        # 指标索引
        metrics = extract_metrics(text)
        for m in metrics:
            metric_index[m].append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "page": page,
                "weight": base_weight * 1.5,
            })

        # 时期索引
        periods = extract_periods(text)
        for p in set(periods):
            period_index[p].append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "page": page,
                "weight": base_weight,
            })

        # 数值索引
        numbers = extract_numbers(text)
        for n in numbers[:10]:
            number_index[n.strip()].append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "page": page,
                "weight": base_weight,
            })

        # 表格索引
        if has_table:
            table_index[doc_id].append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "page": page,
                "weight": 1.5,
            })

        # 条款索引
        clauses = extract_clauses(text)
        for c in clauses:
            clause_index[c].append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "page": page,
                "weight": base_weight * 1.2,
            })

        # 文档 profile
        if doc_id not in doc_profile_index:
            doc_profile_index[doc_id] = {
                "domain": attrs.get("domain", ""),
                "entities": set(),
                "metrics": set(),
                "periods": set(),
            }
        doc_profile_index[doc_id]["entities"].update(
            ent_name for _, ent_name in entities
        )
        doc_profile_index[doc_id]["metrics"].update(metrics)
        doc_profile_index[doc_id]["periods"].update(periods)

    # 转换 set 为 list 以便 JSON 序列化
    for doc_id in doc_profile_index:
        for key in ["entities", "metrics", "periods"]:
            doc_profile_index[doc_id][key] = list(doc_profile_index[doc_id][key])

    symbolic_index = {
        "keyword_index": dict(keyword_index),
        "metric_index": dict(metric_index),
        "period_index": dict(period_index),
        "number_index": dict(number_index),
        "table_index": dict(table_index),
        "clause_index": dict(clause_index),
        "doc_profile_index": doc_profile_index,
    }

    return symbolic_index


def get_symbolic_stats(symbolic_index: dict) -> dict:
    """获取符号索引统计信息"""
    stats = {}
    for key, index in symbolic_index.items():
        if isinstance(index, dict):
            stats[key] = {
                "unique_symbols": len(index),
                "total_entries": sum(
                    len(v) if isinstance(v, list) else 1
                    for v in index.values()
                ),
            }
    return stats
