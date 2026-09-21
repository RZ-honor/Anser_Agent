"""
图检索模块 - 基于图的 BFS 检索（无embedding版）
纯关键词 + 实体匹配，支持索引加速

新增：多路符号召回 + 加权融合 + 优先队列图扩展
"""
import heapq
import os
import json
from collections import deque, defaultdict
import networkx as nx

from graph_builder import get_doc_id_from_chunk
from entity_extractor import extract_question_entities
from symbolic_memory import extract_metrics, extract_periods, extract_numbers, extract_clauses
from config import (
    BFS_MAX_DEPTH, TOP_K_CHUNKS, DEPTH_DECAY, DOC_TYPE_BOOST, TABLE_BOOST,
    GRAPH_SEARCH_MAX_VISITS, SYMBOLIC_RETRIEVAL_ENABLED, SYMBOLIC_WEIGHT, LEGACY_WEIGHT,
    OUTPUT_DIR,
)


def graph_retrieve(
    question: str,
    options: dict,
    G: nx.DiGraph,
    doc_ids: list[str] | None = None,
    qid: str | None = None,
    domain: str = "",
) -> list[dict]:
    """
    图检索：BM25 + jieba 分词（按领域优化）

    Args:
        question: 问题文本
        options: 选项字典
        G: 异构图
        doc_ids: A榜提供的参考文档ID
        qid: 题目 ID（用于 debug 输出）
        domain: 文档领域（用于过滤和优化检索）

    Returns:
        [{"chunk_id": str, "text": str, "score": float, "doc_id": str}]
    """
    from bm25_search import bm25_search

    # 使用 BM25 + jieba 检索（按领域分索引）
    results = bm25_search(G, question, options, doc_ids=doc_ids, domain=domain, top_k=TOP_K_CHUNKS)

    # debug 输出
    if qid:
        _save_retrieval_debug(qid, question, {}, {}, results)

    return results

    # 9. 证据打包
    packed_chunks = []
    for chunk_id, score in scored:
        if not G.has_node(chunk_id):
            continue
        node_data = G.nodes[chunk_id]
        packed_chunks.append({
            "chunk_id": chunk_id,
            "text": node_data.get("text", ""),
            "score": score,
            "doc_id": node_data.get("doc_id", get_doc_id_from_chunk(chunk_id)),
            "has_table": node_data.get("has_table", False),
        })

    packed_chunks = _evidence_packing(packed_chunks, max_chunks=TOP_K_CHUNKS)

    # 10. 输出检索 debug 信息
    if qid:
        _save_retrieval_debug(qid, question, {}, {}, packed_chunks)

    return packed_chunks


def _find_docs_by_entities(G: nx.DiGraph, q_entities: dict) -> list[str]:
    """通过实体索引找到相关文档（B-group 关键策略）"""
    entity_index = G.graph.get("entity_index", {})
    doc_scores = {}  # doc_id -> score

    for key in ["companies", "regulations", "products", "metrics"]:
        for name in q_entities.get(key, []):
            # 精确匹配
            if name in entity_index:
                for chunk_id in entity_index[name]:
                    doc_id = G.nodes[chunk_id].get("doc_id", get_doc_id_from_chunk(chunk_id))
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 2.0
            # 模糊匹配
            for ent_name, chunk_ids in entity_index.items():
                if name in ent_name or ent_name in name:
                    for chunk_id in chunk_ids:
                        doc_id = G.nodes[chunk_id].get("doc_id", get_doc_id_from_chunk(chunk_id))
                        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0

    # 按分数排序，返回 top 文档
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:5]]


def _get_docs_by_domain(G: nx.DiGraph, domain: str) -> list[str]:
    """获取指定领域的所有文档 ID"""
    doc_ids = []
    for node, attrs in G.nodes(data=True):
        if attrs.get("type") == "document":
            doc_domain = attrs.get("domain", "")
            if doc_domain == domain:
                doc_ids.append(node)
    return doc_ids


def _get_chunks_from_docs(G: nx.DiGraph, doc_ids: list[str]) -> list[str]:
    """从指定文档中获取所有 chunk 节点"""
    chunks = []
    for doc_id in doc_ids:
        if G.has_node(doc_id):
            for neighbor in G.neighbors(doc_id):
                if G.nodes[neighbor].get("type") == "chunk":
                    chunks.append(neighbor)
        else:
            # 模糊匹配
            for node in G.nodes():
                if G.nodes[node].get("type") == "document":
                    node_doc_id = node
                    title = G.nodes[node].get("title", "")
                    if (doc_id in node_doc_id or
                        doc_id in title or
                        node_doc_id.startswith(doc_id)):
                        for neighbor in G.neighbors(node):
                            if G.nodes[neighbor].get("type") == "chunk":
                                chunks.append(neighbor)
                        break
    return chunks


def _find_chunks_by_entities_indexed(G: nx.DiGraph, q_entities: dict) -> list[str]:
    """通过实体索引找到相关 chunks（O(1) 查找）"""
    entity_index = G.graph.get("entity_index", {})
    matched_chunks = []

    for key in ["companies", "regulations", "products", "metrics"]:
        for name in q_entities.get(key, []):
            # 精确匹配
            if name in entity_index:
                matched_chunks.extend(entity_index[name])
            # 模糊匹配
            for ent_name, chunk_ids in entity_index.items():
                if name in ent_name or ent_name in name:
                    matched_chunks.extend(chunk_ids)

    # 如果实体匹配不到，用关键词匹配
    if not matched_chunks:
        keywords = q_entities.get("keywords", [])
        for node, attrs in G.nodes(data=True):
            if attrs.get("type") != "chunk":
                continue
            text = attrs.get("text", "")
            for kw in keywords:
                if kw in text:
                    matched_chunks.append(node)
                    break

    return list(set(matched_chunks))


def _bfs_expand(
    G: nx.DiGraph,
    start_chunks: list[str],
    q_entities: dict,
) -> list[tuple[str, int]]:
    """BFS 扩展，返回 (chunk_id, depth) 对"""
    visited = set()
    relevant = {}
    queue = deque()

    for chunk_id in start_chunks:
        queue.append((chunk_id, 0))
        relevant[chunk_id] = 0

    while queue:
        node, depth = queue.popleft()
        if depth >= BFS_MAX_DEPTH:
            continue
        if node in visited:
            continue
        visited.add(node)

        for neighbor in G.neighbors(node):
            if neighbor in visited:
                continue

            edge_data = G.edges[node, neighbor]
            relation = edge_data.get("relation", "")

            # 沿 "mentioned_in" 边：entity -> chunk
            if relation == "mentioned_in" and G.nodes[neighbor].get("type") == "chunk":
                if neighbor not in relevant or depth + 1 < relevant[neighbor]:
                    relevant[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

            # 沿 "next_chunk"/"prev_chunk" 边扩展（仅浅层）
            elif relation in ("next_chunk", "prev_chunk") and depth < 1:
                if G.nodes[neighbor].get("type") == "chunk":
                    if neighbor not in relevant or depth + 1 < relevant[neighbor]:
                        relevant[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))

    return list(relevant.items())


def _apply_numeric_boost(G, question, options, scored, start_chunks=None):
    """使用 numeric_index 对包含选项数字的 chunk 加分，并补入缺失的高价值 chunk"""
    import re
    numeric_index = G.graph.get("numeric_index", {})
    if not numeric_index:
        return scored

    # 从选项中提取数字
    option_nums = set()
    for opt_text in options.values():
        for m in re.finditer(r'(\d+(?:\.\d+)?)\s*%', opt_text):
            option_nums.add(m.group(1))
        for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:亿|万)', opt_text):
            option_nums.add(m.group(1))
        for m in re.finditer(r'(\d+(?:\.\d+)?)', opt_text):
            option_nums.add(m.group(1))

    # 从问题中提取关键数字（排除年份）
    question_nums = set()
    for m in re.finditer(r'(\d+(?:\.\d+)?)', question):
        num = m.group(1)
        if not re.match(r'^(19|20)\d{2}$', num):
            question_nums.add(num)

    all_nums = option_nums | question_nums
    if not all_nums:
        return scored

    # 找到包含这些数字的 chunk 及匹配数
    num_match_count = {}
    for num_str in all_nums:
        chunks = numeric_index.get(num_str, [])
        for cid in chunks:
            num_match_count[cid] = num_match_count.get(cid, 0) + 1

    # 已在 scored 中的 chunk_id
    scored_ids = {cid for cid, _ in scored}
    start_set = set(start_chunks) if start_chunks else set()

    # 1. 对已有的 chunk 应用乘法 boost
    boosted = []
    for chunk_id, score in scored:
        if chunk_id in num_match_count:
            boost_multiplier = 1.0 + num_match_count[chunk_id] * 5.0
            boosted.append((chunk_id, score * boost_multiplier))
        else:
            boosted.append((chunk_id, score))

    # 2. 补入 numeric_index 中匹配但不在 scored 中的 chunk（限于 start_chunks 范围）
    for cid, match_count in num_match_count.items():
        if cid not in scored_ids and cid in start_set:
            boosted.append((cid, match_count * 50.0))

    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted


def _inverted_index_search(G, question, options, start_chunks=None):
    """使用倒排索引和短语索引进行直接检索（含 IDF 权重）"""
    import re
    import math

    inverted_index = G.graph.get("inverted_index", {})
    phrase_index = G.graph.get("phrase_index", {})
    numeric_index = G.graph.get("numeric_index", {})

    start_set = set(start_chunks) if start_chunks else set()
    N = max(len(start_set), 1)  # 总 chunk 数

    # 从问题和选项中提取搜索词
    full_text = question + " " + " ".join(options.values())

    # 提取中文短语（2-6字）
    search_phrases = set()
    stopwords = {"的", "了", "在", "是", "有", "和", "与", "或", "及", "其中",
                 "一份", "第二份", "文档", "文件", "以下", "下列", "描述", "说法",
                 "选项", "关于", "涉及", "包含", "明确", "公司", "规定", "应当",
                 "以下", "哪些", "正确", "关于", "情况", "信息", "相关"}
    for m in re.finditer(r'[一-鿿]{2,6}', full_text):
        phrase = m.group()
        if phrase not in stopwords:
            search_phrases.add(phrase)

    # 统计每个 chunk 的匹配分数
    chunk_scores = {}

    for phrase in search_phrases:
        # 计算 IDF 权重：出现该词的 chunk 越少，权重越高
        pi_chunks = phrase_index.get(phrase, [])
        ii_chunks = inverted_index.get(phrase, [])

        # IDF = log(N / df)，df = 出现该词的 chunk 数
        df_pi = len(pi_chunks)
        df_ii = len(ii_chunks)

        if df_pi > 0:
            idf = math.log(N / df_pi) + 1.0  # +1 平滑
            weight = min(idf, 10.0)  # 上限 10
        else:
            weight = 0

        # 短语匹配（权重高）
        for cid in pi_chunks:
            if cid in start_set:
                chunk_scores[cid] = chunk_scores.get(cid, 0) + weight * 3.0

        # 倒排索引匹配（权重中）
        if df_ii > 0 and df_ii != df_pi:
            idf_ii = math.log(N / df_ii) + 1.0
            weight_ii = min(idf_ii, 10.0)
            for cid in ii_chunks:
                if cid in start_set:
                    chunk_scores[cid] = chunk_scores.get(cid, 0) + weight_ii

    # 数值匹配（权重最高，固定 50 分）
    for m in re.finditer(r'(\d+(?:\.\d+)?)', full_text):
        num = m.group(1)
        if num in numeric_index:
            for cid in numeric_index[num]:
                if cid in start_set:
                    chunk_scores[cid] = chunk_scores.get(cid, 0) + 50.0

    return chunk_scores


def _score_chunks(
    chunks_with_depth: list[tuple[str, int]],
    q_entities: dict,
    G: nx.DiGraph,
) -> list[tuple[str, float]]:
    """
    对 chunks 评分排序（改进版：含文档类型匹配、表格加分）
    """
    matched_entity_names = set()
    for key in ["companies", "regulations", "products", "metrics"]:
        matched_entity_names.update(q_entities.get(key, []))

    # 推断题目可能涉及的文档类型
    target_domain = _infer_domain(q_entities)

    scored = []

    for chunk_id, depth in chunks_with_depth:
        if not G.has_node(chunk_id):
            continue

        node_data = G.nodes[chunk_id]
        chunk_entities = node_data.get("entities", [])
        chunk_text = node_data.get("text", "")
        chunk_domain = node_data.get("domain", "")
        has_table = node_data.get("has_table", False)

        # 1. 实体匹配分
        entity_count = 0
        chunk_entity_names = set()
        for ent_type, ent_name in chunk_entities:
            for key in ["companies", "regulations", "products", "metrics"]:
                for qe in q_entities.get(key, []):
                    if qe == ent_name or qe in ent_name or ent_name in qe:
                        entity_count += 1
                        chunk_entity_names.add(ent_name)
            for kw in q_entities.get("keywords", []):
                if kw in ent_name or ent_name in kw:
                    entity_count += 0.5
                    chunk_entity_names.add(ent_name)
        entity_score = min(entity_count / 3.0, 1.0) * 2.0

        # 2. 关键词匹配分（排除已在实体中计分的）
        text_kw_score = 0
        for kw in q_entities.get("keywords", []):
            if kw in chunk_text and kw not in chunk_entity_names:
                text_kw_score += 0.3

        # 3. 深度衰减
        depth_decay = DEPTH_DECAY ** depth

        # 4. 文档类型加分
        domain_boost = 1.0
        if target_domain and chunk_domain == target_domain:
            domain_boost = DOC_TYPE_BOOST

        # 5. 表格加分（数值题证据通常在表格中）
        table_boost = TABLE_BOOST if has_table else 1.0

        # 最终得分
        final_score = (entity_score + text_kw_score) * depth_decay * domain_boost * table_boost

        scored.append((chunk_id, final_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _infer_domain(q_entities: dict) -> str:
    """从问题实体推断可能的文档领域"""
    keywords = q_entities.get("keywords", [])
    all_text = " ".join(keywords)

    domain_keywords = {
        "insurance": ["保险", "保单", "保费", "赔付", "免赔", "保障", "投保", "被保", "理赔", "年金", "寿险"],
        "financial_contracts": ["债券", "募集", "发行人", "承销", "票面", "利率", "违约", "偿付"],
        "financial_reports": ["营业收入", "净利润", "总资产", "净资产", "现金流", "每股", "毛利率", "年报"],
        "regulatory": ["证监会", "银保监", "管理办法", "规定", "准则", "指引", "规则", "处罚"],
        "research": ["研报", "评级", "目标价", "投资建议", "行业分析", "增持", "买入"],
    }

    for domain, kws in domain_keywords.items():
        for kw in kws:
            if kw in all_text:
                return domain

    return ""


# ============ 符号检索函数 ============

def _extract_question_symbols(question: str, options: dict = None) -> dict:
    """从问题和选项中提取符号特征"""
    full_text = question
    if options:
        full_text += " " + " ".join(options.values())

    return {
        "metrics": extract_metrics(full_text),
        "periods": extract_periods(full_text),
        "numbers": extract_numbers(full_text),
        "clauses": extract_clauses(full_text),
        "keywords": [w for w in full_text.split() if len(w) >= 2],
    }


def _symbolic_recall(G: nx.DiGraph, symbols: dict) -> dict:
    """多路符号召回，返回 chunk_id -> score"""
    symbolic_index = G.graph.get("symbolic_index", {})
    if not symbolic_index:
        return {}

    scores = defaultdict(float)

    # 指标召回
    for metric in symbols.get("metrics", []):
        for entry in symbolic_index.get("metric_index", {}).get(metric, []):
            scores[entry["chunk_id"]] += entry["weight"]

    # 时期召回
    for period in symbols.get("periods", []):
        for entry in symbolic_index.get("period_index", {}).get(period, []):
            scores[entry["chunk_id"]] += entry["weight"]

    # 数值召回
    for number in symbols.get("numbers", []):
        for entry in symbolic_index.get("number_index", {}).get(number, []):
            scores[entry["chunk_id"]] += entry["weight"]

    # 条款召回
    for clause in symbols.get("clauses", []):
        for entry in symbolic_index.get("clause_index", {}).get(clause, []):
            scores[entry["chunk_id"]] += entry["weight"]

    # 关键词召回（权重较低）
    for kw in symbols.get("keywords", [])[:10]:
        for entry in symbolic_index.get("keyword_index", {}).get(kw, []):
            scores[entry["chunk_id"]] += entry["weight"] * 0.3

    return dict(scores)


def _weighted_fusion(symbolic_scores: dict, legacy_scores: dict,
                     symbolic_weight: float = 0.6, legacy_weight: float = 0.4) -> dict:
    """融合符号检索和旧检索分数"""
    all_chunk_ids = set(symbolic_scores.keys()) | set(legacy_scores.keys())
    fused = {}
    for chunk_id in all_chunk_ids:
        sym_score = symbolic_scores.get(chunk_id, 0)
        leg_score = legacy_scores.get(chunk_id, 0)
        fused[chunk_id] = sym_score * symbolic_weight + leg_score * legacy_weight
    return fused


def _weighted_graph_expansion(G: nx.DiGraph, start_chunks: list,
                              max_visits: int = 200) -> list:
    """优先队列图扩展，返回 (chunk_id, depth) 列表"""
    visited = set()
    result = []
    heap = []  # (negative_score, chunk_id, depth)

    for chunk_id in start_chunks:
        if chunk_id not in visited:
            score = G.nodes[chunk_id].get("score", 1.0)
            heapq.heappush(heap, (-score, chunk_id, 0))

    while heap and len(visited) < max_visits:
        neg_score, chunk_id, depth = heapq.heappop(heap)
        if chunk_id in visited:
            continue
        visited.add(chunk_id)
        result.append((chunk_id, depth))

        # 扩展邻居
        for neighbor in G.neighbors(chunk_id):
            if neighbor not in visited:
                edge_data = G.edges[chunk_id, neighbor]
                relation = edge_data.get("relation", "")

                # 根据边类型决定扩展优先级
                if relation in ("next_chunk", "prev_chunk"):
                    neighbor_score = -neg_score * 0.8
                elif relation == "mentioned_in":
                    neighbor_score = -neg_score * 0.6
                else:
                    neighbor_score = -neg_score * 0.4

                heapq.heappush(heap, (-neighbor_score, neighbor, depth + 1))

    return result


def _evidence_packing(scored_chunks: list, max_chunks: int = 8) -> list:
    """证据打包：去重、优先表格、控制数量"""
    seen_docs = set()
    packed = []
    table_chunks = []
    non_table_chunks = []

    for chunk in scored_chunks:
        if chunk.get("has_table"):
            table_chunks.append(chunk)
        else:
            non_table_chunks.append(chunk)

    # 表格优先
    for chunk in table_chunks:
        if len(packed) >= max_chunks:
            break
        doc_id = chunk.get("doc_id", "")
        if doc_id not in seen_docs or len(packed) < 3:
            packed.append(chunk)
            seen_docs.add(doc_id)

    # 补充非表格
    for chunk in non_table_chunks:
        if len(packed) >= max_chunks:
            break
        doc_id = chunk.get("doc_id", "")
        if doc_id not in seen_docs or len(packed) < 5:
            packed.append(chunk)
            seen_docs.add(doc_id)

    return packed


def _save_retrieval_debug(qid: str, question: str, symbolic_scores: dict,
                          legacy_scores: dict, top_chunks: list):
    """保存检索调试信息"""
    debug_dir = os.path.join(OUTPUT_DIR, "retrieval_debug")
    os.makedirs(debug_dir, exist_ok=True)

    debug_info = {
        "qid": qid,
        "question": question[:200],
        "symbolic_recall_count": len(symbolic_scores),
        "legacy_recall_count": len(legacy_scores),
        "top_chunks": [
            {
                "chunk_id": c.get("chunk_id", ""),
                "doc_id": c.get("doc_id", ""),
                "score": c.get("score", 0),
                "text_preview": c.get("text", "")[:100],
            }
            for c in top_chunks
        ],
    }

    debug_path = os.path.join(debug_dir, f"{qid}.json")
    with open(debug_path, 'w', encoding='utf-8') as f:
        json.dump(debug_info, f, ensure_ascii=False, indent=2)


def _fallback_keyword_retrieve(
    question: str,
    G: nx.DiGraph,
    q_entities: dict,
) -> list[dict]:
    """回退方案：全局关键词检索"""
    results = []
    keywords = q_entities.get("keywords", [])

    for node, attrs in G.nodes(data=True):
        if attrs.get("type") != "chunk":
            continue
        text = attrs.get("text", "")
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            results.append({
                "chunk_id": node,
                "text": text,
                "score": score,
                "doc_id": attrs.get("doc_id", get_doc_id_from_chunk(node)),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:TOP_K_CHUNKS]


if __name__ == "__main__":
    from graph_builder import load_graph
    from config import CACHE_DIR
    import os

    graph_path = os.path.join(CACHE_DIR, "graph.json")
    if os.path.exists(graph_path):
        G = load_graph(graph_path)

        test_question = "比亚迪2025年营业收入是多少？"
        test_options = {"A": "3000亿", "B": "3847亿", "C": "4000亿", "D": "4500亿"}

        results = graph_retrieve(test_question, test_options, G, doc_ids=["annual_byd_2025_report"])

        print(f"\n检索结果 ({len(results)} 条):")
        for r in results:
            print(f"  [{r['score']:.2f}] {r['doc_id']}: {r['text'][:80]}...")
    else:
        print(f"图文件不存在: {graph_path}")
