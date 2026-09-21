"""
图构建模块 - 构建三层异构图（无embedding版）
Layer 1: 文本层 (chunks)
Layer 2: 结构层 (document -> section -> paragraph -> chunk)
Layer 3: 实体层 (company, regulation, product, metric, ...)
"""
import os
import json
import networkx as nx

from config import CACHE_DIR
from entity_extractor import extract_entities


def build_graph(all_docs: dict) -> nx.DiGraph:
    """
    构建三层异构图（纯文本/实体，不使用embedding）

    Args:
        all_docs: parse_all_documents() 的输出

    Returns:
        nx.DiGraph 异构图，附带 G.graph["entity_index"] 和 G.graph["keyword_index"]
    """
    G = nx.DiGraph()
    chunk_count = 0

    # 实体索引：entity_name -> [chunk_id, ...]
    entity_index = {}
    # 关键词索引：keyword -> [chunk_id, ...]
    keyword_index = {}
    # 短语索引：phrase -> [chunk_id, ...]（用于精确短语匹配）
    phrase_index = {}
    # 倒排索引：word -> [chunk_id, ...]（用于全文关键词搜索）
    inverted_index = {}

    print("构建图结构...")

    for domain_name, domain_docs in all_docs.items():
        for doc_id, doc_data in domain_docs.items():
            # === 文档节点 ===
            # 容错：部分文档（如strict_v3系列）可能缺少filename/path字段
            G.add_node(doc_id, type="document", domain=domain_name,
                       title=doc_data.get("filename", doc_id), path=doc_data.get("path", ""))

            # === Chunk 节点 ===
            chunk_ids_in_doc = []
            for chunk in doc_data["chunks"]:
                chunk_id = f"{doc_id}_c{chunk['chunk_id']}"
                chunk_text = chunk["text"]

                # 提取实体
                entities = extract_entities(chunk_text)

                # 标记是否含表格
                has_table = chunk.get("has_table", False)

                G.add_node(chunk_id,
                           type="chunk",
                           text=chunk_text,
                           doc_id=doc_id,
                           domain=domain_name,
                           tokens=len(chunk_text),
                           has_table=has_table,
                           page=chunk.get("page", 0),
                           page_start=chunk.get("page_start", chunk.get("page", 0)),
                           page_end=chunk.get("page_end", chunk.get("page", 0)),
                           chunk_type=chunk.get("chunk_type", "table" if has_table else "text"),
                           section_title=chunk.get("section_title", ""),
                           table_context_before=chunk.get("table_context_before", ""),
                           entities=[(e["type"], e["name"]) for e in entities])

                # 文档 → chunk 边
                G.add_edge(doc_id, chunk_id, relation="contains")

                # 实体 → chunk 边（单向：entity -> chunk）
                for ent in entities:
                    ent_id = f"{ent['type']}:{ent['name']}"
                    if not G.has_node(ent_id):
                        G.add_node(ent_id, type="entity",
                                   entity_type=ent["type"], name=ent["name"])
                    G.add_edge(ent_id, chunk_id, relation="mentioned_in")

                    # 更新实体索引
                    if ent["name"] not in entity_index:
                        entity_index[ent["name"]] = []
                    if chunk_id not in entity_index[ent["name"]]:
                        entity_index[ent["name"]].append(chunk_id)

                # 短语索引（从 chunk_text 中提取2-10字的有意义短语）
                import re as _re_build
                # 提取中文短语（2-10字，覆盖更长实体名）
                for m in _re_build.finditer(r'[一-鿿]{2,10}', chunk_text):
                    phrase = m.group()
                    # 过滤常见无意义词
                    if phrase in ("的", "了", "在", "是", "有", "和", "与", "或", "及",
                                 "其中", "一份", "第二份", "文档", "文件", "以下", "下列",
                                 "描述", "说法", "选项", "关于", "涉及", "包含", "明确",
                                 "公司", "规定", "应当", "可以", "进行", "通过", "使用",
                                 "情况", "相关", "以下", "说明", "介绍", "要求", "标准"):
                        continue
                    phrase_index.setdefault(phrase, [])
                    if chunk_id not in phrase_index[phrase]:
                        phrase_index[phrase].append(chunk_id)

                # 提取实体名称作为短语（确保完整实体名被索引）
                for ent in entities:
                    ent_name = ent["name"]
                    if len(ent_name) >= 2:
                        phrase_index.setdefault(ent_name, [])
                        if chunk_id not in phrase_index[ent_name]:
                            phrase_index[ent_name].append(chunk_id)

                # 倒排索引（分词：按标点和空格切分，取≥2字的词）
                words = _re_build.split(r'[，。、\s\|\n]+', chunk_text)
                for w in words:
                    w = w.strip()
                    if len(w) >= 2 and len(w) <= 20:
                        inverted_index.setdefault(w, [])
                        if chunk_id not in inverted_index[w]:
                            inverted_index[w].append(chunk_id)

                chunk_count += 1
                chunk_ids_in_doc.append(chunk_id)

            # === Chunk 顺序边（双向）===
            for i in range(len(chunk_ids_in_doc) - 1):
                G.add_edge(chunk_ids_in_doc[i], chunk_ids_in_doc[i+1], relation="next_chunk")
                G.add_edge(chunk_ids_in_doc[i+1], chunk_ids_in_doc[i], relation="prev_chunk")

        print(f"  {domain_name}: {len(domain_docs)} 文档, "
              f"{sum(len(d['chunks']) for d in domain_docs.values())} chunks")

    # 构建数值索引、标题索引、表格标记、位置标记
    numeric_index = {}
    heading_index = {}
    table_chunk_ids = []
    chunk_position = {}

    import re as _re

    def _normalize_num(s):
        """归一化数字：去除逗号"""
        return s.replace(",", "")

    for domain_name, domain_docs in all_docs.items():
        for doc_id, doc_data in domain_docs.items():
            chunk_ids_in_doc = []
            for chunk in doc_data["chunks"]:
                chunk_id = f"{doc_id}_c{chunk['chunk_id']}"
                chunk_text = chunk.get("text", "")
                chunk_ids_in_doc.append(chunk_id)

                # 数值索引（归一化后存储）
                for m in _re.finditer(r'(\d+(?:\.\d+)?)\s*%', chunk_text):
                    raw = _normalize_num(m.group(1))
                    for key in [raw + "%", raw]:
                        numeric_index.setdefault(key, [])
                        if chunk_id not in numeric_index[key]:
                            numeric_index[key].append(chunk_id)
                for m in _re.finditer(r'(\d+(?:\.\d+)?)\s*亿元', chunk_text):
                    raw = _normalize_num(m.group(1))
                    for key in [raw, raw + "亿"]:
                        numeric_index.setdefault(key, [])
                        if chunk_id not in numeric_index[key]:
                            numeric_index[key].append(chunk_id)
                for m in _re.finditer(r'(\d+(?:\.\d+)?)\s*万元', chunk_text):
                    raw = _normalize_num(m.group(1))
                    for key in [raw, raw + "万"]:
                        numeric_index.setdefault(key, [])
                        if chunk_id not in numeric_index[key]:
                            numeric_index[key].append(chunk_id)
                for m in _re.finditer(r'(\d{1,3}(?:,\d{3})+(?:\.\d+)?)', chunk_text):
                    clean = _normalize_num(m.group(1))
                    numeric_index.setdefault(clean, [])
                    if chunk_id not in numeric_index[clean]:
                        numeric_index[clean].append(chunk_id)
                for m in _re.finditer(r'(?<!\d)(\d{2,}(?:\.\d+)?)(?!\d)', chunk_text):
                    num = _normalize_num(m.group(1))
                    if _re.match(r'^(19|20)\d{2}$', num):
                        continue
                    numeric_index.setdefault(num, [])
                    if chunk_id not in numeric_index[num]:
                        numeric_index[num].append(chunk_id)

                # 标题索引
                for m in _re.finditer(r'\[标题\]\s*(.+)', chunk_text):
                    heading_text = m.group(1).strip()
                    for kw in _re.split(r'[，。、\s]+', heading_text):
                        if len(kw) >= 2:
                            heading_index.setdefault(kw, [])
                            if chunk_id not in heading_index[kw]:
                                heading_index[kw].append(chunk_id)

                # 表格标记
                if chunk.get("has_table", False) or "[表格" in chunk_text:
                    table_chunk_ids.append(chunk_id)

            # 位置标记
            num_chunks = len(chunk_ids_in_doc)
            head_cutoff = max(1, int(num_chunks * 0.2))
            tail_cutoff = max(head_cutoff + 1, int(num_chunks * 0.8))
            for i, cid in enumerate(chunk_ids_in_doc):
                if i < head_cutoff:
                    chunk_position[cid] = "head"
                elif i >= tail_cutoff:
                    chunk_position[cid] = "tail"
                else:
                    chunk_position[cid] = "body"

    # 保存索引到图属性
    G.graph["entity_index"] = entity_index
    G.graph["keyword_index"] = keyword_index
    G.graph["numeric_index"] = numeric_index
    G.graph["heading_index"] = heading_index
    G.graph["table_chunk_ids"] = table_chunk_ids
    G.graph["chunk_position"] = chunk_position
    G.graph["phrase_index"] = phrase_index
    G.graph["inverted_index"] = inverted_index

    # 构建多尺度符号索引（shadow mode）
    try:
        from symbolic_memory import build_symbolic_index, get_symbolic_stats
        symbolic_index = build_symbolic_index(G)
        G.graph["symbolic_index"] = symbolic_index
        stats = get_symbolic_stats(symbolic_index)
        print(f"  符号索引: {sum(s['unique_symbols'] for s in stats.values())} 个符号")
    except Exception as e:
        print(f"  [警告] 符号索引构建失败: {e}")
        G.graph["symbolic_index"] = {}

    print(f"\n图构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print(f"  实体索引: {len(entity_index)} 个实体")
    return G


def save_graph(G: nx.DiGraph, path: str, save_db: bool = True, db_path: str = None):
    """保存图到文件（含索引），同时保存到 SQLite"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"nodes": [], "edges": [], "graph_attrs": {}}

    for node, attrs in G.nodes(data=True):
        node_data = {"id": node}
        for k, v in attrs.items():
            if isinstance(v, tuple):
                node_data[k] = [list(e) if isinstance(e, tuple) else e for e in v]
            elif isinstance(v, list):
                node_data[k] = [list(e) if isinstance(e, tuple) else e for e in v]
            elif isinstance(v, (dict, str, int, float, bool, type(None))):
                node_data[k] = v
        data["nodes"].append(node_data)

    for u, v, attrs in G.edges(data=True):
        edge_data = {"source": u, "target": v}
        edge_data.update(attrs)
        data["edges"].append(edge_data)

    # 保存图级属性（索引等）
    for k, v in G.graph.items():
        if isinstance(v, dict):
            data["graph_attrs"][k] = v

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"图已保存: {path}")

    # 同时保存到 SQLite（如果启用）
    if save_db:
        try:
            from index_db import IndexDB
            if db_path is None:
                db_path = os.path.join(os.path.dirname(path), "indexes.db")
            with IndexDB(db_path) as db:
                db.create_tables()
                db.migrate_from_graph(G)
                print(f"索引已保存到 SQLite: {db_path}")
        except Exception as e:
            print(f"[警告] SQLite 保存失败: {e}")


def load_graph(path: str) -> nx.DiGraph:
    """从文件加载图（含索引）"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node_data in data["nodes"]:
        node_id = node_data.pop("id")
        if "entities" in node_data:
            node_data["entities"] = [tuple(e) for e in node_data["entities"]]
        G.add_node(node_id, **node_data)

    for edge_data in data["edges"]:
        source = edge_data.pop("source")
        target = edge_data.pop("target")
        G.add_edge(source, target, **edge_data)

    # 恢复图级属性
    for k, v in data.get("graph_attrs", {}).items():
        G.graph[k] = v

    print(f"图已加载: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    if "entity_index" in G.graph:
        print(f"  实体索引: {len(G.graph['entity_index'])} 个实体")
    return G


def get_doc_id_from_chunk(chunk_id: str) -> str:
    """从 chunk_id 提取 doc_id"""
    return chunk_id.rsplit("_c", 1)[0]


if __name__ == "__main__":
    from pdf_parser import parse_all_documents
    from config import RAW_DOCS_DIR, CACHE_DIR

    print("解析文档...")
    all_docs = parse_all_documents(RAW_DOCS_DIR)

    print("\n构建图...")
    G = build_graph(all_docs)

    graph_path = os.path.join(CACHE_DIR, "graph.json")
    save_graph(G, graph_path)
