"""
领域图管理器 - 按领域分离图和数据库

设计思路：
1. 每个领域有独立的图和数据库
2. 检索时直接使用对应领域的图和数据库
3. 提高检索效率和准确率
"""
import os
import json
import networkx as nx

from config import CACHE_DIR


class DomainGraphManager:
    """领域图管理器"""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.domain_graphs = {}  # 领域 -> 图
        self.domain_db_paths = {}  # 领域 -> 数据库路径

    def build_domain_graphs(self, all_docs: dict):
        """按领域构建独立的图"""
        from graph_builder import build_graph, save_graph

        print("按领域构建独立图...")

        for domain_name, domain_docs in all_docs.items():
            if not domain_docs:
                continue

            print(f"\n构建 {domain_name} 领域图...")

            # 构建该领域的图
            domain_all_docs = {domain_name: domain_docs}
            G = build_graph(domain_all_docs)

            # 保存图和数据库
            graph_path = os.path.join(self.cache_dir, f"graph_{domain_name}.json")
            db_path = os.path.join(self.cache_dir, f"indexes_{domain_name}.db")
            save_graph(G, graph_path, save_db=True, db_path=db_path)

            # 保存到实例变量
            self.domain_graphs[domain_name] = G
            self.domain_db_paths[domain_name] = db_path

            print(f"  图已保存: {graph_path}")
            print(f"  数据库已保存: {db_path}")

    def load_domain_graph(self, domain: str) -> nx.DiGraph:
        """加载指定领域的图"""
        if domain in self.domain_graphs:
            return self.domain_graphs[domain]

        graph_path = os.path.join(self.cache_dir, f"graph_{domain}.json")
        if os.path.exists(graph_path):
            from graph_builder import load_graph
            G = load_graph(graph_path)
            self.domain_graphs[domain] = G
            return G

        return None

    def get_domain_db_path(self, domain: str) -> str:
        """获取指定领域的数据库路径"""
        if domain in self.domain_db_paths:
            return self.domain_db_paths[domain]

        db_path = os.path.join(self.cache_dir, f"indexes_{domain}.db")
        if os.path.exists(db_path):
            self.domain_db_paths[domain] = db_path
            return db_path

        return None

    def search_domain(self, domain: str, question: str, options: dict,
                     doc_ids: list = None, top_k: int = 8) -> list:
        """在指定领域中搜索"""
        # 加载领域图
        G = self.load_domain_graph(domain)
        if not G:
            print(f"[错误] 未找到 {domain} 领域的图")
            return []

        # 使用领域特定的搜索引擎
        from bm25_search import bm25_search
        results = bm25_search(G, question, options, doc_ids=doc_ids, domain=domain, top_k=top_k)

        return results

    def list_domains(self) -> list:
        """列出所有可用的领域"""
        domains = []

        # 检查缓存目录中的图文件
        for filename in os.listdir(self.cache_dir):
            if filename.startswith("graph_") and filename.endswith(".json"):
                domain = filename[6:-5]  # 去掉 "graph_" 和 ".json"
                domains.append(domain)

        return domains


# 全局实例
_domain_manager = None


def get_domain_manager() -> DomainGraphManager:
    """获取全局领域图管理器"""
    global _domain_manager
    if _domain_manager is None:
        _domain_manager = DomainGraphManager()
    return _domain_manager


def build_all_domain_graphs():
    """构建所有领域的图"""
    from preparsed_loader import load_preparsed_documents
    from pdf_parser import chunk_blocks

    # 加载预解析文档
    print("加载预解析文档...")
    docs = load_preparsed_documents()

    # 对每个文档进行分块
    print("\n分块处理...")
    for domain, domain_docs in docs.items():
        for doc_id, doc_data in domain_docs.items():
            pages = doc_data.get('pages', [])
            if pages:
                chunks = chunk_blocks(pages, doc_id)
                doc_data['chunks'] = chunks
                print(f'  {domain}/{doc_id}: {len(chunks)} chunks')

    # 构建领域图
    manager = get_domain_manager()
    manager.build_domain_graphs(docs)

    return manager


if __name__ == "__main__":
    # 构建所有领域的图
    manager = build_all_domain_graphs()

    # 列出所有领域
    domains = manager.list_domains()
    print(f"\n可用领域: {domains}")
