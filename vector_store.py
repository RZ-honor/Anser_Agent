"""
向量存储层（改进.md §3.4 向量兜底混合召回）

设计：
- Chroma PersistentClient 嵌入式存储（cache/chroma_db），无独立服务进程
- DashScope text-embedding-v4 生成向量（官方端点，与问答 MaaS 端点额度分离）
- 建库一次持久化，重启免重嵌入（Chroma 内部持久化即缓存，不做 JSON 向量缓存）
- 查询时仅 1 次 embedding API 调用（问题文本），延迟可控

用途：BM25 召回失败（术语映射不稳定、口语化表述）时提供语义兜底，
与符号检索（BM25+数值+短语+实体）形成混合召回。
"""
import os
import sys
import time
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chromadb
from openai import OpenAI

from config import (EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, CHROMA_DIR,
                    EMBEDDING_API_KEY)

# DashScope 兼容端点（embedding 走独立账户，密钥用 EMBEDDING_API_KEY）
EMBEDDING_BASE_URL = os.environ.get(
    "EMBEDDING_BASE_URL", "https://maas.qianwenaiapi.com/compatible-mode/v1")
# 建库时截断文本长度（截尾保存成本，前 500 字已含指标+数值主体信息）
EMBED_MAX_CHARS = 500


class FinanceVectorStore:
    """金融文档向量库（Chroma + DashScope embedding）"""

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or CHROMA_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="chunks_v1", metadata={"hnsw:space": "cosine"})
        self._embedder = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

    # ============ 建库 ============

    def build_from_domain_systems(self, domain_systems: dict):
        """从领域索引系统导入全部 chunk（增量：已入库的 chunk 跳过）

        Args:
            domain_systems: {domain: NewRetrievalSystem}，读取 chunk_data
        """
        docs, ids, metas = [], [], []
        existing = set(self.collection.get()["ids"]) if self.collection.count() else set()
        for domain, system in domain_systems.items():
            for chunk_id, data in system.chunk_data.items():
                if chunk_id in existing:
                    continue
                docs.append(data["text"][:EMBED_MAX_CHARS])
                ids.append(chunk_id)
                metas.append({"domain": domain, "doc_id": data.get("doc_id", ""),
                              "has_table": bool(data.get("has_table", False))})
        print(f"待嵌入 {len(ids)} 条（已存在 {len(existing)} 条）")
        self._upsert_batched(docs, ids, metas)

    def _embed_with_retry(self, texts: list[str], retries: int = 3) -> list[list[float]]:
        """批量 embedding（带重试与限流退避）"""
        for attempt in range(retries):
            try:
                resp = self._embedder.embeddings.create(model=EMBEDDING_MODEL, input=texts)
                return [d.embedding for d in resp.data]
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 * (attempt + 1)
                print(f"  [embedding 重试 {attempt + 1}] {str(e)[:80]}，{wait}s 后重试")
                time.sleep(wait)

    def _upsert_batched(self, docs: list[str], ids: list[str], metas: list[dict]):
        """分批 embedding + 入库（批次大小取 EMBEDDING_BATCH_SIZE）

        单批失败不中止已入库的进度（upsert 增量持久化），报错后可重跑续传
        """
        total = len(ids)
        t0 = time.time()
        for i in range(0, total, EMBEDDING_BATCH_SIZE):
            batch_docs = docs[i:i + EMBEDDING_BATCH_SIZE]
            batch_ids = ids[i:i + EMBEDDING_BATCH_SIZE]
            batch_metas = metas[i:i + EMBEDDING_BATCH_SIZE]
            try:
                embeddings = self._embed_with_retry(batch_docs)
                self.collection.upsert(ids=batch_ids, documents=batch_docs,
                                       metadatas=batch_metas, embeddings=embeddings)
            except Exception as e:
                # 账户欠费/额度耗尽等持续型错误：报告进度后中断（重跑自动续传）
                print(f"\n[中断] 第 {i} 条批次嵌入失败: {str(e)[:120]}")
                print(f"向量库已有 {self.collection.count()}/{total + self.collection.count()} 条，"
                      f"解决账户问题后重新运行 --build 即可续传")
                return False
            done = min(i + EMBEDDING_BATCH_SIZE, total)
            if done % 200 == 0 or done == total:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  建库进度: {done}/{total} ({rate:.0f} 条/秒)")
        print(f"建库完成，共 {self.collection.count()} 条")
        return True

    # ============ 查询 ============

    def query(self, text: str, domain: str = None, doc_ids: list = None,
              top_k: int = 10) -> list[dict]:
        """语义检索：问题文本 -> 向量 -> 最近邻 chunk

        Args:
            domain: 限定领域（metadata 过滤）
            doc_ids: 限定文档（metadata 过滤，多文档题时缩小范围）
            top_k: 返回条数

        Returns:
            [{"chunk_id", "text", "doc_id", "distance"}]，distance 越小越相似
        """
        if self.collection.count() == 0:
            return []
        q_emb = self._embed_with_retry([text[:EMBED_MAX_CHARS]])[0]
        where = {}
        if domain and doc_ids:
            where = {"$and": [{"domain": {"$eq": domain}},
                              {"doc_id": {"$in": list(doc_ids)}}]}
        elif domain:
            where = {"domain": {"$eq": domain}}
        elif doc_ids:
            where = {"doc_id": {"$in": list(doc_ids)}}
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k,
                                    where=where or None)
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            hits.append({"chunk_id": cid, "text": doc, "doc_id": meta.get("doc_id", ""),
                         "distance": dist})
        return hits


_STORE_CACHE = {}


def get_vector_store() -> FinanceVectorStore:
    """进程级单例加载（避免重复初始化 Chroma）"""
    if "store" not in _STORE_CACHE:
        _STORE_CACHE["store"] = FinanceVectorStore()
    return _STORE_CACHE["store"]


if __name__ == "__main__":
    # 建库入口：python vector_store.py --build
    if "--build" in sys.argv:
        from new_retrieval_system import NewRetrievalSystem  # noqa: F401
        from run_100 import load_or_build_indexes
        systems = load_or_build_indexes()
        store = FinanceVectorStore()
        store.build_from_domain_systems(systems)
    else:
        print("用法: python vector_store.py --build  （构建向量库）")
