"""
FastAPI 常驻检索与问答服务（改进.md §五）

启动方式：
  uvicorn serve.api:app --host 0.0.0.0 --port 8000

接口：
- GET  /health  健康检查（索引/向量库状态）
- GET  /search  混合检索（BM25+符号加分+表格保底+向量兜底）
- POST /ask     完整问答（多轮工具调用 + 引用校验 + 拒答检测）

设计要点：
- 索引在 lifespan 启动时一次性加载（常驻内存，请求零加载延迟）
- 向量库按需惰性加载（未建库时 /search 自动降级为纯符号检索）
- 请求模型预留 acl_tags 字段（后续按主体做访问控制时使用）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from config import VECTOR_RECALL_ENABLED

# 常驻状态：领域索引系统 / 检索器
_STATE = {"systems": {}, "ready": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载索引缓存（约 2 秒），关闭时释放"""
    from run_100 import load_or_build_indexes
    _STATE["systems"] = load_or_build_indexes()
    _STATE["ready"] = True
    yield
    _STATE["systems"] = {}


app = FastAPI(title="金融文档问答检索服务", version="1.0", lifespan=lifespan)


# ============ 请求/响应模型 ============

class SearchRequest(BaseModel):
    question: str = Field(..., description="问题文本")
    options: dict = Field(default_factory=dict, description="选项 {A:..., B:...}")
    domain: str = Field(..., description="领域，如 financial_reports")
    doc_ids: Optional[list] = Field(None, description="限定文档 id 列表")
    top_k: int = Field(8, ge=1, le=30)
    acl_tags: Optional[list] = Field(None, description="预留：按主体做访问控制的标签")


class AskRequest(SearchRequest):
    answer_format: str = Field("mcq", description="题型：mcq / multi / tf")
    max_rounds: Optional[int] = Field(None, description="工具调用轮数上限（默认按题型配置）")


def _get_system(domain: str):
    system = _STATE["systems"].get(domain)
    if system is None:
        raise KeyError(f"未知领域: {domain}")
    return system


# ============ 接口 ============

@app.get("/health")
def health():
    return {
        "ready": _STATE["ready"],
        "domains": list(_STATE["systems"].keys()),
        "vector_recall_enabled": VECTOR_RECALL_ENABLED,
        "vector_db_count": _vector_count(),
    }


def _vector_count() -> int:
    """向量库条数（未建库返回 0）"""
    try:
        from vector_store import get_vector_store
        return get_vector_store().collection.count()
    except Exception:
        return 0


@app.get("/search")
def search(question: str, domain: str, top_k: int = 8,
           doc_ids: Optional[str] = None):
    """混合检索（GET 简化版：doc_ids 用逗号分隔）"""
    system = _get_system(domain)
    ids = [d for d in (doc_ids or "").split(",") if d] or None
    results = system.search(question, {}, doc_ids=ids, top_k=top_k)
    return {"results": results}


@app.post("/search")
def search_post(req: SearchRequest):
    """混合检索（POST 完整版：支持选项文本参与符号加分）"""
    system = _get_system(req.domain)
    results = system.search(req.question, req.options,
                            doc_ids=req.doc_ids, top_k=req.top_k)
    return {"results": results}


@app.post("/ask")
def ask(req: AskRequest):
    """完整问答：检索 + 多轮工具调用 + 引用校验 + 拒答检测"""
    from model_driven_retrieval import ModelDrivenRetriever
    system = _get_system(req.domain)
    retriever = ModelDrivenRetriever(retrieval_system=system, mode="offline_competition")
    initial_evidence = system.search(req.question, req.options,
                                     doc_ids=req.doc_ids, top_k=15)
    result = retriever.answer_question(
        question=req.question, options=req.options,
        answer_format=req.answer_format, initial_evidence=initial_evidence,
        domain=req.domain, doc_ids=req.doc_ids, max_rounds=req.max_rounds)
    return result
