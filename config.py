"""
配置文件 - AFAC2025 金融长文档问答 Agent
"""
import os

# 加载项目根目录的 .env（密钥集中管理，.env 不入版本库）
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.environ.get("DATASET_DIR", "")

# 自动检测数据集目录
if not DATASET_DIR:
    candidates = [
        os.path.join(BASE_DIR, "public_dataset_a", "public_dataset_upload"),
        os.path.join(BASE_DIR, "dataset"),
        os.path.join(BASE_DIR, "data"),
        os.path.join(BASE_DIR, "public_dataset_a"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            DATASET_DIR = c
            break
    if not DATASET_DIR:
        DATASET_DIR = os.path.join(BASE_DIR, "public_dataset_a", "public_dataset_upload")

# 自动检测 questions 目录（兼容 group_a / 不同结构）
QUESTIONS_DIR = ""
for q_candidate in [
    os.path.join(DATASET_DIR, "questions", "group_a"),
    os.path.join(DATASET_DIR, "questions"),
    os.path.join(BASE_DIR, "questions", "group_a"),
    os.path.join(BASE_DIR, "questions"),
]:
    if os.path.isdir(q_candidate):
        QUESTIONS_DIR = q_candidate
        break
if not QUESTIONS_DIR:
    QUESTIONS_DIR = os.path.join(DATASET_DIR, "questions", "group_a")

RAW_DOCS_DIR = os.environ.get("RAW_DOCS_DIR", os.path.join(DATASET_DIR, "raw"))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
VLM_TEXT_DIR = os.path.join(CACHE_DIR, "vlm_text")      # VLM 解析纯文本归档
VLM_MARKDOWN_DIR = os.path.join(CACHE_DIR, "vlm_markdown")  # VLM 解析 Markdown 归档

# 领域映射
DOMAINS = {
    "financial_contracts": "金融合同",
    "financial_reports": "财务报表",
    "insurance": "保险条款",
    "regulatory": "监管法规",
    "research": "行业研报",
}

# ============ API 配置 ============
# 密钥一律从环境变量 / .env 读取，严禁硬编码（参考 .env.example）
# 问答走 OpenAI 兼容端点（示例：AMD Radeon 开发者端点 Qwen3.8-27B）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen3.8-27B")
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
# embedding 走 DashScope 官方端点（与问答不同账户体系，密钥独立）
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
# 部分端点不认识 enable_thinking 参数，测试确认后再启用
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "0") == "1"
# 统一 extra_body，所有 chat.completions.create 调用复用
QWEN_EXTRA_BODY = {"enable_thinking": ENABLE_THINKING} if ENABLE_THINKING else {}
# 调用接口：1 = OpenAI Responses API（/v1/responses，dasuapi 等）；0 = chat.completions
USE_RESPONSES_API = os.environ.get("USE_RESPONSES_API", "1") == "1"

# ============ 分块配置 ============
CHUNK_MAX_CHARS = 1500       # 每个 chunk 最大字符数（中文≈1字符/1token）
CHUNK_OVERLAP_CHARS = 200    # chunk 重叠字符数
CHUNK_MIN_LENGTH = 30        # 最小段落长度（降低以保留重要短文本）

# ============ 图配置 ============
BFS_MAX_DEPTH = 3            # BFS 最大遍历深度
TOP_K_CHUNKS = 8             # 最终证据段落数
DEPTH_DECAY = 0.85           # BFS深度衰减因子

# ============ SCNET OCR 配置（备用）============
SCNET_API_KEY = os.environ.get("SCNET_API_KEY", "sk-Mzc4LTIxNzkwNDM3OTQ3LTE3NzQ0MzA0MDA0NjY=")
SCNET_BASE_URL = "https://api.scnet.cn/api/llm/v1"

# ============ MinerU 2.5 Pro VLM 配置（AMD MI300X / ROCm 专用）============
MINERU_MODEL_DIR = os.environ.get("MINERU_MODEL_DIR", "/work/home/acu58ahr8z/agenitic_mem/rag-enhancement/moudles")

# VLM 推理配置：默认面向单卡 MI300X，可通过环境变量覆盖
VLM_DPI = int(os.environ.get("VLM_DPI", "180"))
VLM_DOC_BATCH_SIZE = int(os.environ.get("VLM_DOC_BATCH_SIZE", "5"))
VLM_PAGE_WINDOW = int(os.environ.get("VLM_PAGE_WINDOW", "96"))
VLM_CONCURRENCY = int(os.environ.get("VLM_CONCURRENCY", "48"))
VLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("VLM_GPU_MEMORY_UTILIZATION", "0.92"))
VLM_MAX_MODEL_LEN = int(os.environ.get("VLM_MAX_MODEL_LEN", "4096"))
VLM_MAX_NUM_SEQS = int(os.environ.get("VLM_MAX_NUM_SEQS", str(VLM_CONCURRENCY)))

# ============ Token 预算 ============
TOKEN_BUDGET_TOTAL = 5_000_000
TOKEN_PER_QUESTION = 20_000  # 每题预算

# ============ 向量检索配置（Chroma + DashScope embedding）============
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3.7-text-embedding-flash")
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "10"))   # DashScope 批量上限 10
CHROMA_DIR = os.path.join(CACHE_DIR, "chroma_db")          # Chroma 持久化目录
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, "embeddings_cache.json")  # embedding 本地缓存（防重复计费）
VECTOR_RECALL_ENABLED = os.environ.get("VECTOR_RECALL_ENABLED", "1") == "1"  # 向量兜底召回开关（消融用）
VECTOR_FALLBACK_TOP_K = int(os.environ.get("VECTOR_FALLBACK_TOP_K", "10"))   # 向量兜底召回条数

# ============ Agent 预算控制（改进.md §4）============
MAX_ROUNDS_BY_FORMAT = {"multi": 5, "mcq": 3, "tf": 2}   # 按题型分级的工具调用轮数上限
MAX_ROUNDS_BY_DIFFICULTY = {"multi_doc": 5}              # 多文档题放宽轮数
PER_QUESTION_TOKEN_LIMIT = TOKEN_PER_QUESTION            # 每题 token 熔断阈值
TOOL_CALL_FAILURE_LIMIT = 3                              # 工具连续失败上限，超过后降级为仅用已有证据作答
CITATION_MODE = os.environ.get("CITATION_MODE", "record")          # record=仅记录引用校验结果 / strict=无引用数字触发重写
REFUSAL_MODE = os.environ.get("REFUSAL_MODE", "guess_fallback")    # guess_fallback=拒答信号+最佳猜测 / strict=输出拒答文本
# 零锚点强制拒答：模型选中的选项其强锚点（数值/年份/条款号）均不在证据池原文中，
# 且模型无有效引用时，判定为"选项无文档依据"，强制拒答而非猜测（金融场景可信优先）
ZERO_EVIDENCE_REFUSAL = os.environ.get("ZERO_EVIDENCE_REFUSAL", "1") == "1"
# 多选锚点验证后处理：逐选项检查"数值锚点+指标词"是否在该题文档 chunk 中共现，
# 真表述（原文数值）必命中、篡改/无据表述必不命中；验证支持选项≥2 时以验证结果为准
MULTI_ANCHOR_VERIFY = os.environ.get("MULTI_ANCHOR_VERIFY", "1") == "1"

# ============ 陷阱题分流（题目自带"若文档未提及，请拒答"提示）============
# 核心策略：陷阱题坚决拒答，正常题永不拒答（拒答提示词只在陷阱题文本中出现）
TRAP_DETECTION_ENABLED = os.environ.get("TRAP_DETECTION_ENABLED", "1") == "1"  # 总开关，0=完全回退旧行为
TRAP_REFUSAL_POLICY = os.environ.get("TRAP_REFUSAL_POLICY", "auto_refuse")     # auto_refuse=拒答信号或答不出时输出REFUSED / legacy=旧行为
TRAP_MAX_ROUNDS = int(os.environ.get("TRAP_MAX_ROUNDS", "2"))                  # 陷阱题工具轮数上限（省token，陷阱题搜多了也答不出）
TRAP_TOKEN_EARLY_STOP = float(os.environ.get("TRAP_TOKEN_EARLY_STOP", "0.5"))  # 陷阱题token早停比例（×PER_QUESTION_TOKEN_LIMIT，0=禁用）
# 拒答提示词表：已验证"若文档未提及"对45陷阱题全覆盖、正常题0误命中；正式赛题措辞不同时可在此扩展
TRAP_HINT_KEYWORDS = ["若文档未提及", "请拒答", "未提及请", "若未提及"]

# ============ 答案配置 ============
ANSWER_FORMATS = {
    "mcq": ["A", "B", "C", "D"],
    "multi": ["A", "B", "C", "D"],
    "tf": ["A", "B"],
}

# ============ 检索优化参数 ============
DOC_TYPE_BOOST = 1.5       # 文档类型匹配加分
TABLE_BOOST = 1.3          # 表格 chunk 加分
POSITION_BOOST = 1.2       # 位置权重（文档开头/结尾）

# ============ 问答优化参数 ============
MAX_RETRIES = 3            # API 调用重试次数
EVIDENCE_TOKEN_RATIO = 0.6  # 证据占 token 预算的比例
QA_AUDIT_MODE = False      # 默认关闭审计模式；CLI --audit 可覆盖

# ============ 图搜索优化参数 ============
GRAPH_SEARCH_MAX_VISITS = 200      # 全局访问预算
SYMBOLIC_RETRIEVAL_ENABLED = True   # 是否启用符号检索
SYMBOLIC_WEIGHT = 0.6              # 符号检索权重
LEGACY_WEIGHT = 0.4                # 旧检索权重
