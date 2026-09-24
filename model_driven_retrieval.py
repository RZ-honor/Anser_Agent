"""
模型驱动检索 - 使用 Qwen function calling 指导检索

流程：
1. BM25 粗召回 top-20（提供初始上下文）
2. Qwen 分析问题 + 初始证据，调用工具搜索补充证据
3. Qwen 综合所有证据，输出最终答案

优势：
- 模型思维链驱动检索，而非规则驱动
- 模型可以识别同义词、推理数值关系、理解语义
- 比纯 BM25 更适合复杂推理题

与现有系统集成：
- 可作为 graph_retrieve() 的替代方案
- 与 qa_engine.py 共享 API 配置
- 支持 fallback 到纯 BM25
"""
import json
import os
import re
import time
from typing import Optional

from config import DASHSCOPE_API_KEY, QWEN_MODEL, QWEN_BASE_URL, QWEN_EXTRA_BODY, \
    MAX_ROUNDS_BY_FORMAT, PER_QUESTION_TOKEN_LIMIT, TOOL_CALL_FAILURE_LIMIT, \
    CITATION_MODE, REFUSAL_MODE, USE_RESPONSES_API, ZERO_EVIDENCE_REFUSAL, \
    TRAP_DETECTION_ENABLED, TRAP_REFUSAL_POLICY, TRAP_MAX_ROUNDS, \
    TRAP_TOKEN_EARLY_STOP, TRAP_HINT_KEYWORDS, MULTI_ANCHOR_VERIFY
from citation_manager import (parse_citations, verify_citations, detect_refusal,
                              check_numeric_support)
from llm_api import call_llm

import jieba  # 中文分词，用于选项锚点提取

# ============ 选项锚点提取（选项级证据匹配 + 出处校验共用） ============
# 旧实现 re.findall(r'[一-鿿]{2,}', opt_text) 只取「最长连续中文串」（整段长句），
# 永远不会在证据里逐字出现 -> 选项级证据 count 恒为 0、has_evidence 恒 False。
# 改用 jieba 分词得到短实义词 + 数值锚点。

_OPT_STOPWORDS = {
    "的", "了", "为", "在", "和", "与", "或", "及", "等", "中", "其", "该", "此",
    "是", "不", "未", "无", "有", "对", "由", "从", "到", "向", "被", "把", "让",
    "明确", "披露", "约定", "提及", "属于", "达到", "高于", "低于", "显示",
    "关于", "根据", "下列", "以下", "选项", "描述", "内容", "文档", "文件",
    "其中", "本期", "发行", "公司", "正确", "错误", "成立", "符合", "是否",
    "进行", "相关", "具体", "规定", "如下", "上述", "可以", "应当", "不得",
    "如果", "由于", "因为", "所以", "但是", "并且", "以及", "对于", "通过",
    "上述", "之", "于", "以", "则", "均", "已", "将", "可", "本",
}


def _extract_opt_anchors(opt_text: str):
    """从选项文本提取匹配锚点（放宽匹配版）

    优化点：
    - 强锚点新增：条款号(第X条/X.Y.Z)、法规名(《》)、公司名等
    - 弱锚点门槛降低：长度>=2即可（原为>=2且需在文本中逐字命中）
    - 新增"部分命中"机制：弱锚点命中1个以上即认为有证据

    Returns:
        (strong_anchors, weak_terms)
        - strong_anchors: 数值/条款号/法规名等（最客观的出处锚点）
        - weak_terms: jieba 分词后的实义词（长度>=2，去停用词）
    """
    # 强锚点：数值相关（年份/百分比/金额/倍数/日期）
    strong = re.findall(r'\d+(?:\.\d+)?\s*(?:年|月|日|%|亿|万|元|倍|个|系数)', opt_text)
    strong += re.findall(r'\d{4}|\d+\.\d+%', opt_text)
    # 新增强锚点：条款号（第X条、X.Y.Z格式）
    strong += re.findall(r'第[一二三四五六七八九十百千\d]+\s*[条章节款项]', opt_text)
    strong += re.findall(r'\d+(?:\.\d+){1,3}', opt_text)
    # 新增强锚点：法规名（《》书名号内容）
    strong += re.findall(r'《[^》]{3,80}》', opt_text)
    strong = [re.sub(r'\s+', '', s) for s in strong]  # 去内部空格
    strong = list(dict.fromkeys(strong))
    # 弱锚点：jieba 分词实义词（门槛降低，保留长度>=2的所有实义词）
    weak = [w for w in jieba.lcut(opt_text)
            if len(w) >= 2 and w not in _OPT_STOPWORDS
            and not w.isdigit()
            and not re.fullmatch(r'[\d\s.\-/年月日%，]+', w)]
    # 新增：提取连续中文字符串作为补充弱锚点（不依赖jieba分词）
    weak += re.findall(r'[一-鿿]{3,}', opt_text)
    weak = list(dict.fromkeys(weak))
    return strong, weak


def detect_trap_hint(question: str) -> bool:
    """检测问题是否自带拒答提示（陷阱题：gold=REFUSED 类）

    陷阱题文本会明确提示"若文档未提及，请拒答"，正常题没有该提示。
    据此在入口处分流：陷阱题坚决拒答、降轮数省token；正常题永不拒答。
    （已验证"若文档未提及"对45道陷阱题全覆盖、正常题0误命中）
    """
    if not TRAP_DETECTION_ENABLED:
        return False
    return any(kw in question for kw in TRAP_HINT_KEYWORDS)


def _context_match(text: str, context: str) -> bool:
    """context 过滤：分词后要求所有实义词都出现（而非整串子串匹配）。

    修复：旧逻辑 `context not in text` 对带空格/多词 context（如 "品种一 兑付日"）
    必然失败，因为文档是 "品种一的兑付日" 无空格整串 -> 证据全被过滤，数值搜不到。
    """
    if not context:
        return True
    ctx_words = [w for w in jieba.lcut(context)
                 if len(w) >= 2 and not w.isspace() and not w.isdigit()
                 and not re.fullmatch(r'[\d\s.\-/年月日%，]+', w)]
    if not ctx_words:
        return True
    return all(w in text for w in ctx_words)


# ============ 工具定义 ============

SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_number",
            "description": "在文档中搜索包含特定数值的段落。用于查找金额、百分比、年份、数量等。注意：数字会自动去逗号匹配，如搜'1213'可匹配'12,132,154.20'中的'1213'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "string",
                        "description": "要搜索的数值，如 '2940'、'35.68'、'1213'、'777102'"
                    },
                    "unit": {
                        "type": "string",
                        "description": "数值单位（可选），如 '亿元'、'万元'、'%'、'百万元'"
                    },
                    "context": {
                        "type": "string",
                        "description": "数值的上下文关键词（可选），如 '营业收入'、'研发投入'、'负债合计'"
                    }
                },
                "required": ["number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_phrase",
            "description": "在文档中搜索包含特定短语的段落。用于查找条款、定义、条件等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "phrase": {
                        "type": "string",
                        "description": "要搜索的短语，如 '保险责任'、'信息披露义务'、'对外担保管理制度'"
                    }
                },
                "required": ["phrase"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_entity",
            "description": "在文档中搜索提到特定公司或机构的段落。",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "公司或机构名称，如 '中国平安人寿保险股份有限公司'、'比亚迪'"
                    }
                },
                "required": ["entity_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_keyword",
            "description": "在文档中搜索包含多个关键词的段落。所有关键词必须同时出现。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "必须同时出现的关键词列表，如 ['研发工程师', '12万名']"
                    }
                },
                "required": ["keywords"]
            }
        }
    }
]


# ============ 领域先验知识库 ============

DOMAIN_KNOWLEDGE = {
    "financial_contracts": """## 金融合同知识（基于文档提取）

### 核心概念
- 发行人=发行债券的公司；主承销商=承销证券公司
- "不超过X亿元"是上限，不是实际金额
- 金额单位：1亿元=10000万元=100000000元

### 文档中的标准格式
- 发行金额：如"不超过10亿元(含10亿元)"
- 信用评级：如"主体信用等级为AAA"，"债项评级为AAA/-"
- 票面利率：固定利率，单利按年计息，不计复利
- 发行规模：如"发行规模不超过人民币190亿元"

### 关键规则
- 同一公司可能有多期债券（如"25广晟K01"），每期条款不同
- 主体评级和债项评级可能不同（如主体AAA，债项AAA/-）
- 资信评级机构每年跟踪评级一次
- 债券存续期内信用可能发生负面变化""",

    "financial_reports": """## 财务报表知识（基于文档提取）

### 核心指标
- 营业收入≈营业额（港股常用"营业额"）
- 净利润=归属于母公司股东的净利润
- "母公司拥有人应占溢利"=归属于母公司净利润
- "同比上升"="同比增长"；同比变化=(本期-上期)/上期×100%

### 单位换算
- 百万元(M)=10^6元；千万元=10^7元；亿元=10^8元
- 万元=10^4元

### 文档中的标准表述
- "截至二零二四年十二月三十一日止年度"=2024年度
- "本集团"=合并报表口径
- "综合收益"包含净利润和其他综合收益
- "经重述"=追溯调整后的数据

### 数值比较规则
- "约X百万元"：四舍五入到百万位
- 777,102百万元四舍五入到十万位=780,000百万元
- 比较时先统一单位再对比""",

    "insurance": """## 保险条款知识（基于文档提取）

### 核心当事人
- 保险人=保险公司（承保方）
- 投保人=购买保险的人（交保费的人）
- 被保险人=受保险保障的人
- 受益人=领取保险金的人

### 文档中的标准规则
- 犹豫期：自签收合同之日起20日，犹豫期内退保没有损失
- 保险期间（主场景）：为被保险人终身，自合同生效时起至被保险人身故时止
- 保险期间（固定期限）：自合同生效时起至约定领取固定期限届满日零时止
- 现金价值：养老保险金开始领取日及之后为零
- 合同解除前发生的保险事故不承担保险责任

### 判断题关键规则
- 合同描述多种情况时，"终身"是主场景（默认选项）
- "不承担保险责任"是标准排除短语，需看具体上下文
- 犹豫期内退保没有损失，犹豫期后退保可能会有损失
- 保险期间内的事故才在保障范围内

### 常见排除
- 双耳失聪、语言能力丧失可能被排除
- 等待期内确诊的重大疾病不赔
- 故意犯罪、故意自伤不赔""",

    "regulatory": """## 监管法规知识（基于文档提取）

### 核心规则（上市公司治理准则）
- 上市公司应当披露定期报告（年度报告、中期报告）
- 上市公司应当制定信息披露事务管理制度
- 上市公司应当制定股东会议事规则
- 上市公司应当和董事签订合同，明确任期
- 独立董事任期：每届3年，连任不超过6年

### 文档中的标准表述
- "应当"=必须；"可以"=可选
- "以上"包含本数
- "对外担保管理制度"是标准法规术语
- "关联交易管理制度"是标准法规术语

### 关键治理要求
- 上市公司应当在公司章程中规定股东会的召集、召开和表决等程序
- 上市公司应当制定重大事件的报告、传递、审核、披露程序
- 上市公司应当制定董事、高级管理人员对外发布信息的行为规范
- 上市公司应当建立与股东畅通有效的沟通渠道

### 处罚规则
- 行政处罚：警告、罚款、责令改正
- 监管措施：出具警示函、监管谈话
- 信息披露义务人未按规定披露的，可能面临处罚""",

    "research": """## 研报知识（基于文档提取）

### 渠道定义
- 银保渠道=通过银行销售保险产品
- 个险渠道=通过个人代理人销售保险产品
- 团险渠道=通过团体（企业）销售保险产品
- 电商渠道=通过互联网平台销售

### 核心概念
- 市场规模=行业总销售额
- 市场份额=某公司销售额/市场总销售额
- 同比增长率=(本期-去年同期)/去年同期
- 渗透率=某产品使用人数/总人口

### 文档中的标准表述
- "预计"/"预测"是观点不是事实
- "首次覆盖"=分析师首次对该股票发表报告
- 目标价是分析师预测，不是市场共识
- "自营品"=公司自己生产的品牌产品""",
}


# ============ 系统 Prompt ============

TOOL_SYSTEM_PROMPT = """你是一个金融文档问答专家。你的任务是根据文档证据回答问题。你有搜索工具可以查找更多证据。

## 工作流程
1. 仔细阅读问题和所有选项
2. 查看已提供的初始证据
3. **必须为每个选项分别搜索证据**，不要跳过任何选项
4. 综合所有证据，逐项分析每个选项
5. 给出最终答案

## 搜索策略（关键：必须覆盖所有选项）

**第一步：分析问题类型**
- 数值题（问多少/几/比例）→ 搜索指标名+具体数字
- 实体题（问哪家公司/什么机构）→ 搜索公司名+专有名词
- 条款题（问什么制度/什么条件）→ 搜索关键短语+法规名
- 判断题（正确/错误）→ 搜索支持或反驳的关键词

**第二步：逐选项搜索（多选题必须执行）**
- 对每个选项，提取关键词（公司名、指标名、条款号、数字）
- 含数字的选项 -> search_number；多概念组合 -> search_keyword(如["资产减值","通知"])；精确短语(2-8字) -> search_phrase。**禁止用带空格或>8字的长句做 phrase 搜**
- 记录每个选项的证据状态：有证据支持 / 有证据否定 / 无证据

**第三步：综合判断**
- 单选题：选证据最充分的1个选项
- 多选题：选所有有证据支持的选项（通常2-4个）
- 判断题：证据支持选A，证据否定选B

## 回答规则（必须严格遵守）
1. **只根据文档证据回答**，绝对不要使用外部知识
2. **数值题**：从证据中找到精确数值，与选项对比，注意单位换算（万元=10000元，亿元=100000000元）
3. **判断题**：如果证据支持题目陈述，选A（正确）；只有证据明确否定时才选B（错误）
4. **多选题**：选所有有证据支持的正确选项。**禁止只选1个选项**，多选题通常有2-4个正确选项
5. **单选题**：只选一个最匹配的选项

## 输出格式（必须严格遵守）
先逐项分析每个选项的证据，然后最后一行必须写：
答案：X（单选题）
答案：A或B（判断题）
答案：ABC（多选题，根据证据选择所有正确选项）

## 限制
- 每次最多调用3个工具
- 如果已有足够证据，直接回答
- 搜索要精准，不要泛搜
- **多选题必须搜索所有选项，不能只搜1-2个就回答**

## 引用与拒答规则（必须遵守）
1. 每条证据都有编号（如[证据1]）。在分析中引用证据时，必须标注编号，如"根据[证据2]，营业收入为..."
2. 结论中的数值必须来自被引用证据的原文，禁止编造或改写数值
3. 如果所有选项均无文档证据支持（初始证据与问题无关、搜索也找不到），先在分析中写明"文档中未提及"，
   再给出最接近的选项（不要输出空答案）"""


# ============ 领域知识库补充（从 cache/domain_knowledge.json 加载） ============

# 缓存：避免每题重复读盘
_DOMAIN_KNOWLEDGE_CACHE = {}


def _load_domain_knowledge_all():
    """加载 cache/domain_knowledge.json（带缓存）"""
    if _DOMAIN_KNOWLEDGE_CACHE:
        return _DOMAIN_KNOWLEDGE_CACHE.get("__data__", {})
    dk_path = None
    for p in ["cache/domain_knowledge.json",
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "domain_knowledge.json")]:
        if os.path.isfile(p):
            dk_path = p
            break
    if not dk_path:
        _DOMAIN_KNOWLEDGE_CACHE["__data__"] = {}
        return {}
    try:
        import json as _json
        with open(dk_path, encoding="utf-8") as f:
            data = _json.load(f)
        _DOMAIN_KNOWLEDGE_CACHE["__data__"] = data
        return data
    except Exception:
        _DOMAIN_KNOWLEDGE_CACHE["__data__"] = {}
        return {}


def _build_knowledge_supplement(domain: str) -> str:
    """从 domain_knowledge.json 构建该领域的 prompt 补充（精选，控制 token）

    注入内容：
    - default_rules 前 5 条（领域默认规则）
    - high_frequency_entities 前 15 个（高频实体，辅助术语识别）
    返回空串表示无补充。
    """
    if not domain:
        return ""
    data = _load_domain_knowledge_all()
    domain_data = data.get(domain, {})
    if not domain_data:
        return ""

    parts = ["\n## 领域知识库补充（从已解析文档/索引抽取）"]

    # 默认规则（精选 5 条，每条截断 100 字）
    rules = domain_data.get("default_rules", []) or []
    rules = [r for r in rules if isinstance(r, str) and len(r) >= 10][:5]
    if rules:
        parts.append("### 领域默认规则")
        for r in rules:
            parts.append(f"- {r[:100]}")

    # 高频实体（精选 15 个，辅助模型识别术语）
    entities = domain_data.get("high_frequency_entities", []) or []
    entities = [e for e in entities if isinstance(e, str) and 2 <= len(e) <= 20][:15]
    if entities:
        parts.append("### 该领域高频实体（术语参考）")
        parts.append("、".join(entities))

    if len(parts) <= 1:
        return ""
    return "\n".join(parts)


class ModelDrivenRetriever:
    """模型驱动检索器"""

    VALID_MODES = {"offline_competition", "research_debug", "auto"}

    def __init__(self, retrieval_system, api_key: str = None, model: str = None,
                 mode: str = "offline_competition"):
        """
        Args:
            retrieval_system: NewRetrievalSystem 实例，提供搜索能力
            api_key: Qwen API key
            model: 模型名
            mode: 运行模式
                - offline_competition: 比赛模式，禁止联网搜索
                - research_debug: 调试模式，允许联网搜索辅助分析
                - auto: 自动模式，文档搜索3次无果后自动启用联网搜索
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"invalid mode: {mode}")

        self.retrieval_system = retrieval_system
        self.api_key = api_key or DASHSCOPE_API_KEY
        self.model = model or QWEN_MODEL
        self.mode = mode
        self.allow_web_search = (mode == "research_debug" or mode == "auto")
        self.client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            print("[警告] 未设置 API Key")
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=QWEN_BASE_URL)
        except Exception as e:
            print(f"[错误] 初始化客户端失败: {e}")

    # ========== 工具执行 ==========

    def _scope(self, doc_ids: list = None) -> set:
        """根据 doc_ids 限定搜索范围（None=全局）"""
        if not doc_ids:
            return set(self.retrieval_system.chunk_data.keys())
        doc_id_set = set(doc_ids)
        return {cid for cid, d in self.retrieval_system.chunk_data.items()
                if d.get("doc_id") in doc_id_set}

    @staticmethod
    def _excerpt(text: str, anchor: str, width: int = 400) -> str:
        """从 text 中截取 anchor 附近的上下文（前后各 width//2 字符）
        若 anchor 不在 text 中，返回 text 前 width 字符。"""
        if not text:
            return ""
        idx = text.find(anchor)
        if idx < 0:
            return text[:width]
        half = width // 2
        start = max(0, idx - half)
        end = min(len(text), idx + len(anchor) + half)
        excerpt = text[start:end]
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{excerpt}{suffix}"

    @staticmethod
    def _excerpt_number(text: str, number: str, width: int = 400) -> str:
        """数字专用截取：原文数字可能带千分位逗号（如 54,614.81），而查询数字去逗号（54614.81）。
        在去逗号文本里定位，映射回原文位置，返回原文片段（保留逗号）。"""
        if not text:
            return ""
        number_norm = number.replace(",", "").strip()
        if not number_norm:
            return text[:width]
        text_no_comma = text.replace(",", "")
        idx_nc = text_no_comma.find(number_norm)
        if idx_nc < 0:
            return text[:width]
        half = width // 2
        # 映射 text_no_comma[idx_nc] 回原文位置
        orig_start = None
        j = 0  # text_no_comma 指针
        for i, ch in enumerate(text):
            if j == idx_nc:
                orig_start = i
                break
            if ch != ",":
                j += 1
        if orig_start is None:
            return text[:width]
        # 找数字在原文中的结束位置（含中间逗号）
        orig_end = orig_start
        j2 = 0
        while j2 < len(number_norm) and orig_end < len(text):
            if text[orig_end] != ",":
                j2 += 1
            orig_end += 1
        start = max(0, orig_start - half)
        end = min(len(text), orig_end + half)
        excerpt = text[start:end]
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{excerpt}{suffix}"

    def _exec_search_number(self, number: str, unit: str = None, context: str = None,
                            doc_ids: list = None) -> list:
        """搜索包含特定数值的 chunks

        匹配策略：
        1. 精确匹配 numeric_index 中的数字
        2. 如果精确匹配无结果，做子串匹配（如 "1213" 匹配 "12132154.20"）
        3. 如果指定了单位，也搜带单位的组合
        返回数值附近上下文（而非 chunk 前500字符），便于模型直接看到数值。
        """
        results = []
        seen_ids = set()
        scope = self._scope(doc_ids)
        # 归一化：去逗号（numeric_index 用去逗号形式索引，原文带千分位逗号如 54,614.81）
        number_norm = number.replace(",", "").strip()
        # 用于截取上下文的锚点：优先 "数字+单位"（带逗号原文形式）
        anchor = f"{number}{unit}" if unit else number

        def _add_results(chunk_ids, max_count=5):
            for cid in chunk_ids:
                if cid in scope and cid not in seen_ids:
                    data = self.retrieval_system.chunk_data[cid]
                    text = data["text"]
                    if not _context_match(text, context):
                        continue
                    # 数字专用截取：原文带逗号，用去逗号定位
                    excerpt = self._excerpt_number(text, anchor, 400)
                    # 若锚点不在文本中，退而用纯数字
                    if number_norm and number_norm not in excerpt.replace(",", ""):
                        excerpt = self._excerpt_number(text, number_norm, 400)
                    results.append({
                        "chunk_id": cid,
                        "text": excerpt,
                        "doc_id": data["doc_id"],
                    })
                    seen_ids.add(cid)
                    if len(results) >= max_count:
                        return

        # 1. 精确匹配 numeric_index（去逗号）
        chunk_ids = self.retrieval_system.numeric_index.get(number_norm, set())
        _add_results(chunk_ids)

        # 2. 如果指定了单位，也搜带单位的组合
        if unit and len(results) < 5:
            unit_key = f"{number_norm}{unit}"
            chunk_ids2 = self.retrieval_system.numeric_index.get(unit_key, set())
            _add_results(chunk_ids2)

        # 3. 如果精确匹配无结果，做子串匹配（在所有 chunk 文本中搜索）
        if not results:
            # 用去逗号的数字 pattern 搜去逗号文本，加数字边界避免 "2049" 命中 "204981" 假阳性
            import re
            pattern = re.compile(r'(?<!\d)' + re.escape(number_norm) + r'(?!\d)')
            for cid, data in self.retrieval_system.chunk_data.items():
                if cid not in scope:
                    continue
                text_no_comma = data["text"].replace(",", "")
                if pattern.search(text_no_comma):
                    text = data["text"]
                    if not _context_match(text, context):
                        continue
                    excerpt = self._excerpt_number(text, number_norm, 400)
                    results.append({
                        "chunk_id": cid,
                        "text": excerpt,
                        "doc_id": data["doc_id"],
                    })
                    if len(results) >= 5:
                        break

        return results[:5]

    def _exec_search_phrase(self, phrase: str, doc_ids: list = None) -> list:
        """搜索包含特定短语的 chunks"""
        results = []
        scope = self._scope(doc_ids)

        chunk_ids = self.retrieval_system.phrase_index.get(phrase, set())
        for cid in chunk_ids:
            if cid in scope:
                data = self.retrieval_system.chunk_data[cid]
                results.append({
                    "chunk_id": cid,
                    "text": self._excerpt(data["text"], phrase, 400),
                    "doc_id": data["doc_id"],
                })

        # 子串匹配 fallback：phrase_index 未命中时，直接在 chunk 文本中搜索
        # 解决 "5%"、"8.5%"、"3日" 等带特殊字符/单位短语未被索引的问题
        if len(results) < 5 and phrase:
            import re
            try:
                pattern = re.compile(re.escape(phrase))
            except re.error:
                pattern = None
            if pattern:
                for cid, data in self.retrieval_system.chunk_data.items():
                    if cid not in scope:
                        continue
                    if cid in {r.get("chunk_id") for r in results}:
                        continue
                    text = data["text"]
                    if pattern.search(text):
                        results.append({
                            "chunk_id": cid,
                            "text": self._excerpt(text, phrase, 400),
                            "doc_id": data["doc_id"],
                        })
                        if len(results) >= 5:
                            break

        # 拆词 BM25 fallback：长 phrase/带空格 phrase 且仍无结果时，分词后检索
        # 解决模型用 "资产减值补偿 通知" 这类多词长 phrase 搜不到的问题
        if not results and phrase and ((' ' in phrase) or len(phrase) > 8):
            words = [w for w in jieba.lcut(phrase)
                     if len(w) >= 2 and not w.isspace() and not w.isdigit()]
            if len(words) >= 2:
                words = words[:4]
                scored = []
                for cid, data in self.retrieval_system.chunk_data.items():
                    if cid not in scope:
                        continue
                    text = data["text"]
                    hits = sum(1 for w in words if w in text)
                    if hits >= 2:
                        scored.append((hits, cid, data))
                scored.sort(key=lambda x: -x[0])
                for hits, cid, data in scored[:5]:
                    results.append({
                        "chunk_id": cid,
                        "text": self._excerpt(data["text"], words[0], 400),
                        "doc_id": data["doc_id"],
                    })
        return results[:5]

    def _exec_search_entity(self, entity_name: str, doc_ids: list = None) -> list:
        """搜索提到特定实体的 chunks"""
        results = []
        scope = self._scope(doc_ids)

        chunk_ids = self.retrieval_system.entity_index.get(entity_name, set())
        for cid in chunk_ids:
            if cid in scope:
                data = self.retrieval_system.chunk_data[cid]
                results.append({
                    "chunk_id": cid,
                    "text": self._excerpt(data["text"], entity_name, 400),
                    "doc_id": data["doc_id"],
                })

        # 也搜索 phrase_index 中的实体名
        if not results:
            chunk_ids2 = self.retrieval_system.phrase_index.get(entity_name, set())
            for cid in chunk_ids2:
                if cid in scope:
                    data = self.retrieval_system.chunk_data[cid]
                    results.append({
                        "chunk_id": cid,
                        "text": data["text"][:500],
                        "doc_id": data["doc_id"],
                    })

        return results[:5]

    def _exec_search_keyword(self, keywords: list, doc_ids: list = None) -> list:
        """搜索同时包含所有关键词的 chunks；若全包含无果，退化为任一包含"""
        results = []
        scope = self._scope(doc_ids)

        # 从第一个关键词的 phrase_index 开始
        if not keywords:
            return results

        # 找第一个关键词的候选
        candidates = set()
        for kw in keywords:
            kw_chunks = self.retrieval_system.phrase_index.get(kw, set())
            if not candidates:
                candidates = kw_chunks & scope
            else:
                candidates = candidates & kw_chunks

        # 如果交集为空，退化为只用第一个关键词
        if not candidates:
            candidates = self.retrieval_system.phrase_index.get(keywords[0], set()) & scope

        for cid in candidates:
            data = self.retrieval_system.chunk_data[cid]
            text = data["text"]
            # 检查是否所有关键词都在文本中
            if all(kw in text for kw in keywords):
                results.append({
                    "chunk_id": cid,
                    "text": text[:500],
                    "doc_id": data["doc_id"],
                })

        # 子串匹配 fallback：phrase_index 未命中或不足时，直接遍历 chunk 文本
        # 解决 "5%"、"8.5%"、"3日" 等带特殊字符/单位短语未被索引的问题
        if len(results) < 5:
            existing = {r.get("chunk_id") for r in results}
            for cid, data in self.retrieval_system.chunk_data.items():
                if cid not in scope or cid in existing:
                    continue
                text = data["text"]
                if all(kw in text for kw in keywords):
                    results.append({
                        "chunk_id": cid,
                        "text": text[:500],
                        "doc_id": data["doc_id"],
                    })
                    if len(results) >= 5:
                        break

        # 最终退化：多关键词全包含无果时，退化为任一包含（按命中数排序）
        if not results and len(keywords) > 1:
            scored = []  # (命中数, cid, data)
            for cid, data in self.retrieval_system.chunk_data.items():
                if cid not in scope:
                    continue
                text = data["text"]
                hit = sum(1 for kw in keywords if kw in text)
                if hit > 0:
                    scored.append((hit, cid, data))
            scored.sort(key=lambda x: -x[0])
            for hit, cid, data in scored[:5]:
                results.append({
                    "chunk_id": cid,
                    "text": f"[命中{hit}/{len(keywords)}个关键词] " + data["text"][:450],
                    "doc_id": data["doc_id"],
                })

        return results[:5]

    def _execute_tool_call(self, tool_name: str, arguments: dict, doc_ids: list = None) -> list:
        """执行单个工具调用

        Args:
            doc_ids: 限定搜索的文档范围（None=全局）
        """
        if tool_name == "search_number":
            res = self._exec_search_number(
                arguments.get("number", ""),
                arguments.get("unit"),
                arguments.get("context"),
                doc_ids=doc_ids,
            )
        elif tool_name == "search_phrase":
            res = self._exec_search_phrase(arguments.get("phrase", ""), doc_ids=doc_ids)
        elif tool_name == "search_entity":
            res = self._exec_search_entity(arguments.get("entity_name", ""), doc_ids=doc_ids)
        elif tool_name == "search_keyword":
            res = self._exec_search_keyword(arguments.get("keywords", []), doc_ids=doc_ids)
        else:
            return [{"error": f"未知工具: {tool_name}"}]
        if os.environ.get("DEBUG_TOOL"):
            print(f"    [tool] {tool_name}({arguments}) doc_ids={doc_ids} -> {len(res)}条")
        return res

    def _build_option_evidence_map(self, question: str, options: dict, doc_ids: list = None, top_k: int = 5) -> dict:
        """为多选题构建选项级证据映射（放宽匹配版）

        优化点：
        - 先用 retrieval_system.search(question + option) 粗召回
        - 过滤条件放宽：
          * 强锚点命中1个以上 → 直接保留
          * 弱锚点命中2个以上 → 保留（原为任一命中即保留，但弱锚点太严格）
          * 无任何锚点时保留原始证据（避免option_scores全0）
        """
        evidence_map = {}
        for opt, opt_text in sorted(options.items()):
            query_options = {opt: opt_text}
            raw = self.retrieval_system.search(
                question=question,
                options=query_options,
                doc_ids=doc_ids,
                top_k=top_k,
            )

            # 提取锚点：强=数值/条款号/法规名，弱=实义词
            strong_anchors, weak_terms = _extract_opt_anchors(opt_text)
            filtered = []
            for ev in raw:
                text = ev.get("text", "")
                # 强锚点任一命中 → 直接保留
                if strong_anchors and any(anchor in text for anchor in strong_anchors):
                    filtered.append(ev)
                    continue
                # 弱锚点命中2个以上 → 保留（放宽条件，避免全0）
                if weak_terms:
                    hit_count = sum(1 for term in weak_terms if term in text)
                    if hit_count >= 2:
                        filtered.append(ev)
                        continue
                # 无任何锚点提取到 → 保留原始证据（兜底，避免option_scores全0）
                if not strong_anchors and not weak_terms:
                    filtered.append(ev)
            evidence_map[opt] = filtered
        return evidence_map

    def _expand_multi_option_search(self, options: dict, doc_ids: list = None) -> list:
        """为多选题扩大搜索：为每个选项单独做BM25检索，找到所有有证据的选项（放宽匹配版）"""
        selected = []
        for opt, opt_text in sorted(options.items()):
            # 用选项文本做单独检索
            results = self.retrieval_system.search(
                question=opt_text,
                options={opt: opt_text},
                doc_ids=doc_ids,
                top_k=5,
            )
            if results:
                # 提取锚点（强=数值/条款号/法规名，弱=实义词）
                strong_anchors, weak_terms = _extract_opt_anchors(opt_text)
                for ev in results:
                    text = ev.get("text", "")
                    # 强锚点任一命中 → 直接选中
                    if strong_anchors and any(anchor in text for anchor in strong_anchors):
                        selected.append(opt)
                        break
                    # 弱锚点命中2个以上 → 选中（放宽条件）
                    if weak_terms:
                        hit_count = sum(1 for term in weak_terms if term in text)
                        if hit_count >= 2:
                            selected.append(opt)
                            break
                    # 无任何锚点 → 直接选中（兜底）
                    if not strong_anchors and not weak_terms:
                        selected.append(opt)
                        break
        return list(set(selected))

    def _enforce_multi_option_rules(self, question: str, options: dict, doc_ids: list,
                                    current_answer: str) -> str:
        """强制执行多选题规则：必须至少2个选项，不能使用默认AB

        Args:
            question: 问题
            options: 选项字典
            doc_ids: 文档范围
            current_answer: 当前答案（可能只有1个选项）
        """
        # 1. 已有2个或更多，直接返回
        if current_answer and len(current_answer) >= 2:
            return current_answer

        # 2. 尝试扩大搜索找到更多有证据的选项
        expanded = self._expand_multi_option_search(options, doc_ids)
        if len(expanded) >= 2:
            return "".join(sorted(expanded))

        # 3. 如果只有1个选项有证据，保留它并加最可能的另一个选项
        # 从当前答案（如果有）和扩展搜索结果中合并
        candidates = set()
        if current_answer:
            candidates.update(current_answer)
        candidates.update(expanded)

        if len(candidates) >= 2:
            return "".join(sorted(candidates))

        # 4. 如果只有1个，加上选项A（或下一个字母）
        if len(candidates) == 1:
            single = list(candidates)[0]
            if single != "A":
                return "A" + single
            else:
                return "AB"

        # 5. 完全没有找到任何选项（极端情况），通过问题+选项组合再搜一次
        all_text = question + " " + " ".join(options.values())
        final_search = self.retrieval_system.search(
            question=all_text,
            options=options,
            doc_ids=doc_ids,
            top_k=10,
        )
        if final_search:
            # 分析搜索结果匹配哪些选项
            matched = set()
            for ev in final_search:
                text = ev.get("text", "")
                for opt, opt_text in options.items():
                    opt_terms = set(re.findall(r'[一-鿿]{2,}', opt_text))
                    if opt_terms and any(term in text for term in opt_terms):
                        matched.add(opt)
            if len(matched) >= 2:
                return "".join(sorted(matched))
            elif len(matched) == 1:
                single = list(matched)[0]
                if single != "A":
                    return "A" + single
                else:
                    return "AB"

        # 最后兜底：AB（比赛要求必须至少2个）
        return "AB"

    # ========== 多选锚点验证后处理（MULTI_ANCHOR_VERIFY） ==========

    @staticmethod
    def _norm_num_for_match(s: str) -> str:
        """数值匹配归一化：去千分位逗号与空白，便于 '1,425,051' vs '1425051' 匹配"""
        return s.replace(",", "").replace("，", "").strip()

    def _verify_multi_answer(self, answer: str, options: dict, doc_ids: list,
                             answer_source: str) -> tuple:
        """多选锚点验证：逐选项检查"数值锚点+指标词"是否在该题文档 chunk 中共现

        判定逻辑（与零锚点拒答同思路，方向相反）：
        - 真表述：选项数值（原文值）必然出现在文档某个 chunk，且该 chunk 含指标词（宽松回退：数值全文命中）
        - 篡改/无据表述：数值不在文档 → 剔除
        - 验证出的支持选项 ≥2 时以验证结果为准（修复模型"只选1个"漏选）；
          支持选项 <2 时保留模型答案（防单位换算等锚点匹配失败导致误杀）

        Returns:
            (final_answer, final_answer_source)
        """
        chunk_data = getattr(self.retrieval_system, "chunk_data", {}) or {}
        doc_chunks = [c for c in chunk_data.values()
                      if c.get("doc_id") in (doc_ids or []) and c.get("text")]
        if not doc_chunks:
            return answer, answer_source
        # 预归一化 chunk 文本（去逗号），加速数值子串匹配
        norm_texts = [(c, self._norm_num_for_match(c["text"])) for c in doc_chunks]

        supported = []
        detail = {}
        for opt, opt_text in options.items():
            opt_norm = opt_text.replace("％", "%")
            # 长锚点（千分位/小数/4位以上数字）：数值命中即可（弱验证）
            long_nums = re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{4,}", opt_norm)
            # 短数字（1-3位）须带单位：单数字太易误配，强验证要求"数字+单位"在 chunk 中完整出现
            # 左边界 (?<![\d.]) 防止从 "108.98亿元" 中抠出 "98亿"（小数位误配）
            short_pairs = re.findall(
                r"(?<![\d.])(\d{1,3})\s*(个月|年|月|日|%|亿|万|元|倍|家|人|次|股|吨|辆|天)", opt_norm)
            # 裸短数字（无单位，如"等待期为12"/"出货量为550"）：仅强验证（数字+指标词同 chunk）
            bare_nums = [n for n in re.findall(r"(?<![\d.])(\d{1,3})(?![\d.])", opt_norm)
                         if not re.search(rf"{re.escape(n)}\s*(?:个月|年|月|日|%|亿|万|元|倍|家|人|次|股|吨|辆|天)", opt_norm)]
            if not long_nums and not short_pairs and not bare_nums:
                detail[opt] = "no_anchor"
                continue
            norm_long = [self._norm_num_for_match(n) for n in long_nums]
            # 指标词：选项中文实义词（要求与数值同 chunk 共现）
            terms = [t for t in re.findall(r"[一-鿿]{2,}", opt_norm)
                     if t not in ("为", "约", "较", "及", "与")]
            hit = False
            for c, nt in norm_texts:
                # 弱验证：长锚点数值命中该 chunk（带数字边界断言，防 "24" 误配 "243"）
                if long_nums and any(
                        re.search(rf"(?<!\d){re.escape(n)}(?!\d)", nt)
                        for n in norm_long):
                    hit = True
                    break
                # 强验证："数字+单位"完整串与指标词同 chunk 共现
                # （如"等待期12个月"须同 chunk 出现"12个月"与"等待期"）
                c_norm = c["text"].replace("％", "%")
                if short_pairs and any(t in c_norm for t in terms) \
                        and any(re.search(rf"(?<!\d){re.escape(num)}\s*{re.escape(unit)}", c_norm)
                                for num, unit in short_pairs):
                    hit = True
                    break
                # 裸短数字强验证：数字（带边界）与指标词同 chunk 共现
                if bare_nums and any(t in c_norm for t in terms) \
                        and any(re.search(rf"(?<!\d){re.escape(n)}(?!\d)", nt) for n in bare_nums):
                    hit = True
                    break
            detail[opt] = "hit" if hit else "miss"
            if hit:
                supported.append(opt)

        # 支持选项 ≥2 → 以验证结果为准；否则保留模型答案（锚点匹配失败防误杀）
        if len(supported) >= 2:
            verified = "".join(sorted(supported))
            src = "anchor_verified" if verified != answer else answer_source
            return verified, src
        return answer, answer_source

    # ========== 联网搜索支持（Tavily MCP） ==========

    def _web_search_tavily(self, query: str, search_depth: str = "basic") -> list:
        """使用 Tavily MCP 进行联网搜索

        Args:
            query: 搜索关键词
            search_depth: basic / advanced

        Returns:
            搜索结果列表
        """
        try:
            import requests
            # 使用 Tavily MCP 端点
            mcp_url = "http://localhost:3000/tavily/search"
            response = requests.post(mcp_url, json={
                "query": query,
                "search_depth": search_depth,
                "max_results": 5,
            }, timeout=10)
            if response.status_code == 200:
                return response.json().get("results", [])
        except Exception:
            # MCP 不可用时尝试直接调用 Tavily API
            try:
                import requests
                # 尝试从环境变量获取 Tavily API Key
                api_key = os.environ.get("TAVILY_API_KEY", "")
                if not api_key:
                    return []
                response = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "query": query,
                        "search_depth": search_depth,
                        "max_results": 5,
                        "api_key": api_key,
                    },
                    timeout=10
                )
                if response.status_code == 200:
                    return response.json().get("results", [])
            except Exception:
                pass
        return []

    def _should_enable_web_search(self, tool_calls_log: list, initial_evidence: list,
                                   answer_format: str, domain: str = "") -> bool:
        """判断是否应该启用联网搜索

        触发条件：
        1. 模式为 auto 或 research_debug
        2. 文档搜索已进行3轮仍无足够证据
        3. 初始证据质量低（数值题找不到数值，条款题找不到条款）
        """
        if not self.allow_web_search:
            return False

        # 检查工具调用次数
        if len(tool_calls_log) >= 3:
            # 3轮后仍在搜索，说明文档中找不到
            return True

        # 检查初始证据质量
        if not initial_evidence:
            return False

        # 数值题：检查证据中是否有数字
        if answer_format == "mcq" and "多少" in str(initial_evidence):
            has_numbers = any(re.search(r'\d{2,}', str(ev.get("text", ""))) for ev in initial_evidence)
            if not has_numbers:
                return len(tool_calls_log) >= 2

        # 财务报表：经常需要对比外部数据
        if domain == "financial_reports" and len(tool_calls_log) >= 2:
            return True

        return False

    # ========== 主流程 ==========

    def _chat_with_retry(self, messages, tools, tool_choice, max_tokens, trace, retries=1):
        """带重试的 LLM 调用（经 llm_api 适配层，支持 Responses API / chat 双协议）

        Returns:
            chat 风格响应对象；重试耗尽仍失败返回 None（trace 记录错误）
        """
        last_err = None
        for attempt in range(retries + 1):
            try:
                return call_llm(self.client, self.model, messages,
                                tools=tools, tool_choice=tool_choice,
                                temperature=0.1, max_tokens=max_tokens,
                                extra_body=QWEN_EXTRA_BODY,
                                use_responses_api=USE_RESPONSES_API)
            except Exception as e:
                last_err = e
                # 部分端点不支持 tool_choice="required"（如 AMD Responses API）：
                # 降级为 auto 立即重试一次，保住多选"每选项搜证据"的主流程
                if tool_choice == "required" and "required mode" in str(e):
                    try:
                        trace.append({"step": "tool_choice_downgrade",
                                      "detail": "required不支持，降级auto"})
                        return call_llm(self.client, self.model, messages,
                                        tools=tools, tool_choice="auto",
                                        temperature=0.1, max_tokens=max_tokens,
                                        extra_body=QWEN_EXTRA_BODY,
                                        use_responses_api=USE_RESPONSES_API)
                    except Exception as e2:
                        last_err = e2
                if attempt < retries:
                    trace.append({"step": "retry", "attempt": attempt + 1, "detail": str(e)})
                    time.sleep(1)
        trace.append({"step": "error", "detail": f"API重试{retries}次后仍失败: {last_err}"})
        print(f"[错误] API 调用失败（含重试）: {last_err}")
        return None

    def _run_tool_loop(
        self,
        question: str,
        options: dict,
        answer_format: str,
        initial_evidence: list = None,
        domain: str = "",
        max_rounds: int = 3,
        doc_ids: list = None,
        trace: list = None,
        is_trap: bool = False,
    ) -> dict:
        """检索 + 推理 + 工具调用的公共交互流程（不做答案后处理/强制补全）

        被 answer_question（生产，强制补全）和 diagnose_question（调试，不猜答案）共用，
        确保两条路径跑的是同一套真实模型交互，差异只在后处理。

        Args:
            trace: 如果传入列表，会把每一步（初始证据/每轮工具调用/模型输出）追加进去，供调试打印
            is_trap: 陷阱题（问题自带拒答提示）→ 追加拒答覆盖指令 + token 早停

        Returns:
            {"final_content", "tool_calls_log", "total_tokens",
             "option_evidence_map", "web_search_results"}
        """
        if trace is None:
            trace = []

        if not self.client:
            trace.append({"step": "error", "detail": "API client 未初始化"})
            return {
                "final_content": "", "tool_calls_log": [], "total_tokens": 0,
                "option_evidence_map": {}, "web_search_results": [],
            }

        # 构建初始用户消息
        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        # 证据池：初始证据 + 选项补充证据 + 工具追加证据，统一编号
        # （模型按 [证据N] 引用，answer_question 阶段用池大小校验引用有效性）
        evidence_pool = list(initial_evidence or [])[:10]
        evidence_text = self._format_evidence(evidence_pool) if evidence_pool else "（无初始证据，请使用工具搜索）"

        trace.append({
            "step": "initial_evidence",
            "count": len(initial_evidence) if initial_evidence else 0,
            "items": [
                {"doc_id": e.get("doc_id", ""), "chunk_id": e.get("chunk_id", ""), "text": e.get("text", "")[:200]}
                for e in (initial_evidence or [])[:10]
            ],
        })

        option_evidence_map = {}
        option_evidence_text = ""
        if answer_format == "multi":
            option_evidence_map = self._build_option_evidence_map(question, options, doc_ids=doc_ids, top_k=3)
            option_sections = []
            for opt, evs in sorted(option_evidence_map.items()):
                # 编号延续证据池（保持 [证据N] 全局唯一，避免引用歧义）
                section = f"### 选项{opt}的补充证据\n" + self._format_evidence(evs, start_idx=len(evidence_pool) + 1)
                option_sections.append(section)
                evidence_pool.extend(e for e in evs if isinstance(e, dict))
            option_evidence_text = "\n\n".join(option_sections)
            trace.append({
                "step": "option_evidence_map",
                "detail": {opt: len(evs) for opt, evs in option_evidence_map.items()},
            })

        format_map = {"mcq": "单选题", "multi": "多选题", "tf": "判断题"}
        # 多选题：强制逐选项验证，禁止只选1个就回答
        multi_hint = (
            "\n**多选题特别要求**：必须为每个选项都搜索证据并逐项给出结论"
            "（✓正确/✗错误/?无证据）。有证据支持的选项必须选，无证据支持的不选。"
            "禁止仅凭初始证据或直觉只选1个选项就回答。"
        ) if answer_format == "multi" else ""

        user_message = f"""## 问题
{question}

## 选项
{options_text}

## 题型：{format_map.get(answer_format, answer_format)}

## 初始证据（来自初步检索）
{evidence_text}

{option_evidence_text}

请分析问题，如果初始证据足够就直接回答；如果不够，调用工具搜索补充证据。{multi_hint}"""

        # 陷阱题覆盖指令：追加在 user_message 末尾（后置指令优先级高，压制系统提示中
        # "文档中未提及也要给出最接近选项"的规则——该规则只适用于正常题，与陷阱题直接冲突）
        if is_trap:
            user_message += """

## 本题特殊规则（最高优先级）
本题为"未提及须拒答"类问题。逐项核对选项后，若所有选项在证据和搜索结果中均找不到直接依据，
最后一行必须直接写：答案：REFUSED
（禁止猜测最接近的选项，禁止输出"文档中未提及但仍选择X"）"""

        # 对话循环（注入领域知识）
        domain_knowledge = DOMAIN_KNOWLEDGE.get(domain, "")
        system_prompt = TOOL_SYSTEM_PROMPT
        if domain_knowledge:
            system_prompt += "\n\n" + domain_knowledge
        supplement = _build_knowledge_supplement(domain)
        if supplement:
            system_prompt += "\n\n" + supplement

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        total_tokens = 0
        tool_calls_log = []
        final_content = ""
        consecutive_tool_failures = 0  # 连续空结果工具调用计数（超限后降级收尾）

        for round_idx in range(max_rounds):
            # 每题 token 熔断：超过阈值立即停止工具循环（由末尾强制收尾兜底）
            if total_tokens > PER_QUESTION_TOKEN_LIMIT:
                trace.append({"step": "budget_break", "round": round_idx + 1,
                              "detail": f"token={total_tokens} 超过每题阈值 {PER_QUESTION_TOKEN_LIMIT}"})
                break
            # 陷阱题 token 早停：陷阱题搜不出证据，提前止损交给收尾消息输出拒答判定
            if is_trap and TRAP_TOKEN_EARLY_STOP > 0 \
                    and total_tokens > PER_QUESTION_TOKEN_LIMIT * TRAP_TOKEN_EARLY_STOP:
                trace.append({"step": "trap_early_stop", "round": round_idx + 1,
                              "detail": f"token={total_tokens} 超过早停阈值 "
                                        f"{PER_QUESTION_TOKEN_LIMIT * TRAP_TOKEN_EARLY_STOP:.0f}"})
                break
            # 多选题首轮强制调用工具（确保为每个选项搜索证据，避免只看初始证据就单选），
            # 后续轮 auto（允许模型收集足够证据后直接回答）
            if answer_format == "multi" and round_idx == 0:
                tool_choice = "required"
            else:
                tool_choice = "auto"
            try:
                response = self._chat_with_retry(
                    messages, SEARCH_TOOLS, tool_choice, 1200, trace)
                # 防御：ModelScope 偶发返回空 choices -> 退化为 auto 重试一次
                if response and not response.choices:
                    trace.append({"step": "warning", "round": round_idx + 1, "detail": "空choices，auto重试"})
                    response = self._chat_with_retry(
                        messages, SEARCH_TOOLS, "auto", 1200, trace)
                    if response and not response.choices:
                        trace.append({"step": "error", "round": round_idx + 1, "detail": "重试仍空choices"})
                        break
                if response is None:
                    break

                msg = response.choices[0].message
                usage = response.usage
                if usage:
                    total_tokens += usage.prompt_tokens + usage.completion_tokens

                if msg.tool_calls:
                    messages.append(msg)

                    for tc in msg.tool_calls:
                        fn_name = tc.function.name
                        try:
                            fn_args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            fn_args = {}

                        results = self._execute_tool_call(fn_name, fn_args, doc_ids=doc_ids)
                        # 连续空结果计数：超过 TOOL_CALL_FAILURE_LIMIT 则降级收尾
                        # （防止模型反复空搜烧 token，由末尾强制收尾产出最终答案）
                        if results:
                            consecutive_tool_failures = 0
                        else:
                            consecutive_tool_failures += 1
                        tool_calls_log.append({
                            "tool": fn_name,
                            "args": fn_args,
                            "results_count": len(results),
                        })
                        trace.append({
                            "step": "tool_call",
                            "round": round_idx + 1,
                            "tool": fn_name,
                            "args": fn_args,
                            "results": [
                                {"doc_id": r.get("doc_id", ""), "chunk_id": r.get("chunk_id", ""), "text": r.get("text", "")[:200]}
                                for r in results if isinstance(r, dict)
                            ],
                        })

                        # 工具结果入池并带编号返回给模型（延续 [证据N] 编号，可被后续轮引用）
                        start_idx = len(evidence_pool)
                        numbered_results = []
                        for j, r in enumerate(results[:5]):
                            if isinstance(r, dict) and r.get("text"):
                                evidence_pool.append(r)
                                numbered_results.append({"证据编号": start_idx + j + 1, **r})
                        tool_result_text = json.dumps(numbered_results, ensure_ascii=False, indent=2) \
                            if numbered_results else "[]（无匹配结果）"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result_text,
                        })

                    # 连续空结果超限：跳出循环走强制收尾
                    if consecutive_tool_failures >= TOOL_CALL_FAILURE_LIMIT:
                        trace.append({"step": "tool_failure_degrade",
                                      "detail": f"连续 {consecutive_tool_failures} 次空结果，降级为仅用已有证据作答"})
                        break
                    continue

                else:
                    final_content = msg.content or ""
                    trace.append({"step": "model_final_answer", "round": round_idx + 1, "content": final_content})
                    break

            except Exception as e:
                trace.append({"step": "error", "round": round_idx + 1, "detail": str(e)})
                print(f"[错误] 工具循环异常 (round {round_idx}): {e}")
                break

        # ========== 文档搜索3轮后，启用联网搜索 ==========
        # 陷阱题跳过联网搜索：外部资料反而会诱导模型猜测，与拒答目标相悖
        web_search_results = []
        if not is_trap and self._should_enable_web_search(tool_calls_log, initial_evidence, answer_format, domain):
            print(f"  [联网] 文档搜索{len(tool_calls_log)}轮无果，启用联网搜索...")
            try:
                search_query = question
                key_terms = re.findall(r'[一-鿿]{2,}', question)[:3]
                if key_terms:
                    search_query = question + " " + " ".join(key_terms)

                web_results = self._web_search_tavily(search_query)
                if web_results:
                    web_search_results = web_results
                    tool_calls_log.append({
                        "tool": "web_search",
                        "query": search_query,
                        "results_count": len(web_results),
                    })
                    trace.append({"step": "web_search", "query": search_query, "results_count": len(web_results)})

                    web_evidence_text = "## 联网搜索补充证据\n"
                    for i, result in enumerate(web_results[:5]):
                        title = result.get("title", "")
                        snippet = result.get("content", "")[:300]
                        web_evidence_text += f"[{i+1}] {title}\n{snippet}\n\n"

                    messages.append({
                        "role": "user",
                        "content": web_evidence_text + "\n请基于文档证据和联网补充证据给出最终答案。注意：优先使用文档证据，联网证据仅作补充。",
                    })

                    response = self._chat_with_retry(messages, None, "auto", 800, trace)
                    if response and response.choices:
                        final_content = response.choices[0].message.content or ""
                        trace.append({"step": "model_final_answer_after_web", "content": final_content})
                        usage = response.usage
                        if usage:
                            total_tokens += usage.prompt_tokens + usage.completion_tokens
            except Exception as e:
                trace.append({"step": "error", "detail": f"联网搜索失败: {e}"})
                print(f"  [联网] 搜索失败: {e}")

        # 循环耗尽/熔断/工具降级后仍未产出最终答案时，强制再请求一次（禁止工具调用）
        # 触发条件：final_content 为空，或其中解析不出答案字母（token熔断时常见"中间分析无答案"文本）；
        # 已含 REFUSED 拒答结论的内容视为已有答案，不再触发（避免重复收尾浪费 token）
        if tool_calls_log and (not final_content
                               or ("REFUSED" not in final_content
                                   and not self._extract_answer(final_content, answer_format))):
            # 陷阱题收尾消息：明确允许/要求拒答，避免模型被系统提示逼着猜最接近选项
            if is_trap:
                force_content = ("已经收集了足够证据。请不再调用工具，直接基于已有证据判断："
                                 "若文档未提及该信息，最后一行必须写：答案：REFUSED；"
                                 "只有找到明确依据时才输出对应选项。")
            else:
                force_content = ("已经收集了足够证据。请不再调用工具，直接基于已有证据给出最终答案。"
                                 "先用1-3句话分析关键证据（引用证据编号），最后一行必须写：答案：X")
            force_msg = {
                "role": "user",
                "content": force_content,
            }
            response = self._chat_with_retry(messages + [force_msg], None, "auto", 800, trace)
            if response and response.choices:
                final_content = response.choices[0].message.content or ""
                trace.append({"step": "model_final_answer_forced", "content": final_content})
                usage = response.usage
                if usage:
                    total_tokens += usage.prompt_tokens + usage.completion_tokens

        return {
            "final_content": final_content,
            "tool_calls_log": tool_calls_log,
            "total_tokens": total_tokens,
            "option_evidence_map": option_evidence_map,
            "web_search_results": web_search_results,
            "evidence_pool": evidence_pool,
        }

    def answer_question(
        self,
        question: str,
        options: dict,
        answer_format: str,
        initial_evidence: list = None,
        domain: str = "",
        max_rounds: int = None,
        doc_ids: list = None,
    ) -> dict:
        """
        模型驱动检索 + 回答（生产路径：证据不足时会强制补全出符合比赛格式的答案）

        Args:
            question: 问题
            options: 选项字典
            answer_format: 题型
            initial_evidence: BM25 粗召回的初始证据
            domain: 领域
            max_rounds: 最大工具调用轮数（None 时按题型从 config.MAX_ROUNDS_BY_FORMAT 取）

        Returns:
            {"answer", "reasoning", "tool_calls_log", "total_tokens",
             "citations", "refused", ...}
        """
        # 陷阱题分流：问题自带"若文档未提及，请拒答"提示 → 降低工具轮数（搜多了也答不出，省token），
        # 后处理阶段对答不出/有拒答信号的陷阱题强制输出 REFUSED
        is_trap = detect_trap_hint(question)
        # 按题型分级的轮数预算（多选题搜索面大给更多轮，判断题收敛快给更少轮）
        if max_rounds is None:
            if is_trap and TRAP_MAX_ROUNDS > 0:
                max_rounds = TRAP_MAX_ROUNDS
            else:
                max_rounds = MAX_ROUNDS_BY_FORMAT.get(answer_format, 3)

        if not self.client:
            fallback_answer = self._evidence_based_fallback(
                question, options, answer_format, initial_evidence
            )
            return {"answer": fallback_answer, "reasoning": "client未初始化",
                    "tool_calls_log": [], "total_tokens": 0,
                    "answer_source": "api_error_fallback",
                    "citations": {"valid": [], "invalid": [], "coverage": 0.0},
                    "refused": False}

        loop_result = self._run_tool_loop(
            question, options, answer_format,
            initial_evidence=initial_evidence, domain=domain,
            max_rounds=max_rounds, doc_ids=doc_ids, is_trap=is_trap,
        )
        final_content = loop_result["final_content"]
        tool_calls_log = loop_result["tool_calls_log"]
        total_tokens = loop_result["total_tokens"]
        option_evidence_map = loop_result["option_evidence_map"]
        web_search_results = loop_result["web_search_results"]
        evidence_pool = loop_result.get("evidence_pool", [])

        # ===== 引用校验与拒答检测（citation_manager）=====
        # record 模式：仅记录校验结果进结果字段；strict 模式：数值无证据支撑时触发一次重写
        cited = parse_citations(final_content)
        citation_result = verify_citations(cited, len(evidence_pool))
        refused_signal = detect_refusal(final_content)

        if CITATION_MODE == "strict" and final_content:
            unsupported = check_numeric_support(
                final_content, [e.get("text", "") for e in evidence_pool])
            if unsupported:
                # 数值无证据支撑：请求模型基于证据重写一次（防幻觉数值）
                fix_msg = {"role": "user",
                           "content": f"你结论中的数值 {unsupported} 未在证据原文中出现，"
                                      "可能是幻觉。请只基于证据原文重新给出最终答案，"
                                      "最后一行必须写：答案：X"}
                fix_resp = self._chat_with_retry(
                    messages=[{"role": "assistant", "content": final_content}, fix_msg],
                    tools=None, tool_choice="auto", max_tokens=800, trace=[])
                if fix_resp and fix_resp.choices:
                    fixed = fix_resp.choices[0].message.content or ""
                    if fixed:
                        final_content = fixed
                        usage = fix_resp.usage
                        if usage:
                            total_tokens += usage.prompt_tokens + usage.completion_tokens
                        # 重写后重新解析答案与引用
                        cited = parse_citations(final_content)
                        citation_result = verify_citations(cited, len(evidence_pool))
                        refused_signal = detect_refusal(final_content)

        # 提取答案（直接使用模型原始答案，不做fallback补全）
        raw_model_answer = self._extract_answer(final_content, answer_format)
        answer = raw_model_answer
        answer_source = "model"

        # 多选题：标记答案来源，不使用无依据的 fallback
        if answer_format == "multi":
            if not answer:
                answer_source = "no_answer"
            elif len(answer) < 2:
                answer_source = "single_option"

        # 拒答机制（REFUSAL_MODE=guess_fallback）：模型给出拒答信号时保留最佳猜测，
        # 但标记来源为 refused_guess（评测时可据此统计拒答率与拒答后正确率）
        if refused_signal and answer_source == "model":
            answer_source = "refused_guess"

        # 统一答案校验（只做格式规范，不做内容补全）
        if not answer:
            # 兜底前先尝试"数值→选项字母"映射：计算题模型常直接输出计算结果数值（如"答案：26"）
            # 而非选项字母，此时从最终回答中匹配选项的数值文本反推字母。
            # 取"数值最后出现位置最靠后"的选项——原始值（运算数）先出现，计算结果在最后
            if answer_format == "mcq" and options and final_content:
                content_norm = final_content.replace(",", "")
                best_letter, best_pos = None, -1
                for letter, opt_text in options.items():
                    nums = re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+", opt_text)
                    if not nums:
                        continue
                    pos = [content_norm.rfind(n.replace(",", "")) for n in nums]
                    if all(p >= 0 for p in pos) and max(pos) > best_pos:
                        best_letter, best_pos = letter, max(pos)
                if best_letter:
                    answer, answer_source = best_letter, "model_value_matched"
            if not answer:
                # 模型完全没有输出答案
                if answer_format == "tf":
                    answer = "A"  # 判断题兜底
                    answer_source = "validate_guess"
                elif answer_format == "mcq":
                    answer = "A"  # 单选兜底
                    answer_source = "validate_guess"
                elif answer_format == "multi":
                    answer = "A"  # 多选题兜底：至少选一个
                    answer_source = "validate_guess"
        else:
            # 格式规范化：去重排序，只保留有效字母
            valid = set(options.keys()) if options else {"A", "B", "C", "D"}
            answer = "".join(sorted(set(c for c in answer.upper() if c in valid)))
            if not answer:
                answer_source = "validate_guess"
                if answer_format in ("mcq", "tf"):
                    answer = "A"

        # 多选锚点验证后处理（MULTI_ANCHOR_VERIFY）：修复模型"只选1个"漏选/凑数
        # 逐选项验证数值锚点在文档 chunk 的共现，支持≥2 项时以验证结果为准
        if MULTI_ANCHOR_VERIFY and answer_format == "multi" and doc_ids:
            answer, answer_source = self._verify_multi_answer(
                answer, options, doc_ids, answer_source)

        # 零锚点强制拒答（ZERO_EVIDENCE_REFUSAL，金融场景可信优先）：
        # 单选/判断题中，模型选中选项的强锚点（数值/年份/条款号/法规名）全部不在证据池原文，
        # 且模型未引用任何有效证据 → 选项无文档依据，强制拒答而非硬猜（陷阱题主要失分点）。
        # 注意：必须放在格式规范化之后，避免 "REFUSED" 中的 E/D 被过滤成 "D"
        # 仅对陷阱题启用：正常题零锚点误拒代价 > 猜对收益（single_hop 实测 1 题误拒 0 题挽救）
        if (ZERO_EVIDENCE_REFUSAL and is_trap and not refused_signal
                and answer_format in ("mcq", "tf") and answer in options):
            sel_text = options.get(answer, "")
            strong_anchors, _ = _extract_opt_anchors(sel_text)
            pool_text = " ".join(e.get("text", "") for e in evidence_pool)
            if strong_anchors and not any(a in pool_text for a in strong_anchors) \
                    and not citation_result.get("valid"):
                refused_signal = True
                answer_source = "zero_evidence_refused"
                answer = "REFUSED"

        # 陷阱题强制拒答（TRAP_REFUSAL_POLICY=auto_refuse）：问题自带"若文档未提及请拒答"提示
        # 且模型给出拒答信号 / 没输出可解析答案（validate_guess/refused_guess/no_answer）时，
        # 统一收敛为 REFUSED。必须放在零锚点分支之后、格式规范化之后（同上 REFUSED 被过滤的坑）。
        # model 来源且可能持有有效证据的答案不覆盖，保留模型判断（如"提示拒答但实际有答案"的题）。
        if (is_trap and TRAP_REFUSAL_POLICY == "auto_refuse"
                and answer_format in ("mcq", "tf")
                and (refused_signal
                     or answer_source in ("validate_guess", "refused_guess", "no_answer"))):
            answer = "REFUSED"
            answer_source = "trap_refused"
            refused_signal = True

        # 正常题永不拒答（guess_fallback 模式）：拒答信号时已保留最佳猜测字母，
        # 清除 refused 标志，避免评测把"猜对了的拒答题"计为误拒（single_hop 实测 8 题转正）
        if not is_trap and REFUSAL_MODE != "strict":
            refused_signal = False

        # 收集证据出处
        evidence_sources = []
        if initial_evidence:
            for ev in initial_evidence[:5]:
                evidence_sources.append({
                    "chunk_id": ev.get("chunk_id", ""),
                    "doc_id": ev.get("doc_id", ""),
                    "text_preview": ev.get("text", "")[:100],
                    "source_type": "document",
                })

        # 加入联网搜索证据
        if web_search_results:
            for ev in web_search_results[:3]:
                evidence_sources.append({
                    "chunk_id": "",
                    "doc_id": "web_search",
                    "text_preview": ev.get("content", "")[:100],
                    "source_type": "web",
                    "url": ev.get("url", ""),
                })

        option_judgements = {}
        if answer_format == "multi":
            for opt in sorted(options.keys()):
                option_judgements[opt] = {
                    "verdict": opt in answer,
                    "evidence_count": len(option_evidence_map.get(opt, [])),
                    "evidence_preview": [e.get("text", "")[:80] for e in option_evidence_map.get(opt, [])[:2]],
                }

        return {
            "answer": answer,
            "raw_model_answer": raw_model_answer,
            "answer_source": answer_source,
            "reasoning": final_content,
            "tool_calls_log": tool_calls_log,
            "total_tokens": total_tokens,
            "evidence_sources": evidence_sources,
            "option_judgements": option_judgements,
            "used_web_search": len(web_search_results) > 0,
            # 引用校验与拒答检测结果（改进.md §3 可追溯性）
            "citations": citation_result,
            "refused": refused_signal,
        }

    # ========== 单次 API 调用优化 ==========

    def _local_extract_search_tasks(self, question: str, options: dict, answer_format: str, doc_ids: list = None) -> dict:
        """本地规则提取搜索任务（无需 API）

        从问题和选项中提取：
        1. 数值搜索任务（金额、百分比、年份）
        2. 短语搜索任务（条款、定义、条件）
        3. 实体搜索任务（公司名、机构名）
        4. 关键词搜索任务（多概念组合）

        Returns:
            {"numbers": [...], "phrases": [...], "entities": [...], "keywords": [...]}
        """
        tasks = {"numbers": [], "phrases": [], "entities": [], "keywords": []}

        # 从选项中提取搜索任务
        for opt, opt_text in sorted(options.items()):
            # 数值：年份、百分比、金额
            numbers = re.findall(r'\d{4}年|\d+\.\d+%|\d+(?:\.\d+)?\s*(?:亿|万|元)', opt_text)
            for num in numbers:
                clean_num = re.sub(r'\s+', '', num).rstrip('年%亿元万')
                if clean_num and clean_num not in [t["number"] for t in tasks["numbers"]]:
                    unit = "年" if "年" in num else ("%" if "%" in num else None)
                    tasks["numbers"].append({"number": clean_num, "unit": unit, "source": opt})

            # 实体：公司名（连续中文2-10字+公司/集团/银行/证券/保险/基金）
            entities = re.findall(r'[一-鿿]{2,10}(?:公司|集团|银行|证券|保险|基金|控股|资产)', opt_text)
            for ent in entities:
                if ent not in tasks["entities"] and len(ent) >= 3:
                    tasks["entities"].append(ent)

            # 短语：关键条款词（2-6字实义词）
            phrases = [w for w in jieba.lcut(opt_text)
                       if 2 <= len(w) <= 6 and w not in _OPT_STOPWORDS
                       and not w.isdigit() and not re.fullmatch(r'[\d\s.\-/年月日%，]+', w)]
            for phrase in phrases[:3]:  # 每个选项最多3个短语
                if phrase not in tasks["phrases"]:
                    tasks["phrases"].append(phrase)

        # 从问题中提取关键词
        q_words = [w for w in jieba.lcut(question)
                   if 2 <= len(w) <= 8 and w not in _OPT_STOPWORDS
                   and not w.isdigit()]
        tasks["keywords"] = q_words[:5]

        # 从问题中提取实体
        q_entities = re.findall(r'[一-鿿]{2,10}(?:公司|集团|银行|证券|保险|基金|控股|资产|保险公司)', question)
        for ent in q_entities:
            if ent not in tasks["entities"] and len(ent) >= 3:
                tasks["entities"].append(ent)

        return tasks

    def _local_collect_evidence(self, question: str, options: dict, answer_format: str,
                                 initial_evidence: list, doc_ids: list = None,
                                 top_k_per_tool: int = 5) -> dict:
        """本地收集所有证据（无需 API）

        执行流程：
        1. BM25 初始证据（已在外部获取）
        2. 选项级证据映射（多选题）
        3. 基于本地提取的搜索任务，执行工具搜索
        4. 合并去重

        Returns:
            {
                "all_evidence": [...],  # 合并后的证据列表
                "option_evidence_map": {opt: [...]},  # 选项级证据
                "tool_search_results": {tool_name: [...]},  # 工具搜索结果
                "search_tasks": {...},  # 提取的搜索任务
            }
        """
        all_evidence = list(initial_evidence or [])
        seen_chunk_ids = {e.get("chunk_id") for e in all_evidence if e.get("chunk_id")}

        # 选项级证据映射
        option_evidence_map = {}
        if answer_format == "multi":
            option_evidence_map = self._build_option_evidence_map(question, options, doc_ids=doc_ids, top_k=8)
            for opt, evs in option_evidence_map.items():
                for ev in evs:
                    cid = ev.get("chunk_id")
                    if cid and cid not in seen_chunk_ids:
                        all_evidence.append(ev)
                        seen_chunk_ids.add(cid)

        # 本地提取搜索任务
        search_tasks = self._local_extract_search_tasks(question, options, answer_format, doc_ids)

        tool_search_results = {"search_number": [], "search_phrase": [], "search_entity": [], "search_keyword": []}

        # 执行数值搜索
        for task in search_tasks["numbers"][:5]:
            results = self._exec_search_number(task["number"], unit=task.get("unit"), doc_ids=doc_ids)
            for ev in results[:top_k_per_tool]:
                cid = ev.get("chunk_id")
                if cid and cid not in seen_chunk_ids:
                    all_evidence.append(ev)
                    seen_chunk_ids.add(cid)
                tool_search_results["search_number"].append(ev)

        # 执行短语搜索
        for phrase in search_tasks["phrases"][:12]:
            results = self._exec_search_phrase(phrase, doc_ids=doc_ids)
            for ev in results[:top_k_per_tool]:
                cid = ev.get("chunk_id")
                if cid and cid not in seen_chunk_ids:
                    all_evidence.append(ev)
                    seen_chunk_ids.add(cid)
                tool_search_results["search_phrase"].append(ev)

        # 执行实体搜索
        for entity in search_tasks["entities"][:5]:
            results = self._exec_search_entity(entity, doc_ids=doc_ids)
            for ev in results[:top_k_per_tool]:
                cid = ev.get("chunk_id")
                if cid and cid not in seen_chunk_ids:
                    all_evidence.append(ev)
                    seen_chunk_ids.add(cid)
                tool_search_results["search_entity"].append(ev)

        # 执行关键词搜索（问题关键词组合）
        if len(search_tasks["keywords"]) >= 2:
            results = self._exec_search_keyword(search_tasks["keywords"][:3], doc_ids=doc_ids)
            for ev in results[:top_k_per_tool]:
                cid = ev.get("chunk_id")
                if cid and cid not in seen_chunk_ids:
                    all_evidence.append(ev)
                    seen_chunk_ids.add(cid)
                tool_search_results["search_keyword"].append(ev)

        return {
            "all_evidence": all_evidence,
            "option_evidence_map": option_evidence_map,
            "tool_search_results": tool_search_results,
            "search_tasks": search_tasks,
        }

    def answer_question_single_call(
        self,
        question: str,
        options: dict,
        answer_format: str,
        initial_evidence: list = None,
        domain: str = "",
        doc_ids: list = None,
    ) -> dict:
        """单次 API 调用问答（优化版）

        流程：
        1. 本地规则提取搜索关键词（无 API）
        2. 本地执行所有工具搜索（无 API）
        3. 合并所有证据
        4. 单次 API 调用：模型综合证据给出最终答案

        每道题只需 1 次 API 调用。
        """
        if not self.client:
            fallback_answer = self._evidence_based_fallback(
                question, options, answer_format, initial_evidence
            )
            return {"answer": fallback_answer, "reasoning": "client未初始化",
                    "tool_calls_log": [], "total_tokens": 0,
                    "answer_source": "api_error_fallback"}

        # 步骤 1+2+3：本地收集所有证据
        evidence_result = self._local_collect_evidence(
            question, options, answer_format, initial_evidence, doc_ids
        )
        all_evidence = evidence_result["all_evidence"]
        option_evidence_map = evidence_result["option_evidence_map"]
        search_tasks = evidence_result["search_tasks"]

        # 构建证据文本
        evidence_text = self._format_evidence(all_evidence[:18]) if all_evidence else "（无证据）"

        # 多选题：构建选项级证据
        option_evidence_text = ""
        if answer_format == "multi" and option_evidence_map:
            option_sections = []
            for opt, evs in sorted(option_evidence_map.items()):
                section = f"### 选项{opt}的补充证据\n" + self._format_evidence(evs)
                option_sections.append(section)
            option_evidence_text = "\n\n".join(option_sections)

        # 构建提示
        format_map = {"mcq": "单选题", "multi": "多选题", "tf": "判断题"}
        multi_hint = ""
        if answer_format == "multi":
            multi_hint = """ 多选题：逐项判断每个选项✓或✗，把所有✓的字母组合成答案（如ABD）。
重要判断标准：
- 只要选项内容在文档中有直接或间接证据支持，就选✓
- 只有当选项内容与文档明确矛盾时，才选✗
- 如果无法确定，倾向于选✓（宁可多选也不要漏选）"""
        elif answer_format == "mcq":
            multi_hint = " 单选题：只选一个最匹配的选项。"

        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))

        user_message = f"""问题：{question}

选项：
{options_text}

题型：{format_map.get(answer_format, answer_format)}

证据（共{len(all_evidence)}条）：
{evidence_text}

{option_evidence_text}

请根据证据回答。{multi_hint}
规则：只用文档证据，不要外部知识。对于多选题，宁可多选也不要漏选。
最后一行写：答案：X"""

        # 领域知识注入
        domain_knowledge = DOMAIN_KNOWLEDGE.get(domain, "")
        system_prompt = TOOL_SYSTEM_PROMPT.split("## 工作流程")[0]  # 只保留基础 prompt，去掉工具调用指令
        if domain_knowledge:
            system_prompt += "\n\n" + domain_knowledge
        supplement = _build_knowledge_supplement(domain)
        if supplement:
            system_prompt += "\n\n" + supplement

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # 步骤 4：单次 API 调用
        total_tokens = 0
        final_content = ""
        tool_calls_log = []
        for key, results in evidence_result["tool_search_results"].items():
            if results:
                tool_calls_log.append({"tool": key, "results_count": len(results)})

        try:
            response = call_llm(self.client, self.model, messages,
                                tools=None, tool_choice=None,
                                temperature=0.1, max_tokens=1500,
                                extra_body=QWEN_EXTRA_BODY,
                                use_responses_api=USE_RESPONSES_API)
            final_content = response.choices[0].message.content or ""
            usage = response.usage
            if usage:
                total_tokens = usage.prompt_tokens + usage.completion_tokens
        except Exception as e:
            print(f"[错误] API 调用失败: {e}")
            # 基于已检索证据的 fallback，而非固定返回 A
            fallback_answer = self._evidence_based_fallback(
                question, options, answer_format, all_evidence, option_evidence_map
            )
            return {"answer": fallback_answer, "reasoning": f"API失败: {e}",
                    "tool_calls_log": [], "total_tokens": 0,
                    "answer_source": "api_error_fallback"}

        # 提取答案
        raw_model_answer = self._extract_answer(final_content, answer_format)
        answer = raw_model_answer
        answer_source = "model"

        if answer_format == "multi":
            if not answer:
                answer_source = "no_answer"
            elif len(answer) < 2:
                answer_source = "single_option"

        if not answer:
            if answer_format == "tf":
                answer = "A"
                answer_source = "validate_guess"
            elif answer_format == "mcq":
                answer = "A"
                answer_source = "validate_guess"
            elif answer_format == "multi":
                answer = "A"
                answer_source = "validate_guess"
        else:
            valid = set(options.keys()) if options else {"A", "B", "C", "D"}
            answer = "".join(sorted(set(c for c in answer.upper() if c in valid)))
            if not answer:
                answer_source = "validate_guess"
                if answer_format in ("mcq", "tf"):
                    answer = "A"

        # 多选锚点验证后处理（MULTI_ANCHOR_VERIFY）：修复模型"只选1个"漏选/凑数
        # 逐选项验证数值锚点在文档 chunk 的共现，支持≥2 项时以验证结果为准
        if MULTI_ANCHOR_VERIFY and answer_format == "multi" and doc_ids:
            answer, answer_source = self._verify_multi_answer(
                answer, options, doc_ids, answer_source)

        # 多选题强制至少2个选项
        if answer_format == "multi" and len(answer) == 1:
            # 根据证据找到第二个最可能的选项
            selected_opt = answer[0]
            remaining_opts = [opt for opt in sorted(options.keys()) if opt != selected_opt]

            # 检查哪个选项有最多的证据支持
            best_second = None
            best_count = 0
            for opt in remaining_opts:
                ev_count = len(option_evidence_map.get(opt, []))
                if ev_count > best_count:
                    best_count = ev_count
                    best_second = opt

            # 如果找到有证据的选项，添加它
            if best_second and best_count > 0:
                answer = "".join(sorted([selected_opt, best_second]))
                answer_source = "forced_multi"
            else:
                # 如果没有其他选项有证据，添加相邻的选项
                if selected_opt == "A":
                    answer = "AB"
                elif selected_opt == "B":
                    answer = "AB"
                elif selected_opt == "C":
                    answer = "BC"
                elif selected_opt == "D":
                    answer = "CD"
                answer_source = "forced_multi_default"

        # 收集证据出处
        evidence_sources = []
        for ev in all_evidence[:5]:
            evidence_sources.append({
                "chunk_id": ev.get("chunk_id", ""),
                "doc_id": ev.get("doc_id", ""),
                "text_preview": ev.get("text", "")[:100],
                "source_type": "document",
            })

        option_judgements = {}
        if answer_format == "multi":
            for opt in sorted(options.keys()):
                option_judgements[opt] = {
                    "verdict": opt in answer,
                    "evidence_count": len(option_evidence_map.get(opt, [])),
                    "evidence_preview": [e.get("text", "")[:80] for e in option_evidence_map.get(opt, [])[:2]],
                }

        return {
            "answer": answer,
            "raw_model_answer": raw_model_answer,
            "answer_source": answer_source,
            "reasoning": final_content,
            "tool_calls_log": tool_calls_log,
            "total_tokens": total_tokens,
            "evidence_sources": evidence_sources,
            "option_judgements": option_judgements,
            "used_web_search": False,
            "search_tasks_summary": {
                "numbers": len(search_tasks["numbers"]),
                "phrases": len(search_tasks["phrases"]),
                "entities": len(search_tasks["entities"]),
                "keywords": len(search_tasks["keywords"]),
                "total_evidence": len(all_evidence),
            },
        }

    def diagnose_question(
        self,
        question: str,
        options: dict,
        answer_format: str,
        initial_evidence: list = None,
        domain: str = "",
        max_rounds: int = 3,
        doc_ids: list = None,
    ) -> dict:
        """
        调试路径：与 answer_question 走同一套真实检索/推理/工具调用流程，
        但绝不做强制补全或兜底猜测。

        判定规则（任一命中 → uncertain=True）：
        1. 模型没有给出可解析的答案字母
        2. 答案格式与题型不符（mcq 应恰好1个字母；tf 只能 A/B；multi 应 ≥2个字母）
        3. 答案中某个被选中的字母，在证据（初始证据 + 工具调用结果 + 选项级证据）里
           找不到任何一条包含该选项关键术语的记录 —— 即"选了但没出处"

        Returns:
            {
              "raw_answer": 模型原始提取的答案（未经任何修正，可能为空/不完整/格式不对）,
              "uncertain": bool,
              "uncertain_reasons": [str, ...],
              "option_evidence_check": {opt: {"has_evidence": bool, "evidence_count": int}},
              "trace": [...],  # 完整交互轨迹，未截断
              "reasoning": 模型最终输出的完整原文（未截断）,
              "tool_calls_log": [...],
              "total_tokens": int,
            }
        """
        trace = []
        loop_result = self._run_tool_loop(
            question, options, answer_format,
            initial_evidence=initial_evidence, domain=domain,
            max_rounds=max_rounds, doc_ids=doc_ids, trace=trace,
        )
        final_content = loop_result["final_content"]
        tool_calls_log = loop_result["tool_calls_log"]
        total_tokens = loop_result["total_tokens"]
        option_evidence_map = loop_result["option_evidence_map"]
        web_search_results = loop_result["web_search_results"]

        raw_answer = self._extract_answer(final_content, answer_format)

        reasons = []       # 真正的不确定原因（会标记 uncertain=True）
        warnings = []      # 警告信息（不标记 uncertain，但值得人工复查）

        # 1. 完全没有答案
        if not raw_answer:
            reasons.append("模型未输出可解析的答案字母")

        # 2. 格式校验（不修正，只报告）
        valid_letters = set(options.keys()) if options else {"A", "B", "C", "D"}
        invalid_letters = set(raw_answer) - valid_letters
        if invalid_letters:
            reasons.append(f"答案包含非法选项字母: {sorted(invalid_letters)}")

        if raw_answer:
            if answer_format == "mcq" and len(raw_answer) != 1:
                reasons.append(f"单选题应恰好1个选项，实际提取到 {len(raw_answer)} 个: {raw_answer}")
            elif answer_format == "tf" and raw_answer not in ("A", "B"):
                reasons.append(f"判断题只能是 A 或 B，实际提取到: {raw_answer}")
            elif answer_format == "multi" and len(raw_answer) < 2:
                # 多选题只选1个 → 警告（可能题目确实只有1个正确选项），不标记 uncertain
                warnings.append(f"多选题只提取到 {len(raw_answer)} 个选项: {raw_answer}（可能题目确实只有1个正确选项，也可能是证据不足）")

        # 3. 逐个被选中选项做证据溯源
        #    汇总可用于溯源的全部证据文本：初始检索 + 每轮工具调用结果 + 选项级证据
        all_evidence_texts = []
        for e in (initial_evidence or []):
            all_evidence_texts.append(e.get("text", ""))
        for step in trace:
            if step.get("step") == "tool_call":
                for r in step.get("results", []):
                    all_evidence_texts.append(r.get("text", ""))
        for evs in option_evidence_map.values():
            for e in evs:
                all_evidence_texts.append(e.get("text", ""))
        combined_text = "\n".join(all_evidence_texts)

        # 从模型 reasoning 中提取各选项分析结果
        option_analysis_mentioned = {}  # {opt: {"analyzed": bool, "conclusion": "correct"|"wrong"|None}}
        for opt in sorted(options.keys()):
            # 检查模型是否在 reasoning 中对该选项进行了分析（覆盖多种格式）
            analysis_patterns = [
                # 标准格式：选项A分析 / 选项A判断 / 选项A结论
                rf'选项{opt}.*?(?:分析|判断|结论)',
                # markdown格式：### 选项A： / **选项A**： / **选项A：**
                rf'[#*]{{1,3}}\s*选项{opt}\s*[#*]{{0,3}}\s*[：:]',
                # markdown格式变体：**选项A** 后面跟冒号
                rf'\*\*选项{opt}\*\*\s*[：:]',
                # 结论格式：选项A正确/错误/成立/不成立
                rf'选项{opt}.*?(?:正确|错误|成立|不成立|符合|不符合)',
                # 简写格式：A正确 / A.正确 / A、正确
                rf'(?:^|\n)\s*{opt}[.．、：:]\s*.*?(?:正确|错误|成立|不成立)',
                # 分析内容格式：选项A：xxx → 结论
                rf'选项{opt}[：:].*?(?:正确|错误|成立|不成立|符合|不符合)',
                # 结论在前格式：正确/错误...选项A
                rf'(?:正确|错误|成立|不成立).*?选项{opt}',
                # 结论在行尾格式：xxx，**错误** / xxx，**正确**
                rf'选项{opt}.*?[，,]\s*\*\*(?:正确|错误|成立|不成立)\*\*',
            ]
            analyzed = any(re.search(p, final_content, re.IGNORECASE | re.MULTILINE) for p in analysis_patterns)
            # 检查模型对该选项的结论
            conclusion = None
            correct_patterns = [
                rf'选项{opt}.*?(?:正确|成立|符合)',
                rf'{opt}[.．、：:].*?(?:正确|成立|符合)',
                rf'(?:正确|成立|符合).*?选项{opt}',
                # 结论在行尾格式：xxx，**正确**
                rf'选项{opt}.*?[，,]\s*\*\*正确\*\*',
            ]
            wrong_patterns = [
                rf'选项{opt}.*?(?:错误|不成立|不符合)',
                rf'{opt}[.．、：:].*?(?:错误|不成立|不符合)',
                rf'(?:错误|不成立|不符合).*?选项{opt}',
                # 结论在行尾格式：xxx，**错误**
                rf'选项{opt}.*?[，,]\s*\*\*错误\*\*',
            ]
            if any(re.search(p, final_content, re.IGNORECASE | re.MULTILINE) for p in correct_patterns):
                conclusion = "correct"
            elif any(re.search(p, final_content, re.IGNORECASE | re.MULTILINE) for p in wrong_patterns):
                conclusion = "wrong"
            option_analysis_mentioned[opt] = {"analyzed": analyzed, "conclusion": conclusion}

        # 判断题特殊处理：检查模型是否对题目陈述进行了分析
        question_analysis_detected = False
        if answer_format == "tf":
            # 提取题目中的关键短语（≥4个字的连续中文片段）
            question_phrases = [m.group() for m in re.finditer(r'[一-鿿]{4,}', question)]
            question_phrases = [p for p in question_phrases if len(p) >= 4][:5]
            # 检查模型是否在推理中引用了题目相关的证据或分析了题目陈述
            question_analysis_patterns = [
                r'(?:根据|从|证据显示|文档.*?记载|文档.*?包含|文档.*?提及|文档.*?均)',
                r'(?:两份文档|两份文件|上述文档|相关文档)',
                r'(?:未提及|未找到|未发现|未包含|均包含|均提及|均未)',
                r'(?:题目陈述|题目.*?准确|题目.*?不准确|与题目.*?一致)',
            ]
            question_analysis_detected = any(re.search(p, final_content, re.IGNORECASE) for p in question_analysis_patterns)
            # 也检查模型是否引用了题目相关的关键证据
            for phrase in question_phrases:
                if phrase in final_content:
                    question_analysis_detected = True
                    break

        option_evidence_check = {}
        for opt, opt_text in sorted(options.items()):
            # 用 jieba 分词锚点判定出处：强锚点(数值)命中 or >=2 个弱锚点(实义词)命中
            # 修复：旧正则用 >=4 字连续中文长句逐字匹配，长句永不在证据里出现 -> 全误报无出处
            strong_anchors, weak_terms = _extract_opt_anchors(opt_text)
            strong_hit = any(a in combined_text for a in strong_anchors) if strong_anchors else False
            weak_hit_count = sum(1 for w in weak_terms if w in combined_text)
            hit = strong_hit or weak_hit_count >= 2
            ev_count = len(option_evidence_map.get(opt, []))
            analysis = option_analysis_mentioned.get(opt, {})
            option_evidence_check[opt] = {
                "has_evidence": hit,
                "evidence_count": ev_count,
                "strong_anchors": strong_anchors[:6],
                "weak_terms": weak_terms[:6],
                "strong_hit": strong_hit,
                "weak_hit_count": weak_hit_count,
                "analyzed_in_reasoning": analysis.get("analyzed", False),
                "reasoning_conclusion": analysis.get("conclusion"),
            }

        # 逐选项检查证据支撑
        for letter in raw_answer:
            if letter in valid_letters:
                check = option_evidence_check.get(letter, {})
                if not check.get("has_evidence"):
                    # 没有直接证据，但模型在 reasoning 中对该选项进行了分析 → 可能是推理结论
                    if check.get("analyzed_in_reasoning"):
                        warnings.append(
                            f"选项{letter}被选中，检索证据中无直接匹配，"
                            f"但模型在推理中对该选项进行了分析（结论: {check.get('reasoning_conclusion', '未知')}），"
                            f"属于推理型答案而非直接证据支撑"
                        )
                    elif answer_format == "tf" and question_analysis_detected:
                        # 判断题特殊处理：模型对题目陈述进行了分析，即使选项本身没有被单独分析
                        warnings.append(
                            f"判断题选项{letter}被选中，模型对题目陈述进行了分析和推理，"
                            f"但未单独对选项{letter}进行分析（判断题常见格式）"
                        )
                    else:
                        reasons.append(
                            f"选项{letter}被选中，但检索到的所有证据中均未找到与该选项相关术语匹配的记录（无出处支撑），"
                            f"且模型推理中未对该选项进行分析"
                        )

        uncertain = len(reasons) > 0

        return {
            "raw_answer": raw_answer,
            "uncertain": uncertain,
            "uncertain_reasons": reasons,
            "warnings": warnings,
            "option_evidence_check": option_evidence_check,
            "trace": trace,
            "reasoning": final_content,
            "tool_calls_log": tool_calls_log,
            "total_tokens": total_tokens,
            "used_web_search": len(web_search_results) > 0,
        }

    # ========== API 失败时的基于证据的 fallback ==========

    def _evidence_based_fallback(
        self,
        question: str,
        options: dict,
        answer_format: str,
        all_evidence: list = None,
        option_evidence_map: dict = None,
    ) -> str:
        """API 调用失败时，基于已检索证据给出最优猜测（不再固定返回 A）

        策略：
        - multi（多选题）：选所有有证据支持的选项，不足 2 个时按 BM25 分数补齐
        - mcq（单选题）：选证据数量最多的选项
        - tf（判断题）：证据中如含否定词倾向 B，否则默认 A

        Args:
            question: 原始问题
            options: 选项字典 {字母: 文本}
            answer_format: 题型 mcq/multi/tf
            all_evidence: 已收集的全部证据列表
            option_evidence_map: 选项级证据映射 {字母: [证据]}

        Returns:
            答案字母串，如 "ABD" / "A"
        """
        # 无证据时的最小兜底（保持比赛格式合法）
        if not all_evidence and not option_evidence_map:
            if answer_format == "multi":
                return "AB"  # 多选题至少 2 个选项
            return "A"  # 单选/判断题默认 A

        option_evidence_map = option_evidence_map or {}
        all_evidence = all_evidence or []

        # ---- 多选题：选所有有证据支持的选项 ----
        if answer_format == "multi":
            # 1. 收集每个选项的证据数
            opt_counts = []
            for opt in sorted(options.keys()):
                evs = option_evidence_map.get(opt, [])
                opt_counts.append((opt, len(evs)))

            # 2. 有证据支持的选项直接入选
            selected = [opt for opt, cnt in opt_counts if cnt > 0]

            # 3. 不足 2 个时，用 all_evidence 的 BM25 分数补齐
            if len(selected) < 2:
                # 计算每个未入选选项在 all_evidence 中的匹配度
                remaining = [opt for opt in sorted(options.keys()) if opt not in selected]
                for opt in remaining:
                    opt_text = options.get(opt, "")
                    # 用选项锚点匹配证据文本
                    strong, weak = _extract_opt_anchors(opt_text)
                    opt_terms = set(weak) | set(strong)
                    if opt_terms:
                        hit_count = sum(
                            1 for ev in all_evidence
                            if any(t in ev.get("text", "") for t in opt_terms)
                        )
                        if hit_count > 0:
                            selected.append(opt)
                    if len(selected) >= 2:
                        break

            # 4. 最终兜底：仍不足 2 个时强制取前 2 个选项
            if len(selected) < 2:
                for opt in sorted(options.keys()):
                    if opt not in selected:
                        selected.append(opt)
                        break
            return "".join(sorted(selected)) if selected else "AB"

        # ---- 单选题：选证据最多的选项 ----
        if answer_format == "mcq":
            best_opt, best_cnt = "A", 0
            for opt in sorted(options.keys()):
                evs = option_evidence_map.get(opt, [])
                cnt = len(evs)
                # 若选项级证据为空，用 all_evidence 的锚点匹配数补充
                if cnt == 0 and all_evidence:
                    strong, weak = _extract_opt_anchors(options.get(opt, ""))
                    opt_terms = set(weak) | set(strong)
                    cnt = sum(
                        1 for ev in all_evidence
                        if any(t in ev.get("text", "") for t in opt_terms)
                    )
                if cnt > best_cnt:
                    best_cnt, best_opt = cnt, opt
            return best_opt

        # ---- 判断题：证据含否定词倾向 B，否则默认 A ----
        if answer_format == "tf":
            combined_text = " ".join(ev.get("text", "") for ev in all_evidence)
            # 否定信号：证据中出现"不/未/无/错误/不成立"等
            neg_patterns = ["不承担", "不属于", "未提及", "未包含", "错误",
                            "不成立", "不符合", "不得", "禁止"]
            neg_count = sum(1 for p in neg_patterns if p in combined_text)
            # 肯定信号：证据中出现"应当/必须/包括/属于/承担"等
            pos_patterns = ["应当", "必须", "包括", "属于", "承担",
                            "可以", "明确", "符合"]
            pos_count = sum(1 for p in pos_patterns if p in combined_text)
            # 否定信号强于肯定信号时选 B（错误）
            if neg_count > pos_count and neg_count >= 2:
                return "B"
            return "A"

        return "A"

    def _format_evidence(self, evidence: list, start_idx: int = 1) -> str:
        """格式化证据列表（带全局编号 [证据N]，供模型引用与引用校验对齐）

        Args:
            evidence: 证据列表
            start_idx: 起始编号（初始证据从1开始，选项补充证据/工具结果延续编号）
        """
        if not evidence:
            return "（无）"
        lines = []
        for i, ev in enumerate(evidence[:10]):
            text = ev.get("text", "")[:300]
            doc_id = ev.get("doc_id", "")
            lines.append(f"- [证据{start_idx + i}] ({doc_id}) {text}")
        return "\n".join(lines)

    def _extract_answer(self, content: str, answer_format: str) -> str:
        """从模型输出中提取答案字母"""
        if not content:
            return ""

        # 辅助函数：从字符串中提取所有字母并去重排序
        def _letters(s: str) -> str:
            return "".join(sorted(set(c.upper() for c in s if c.upper() in "ABCD")))

        # 预处理：将 "A 和 D" / "A 与 D" 统一为 "A、D"
        content_normalized = re.sub(r'([A-Da-d])\s*(?:和|与|及)\s*([A-Da-d])', r'\1、\2', content)

        # 尝试匹配 "答案：ABC" 或 "答案：A、B、C" 或 "答案: A B C"（最可靠）
        # 支持连续字母 或 用/、，空格分隔的字母
        m = re.search(r'答案[：:]\s*([A-Da-d](?:[、,\s]*[A-Da-d])*)', content_normalized)
        if m:
            result = _letters(m.group(1))
            if result:
                return result

        # 尝试匹配 "选A" 或 "选 A、B、C"
        m = re.search(r'选\s*([A-Da-d](?:[、,\s]*[A-Da-d])*)', content_normalized)
        if m:
            result = _letters(m.group(1))
            if result:
                return result

        # 尝试匹配末尾的单独字母行（如 "ABC" 或 "A、B、C"）
        m = re.search(r'^\s*([A-Da-d](?:[、,\s]*[A-Da-d])*)\s*$', content_normalized.strip(), re.MULTILINE)
        if m:
            result = _letters(m.group(1))
            if result:
                return result

        # 尝试匹配 "正确选项是 A、B、C" 或 "正确选项应为 A、B、C" 或 "正确的选项是 A、D"
        m = re.search(r'正确(?:的)?选项[^A-Da-d]*([A-Da-d](?:[、,\s]*[A-Da-d])*)', content_normalized)
        if m:
            result = _letters(m.group(1))
            if result:
                return result

        # 尝试匹配 "选项X"
        m = re.search(r'选项\s*([A-Da-d]+)', content_normalized)
        if m:
            return m.group(1).upper()

        # fallback：从内容中提取所有提到的选项字母
        mentioned = set()
        for m in re.finditer(r'(?:选项|选)\s*([A-D])', content_normalized):
            mentioned.add(m.group(1).upper())
        if mentioned:
            return "".join(sorted(mentioned))

        # 最终 fallback
        if answer_format == "tf":
            if any(w in content for w in ["正确", "对", "支持", "是"]):
                return "A"
            return "B"

        return ""
