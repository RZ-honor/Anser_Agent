"""
全新检索系统 - 借鉴Elasticsearch和BM25的设计理念

设计原则：
1. 多字段索引：为不同类型的字段建立不同的索引
2. BM25参数优化：调整k1和b参数
3. 数值匹配优化：支持带逗号的数值匹配
4. 关键词匹配优化：支持精确短语匹配
5. 领域分离：每个领域有独立的索引
"""
import os
import re
import json
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Tuple, Optional

import jieba
from rank_bm25 import BM25Okapi

from config import VECTOR_RECALL_ENABLED, VECTOR_FALLBACK_TOP_K
import finance_aliases


class NewRetrievalSystem:
    """全新检索系统

    Args:
        domain: 领域名
        enable_numeric: 是否启用数值匹配加分
        enable_phrase: 是否启用短语匹配加分
        enable_entity: 是否启用实体匹配加分
    """

    USER_DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "jieba_userdict.txt")
    SYNONYM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "domain_synonyms.json")
    _GLOBAL_TOKENIZER_LOADED = False
    _GLOBAL_SYNONYMS = {}

    def __init__(self, domain: str = None,
                 enable_numeric: bool = True,
                 enable_phrase: bool = True,
                 enable_entity: bool = True):
        self.domain = domain
        self.chunk_data = {}  # chunk_id -> {text, doc_id, has_table, ...}
        self.bm25 = None
        self.chunk_ids = []
        self.chunk_texts = []

        # boost 开关
        self.enable_numeric = enable_numeric
        self.enable_phrase = enable_phrase
        self.enable_entity = enable_entity

        # 多字段索引
        self.numeric_index = defaultdict(set)  # 数值 -> {chunk_id, ...}
        self.phrase_index = defaultdict(set)  # 短语 -> {chunk_id, ...}
        self.entity_index = defaultdict(set)  # 实体 -> {chunk_id, ...}

        # 术语别名映射（从 domain_knowledge.json 加载，用于查询扩展）
        self.alias_map = {}  # alias -> full_name

        # 停用词
        self.stopwords = {
            "的", "了", "在", "是", "有", "和", "与", "或", "及", "其中",
            "一份", "第二份", "文档", "文件", "以下", "下列", "描述", "说法",
            "选项", "关于", "涉及", "包含", "明确", "公司", "规定", "应当",
        }

        # 加载术语别名（必须在 stopwords 初始化之后，因为 _load_term_aliases 会引用它）
        self._load_term_aliases()

    # 跨文档通用词黑名单：这些别名在不同文档指代不同主体，扩展会引入错误
    _GENERIC_ALIAS_BLACKLIST = {
        "发行人", "本公司", "本集团", "公司", "上市公司", "发行主体",
        "本次债券", "本次发行", "本次公司债券", "本期债券", "本期发行", "本期公司债券",
        "报告期", "报告期内", "报告期各期末", "最近三年及一期", "最近三年",
        "释义项", "释义内容", "重大事项", "基本情况", "注",
        "债务人", "债权人", "甲方", "乙方", "丙方", "丁方",
        "标的", "标的资产", "标的股权", "标的股份",
        "审计机构", "会计师事务所", "律师", "评级机构", "主承销商",
        "联席主承销商", "簿记管理人", "受托管理人", "债券持有人", "投资人",
    }

    def _load_term_aliases(self):
        """从 cache/domain_knowledge.json 加载术语别名，构建 alias -> full_name 映射

        用于检索查询扩展：若问题/选项命中别名，把对应全称加入查询，
        解决"术语映射不稳定"导致的召回失败（如问题用简称、文档用全称）。

        质量控制：
        - 同一别名若映射到多个不同全称（冲突），丢弃（歧义大）
        - 通用词（发行人/本公司等跨文档通用别名）不参与扩展
        - 全称过长（>40字）视为误解析，丢弃
        加载失败或文件缺失时静默降级，不影响主检索流程。
        """
        if not self.domain:
            return
        candidates = [
            "cache/domain_knowledge.json",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "domain_knowledge.json"),
        ]
        dk_path = None
        for p in candidates:
            if os.path.isfile(p):
                dk_path = p
                break
        if not dk_path:
            return
        try:
            with open(dk_path, encoding="utf-8") as f:
                dk = json.load(f)
            domain_data = dk.get(self.domain, {})
            term_aliases = domain_data.get("term_aliases", {})

            # 第一遍：收集 alias -> set(full_name)，统计冲突
            alias_to_fulls = defaultdict(set)
            for aliases_key, info in term_aliases.items():
                aliases = re.split(r'[、，,]', aliases_key)
                raw = info.get("raw", "") if isinstance(info, dict) else ""
                full_name = ""
                if "指" in raw:
                    cols = [c.strip() for c in raw.split("|") if c.strip() and c.strip() != "指"]
                    if len(cols) >= 2:
                        full_name = cols[-1]
                if not full_name:
                    continue
                # 全称过长视为误解析（如审计段落被误抽为全称）
                if len(full_name) > 40:
                    continue
                for a in aliases:
                    a = a.strip()
                    if 3 <= len(a) <= 12 and a != full_name:
                        alias_to_fulls[a].add(full_name)

            # 第二遍：只保留无冲突、非通用词的别名
            for alias, fulls in alias_to_fulls.items():
                if alias in self._GENERIC_ALIAS_BLACKLIST:
                    continue
                if alias in self.stopwords:
                    continue
                if len(fulls) != 1:
                    # 冲突：同一别名对应多个全称，跳过
                    continue
                self.alias_map[alias] = next(iter(fulls))
        except Exception:
            self.alias_map = {}

    def build_from_graph(self, G, n_workers: int = 8):
        """从图构建索引（BM25 分词层多线程加速）"""
        print(f"构建 {self.domain} 领域索引...")

        # 收集chunk数据（顺序）
        chunks = []
        for node, attrs in G.nodes(data=True):
            if attrs.get("type") != "chunk":
                continue
            if self.domain and attrs.get("domain") != self.domain:
                continue

            chunk_id = node
            text = attrs.get("text", "")
            doc_id = attrs.get("doc_id", "")
            has_table = attrs.get("has_table", False)

            self.chunk_data[chunk_id] = {
                "text": text,
                "doc_id": doc_id,
                "has_table": has_table,
                "page": attrs.get("page", 0),
                "page_start": attrs.get("page_start", attrs.get("page", 0)),
                "page_end": attrs.get("page_end", attrs.get("page", 0)),
                "chunk_type": attrs.get("chunk_type", "table" if has_table else "text"),
                "section_title": attrs.get("section_title", ""),
                "table_context_before": attrs.get("table_context_before", ""),
            }

            chunks.append((chunk_id, text, attrs))

        # 顺序构建数值/短语/实体索引
        for chunk_id, text, attrs in chunks:
            self._build_numeric_index(chunk_id, text)
            self._build_phrase_index(chunk_id, text)
            self._build_entity_index(chunk_id, attrs)

        # 多线程构建BM25索引（分词是瓶颈）
        plain_chunks = [(cid, text) for cid, text, _ in chunks]
        self._build_bm25_index(plain_chunks, n_workers=n_workers)

        print(f"  索引构建完成: {len(chunks)} chunks, "
              f"{len(self.numeric_index)} 数值, "
              f"{len(self.phrase_index)} 短语, "
              f"{len(self.entity_index)} 实体")

    def _build_numeric_index(self, chunk_id: str, text: str):
        """构建数值索引（双形式索引：原文 + 归一化）

        修复点（P5）：
        - 原实现只索引归一化数字（如"10亿元"→"1000000000"），但文档原文通常保留"10亿元"形式
          归一化数字在 numeric_index 中极少精确匹配，实际命中率很低
        - 现在同时索引：
          1. 原文形式（带单位）：如"10亿元"、"43.24%"、"150万元"
          2. 归一化形式：如"1000000000"、"43.24"、"1500000"
        - 查询时优先匹配原文形式（更精准），归一化形式作为兜底

        修复点（P6 - LaTeX转义预处理）：
        - 文档中百分比常以LaTeX转义形式存储：\(160\%\)
        - 原正则 r'\d+(?:\.\d+)?\s*%' 无法匹配带反斜杠的 \(160\%\)
        - 现在构建索引前先预处理文本，去除LaTeX转义符
        """
        # 预处理：去除LaTeX转义符，确保百分比和数值能被正则匹配
        # \(160\%\) → 160%，\(\d+\) → \d+
        text_clean = text.replace('\\(', '').replace('\\)', '')
        text_clean = text_clean.replace('\\%', '%').replace('\\,', '')
        # 去除逗号后提取纯数字（保留原有逻辑）
        text_no_comma = text_clean.replace(",", "")
        nums = re.findall(r'\d+(?:\.\d+)?', text_no_comma)
        for num in nums:
            self.numeric_index[num].add(chunk_id)

        # === 新增：索引原文形式（带单位）===
        # 匹配"10亿元"、"43.24%"、"150万元"等，保留原文便于精确匹配
        unit_patterns_raw = [
            r'\d+(?:\.\d+)?\s*亿元',
            r'\d+(?:\.\d+)?\s*万元',
            r'\d+(?:\.\d+)?\s*百万元',
            r'\d+(?:\.\d+)?\s*元',
            r'\d+(?:\.\d+)?\s*%',
            r'\d+(?:\.\d+)?\s*亿美元',
        ]
        for pattern in unit_patterns_raw:
            for m in re.finditer(pattern, text_no_comma):
                # 去除空格后索引原文（如"10亿元"而非"10 亿元"）
                raw_form = re.sub(r'\s+', '', m.group())
                self.numeric_index[raw_form].add(chunk_id)

        # === 归一化形式（兜底） ===
        unit_patterns_norm = [
            (r'(\d+(?:\.\d+)?)\s*亿元', 100000000),
            (r'(\d+(?:\.\d+)?)\s*万元', 10000),
            (r'(\d+(?:\.\d+)?)\s*百万元', 1000000),
            (r'(\d+(?:\.\d+)?)\s*元', 1),
        ]
        for pattern, multiplier in unit_patterns_norm:
            for m in re.finditer(pattern, text_no_comma):
                val = str(float(m.group(1)) * multiplier)
                self.numeric_index[val].add(chunk_id)

        # 提取百分比（保留原文 + 数值）
        for m in re.finditer(r'(\d+(?:\.\d+)?)\s*%', text_no_comma):
            self.numeric_index[m.group(1)].add(chunk_id)
            self.numeric_index[m.group(1) + "%"].add(chunk_id)

        # 提取带修饰词的数值（约、超过、超、达到等）
        modifier_patterns = [
            r'约\s*(\d+(?:\.\d+)?)',
            r'超过\s*(\d+(?:\.\d+)?)',
            r'超\s*(\d+(?:\.\d+)?)',
            r'达到\s*(\d+(?:\.\d+)?)',
        ]
        for pattern in modifier_patterns:
            for m in re.finditer(pattern, text_no_comma):
                self.numeric_index[m.group(1)].add(chunk_id)

    def _build_phrase_index(self, chunk_id: str, text: str):
        """构建短语索引（P0 修复：上限从 10 字扩展到 20 字，与查询端对齐）

        修复原因：
        - 索引端原为 {2,10}（10 字上限）
        - 查询端为 {3,20}（20 字上限）
        - 长公司名（如"中国远洋海运集团有限公司"12 字）在索引端无法整体索引
        - 现统一为 {2,20}，确保长术语能作为整体被召回

        P1 增强：同时索引条款号原文（第X条/章/节/款/项），解决 clause 类问题锚点召回失败
        """
        # 提取 2-20 字的连续中文短语（上限与查询端 _add_phrase_boost 的 {3,20} 对齐）
        for m in re.finditer(r'[一-鿿]{2,20}', text):
            phrase = m.group()
            if phrase not in self.stopwords:
                self.phrase_index[phrase].add(chunk_id)

        # P1 新增：索引条款号原文（第X条/章/节/款/项）
        # 解决 clause 类问题（如"第五条主要涉及..."）的锚点召回失败
        for m in re.finditer(r'第[一二三四五六七八九十百千零\d]+\s*[条章节款项号]', text):
            # 去除空格后索引原文（如"第五条"而非"第 五 条"）
            clause_phrase = re.sub(r'\s+', '', m.group())
            self.phrase_index[clause_phrase].add(chunk_id)

    def _build_entity_index(self, chunk_id: str, attrs: dict):
        """构建实体索引"""
        entities = attrs.get("entities", [])
        for ent_type, ent_name in entities:
            self.entity_index[ent_name].add(chunk_id)

    def _build_bm25_index(self, chunks: List[Tuple[str, str]], n_workers: int = 8):
        """构建BM25索引（多线程分词）"""
        self.chunk_ids = []
        self.chunk_texts = []
        self._ensure_tokenizer_loaded()

        # 多线程分词
        results = [None] * len(chunks)

        def tokenize_one(args):
            idx, chunk_id, text = args
            tokens = self._tokenize(text)
            return idx, chunk_id, tokens

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i, (chunk_id, text) in enumerate(chunks):
                futures.append(executor.submit(tokenize_one, (i, chunk_id, text)))

            for future in as_completed(futures):
                idx, chunk_id, tokens = future.result()
                if tokens:
                    results[idx] = (chunk_id, tokens)

        for r in results:
            if r is not None:
                self.chunk_ids.append(r[0])
                self.chunk_texts.append(r[1])

        if self.chunk_texts:
            # 优化BM25参数：k1=1.8增强关键词匹配，b=0.6减少长文档惩罚
            self.bm25 = BM25Okapi(self.chunk_texts, k1=1.8, b=0.6)

    # 领域自定义词典（高频专有名词 + 复合词保护）
    CUSTOM_DICT = [
        # ===== 公司/机构后缀（必须作为原子词）=====
        "股份有限公司", "有限公司", "有限责任公司", "集团有限公司",
        "有限合伙人", "有限合伙",
        # ===== 银行 =====
        "招商银行", "中国银行", "工商银行", "建设银行", "交通银行",
        "兴业银行", "浦发银行", "平安银行", "厦门银行", "招商永隆银行",
        "商业银行", "中央银行", "上市银行",
        # ===== 证券 =====
        "国信证券", "西部证券", "东方证券", "中泰证券", "国盛证券",
        "华创证券", "中原证券", "证券交易所", "证券投资基金",
        "中国证券报", "上海证券报", "深圳证券交易所",
        # ===== 保险 =====
        "新华保险", "中国人寿", "中国平安", "平安人寿", "平安财产",
        "养老保险", "人寿保险", "财产保险", "医疗保险", "工伤保险",
        "生育保险", "失业保险", "社会保险", "商业养老保险",
        "平安智盈金生", "专属商业养老保险",
        "保险责任", "保险期间", "保险事故", "保险合同", "保险费", "保险金",
        "被保险人", "投保人", "受益人", "保险金申请人",
        "责任免除", "现金价值", "犹豫期", "等待期",
        "不承担赔偿责任", "不承担保险责任", "不承担给付保险金责任",
        "偿付能力", "充足率", "核心偿付能力充足率", "综合偿付能力充足率",
        # ===== 财务/会计 =====
        "负债合计", "资产合计", "营业收入", "营业利润", "净利润",
        "净资产", "总资产", "应收账款", "坏账准备",
        "账面余额", "账面价值", "年末余额", "年初余额",
        "研发投入", "研发费用", "研发工程师", "同比上升", "同比下降",
        "手机部件", "汽车业务", "二次充电电池", "市场份额",
        "工业软件", "市场规模", "母公司拥有人", "应占溢利",
        "报告期内", "报告期各期末", "其他综合收益", "其他权益工具",
        "连带责任保证", "基础设施建设",
        # ===== 监管/法律 =====
        "上市公司", "高级管理人员", "实际控制人", "控股股东",
        "信息披露", "信息披露义务", "行政监管措施", "行政法规",
        "对外担保", "管理制度", "独立董事", "董事会", "会计师事务所",
        "行政处罚", "定期报告", "临时报告",
        "公司法", "证券法",
        # ===== 数量+单位（防止数字与单位分离）=====
        "4个月", "6个月", "12个月", "3个月", "1个月",
        "4年", "6年", "3年", "2年", "1年",
        # ===== 研报 =====
        "银保渠道", "个险渠道", "银行竞争", "核心驱动力",
        "自营品", "首款",
        # ===== 通用高频复合词 =====
        "标的公司", "控股子公司", "合营公司", "租赁项目公司",
        "归属于母公司", "归属于上市公司",
        "包括但不限于", "除另有约定外",
    ]

    def _ensure_tokenizer_loaded(self):
        """加载硬编码词典、离线 userdict 和同义词表。"""
        cls = type(self)
        if not cls._GLOBAL_TOKENIZER_LOADED:
            for word in self.CUSTOM_DICT:
                jieba.add_word(word, freq=99999)
            for y in range(2018, 2031):
                jieba.add_word(f"{y}年", freq=99999)
                jieba.add_word(f"{y}年末", freq=99999)
                jieba.add_word(f"{y}年上半年", freq=99999)
                jieba.add_word(f"{y}年下半年", freq=99999)
            for unit in ['个月', '万名', '亿元', '万元', '百万元']:
                for n in ['1', '2', '3', '4', '6', '10', '12']:
                    jieba.add_word(f"{n}{unit}", freq=99999)
            if os.path.isfile(cls.USER_DICT_PATH):
                try:
                    with open(cls.USER_DICT_PATH, encoding="utf-8") as f:
                        jieba.load_userdict(f)
                except Exception:
                    pass
            cls._GLOBAL_TOKENIZER_LOADED = True

        if not cls._GLOBAL_SYNONYMS and os.path.isfile(cls.SYNONYM_PATH):
            try:
                with open(cls.SYNONYM_PATH, encoding="utf-8") as f:
                    data = json.load(f)
                cls._GLOBAL_SYNONYMS = {
                    str(k): [str(x) for x in v[:8]]
                    for k, v in data.items()
                    if isinstance(v, list)
                }
            except Exception:
                cls._GLOBAL_SYNONYMS = {}

    def _expand_query_tokens(self, full_text: str, tokens: List[str]) -> List[str]:
        """基于释义别名、离线同义词、金融别名表扩展查询 token。"""
        expanded = list(tokens)
        extras = []
        for alias, full in self.alias_map.items():
            if alias in full_text:
                extras.append(full)
        for term, syns in type(self)._GLOBAL_SYNONYMS.items():
            if term in full_text:
                extras.extend(syns)
        # 金融别名表（股票简称/证券代码/指标口语 -> 全称/术语），见 finance_aliases.py
        extras.extend(finance_aliases.expand_all(full_text))
        seen = set(expanded)
        for extra in extras[:12]:
            if extra and extra not in seen:
                expanded.append(extra)
                seen.add(extra)
            for t in self._tokenize(extra):
                if t not in seen:
                    expanded.append(t)
                    seen.add(t)
        return expanded

    def _build_query_tokens(self, question: str, options: dict) -> List[str]:
        """构建检索查询 token，保留原词并追加同义词/全称扩展。"""
        full_text = question + " " + " ".join(options.values())
        return self._expand_query_tokens(full_text, self._tokenize(full_text))

    def _tokenize(self, text: str) -> List[str]:
        """分词：预处理保护 + 自定义词典 + jieba + 2-gram 补充

        三层策略：
        1. 预处理：逗号数字归一化
        2. 自定义词典 + jieba：确保金融实体不被切碎
        3. 2-gram：补充连续中文 2-gram 防止长短语被切碎
        """
        self._ensure_tokenizer_loaded()

        # 去除表格标记
        text = re.sub(r'\[表格[^\]]*\]', '', text)
        text = re.sub(r'\[标题\]', '', text)

        # === 第一层：预处理 ===
        # 保护带逗号的数字：12,015,031.17 → 12015031.17
        text = re.sub(r'(\d{1,3}(?:,\d{3})+(?:\.\d+)?)', lambda m: m.group(1).replace(',', ''), text)

        words = list(jieba.cut(text))

        # === 过滤 ===
        tokens = []
        for w in words:
            w = w.strip()
            if not w:
                continue
            if w in self.stopwords:
                continue
            if re.match(r'^[\s\W]+$', w):
                continue
            tokens.append(w)

        # === 第三层：2-gram 补充 ===
        # 对连续中文 token 生成 2-gram，覆盖 jieba 未识别的长短语
        ngram_tokens = []
        cn_buf = []
        for t in tokens:
            if re.match(r'^[一-鿿]{2,}$', t):
                cn_buf.append(t)
            else:
                if len(cn_buf) >= 2:
                    for i in range(len(cn_buf) - 1):
                        ngram_tokens.append(cn_buf[i] + cn_buf[i + 1])
                cn_buf = []
        if len(cn_buf) >= 2:
            for i in range(len(cn_buf) - 1):
                ngram_tokens.append(cn_buf[i] + cn_buf[i + 1])

        tokens.extend(ngram_tokens)
        return tokens

    def search(self, question: str, options: dict, doc_ids: List[str] = None,
               top_k: int = 8, debug: bool = False) -> List[dict]:
        """搜索

        Args:
            debug: 若为 True，每条结果额外附带 score_breakdown 字段

        Returns:
            [{"chunk_id", "text", "score", "doc_id", "has_table", ...}]
        """
        if not self.bm25 or not self.chunk_ids:
            return []

        # 1. 确定搜索范围
        scope = self._get_scope(doc_ids)
        if not scope:
            return []

        # 2. BM25召回
        query_tokens = self._build_query_tokens(question, options)
        if not query_tokens:
            return []

        bm25_scores = self.bm25.get_scores(query_tokens)

        # 构建候选，记录各层分数
        # breakdown: {chunk_id: {"bm25": float, "numeric": float, "phrase": float, "entity": float}}
        breakdown = {}
        candidates = {}
        for i, chunk_id in enumerate(self.chunk_ids):
            if chunk_id in scope:
                s = float(bm25_scores[i])
                candidates[chunk_id] = s
                breakdown[chunk_id] = {"bm25": s, "numeric": 0.0, "phrase": 0.0, "entity": 0.0}

        # 3. 数值匹配加分
        if self.enable_numeric:
            self._add_numeric_boost(candidates, question, options, scope, breakdown)

        # 4. 短语匹配加分
        if self.enable_phrase:
            self._add_phrase_boost(candidates, question, options, scope, breakdown)

        # 5. 实体匹配加分
        if self.enable_entity:
            self._add_entity_boost(candidates, question, options, scope, breakdown)

        # 5.5 表格chunk基础分加成（修复表格chunk BM25天然劣势）
        # 问题：表格chunk文本短（如"第三年 | 2%"），BM25得分天然偏低
        # 解决：对has_table=True的chunk给予基础分加成，避免关键表格数据被挤出top_k
        TABLE_BOOST = 5.0  # 表格chunk基础加成分
        for chunk_id in list(candidates.keys()):
            data = self.chunk_data.get(chunk_id)
            if data and data.get("has_table", False):
                candidates[chunk_id] += TABLE_BOOST
                breakdown[chunk_id]["table_boost"] = TABLE_BOOST

        # 6. 排序返回（含多文档覆盖保证机制）
        # 多文档题：确保每个 doc_id 至少返回 3 条最高分证据
        # 解决 fc_debug_004 的 text03 受托管理人 chunk 未被召回问题
        # 优化：从 2 条提升到 3 条，进一步提升跨文档关键信息召回率
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        selected = []        # 最终选中的 chunk_id 列表
        selected_set = set() # 已选中的 chunk_id 集合
        doc_chunk_count = {}  # 每个 doc_id 已选中的 chunk 数

        # 第一轮：对每个 doc_id 至少取 3 条最高分证据
        # 当 doc_ids 有多个时启用覆盖逻辑，避免低分文档完全被排除
        if doc_ids and len(doc_ids) > 1:
            # 每个 doc_id 取 3 条，确保关键信息（如受托管理人、法条完整内容）有更高概率被召回
            chunks_per_doc = 3
            max_first_round = min(len(doc_ids) * chunks_per_doc, top_k)
            for chunk_id, score in sorted_candidates:
                if len(selected) >= max_first_round:
                    break
                if chunk_id in selected_set:
                    continue
                data = self.chunk_data.get(chunk_id)
                if not data:
                    continue
                doc_id = data["doc_id"]
                current_count = doc_chunk_count.get(doc_id, 0)
                if current_count < chunks_per_doc:
                    selected.append((chunk_id, score))
                    selected_set.add(chunk_id)
                    doc_chunk_count[doc_id] = current_count + 1

        # 第二轮：按分数填满剩余位置
        for chunk_id, score in sorted_candidates:
            if len(selected) >= top_k:
                break
            if chunk_id in selected_set:
                continue
            selected.append((chunk_id, score))
            selected_set.add(chunk_id)

        # 6.4 表格保底：选中结果中无表格 chunk 而候选中有表格时，
        # 用最高分表格候选替换末位（修复纯文本问题挤掉关键表格数据的场景）
        if selected and not any(
                self.chunk_data.get(cid, {}).get("has_table") for cid, _ in selected):
            for cand_id, cand_score in sorted_candidates:
                if cand_id in selected_set:
                    continue
                if self.chunk_data.get(cand_id, {}).get("has_table"):
                    replaced = selected.pop()
                    selected_set.discard(replaced[0])
                    selected.append((cand_id, cand_score))
                    selected_set.add(cand_id)
                    break

        # 6.5 向量兜底混合召回（改进.md §3.4）：语义最近邻中未入选的 chunk 补位末位
        # 失败静默降级（API 不可用/库未建），不影响符号检索主流程
        if VECTOR_RECALL_ENABLED and selected:
            try:
                from vector_store import get_vector_store
                store = get_vector_store()
                v_hits = store.query(question, domain=self.domain, doc_ids=doc_ids,
                                     top_k=VECTOR_FALLBACK_TOP_K)
                for h in v_hits:
                    cid = h.get("chunk_id", "")
                    if cid in selected_set or cid not in self.chunk_data:
                        continue
                    replaced = selected.pop()
                    selected_set.discard(replaced[0])
                    selected.append((cid, candidates.get(cid, 0.5)))
                    selected_set.add(cid)
                    break  # 每题只补 1 条，控制上下文膨胀
            except Exception:
                pass

        # 构建返回结果
        results = []
        for chunk_id, score in selected[:top_k]:
            if chunk_id in self.chunk_data:
                data = self.chunk_data[chunk_id]
                row = {
                    "chunk_id": chunk_id,
                    "text": data["text"],
                    "score": score,
                    "doc_id": data["doc_id"],
                    "has_table": data["has_table"],
                }
                if debug:
                    row["score_breakdown"] = breakdown.get(chunk_id, {})
                    row["query_tokens"] = query_tokens[:80]
                results.append(row)

        return results

    def _get_scope(self, doc_ids: List[str] = None) -> Set[str]:
        """获取搜索范围"""
        if doc_ids:
            scope = set()
            for doc_id in doc_ids:
                for chunk_id, data in self.chunk_data.items():
                    if data["doc_id"] == doc_id:
                        scope.add(chunk_id)
            return scope
        else:
            return set(self.chunk_data.keys())

    def _add_numeric_boost(self, candidates: dict, question: str, options: dict,
                           scope: set, breakdown: dict = None):
        """数值匹配加分（归一化 + 指标上下文联合匹配）

        设计原则：
        - 数值匹配必须伴随指标上下文关键词命中，否则不加分
        - 加分幅度与 BM25 主分可比（~0.5-2.0），不覆盖 BM25 排序
        - 问题中的数值权重高于选项中的数值
        """
        # 提取问题中的指标关键词（jieba 分词 ≥ 2 字）
        metric_keywords = set()
        for w in jieba.cut(question):
            w = w.strip()
            if len(w) >= 2 and w not in self.stopwords:
                metric_keywords.add(w)

        # 从问题中提取数值（高权重）
        q_nums = self._extract_nums_from_text(question)
        # 从选项中提取数值（低权重）
        o_nums = set()
        for opt_text in options.values():
            o_nums.update(self._extract_nums_from_text(opt_text))

        # 问题数值：需匹配指标上下文，boost = 2.5（优化：从1.5提升到2.5）
        for num in q_nums:
            chunk_ids = self.numeric_index.get(num, set())
            for chunk_id in chunk_ids:
                if chunk_id not in scope:
                    continue
                text = self.chunk_data[chunk_id]["text"]
                # 必须包含至少 1 个指标关键词
                metric_hits = sum(1 for kw in metric_keywords if kw in text)
                if metric_hits >= 1:
                    bonus = 2.5  # 优化：增强数值匹配权重
                    candidates[chunk_id] = candidates.get(chunk_id, 0) + bonus
                    if breakdown and chunk_id in breakdown:
                        breakdown[chunk_id]["numeric"] += bonus

        # 选项数值：需匹配指标上下文，boost = 1.0（优化：从0.5提升到1.0）
        for num in o_nums - q_nums:  # 避免重复加分
            chunk_ids = self.numeric_index.get(num, set())
            for chunk_id in chunk_ids:
                if chunk_id not in scope:
                    continue
                text = self.chunk_data[chunk_id]["text"]
                metric_hits = sum(1 for kw in metric_keywords if kw in text)
                if metric_hits >= 2:  # 选项数值要求更高
                    bonus = 1.0  # 优化：增强选项数值匹配
                    candidates[chunk_id] = candidates.get(chunk_id, 0) + bonus
                    if breakdown and chunk_id in breakdown:
                        breakdown[chunk_id]["numeric"] += bonus

    def _extract_nums_from_text(self, text: str) -> set:
        """从文本中提取数值集合（双形式：原文 + 归一化）

        修复点（P5）：
        - 与 _build_numeric_index 对齐，同时提取原文形式和归一化形式
        - 原文形式优先匹配（更精准），归一化形式作为兜底
        """
        nums = set()
        text_no_comma = text.replace(",", "")

        # 普通数字（纯数值）
        nums.update(re.findall(r'\d+(?:\.\d+)?', text_no_comma))

        # === 原文形式（带单位） ===
        # 与索引端保持一致，确保"10亿元"等原文能精确匹配
        raw_patterns = [
            r'\d+(?:\.\d+)?\s*亿元',
            r'\d+(?:\.\d+)?\s*万元',
            r'\d+(?:\.\d+)?\s*百万元',
            r'\d+(?:\.\d+)?\s*元',
            r'\d+(?:\.\d+)?\s*%',
            r'\d+(?:\.\d+)?\s*亿美元',
        ]
        for pattern in raw_patterns:
            for m in re.finditer(pattern, text_no_comma):
                raw_form = re.sub(r'\s+', '', m.group())
                nums.add(raw_form)

        # === 归一化形式（兜底） ===
        unit_patterns = [
            (r'(\d+(?:\.\d+)?)\s*亿元', 100000000),
            (r'(\d+(?:\.\d+)?)\s*万元', 10000),
            (r'(\d+(?:\.\d+)?)\s*百万元', 1000000),
            (r'(\d+(?:\.\d+)?)\s*元', 1),
        ]
        for pattern, multiplier in unit_patterns:
            for m in re.finditer(pattern, text_no_comma):
                nums.add(str(float(m.group(1)) * multiplier))

        # 百分比
        for m in re.finditer(r'(\d+(?:\.\d+)?)\s*%', text_no_comma):
            nums.add(m.group(1))

        # 带修饰词数值
        for pat in [r'约\s*(\d+(?:\.\d+)?)', r'超过\s*(\d+(?:\.\d+)?)',
                     r'超\s*(\d+(?:\.\d+)?)', r'达到\s*(\d+(?:\.\d+)?)']:
            for m in re.finditer(pat, text_no_comma):
                nums.add(m.group(1))

        return nums

    def _add_phrase_boost(self, candidates: dict, question: str, options: dict,
                          scope: set, breakdown: dict = None):
        """短语匹配加分（归一化版本）

        设计原则：
        - 只匹配 3+ 字的高价值短语，2 字词不单独加分
        - 加分幅度与 BM25 主分可比（~0.3-1.5）
        - 长短语权重略高，但有上限
        """
        # 提取问题中 3+ 字的精确短语
        exact_phrases = set()
        for m in re.finditer(r'[一-鿿]{3,10}', question):
            phrase = m.group()
            if phrase not in self.stopwords:
                exact_phrases.add(phrase)

        # 精确短语匹配加分（优化：提升基础权重）
        for phrase in exact_phrases:
            chunk_ids = self.phrase_index.get(phrase, set())
            for chunk_id in chunk_ids:
                if chunk_id in scope:
                    # 优化：3字=0.5, 4字=0.7, 5字+=0.9, 上限2.0（从1.5提升）
                    weight = min(0.15 * len(phrase) + 0.05, 2.0)
                    candidates[chunk_id] = candidates.get(chunk_id, 0) + weight
                    if breakdown and chunk_id in breakdown:
                        breakdown[chunk_id]["phrase"] += weight

    def _add_entity_boost(self, candidates: dict, question: str, options: dict,
                          scope: set, breakdown: dict = None):
        """实体匹配加分（归一化版本）

        设计原则：
        - 只匹配公司/机构类实体
        - 加分幅度 ~0.5-1.0，不覆盖 BM25 排序
        """
        full_text = question + " " + " ".join(options.values())
        entities = []
        for m in re.finditer(r'[一-鿿]{2,15}(?:股份|集团|科技|证券|银行|保险|控股)', full_text):
            entities.append(m.group())

        for entity in entities[:3]:
            chunk_ids = self.entity_index.get(entity, set())
            for chunk_id in chunk_ids:
                if chunk_id in scope:
                    bonus = 1.5  # 优化：从1.0提升到1.5
                    candidates[chunk_id] = candidates.get(chunk_id, 0) + bonus
                    if breakdown and chunk_id in breakdown:
                        breakdown[chunk_id]["entity"] += bonus


def build_domain_indexes(all_docs: dict,
                         enable_numeric: bool = True,
                         enable_phrase: bool = True,
                         enable_entity: bool = True) -> dict:
    """为每个领域构建索引（分词层多线程加速）"""
    from graph_builder import build_graph

    domain_systems = {}

    for domain_name, domain_docs in all_docs.items():
        if not domain_docs:
            continue

        print(f"\n构建 {domain_name} 领域索引...")

        # 构建该领域的图
        domain_all_docs = {domain_name: domain_docs}
        G = build_graph(domain_all_docs)

        # 构建检索系统（内部分词多线程）
        system = NewRetrievalSystem(
            domain=domain_name,
            enable_numeric=enable_numeric,
            enable_phrase=enable_phrase,
            enable_entity=enable_entity,
        )
        system.build_from_graph(G)

        domain_systems[domain_name] = system

    return domain_systems


if __name__ == "__main__":
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

    # 构建领域索引
    domain_systems = build_domain_indexes(docs)

    # 测试
    print("\n测试检索...")
    for domain, system in domain_systems.items():
        print(f"\n{domain}: {len(system.chunk_data)} chunks")
