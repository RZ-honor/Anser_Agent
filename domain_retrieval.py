"""
领域专属检索系统 - 为每个领域构建独立索引
从各自对应的数据文件加载，使用领域专属参数
"""
import os
import re
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import jieba
from rank_bm25 import BM25Okapi
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import List

from domain_config import get_domain_config, get_domain_custom_words, get_domain_bm25_params
from new_retrieval_system import NewRetrievalSystem


class DomainRetrievalSystem(NewRetrievalSystem):
    """领域专属检索系统，继承NewRetrievalSystem并使用领域专属参数

    修复点（P0+P2）：
    - 重写 _add_numeric_boost / _add_phrase_boost / _add_entity_boost
      使其使用 self._numeric_match_score 等领域配置值，而非父类硬编码 2.5/1.0/1.5
    - 加分项量级对齐 BM25 主分（10-25 量级），让领域差异化真正生效
    """

    def __init__(self, domain: str, data_dir: str = None):
        config = get_domain_config(domain)
        super().__init__(
            domain=domain,
            enable_numeric=True,
            enable_phrase=True,
            enable_entity=True,
        )
        self.domain_config = config
        self.data_dir = data_dir or r"D:\PROJECT\tianci\1"

        # 覆盖BM25参数为领域专属值
        self._bm25_k1 = config["bm25_k1"]
        self._bm25_b = config["bm25_b"]

        # 覆盖加分权重为领域专属值
        self._numeric_match_score = config["numeric_match_score"]
        self._rating_match_score = config["rating_match_score"]
        self._entity_match_score = config["entity_match_score"]
        self._keyword_coverage_score = config["keyword_coverage_score"]
        self._table_boost = config["table_boost"]
        self._table_numeric_boost = config["table_numeric_boost"]

        # 检索参数
        self._retrieve_top_k = config["retrieve_top_k"]
        self._retrieve_candidate_k = config["retrieve_candidate_k"]

        # 缓存领域专属条款关键词，供 _add_phrase_boost 加分使用
        self._domain_clause_keywords = set(config.get("clause_keywords", []))
        # 缓存领域定位短语，用于强匹配加分
        self._domain_doc_guarantee = set(config.get("doc_guarantee_phrases", []))
        # 缓存领域噪音标题，用于降权
        self._domain_noise_headings = set(config.get("noise_headings", []))
        # 缓存领域同义词扩展表（来自 domain_optimizers）
        self._domain_synonym_map = self._load_domain_synonyms(domain)

        # 添加领域专属词典
        for word in get_domain_custom_words(domain):
            jieba.add_word(word, freq=99999)

    def _load_domain_synonyms(self, domain: str) -> dict:
        """加载领域同义词映射（来自 domain_optimizers.py）

        将 domain_optimizers 中分散的同义词表合并为统一的 {trigger: [expansions]} 映射，
        用于 _expand_query_tokens 阶段扩展查询 token。

        Returns:
            {trigger_word: [synonym1, synonym2, ...]}
        """
        # 延迟导入避免循环依赖
        try:
            from domain_optimizers import (
                CONTRACT_KEYWORDS, REPORT_KEYWORDS, INSURANCE_KEYWORDS,
                REGULATORY_KEYWORDS, RESEARCH_KEYWORDS,
            )
        except ImportError:
            return {}

        synonym_map = {}
        if domain == "financial_contracts":
            # 合同条款类型同义词
            for trigger, syns in CONTRACT_KEYWORDS.get("clause_types", {}).items():
                synonym_map[trigger] = list(syns)
        elif domain == "financial_reports":
            # 财务指标同义词
            for trigger, syns in REPORT_KEYWORDS.get("metrics", {}).items():
                synonym_map[trigger] = list(syns)
        elif domain == "insurance":
            # 保险术语同义词（犹豫期→冷静期/撤销期 等）
            for trigger, syns in INSURANCE_KEYWORDS.get("term_synonyms", {}).items():
                synonym_map[trigger] = list(syns)
            # 保险产品类型同义词
            for trigger, syns in INSURANCE_KEYWORDS.get("product_types", {}).items():
                synonym_map[trigger] = list(syns)
        elif domain == "regulatory":
            # 监管领域术语同义词
            for trigger, syns in REGULATORY_KEYWORDS.get("regulatory_terms", {}).items():
                synonym_map[trigger] = list(syns)
        elif domain == "research":
            # 研报数据类型同义词
            for trigger, syns in RESEARCH_KEYWORDS.get("data_types", {}).items():
                synonym_map[trigger] = list(syns)
            # 研报行业同义词
            for trigger, syns in RESEARCH_KEYWORDS.get("industries", {}).items():
                synonym_map[trigger] = list(syns)
        return synonym_map

    def _expand_query_tokens(self, full_text: str, tokens: List[str]) -> List[str]:
        """重写查询扩展：在父类基础上追加领域同义词

        修复点（P3）：
        - 父类 _expand_query_tokens 只用了 alias_map（释义别名）和 _GLOBAL_SYNONYMS（全局同义词）
        - 这里追加 domain_optimizers.py 的领域同义词扩展（如保险"犹豫期"→"冷静期/撤销期"）
        - 解决术语映射不稳定导致的召回失败
        """
        # 先调用父类扩展（保留原有 alias_map 和全局同义词逻辑）
        expanded = super()._expand_query_tokens(full_text, tokens)

        # 追加领域同义词扩展
        extras = []
        for trigger, syns in self._domain_synonym_map.items():
            if trigger in full_text:
                extras.extend(syns)

        # 去重并追加（控制扩展数量避免过度膨胀）
        seen = set(expanded)
        for extra in extras[:15]:
            if not extra or extra in seen:
                continue
            expanded.append(extra)
            seen.add(extra)
            # 对长同义词进一步分词后追加
            for t in self._tokenize(extra):
                if t not in seen:
                    expanded.append(t)
                    seen.add(t)
        return expanded

    # ===== 重写三个加分方法：使用领域配置值 =====

    def _add_numeric_boost(self, candidates: dict, question: str, options: dict,
                           scope: set, breakdown: dict = None):
        """数值匹配加分（领域专属权重）

        与父类差异：
        - bonus 使用 self._numeric_match_score（15-25 量级）而非硬编码 2.5
        - 选项数值使用 self._numeric_match_score * 0.4（6-10 量级）而非 1.0
        - 数值权重足以撼动 BM25 主分（典型 60-240）的排序

        P1 优化：对 clause 类问题（如"第X条主要涉及..."）跳过 metric_hits >= 1 约束
        因为条款类问题中"第X条"是关键锚点，不应被指标关键词约束
        """
        import jieba as _jieba

        # P1 新增：检测是否为 clause 类问题（含"第X条/章/节"模式）
        is_clause_question = bool(re.search(r'第[一二三四五六七八九十百千零\d]+\s*[条章节款项号]', question))

        # 提取问题中的指标关键词（用于上下文约束，避免噪声）
        metric_keywords = set()
        for w in _jieba.cut(question):
            w = w.strip()
            if len(w) >= 2 and w not in self.stopwords:
                metric_keywords.add(w)

        q_nums = self._extract_nums_from_text(question)
        o_nums = set()
        for opt_text in options.values():
            o_nums.update(self._extract_nums_from_text(opt_text))

        # 问题数值：需匹配指标上下文，权重使用领域配置值
        q_bonus = self._numeric_match_score  # 15-25 量级
        for num in q_nums:
            # 原始数值匹配
            chunk_ids = self.numeric_index.get(num, set())
            # 单位换算匹配：当数值带小数点时，尝试 ×100/×10000 匹配文档中的百万元/万元单位
            # 解决 fin_debug_002 的"374.48亿元"无法匹配"37,448百万元"问题
            if '.' in num and re.match(r'^\d+\.\d+$', num):
                try:
                    val = float(num)
                    for multiplier in [100, 10000, 1000000]:
                        converted = str(int(val * multiplier))
                        chunk_ids = chunk_ids | self.numeric_index.get(converted, set())
                except ValueError:
                    pass
            for chunk_id in chunk_ids:
                if chunk_id not in scope:
                    continue
                text = self.chunk_data[chunk_id]["text"]
                metric_hits = sum(1 for kw in metric_keywords if kw in text)
                # P1 优化：clause 类问题跳过 metric_hits 约束
                # 因为条款类问题中数值通常是条款编号的一部分（如"第八十五条"中的"85"）
                # 不应被"指标关键词"约束（clause 问题通常无明确指标关键词）
                if is_clause_question or metric_hits >= 1:
                    candidates[chunk_id] = candidates.get(chunk_id, 0) + q_bonus
                    if breakdown and chunk_id in breakdown:
                        breakdown[chunk_id]["numeric"] += q_bonus

        # 选项数值：要求更高指标上下文命中，权重为问题数值的 40%
        o_bonus = self._numeric_match_score * 0.4  # 6-10 量级
        for num in o_nums - q_nums:  # 避免重复加分
            # 原始数值匹配
            chunk_ids = self.numeric_index.get(num, set())
            # 单位换算匹配（同上）
            if '.' in num and re.match(r'^\d+\.\d+$', num):
                try:
                    val = float(num)
                    for multiplier in [100, 10000, 1000000]:
                        converted = str(int(val * multiplier))
                        chunk_ids = chunk_ids | self.numeric_index.get(converted, set())
                except ValueError:
                    pass
            # 大数值（4+ 位整数或含亿/万元）放宽到 1 个上下文命中
            # 解决 fc_debug_009 的 123,485.77 和 res_debug_002 的 2500亿 召回失败
            is_large_num = (
                re.match(r'^\d{4,}', num) or
                any(u in num for u in ['亿', '万元', '百万']) or
                ',' in num  # 带千位分隔符的精确数值
            )
            for chunk_id in chunk_ids:
                if chunk_id not in scope:
                    continue
                text = self.chunk_data[chunk_id]["text"]
                metric_hits = sum(1 for kw in metric_keywords if kw in text)
                # 表格 chunks 通常缺少问题关键词，但对大数值匹配放宽到 0 个上下文命中
                # 解决 fc_debug_009 中表格 chunks（含 123,485.77）无法触发数值加分的问题
                is_table_chunk = self.chunk_data.get(chunk_id, {}).get("has_table", False)
                if is_large_num and is_table_chunk:
                    required_hits = 0
                elif is_large_num:
                    required_hits = 1
                else:
                    required_hits = 2
                if metric_hits >= required_hits:
                    bonus = o_bonus
                    # 大数值额外加权（精确数值匹配价值高）
                    if is_large_num:
                        bonus = o_bonus * 2.0  # 12-20 量级
                        # 表格 chunks 中的精确数值额外加权（解决表格被文字 chunks 掩盖的问题）
                        # 表格 chunks BM25 分数通常很低（40-80），需要足够大的 numeric 加分才能进入 top-10
                        # 加 250 分让表格 chunks 能与 BM25 200+ 的文字 chunks 竞争
                        if is_table_chunk:
                            bonus = 250.0  # 固定大权重，足以让表格 chunks 进入 top-10
                    candidates[chunk_id] = candidates.get(chunk_id, 0) + bonus
                    if breakdown and chunk_id in breakdown:
                        breakdown[chunk_id]["numeric"] += bonus

    # 关键短语强匹配表：这些短语在查询中出现时，对包含完整短语的 chunk 大幅加分
    # 解决"研发投入合计"被"研发费用"chunk 掩盖的问题
    # 解决 fin_debug_005/008 的"研发费用占营业收入比例"召回失败问题
    # 解决 reg_debug_006/008 的法条编号相关 chunk 排名靠后问题
    STRONG_PHRASE_BOOST = {
        # 研发投入相关
        "研发投入合计", "研发投入总额", "研发投入金额", "研发投入资本化",
        "研发投入占营业收入比例", "研发费用占营业收入比例",
        "研发投入总额占营业收入比例", "资本化研发投入占研发投入",
        # 市场规模相关
        "市场规模", "市场空间", "金融信创市场规模",
        # 债券承销相关
        "受托管理人", "牵头主承销商", "簿记管理人",
        # 发行规模相关
        "发行规模", "发行总额", "募集资金总额", "募集资金净额",
        "项目总投资额", "项目投资总额", "拟投入募集资金",
        # 财务指标相关
        "资产负债率", "经营活动产生的现金流量净额",
        "归母净利润", "归属于上市公司股东的净利润",
        "营业收入", "净利润", "营业总收入",
        "分红比例", "每10股派息",
        # 债券条款相关
        "票面利率", "债券期限", "还本付息",
        # 保险相关
        "身故保险金", "现金价值", "犹豫期", "免责条款", "责任免除",
        # 监管法规相关
        "施行日期", "保存期限", "保存十年", "客户身份资料", "交易记录",
        "调任", "无需提交变更申请", "非银行支付机构",
        # 研报相关
        "AI光模块", "光模块市场规模", "云服务厂商资本开支", "CPO方案",
        "价值量占比", "贴片", "引线键合", "耦合", "封装", "测试",
    }

    def _add_phrase_boost(self, candidates: dict, question: str, options: dict,
                          scope: set, breakdown: dict = None):
        """短语匹配加分（领域专属权重）

        与父类差异：
        - 长短语（5+字）权重提升到 self._rating_match_score（10-12 量级）
        - 命中领域专属 clause_keywords 时追加 0.5 倍权重
        - 表格 chunk 在财报/研报领域额外加 table_boost
        - 关键短语强匹配：STRONG_PHRASE_BOOST 中的短语加 30 分（足以撼动 BM25 主分）
        """
        # 提取问题中 3-20 字的精确短语（扩展到 20 字以支持"研发费用占营业收入比例"等长术语）
        exact_phrases = set()
        for m in re.finditer(r'[一-鿿]{3,20}', question):
            phrase = m.group()
            if phrase not in self.stopwords:
                exact_phrases.add(phrase)

        # 也从选项中提取关键短语（如"受托管理人"、"研发投入合计"）
        for opt_text in options.values():
            for m in re.finditer(r'[一-鿿]{3,20}', opt_text):
                phrase = m.group()
                if phrase not in self.stopwords:
                    exact_phrases.add(phrase)

        # 基础上限 = 领域评级匹配分（10-12 量级）
        base_cap = self._rating_match_score

        for phrase in exact_phrases:
            chunk_ids = self.phrase_index.get(phrase, set())
            for chunk_id in chunk_ids:
                if chunk_id not in scope:
                    continue
                # 短语长度加权：3字=40%, 4字=60%, 5字+=100%（受 base_cap 上限约束）
                ratio = min(0.2 * len(phrase) + 0.2, 1.0)
                weight = base_cap * ratio
                # 命中领域条款关键词时追加 50% 权重
                if phrase in self._domain_clause_keywords:
                    weight *= 1.5
                # 关键短语强匹配：加 40 分（足以撼动 BM25 主分 60-240）
                # 优化：从 30 分提升到 40 分，让"研发费用占营业收入比例"等长术语更易被召回
                if phrase in self.STRONG_PHRASE_BOOST:
                    weight += 40.0
                candidates[chunk_id] = candidates.get(chunk_id, 0) + weight
                if breakdown and chunk_id in breakdown:
                    breakdown[chunk_id]["phrase"] += weight

                # 表格 chunk 额外加 table_boost（财报/研报尤其重要）
                data = self.chunk_data.get(chunk_id, {})
                if data.get("has_table"):
                    table_extra = self._table_boost * 2.0  # 表格加权 2-4 量级
                    candidates[chunk_id] = candidates.get(chunk_id, 0) + table_extra
                    if breakdown and chunk_id in breakdown:
                        breakdown[chunk_id]["phrase"] += table_extra

        # === 新增：直接检查 STRONG_PHRASE_BOOST 短语是否在问题/选项中出现 ===
        # 解决"研发费用占营业收入比例"等长术语在问题中跨字符出现但被正则误判的问题
        # 例如问题"研发投入占营业收入比例的对比"中包含"研发投入占营业收入比例"这一强匹配短语
        full_query = question + " " + " ".join(options.values())
        for boost_phrase in self.STRONG_PHRASE_BOOST:
            if boost_phrase not in full_query:
                continue
            # 该强匹配短语在查询中出现，对包含此短语的 chunk 加分
            chunk_ids = self.phrase_index.get(boost_phrase, set())
            for chunk_id in chunk_ids:
                if chunk_id not in scope:
                    continue
                candidates[chunk_id] = candidates.get(chunk_id, 0) + 40.0
                if breakdown and chunk_id in breakdown:
                    breakdown[chunk_id]["phrase"] += 40.0

    def _add_entity_boost(self, candidates: dict, question: str, options: dict,
                          scope: set, breakdown: dict = None):
        """实体匹配加分（领域专属权重）

        与父类差异：
        - bonus 使用 self._entity_match_score（5-8 量级）而非硬编码 1.5
        - 命中 doc_guarantee_phrases 时追加 0.5 倍权重（提升定位准确性）
        - 噪音标题段落降权（减分）
        """
        full_text = question + " " + " ".join(options.values())
        entities = []
        # 提取公司/机构类实体
        for m in re.finditer(r'[一-鿿]{2,15}(?:股份|集团|科技|证券|银行|保险|控股)', full_text):
            entities.append(m.group())
        # 监管领域：提取法条编号作为实体
        if self.domain == "regulatory":
            entities += re.findall(r'第[一二三四五六七八九十百千\d]+\s*[条章节款项]', full_text)
        # 保险领域：提取产品名（含"险"字）
        if self.domain == "insurance":
            entities += re.findall(r'[一-鿿]{2,8}(?:险|保)', full_text)

        # 实体加分上限：取前 5 个实体，避免过度加分
        entity_bonus = self._entity_match_score  # 5-8 量级
        for entity in entities[:5]:
            chunk_ids = self.entity_index.get(entity, set())
            for chunk_id in chunk_ids:
                if chunk_id not in scope:
                    continue
                candidates[chunk_id] = candidates.get(chunk_id, 0) + entity_bonus
                if breakdown and chunk_id in breakdown:
                    breakdown[chunk_id]["entity"] += entity_bonus

        # 领域定位短语加分（如"现金价值"、"信息披露"等关键术语）
        for phrase in self._domain_doc_guarantee:
            if phrase in full_text:
                chunk_ids = self.phrase_index.get(phrase, set())
                for chunk_id in chunk_ids:
                    if chunk_id not in scope:
                        continue
                    candidates[chunk_id] = candidates.get(chunk_id, 0) + entity_bonus * 0.5
                    if breakdown and chunk_id in breakdown:
                        breakdown[chunk_id]["entity"] += entity_bonus * 0.5

        # 噪音标题段落降权（如"目录"、"重要提示"等）
        if self._domain_noise_headings:
            for chunk_id, data in self.chunk_data.items():
                if chunk_id not in scope or chunk_id not in candidates:
                    continue
                section_title = data.get("section_title", "")
                if any(noise in section_title for noise in self._domain_noise_headings):
                    penalty = self._numeric_match_score * 0.3  # 降权 4-7 分
                    candidates[chunk_id] = candidates.get(chunk_id, 0) - penalty
                    if breakdown and chunk_id in breakdown:
                        breakdown[chunk_id]["entity"] -= penalty

    def build_from_json_dir(self):
        """从JSON目录构建领域专属索引"""
        pattern = self.domain_config["json_pattern"]
        chunk_params = {
            "max_chars": self.domain_config["chunk_max_chars"],
            "overlap_chars": self.domain_config["chunk_overlap_chars"],
            "min_length": self.domain_config["chunk_min_length"],
        }

        print(f"构建 {self.domain} 领域索引 (data_dir={self.data_dir})...")
        print(f"  文件模式: {pattern}")
        print(f"  分块参数: max={chunk_params['max_chars']}, overlap={chunk_params['overlap_chars']}")
        print(f"  BM25参数: k1={self._bm25_k1}, b={self._bm25_b}")

        chunks = []
        files_loaded = 0

        for fname in sorted(os.listdir(self.data_dir)):
            if not re.match(pattern, fname):
                continue
            fpath = os.path.join(self.data_dir, fname)
            if not os.path.isfile(fpath):
                continue

            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)

            doc_id = data.get("doc_id", os.path.splitext(fname)[0])
            files_loaded += 1

            # 提取文本内容并分块
            doc_chunks = self._chunk_document(data, doc_id, chunk_params)
            for chunk in doc_chunks:
                chunk_id = f"{doc_id}_c{chunk['chunk_id']}"
                self.chunk_data[chunk_id] = {
                    "text": chunk["text"],
                    "doc_id": doc_id,
                    "has_table": chunk.get("has_table", False),
                    "page": chunk.get("page", 0),
                    "page_start": chunk.get("page", 0),
                    "page_end": chunk.get("page", 0),
                    "chunk_type": "table" if chunk.get("has_table") else "text",
                    "section_title": chunk.get("section_title", ""),
                    "table_context_before": "",
                }
                chunks.append((chunk_id, chunk["text"]))

        print(f"  加载文件: {files_loaded}")
        print(f"  生成chunks: {len(chunks)}")

        # 构建索引
        # 数值/短语/实体索引（P0 修复：补上 entity_index 构建）
        for chunk_id, text in chunks:
            self._build_numeric_index(chunk_id, text)
            self._build_phrase_index(chunk_id, text)
            # P0 修复：补上 entity_index 构建（之前漏调导致 entity_index 全部为空）
            entities = self._extract_entities_from_text(text)
            for ent_name in entities:
                self.entity_index[ent_name].add(chunk_id)

        # BM25索引（使用领域专属参数）
        self._build_bm25_index_with_params(chunks, self._bm25_k1, self._bm25_b)

        print(f"  索引构建完成: {len(chunks)} chunks, "
              f"{len(self.numeric_index)} 数值, "
              f"{len(self.phrase_index)} 短语, "
              f"{len(self.entity_index)} 实体")  # 新增实体统计

    def _extract_entities_from_text(self, text: str) -> list:
        """从文本中提取实体（P0 新增方法，用于补全 entity_index）

        提取策略：
        - 公司/机构类实体：含"股份/集团/科技/证券/银行/保险/控股"后缀
        - 监管领域法条编号：第X条/章/节/款/项
        - 保险产品名：含"险"字
        - 长公司名（6-20 字）：作为整体索引，解决长术语匹配失效

        Returns:
            实体名称列表（去重）
        """
        entities = set()
        # 公司/机构类实体（与查询端 _add_entity_boost 的正则对齐）
        for m in re.finditer(r'[一-鿿]{2,15}(?:股份|集团|科技|证券|银行|保险|控股)', text):
            entities.add(m.group())
        # 监管领域法条编号
        for m in re.finditer(r'第[一二三四五六七八九十百千\d]+\s*[条章节款项号]', text):
            entities.add(re.sub(r'\s+', '', m.group()))  # 去除空格
        # 保险产品名
        if self.domain == "insurance":
            for m in re.finditer(r'[一-鿿]{2,8}(?:险|保)', text):
                entities.add(m.group())
        # 长公司名（6-20 字连续中文，解决 phrase_index 上限不一致导致的长公司名匹配失效）
        for m in re.finditer(r'[一-鿿]{6,20}(?:有限公司|股份有限公司|集团)', text):
            entities.add(m.group())
        return list(entities)

    def _chunk_document(self, data: dict, doc_id: str, params: dict) -> list:
        """将文档分块（领域专属策略）"""
        max_chars = params["max_chars"]
        overlap_chars = params["overlap_chars"]
        min_length = params["min_length"]

        chunks = []
        chunk_id = 0

        # 提取pages
        pages = data.get("pages", [])
        if not pages:
            # 如果没有pages结构，尝试直接解析
            text = json.dumps(data, ensure_ascii=False)
            pages = [{"page_num": 1, "blocks": [{"type": "text", "text": text}]}]

        for page in pages:
            page_num = page.get("page_num", 0)
            blocks = page.get("blocks", [])

            if not blocks:
                text = page.get("text", "")
                if text and len(text.strip()) >= min_length:
                    for para in re.split(r'\n{2,}', text):
                        para = para.strip()
                        if len(para) >= min_length:
                            chunks.append({
                                "chunk_id": chunk_id,
                                "text": para,
                                "page": page_num,
                                "has_table": False,
                            })
                            chunk_id += 1
                continue

            current_text = ""
            for block in blocks:
                btype = block.get("type", "text")
                btext = block.get("text", "")

                if btype == "table":
                    # 表格独立成chunk
                    if current_text and len(current_text.strip()) >= min_length:
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": current_text.strip(),
                            "page": page_num,
                            "has_table": False,
                        })
                        chunk_id += 1
                        # overlap
                        if overlap_chars > 0 and len(current_text) > overlap_chars:
                            current_text = current_text[-overlap_chars:]
                        else:
                            current_text = ""

                    # 表格作为独立chunk
                    table_text = block.get("markdown", block.get("html", btext))
                    if table_text and len(table_text.strip()) >= min_length:
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": table_text.strip(),
                            "page": page_num,
                            "has_table": True,
                        })
                        chunk_id += 1
                else:
                    # 文本块累积
                    if current_text and len(current_text) + len(btext) > max_chars:
                        if len(current_text.strip()) >= min_length:
                            chunks.append({
                                "chunk_id": chunk_id,
                                "text": current_text.strip(),
                                "page": page_num,
                                "has_table": False,
                            })
                            chunk_id += 1
                            if overlap_chars > 0 and len(current_text) > overlap_chars:
                                current_text = current_text[-overlap_chars:]
                            else:
                                current_text = ""
                    current_text = current_text + "\n" + btext if current_text else btext

            # 剩余文本
            if current_text and len(current_text.strip()) >= min_length:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_text.strip(),
                    "page": page_num,
                    "has_table": False,
                })
                chunk_id += 1

        return chunks

    def _build_bm25_index_with_params(self, chunks, k1, b):
        """使用领域专属BM25参数构建索引"""
        self.chunk_ids = []
        self.chunk_texts = []
        self._ensure_tokenizer_loaded()

        # 多线程分词
        results = [None] * len(chunks)

        def tokenize_one(args):
            idx, chunk_id, text = args
            tokens = self._tokenize(text)
            return idx, chunk_id, tokens

        with ThreadPoolExecutor(max_workers=8) as executor:
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
            self.bm25 = BM25Okapi(self.chunk_texts, k1=k1, b=b)

    def search(self, question: str, options: dict, doc_ids=None,
               top_k: int = None, debug: bool = False):
        """领域专属检索"""
        if top_k is None:
            top_k = self._retrieve_top_k
        return super().search(question, options, doc_ids, top_k, debug)


def build_all_domain_indexes():
    """为所有5个领域构建独立索引"""
    systems = {}
    domains = ["financial_contracts", "financial_reports", "insurance",
               "regulatory", "research"]

    for domain in domains:
        print(f"\n{'='*60}")
        system = DomainRetrievalSystem(domain)
        system.build_from_json_dir()
        systems[domain] = system

    return systems


if __name__ == "__main__":
    # 构建所有领域索引并验证
    systems = build_all_domain_indexes()

    # 验证检索质量
    print(f"\n{'='*60}")
    print("验证检索质量")
    print(f"{'='*60}")

    # 测试fc_a_005: 搜索43.24%
    fc_system = systems["financial_contracts"]
    results = fc_system.search(
        "针对两份文档所涉及的发行主体及交易结构特征",
        {"A": "第一份文档的发行人是广东省广晟控股集团有限公司",
         "B": "第二份文档的内容涉及发行股份购买资产并募集配套资金",
         "C": "两份文档均明确给出了发行人的合并口径资产负债率具体数值",
         "D": "第二份文档中标的公司控股股东的资产负债率为43.24%"},
        doc_ids=["text01", "text10"],
        top_k=5
    )
    print(f"\nfc_a_005 检索结果:")
    for r in results:
        has_4324 = "43.24" in r["text"]
        print(f"  {r['chunk_id']}: score={r['score']:.1f} | 43.24={'✓' if has_4324 else '✗'} | preview={r['text'][:80]}...")
