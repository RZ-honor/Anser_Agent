"""
领域特化检索优化模块

针对5个领域设计不同的检索策略：
1. financial_contracts: 条款精确匹配 + 数值归一化 + 实体识别
2. financial_reports: 财务指标关键词扩展 + 时间范围过滤 + 单位换算
3. insurance: 保险术语同义词 + 条款编号定位 + 责任范围区分
4. regulatory: 法条编号精确匹配 + 施行日期定位 + 义务主体识别
5. research: 数据来源识别 + 数值单位归一化 + 时间范围匹配
"""
import re
from typing import List, Dict, Set


# ============================================================
# 1. 金融合同领域特化
# ============================================================
CONTRACT_KEYWORDS = {
    # 条款类型关键词（用于扩展查询）
    'clause_types': {
        '违约': ['违约责任', '违约金', '逾期利息', '惩罚性赔偿', '违约情形'],
        '赎回': ['有条件赎回', '强制赎回', '赎回条款', '提前赎回'],
        '回售': ['回售选择权', '投资人回售', '回售条款'],
        '转股': ['转股价格', '向下修正', '转股期限', '初始转股价'],
        '评级': ['主体信用评级', '债项评级', 'AAA', 'AA+', 'AA'],
        '发行': ['发行金额', '发行规模', '募集金额', '注册金额'],
        '担保': ['增信', '担保', '保证', '质押', '抵押'],
        '兑付': ['兑付日', '付息日', '到期日', '兑付安排'],
        '中介': ['受托管理人', '主承销商', '保荐机构', '簿记管理人'],
    },
    # 数值单位归一化（元为基准）
    'unit_patterns': [
        (r'(\d+(?:\.\d+)?)\s*亿元', 100000000),
        (r'(\d+(?:\.\d+)?)\s*万元', 10000),
        (r'(\d+(?:\.\d+)?)\s*百万元', 1000000),
    ],
    # 关键实体模式
    'entity_patterns': [
        r'[一-鿿]{2,20}(?:股份有限公司|有限公司|集团公司|控股集团)',
        r'第[一二三四五六七八九十百千零0-9]+[条章节款项号]',
    ],
}


# ============================================================
# 2. 财务报表领域特化
# ============================================================
REPORT_KEYWORDS = {
    # 财务指标关键词（用于扩展查询）
    'metrics': {
        '营收': ['营业收入', '营业总收入', '主营收入', '营收'],
        '利润': ['净利润', '归母净利润', '归属于母公司', '归属于上市公司股东', '利润总额', '营业利润'],
        '资产': ['总资产', '资产合计', '净资产', '归属于母公司股东权益'],
        '现金流': ['经营现金流', '经营活动产生的现金流量净额', '筹资活动', '投资活动'],
        '每股': ['每股收益', '每股净资产', '每股股利', '每股分红'],
        '研发': ['研发投入', '研发费用', '研发占比', '研发投入占营业收入'],
        '分红': ['现金分红', '利润分配', '股利分配', '每10股派'],
        '增速': ['同比增长', '同比下降', '同比', '增速', '增长率'],
    },
    # 时间关键词
    'time_patterns': [
        r'2024年', r'2025年', r'2023年',
        r'报告期', r'本期', r'上年同期',
        r'年末', r'年末余额',
    ],
}


# ============================================================
# 3. 保险条款领域特化
# ============================================================
INSURANCE_KEYWORDS = {
    # 保险术语同义词
    'term_synonyms': {
        '犹豫期': ['犹豫期', '冷静期', '撤销期'],
        '等待期': ['等待期', '观察期', '免赔期'],
        '免赔额': ['免赔额', '起付线', '免赔'],
        '赔付': ['赔付', '给付', '报销', '理赔'],
        '保险责任': ['保险责任', '保障范围', '承保范围'],
        '责任免除': ['责任免除', '免责条款', '除外责任', '不承担'],
        '现金价值': ['现金价值', '退保金', '解约金'],
        '保单贷款': ['保单贷款', '借款', '贷款比例'],
        '宽限期': ['宽限期', '续期', '续保'],
        '退保': ['退保', '解除合同', '终止合同'],
    },
    # 保险产品类型
    'product_types': {
        '重疾险': ['重大疾病', '重疾', '安佑福'],
        '医疗险': ['医疗', 'e生保', '百万医疗'],
        '意外险': ['意外', '营运交通', '预防接种'],
        '家财险': ['家财', '家庭财产', '房屋'],
        '寿险': ['寿险', '智盈金生', '增益宝', '富鸿金生'],
    },
}


# ============================================================
# 4. 监管法规领域特化
# ============================================================
REGULATORY_KEYWORDS = {
    # 法条编号模式
    'article_patterns': [
        r'第[一二三四五六七八九十百千零0-9]+条',
        r'第[一二三四五六七八九十百千零0-9]+款',
        r'第[一二三四五六七八九十百千零0-9]+项',
    ],
    # 监管领域关键词
    'regulatory_terms': {
        '反洗钱': ['反洗钱', '客户身份识别', '受益所有人', '可疑交易', '大额交易'],
        '信息披露': ['信息披露', '定期报告', '临时报告', '董事会审议', '对外披露'],
        '上市公司': ['上市公司治理', '独立董事', '股东大会', '股东会', '董事会'],
        '证券公司': ['证券公司分类', '分类评价', '风险覆盖率'],
        '银行卡': ['银行卡清算', '清算机构', '分支机构'],
        '支付机构': ['非银行支付', '支付机构', '收费标准'],
        '数据安全': ['数据安全', '核心数据', '高敏感性数据'],
    },
    # 时间节点关键词
    'time_keywords': [
        '施行', '实施', '生效', '公布', '发布',
        '个工作日', '个自然日', '日内', '日前',
    ],
    # 义务主体
    'subject_patterns': [
        r'金融机构', r'银行卡清算机构', r'非银行支付机构',
        r'上市公司', r'证券公司', r'保险公司',
    ],
}


# ============================================================
# 5. 行业研报领域特化
# ============================================================
RESEARCH_KEYWORDS = {
    # 数据类型关键词
    'data_types': {
        '市场规模': ['市场规模', '市场容量', '市场总额', '总规模'],
        '增速': ['复合增速', '增长率', '同比', 'CAGR', '复合年增长率'],
        '份额': ['市场份额', '市占率', '占比'],
        '预测': ['预计', '预测', '展望', '前景'],
        '估值': ['PE', 'PB', '估值', '市盈率', '市净率'],
    },
    # 行业关键词
    'industries': {
        '金融科技': ['金融科技', '信创', '银行IT', '金融信息化'],
        '光通信': ['光通信', '光模块', '光器件'],
        '新能源': ['新能源', '电动车', '锂电池', '碳酸锂', '宁德时代'],
        '保险': ['银保渠道', '寿险', '保费'],
        '网络安全': ['网络安全', '安全运营', '等保'],
        '芯片': ['芯片', '芯原', 'IP授权', '芯片设计'],
    },
    # 数值单位（需注意区分）
    'unit_patterns': [
        (r'(\d+(?:\.\d+)?)\s*亿美元', 'billion_usd'),
        (r'(\d+(?:\.\d+)?)\s*亿元人民币', 'billion_rmb'),
        (r'(\d+(?:\.\d+)?)\s*亿元', 'billion_rmb'),
        (r'(\d+(?:\.\d+)?)\s*%', 'percent'),
    ],
}


# ============================================================
# 领域查询扩展函数
# ============================================================

def expand_query_financial_contracts(question: str, options: dict) -> List[str]:
    """金融合同领域查询扩展"""
    full_text = question + " " + " ".join(options.values())
    expansions = []

    # 1. 条款类型扩展
    for clause_type, synonyms in CONTRACT_KEYWORDS['clause_types'].items():
        if clause_type in full_text:
            expansions.extend(synonyms[:3])

    # 2. 提取法条编号
    for pattern in CONTRACT_KEYWORDS['entity_patterns']:
        matches = re.findall(pattern, full_text)
        expansions.extend(matches)

    # 3. 数值提取（带单位）
    for pattern, _ in CONTRACT_KEYWORDS['unit_patterns']:
        matches = re.findall(pattern, full_text)
        expansions.extend(matches)

    # 4. 实体名称提取
    for pattern in CONTRACT_KEYWORDS['entity_patterns']:
        matches = re.findall(pattern, full_text)
        expansions.extend(matches)

    return list(set(expansions))


def expand_query_financial_reports(question: str, options: dict) -> List[str]:
    """财务报表领域查询扩展"""
    full_text = question + " " + " ".join(options.values())
    expansions = []

    # 1. 财务指标扩展
    for metric, synonyms in REPORT_KEYWORDS['metrics'].items():
        if metric in full_text:
            expansions.extend(synonyms[:3])

    # 2. 时间关键词
    for pattern in REPORT_KEYWORDS['time_patterns']:
        matches = re.findall(pattern, full_text)
        expansions.extend(matches)

    # 3. 数值提取
    nums = re.findall(r'\d+(?:\.\d+)?', full_text.replace(',', ''))
    expansions.extend(nums[:10])

    return list(set(expansions))


def expand_query_insurance(question: str, options: dict) -> List[str]:
    """保险条款领域查询扩展"""
    full_text = question + " " + " ".join(options.values())
    expansions = []

    # 1. 保险术语同义词扩展
    for term, synonyms in INSURANCE_KEYWORDS['term_synonyms'].items():
        if term in full_text:
            expansions.extend(synonyms)

    # 2. 保险产品类型识别
    for product_type, keywords in INSURANCE_KEYWORDS['product_types'].items():
        for kw in keywords:
            if kw in full_text:
                expansions.append(product_type)
                expansions.extend(keywords[:2])
                break

    # 3. 数值提取（免赔额、赔付比例等）
    nums = re.findall(r'\d+(?:\.\d+)?', full_text)
    expansions.extend(nums[:8])

    return list(set(expansions))


def expand_query_regulatory(question: str, options: dict) -> List[str]:
    """监管法规领域查询扩展"""
    full_text = question + " " + " ".join(options.values())
    expansions = []

    # 1. 法条编号精确提取（最重要）
    for pattern in REGULATORY_KEYWORDS['article_patterns']:
        matches = re.findall(pattern, full_text)
        expansions.extend(matches)

    # 2. 监管领域关键词扩展
    for reg_type, terms in REGULATORY_KEYWORDS['regulatory_terms'].items():
        for term in terms:
            if term in full_text:
                expansions.extend(terms[:3])
                break

    # 3. 时间节点关键词
    for kw in REGULATORY_KEYWORDS['time_keywords']:
        if kw in full_text:
            expansions.append(kw)

    # 4. 义务主体
    for pattern in REGULATORY_KEYWORDS['subject_patterns']:
        if pattern in full_text:
            expansions.append(pattern)

    # 5. 数值提取（年限、天数、金额）
    nums = re.findall(r'\d+(?:\.\d+)?', full_text)
    expansions.extend(nums[:8])

    return list(set(expansions))


def expand_query_research(question: str, options: dict) -> List[str]:
    """行业研报领域查询扩展"""
    full_text = question + " " + " ".join(options.values())
    expansions = []

    # 1. 数据类型关键词扩展
    for data_type, synonyms in RESEARCH_KEYWORDS['data_types'].items():
        if data_type in full_text:
            expansions.extend(synonyms)

    # 2. 行业关键词扩展
    for industry, keywords in RESEARCH_KEYWORDS['industries'].items():
        for kw in keywords:
            if kw in full_text:
                expansions.extend(keywords[:3])
                break

    # 3. 数值提取（带单位）
    for pattern, _ in RESEARCH_KEYWORDS['unit_patterns']:
        matches = re.findall(pattern, full_text)
        expansions.extend(matches)

    # 4. 年份提取
    years = re.findall(r'20\d{2}年', full_text)
    expansions.extend(years)

    return list(set(expansions))


# ============================================================
# 领域查询扩展映射
# ============================================================
DOMAIN_EXPANDERS = {
    'financial_contracts': expand_query_financial_contracts,
    'financial_reports': expand_query_financial_reports,
    'insurance': expand_query_insurance,
    'regulatory': expand_query_regulatory,
    'research': expand_query_research,
}


def expand_query(domain: str, question: str, options: dict) -> List[str]:
    """根据领域扩展查询关键词"""
    expander = DOMAIN_EXPANDERS.get(domain)
    if expander:
        return expander(question, options)
    return []


# ============================================================
# 领域特化检索策略
# ============================================================
DOMAIN_SEARCH_STRATEGY = {
    'financial_contracts': {
        'top_k_main': 10,           # 主检索返回数
        'top_k_option': 6,          # 选项级检索返回数
        'enable_fulltext': True,    # 启用全文搜索补充
        'fulltext_keywords': ['第', '条', '违约', '赎回', '回售', '转股'],
        'priority': 'clause_exact', # 优先策略：条款精确匹配
    },
    'financial_reports': {
        'top_k_main': 10,
        'top_k_option': 6,
        'enable_fulltext': True,
        'fulltext_keywords': ['营业收入', '净利润', '研发投入', '每股', '分红'],
        'priority': 'numeric_exact', # 优先策略：数值精确匹配
    },
    'insurance': {
        'top_k_main': 10,
        'top_k_option': 8,          # 保险条款选项检索需要更多
        'enable_fulltext': True,
        'fulltext_keywords': ['犹豫期', '免赔额', '保险责任', '责任免除', '等待期', '现金价值'],
        'priority': 'term_synonym', # 优先策略：术语同义词匹配
    },
    'regulatory': {
        'top_k_main': 12,           # 法规需要更多上下文
        'top_k_option': 6,
        'enable_fulltext': True,
        'fulltext_keywords': ['第', '条', '款', '项', '施行', '工作报告'],
        'priority': 'article_exact', # 优先策略：法条编号精确匹配
    },
    'research': {
        'top_k_main': 10,
        'top_k_option': 6,
        'enable_fulltext': True,
        'fulltext_keywords': ['市场规模', '增速', '占比', '预计', '亿元', '亿美元'],
        'priority': 'data_source',  # 优先策略：数据来源匹配
    },
}


def get_search_strategy(domain: str) -> dict:
    """获取领域特化检索策略"""
    return DOMAIN_SEARCH_STRATEGY.get(domain, {
        'top_k_main': 8,
        'top_k_option': 5,
        'enable_fulltext': False,
        'fulltext_keywords': [],
        'priority': 'default',
    })


if __name__ == '__main__':
    # 测试查询扩展
    print("=== 金融合同领域 ===")
    q = "关于违约赔偿和发行规模的描述"
    opts = {'A': '违约金按150%计算', 'B': '发行金额10亿元'}
    print(f"查询: {q}")
    print(f"扩展词: {expand_query('financial_contracts', q, opts)}")

    print("\n=== 保险领域 ===")
    q = "关于犹豫期和免赔额的说法"
    opts = {'A': '犹豫期15日', 'B': '免赔额1万元'}
    print(f"查询: {q}")
    print(f"扩展词: {expand_query('insurance', q, opts)}")

    print("\n=== 监管法规领域 ===")
    q = "根据第二十七条的规定，金融机构应当在30个工作日内提交报告"
    opts = {'A': '正确', 'B': '错误'}
    print(f"查询: {q}")
    print(f"扩展词: {expand_query('regulatory', q, opts)}")
