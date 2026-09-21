"""
实体抽取模块 - 从文本中提取金融实体
使用规则 + 正则，不依赖大模型（节省 Token）
"""
import re
from utils import normalize_entity, extract_chinese_keywords


# ============ 实体类型定义 ============
ENTITY_TYPES = {
    "company": "公司",
    "regulation": "法规",
    "product": "产品",
    "amount": "金额",
    "percentage": "比例",
    "date": "日期",
    "clause": "条款号",
    "metric": "指标",
}


# ============ 正则模式 ============

# 公司名
COMPANY_PATTERNS = [
    r'([一-龥]{2,15}(?:股份|集团|科技|电子|银行|证券|保险|基金|控股)(?:有限公司)?(?:\([一-龥]+\))?)',
    r'([一-龥]{2,10}(?:证券|保险|银行|基金|信托|期货)(?:股份)?有限公司)',
    r'((?:中国|国家)[一-龥]{2,10}(?:有限公司|集团|公司))',
]

# 法规名（增强版：覆盖更多法规命名模式）
REGULATION_PATTERNS = [
    r'(《[^》]{3,80}》)',                                                    # 《法规名》放宽长度到80
    r'((?:中华人民共和国)?[一-龥]{2,30}(?:法|条例|管理办法|实施细则|规定|准则|指引|规则|细则|通知|公告|意见|决定|命令|令))',
    r'((?:中国人民银行|中国证监会|银保监会|国家金融监督管理总局|国务院|财政部|国家税务总局|国家发改委)[一-龥]{2,40})',
    r'([一二三四五六七八九十百千\d]+号令)',                                    # 第X号令
    r'(\d{4}年\d{1,2}月\d{1,2}日[一-龥]{2,20}(?:发布|实施|施行))',           # 日期+法规动作
]

# 金额
AMOUNT_PATTERNS = [
    r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:万)?元',
    r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*亿元',
    r'(?:人民币|港币|美元)\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:万)?元?',
]

# 比例/百分比
PERCENTAGE_PATTERNS = [
    r'(\d+(?:\.\d+)?)\s*%',
    r'百分之\s*(\d+(?:\.\d+)?)',
    r'(\d+(?:\.\d+)?)\s*个百分点',
]

# 日期
DATE_PATTERNS = [
    r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日',
    r'(\d{4})\s*年\s*(\d{1,2})\s*月',
    r'(\d{4})-(\d{2})-(\d{2})',
    r'(\d{4})/(\d{2})/(\d{2})',
]

# 条款号（增强版：覆盖数字编号、条款项、附件等多种格式）
CLAUSE_PATTERNS = [
    r'第\s*([一二三四五六七八九十百千\d]+)\s*条',
    r'第\s*([一二三四五六七八九十百千\d]+)\s*款',
    r'第\s*([一二三四五六七八九十百千\d]+)\s*章',
    r'第\s*([一二三四五六七八九十百千\d]+)\s*节',
    r'第\s*([一二三四五六七八九十百千\d]+)\s*项',
    r'(?:^|\s)(\d+(?:\.\d+){1,3})(?:\s*[|｜])',                              # 1.1.1 | 格式编号
    r'((?:附|附件)[一二三四五六七八九十\d]+)',                                # 附件一/附件1
    r'(第[一二三四五六七八九十百千\d]+号令)',                                  # 第X号令
]

# 产品名（增强版：覆盖更多保险产品命名模式）
PRODUCT_PATTERNS = [
    r'([一-龥]{2,15}(?:保险|基金|理财|债券|信托|资管)(?:产品|计划|账户)?)',
    r'((?:平安|人保|太保|众安|国寿|新华|泰康|太平|太平洋|中国人寿|中国平安|中国太保|中国财险)[一-龥]{2,20})',
    r'([一-龥]{2,10}(?:年金|寿险|重疾|医疗|意外|养老|分红|万能|投连|两全|定期|终身)(?:保险)?)',
    r'([一-龥]{2,15}(?:2024|2025|2026)[一-龥]{0,10}(?:款|版)?(?:保险|产品|计划)?)',  # 年份款产品
    r'([A-Za-z][一-龥]{2,15}(?:保险|产品|计划|账户)?)',                              # 字母开头产品名
    r'(智盈金生|增益宝|长相伴|财富金生|金佑人生|国寿福|康宁终身|平安福|太保寿险|安享百万|百万医疗)',  # 具体产品名
]

# 年份/期间
YEAR_PATTERNS = [
    r'(\d{4})\s*年',
    r'(上年|本年|当年|前年|去年|今年)',
    r'(上年同期|本报告期|报告期|上一报告期)',
]


# 财务指标（扩展版）
METRIC_KEYWORDS = [
    # 基础财务
    "营业收入", "净利润", "总资产", "净资产", "现金流",
    "毛利率", "净利率", "资产负债率", "ROE", "ROA",
    "每股收益", "市盈率", "市净率", "股息率",
    # 研发
    "研发投入", "研发费用", "研发费用率", "研发占比",
    # 银行/保险
    "资本充足率", "不良贷款率", "拨备覆盖率",
    "偿付能力", "综合成本率", "已赚保费", "退保率",
    "现金价值", "身故保险金", "满期保险金", "年金",
    # 细分财务行项目
    "归属于上市公司股东的净利润", "归母净利润",
    "经营活动产生的现金流量净额", "经营活动现金流",
    "投资活动产生的现金流量净额", "投资活动现金流",
    "筹资活动产生的现金流量净额", "筹资活动现金流",
    "新签合同额", "合同负债",
    "基本每股收益", "稀释每股收益",
    "加权平均净资产收益率",
    "归属于上市公司股东的净资产",
    # 收入结构
    "主营业务收入", "其他业务收入",
    "营业成本", "销售费用", "管理费用", "财务费用",
    # 分红
    "每股股利", "每10股", "分红", "派息",
]


def extract_entities(text: str) -> list[dict]:
    """
    从文本中提取实体

    Returns:
        [{"type": str, "name": str, "value": str, "start": int, "end": int}]
    """
    entities = []

    # 公司名（同时存储全称和简称）
    for pattern in COMPANY_PATTERNS:
        for match in re.finditer(pattern, text):
            full_name = match.group(1)
            short_name = normalize_entity(full_name)
            entities.append({
                "type": "company",
                "name": full_name,
                "value": full_name,
                "start": match.start(),
                "end": match.end(),
            })
            if short_name != full_name and len(short_name) >= 2:
                entities.append({
                    "type": "company",
                    "name": short_name,
                    "value": full_name,
                    "start": match.start(),
                    "end": match.end(),
                })

    # 法规名
    for pattern in REGULATION_PATTERNS:
        for match in re.finditer(pattern, text):
            entities.append({
                "type": "regulation",
                "name": match.group(1),
                "value": match.group(1),
                "start": match.start(),
                "end": match.end(),
            })

    # 金额
    for pattern in AMOUNT_PATTERNS:
        for match in re.finditer(pattern, text):
            entities.append({
                "type": "amount",
                "name": f"金额_{match.group(1)}",
                "value": match.group(1).replace(",", ""),
                "start": match.start(),
                "end": match.end(),
            })

    # 比例
    for pattern in PERCENTAGE_PATTERNS:
        for match in re.finditer(pattern, text):
            entities.append({
                "type": "percentage",
                "name": f"比例_{match.group(1)}%",
                "value": match.group(1),
                "start": match.start(),
                "end": match.end(),
            })

    # 条款号
    for pattern in CLAUSE_PATTERNS:
        for match in re.finditer(pattern, text):
            entities.append({
                "type": "clause",
                "name": f"条款_{match.group(1)}",
                "value": match.group(1),
                "start": match.start(),
                "end": match.end(),
            })

    # 产品名
    for pattern in PRODUCT_PATTERNS:
        for match in re.finditer(pattern, text):
            entities.append({
                "type": "product",
                "name": match.group(1),
                "value": match.group(1),
                "start": match.start(),
                "end": match.end(),
            })

    # 年份/期间
    for pattern in YEAR_PATTERNS:
        for match in re.finditer(pattern, text):
            entities.append({
                "type": "period",
                "name": f"期间_{match.group(1)}",
                "value": match.group(1),
                "start": match.start(),
                "end": match.end(),
            })

    # 财务指标
    for metric in METRIC_KEYWORDS:
        for match in re.finditer(re.escape(metric), text):
            entities.append({
                "type": "metric",
                "name": metric,
                "value": metric,
                "start": match.start(),
                "end": match.end(),
            })

    # 去重
    seen = set()
    unique_entities = []
    for ent in entities:
        key = (ent["type"], ent["name"], ent["start"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)

    return unique_entities


def extract_question_entities(question: str, options: dict) -> dict:
    """
    从问题和选项中提取实体

    Returns:
        {
            "companies": [str],
            "regulations": [str],
            "amounts": [str],
            "percentages": [str],
            "clauses": [str],
            "metrics": [str],
            "keywords": [str],
        }
    """
    full_text = question + " " + " ".join(options.values())
    entities = extract_entities(full_text)

    result = {
        "companies": [],
        "regulations": [],
        "products": [],
        "amounts": [],
        "percentages": [],
        "clauses": [],
        "dates": [],
        "periods": [],
        "metrics": [],
        "keywords": [],
    }

    for ent in entities:
        if ent["type"] == "company":
            result["companies"].append(ent["name"])
        elif ent["type"] == "regulation":
            result["regulations"].append(ent["name"])
        elif ent["type"] == "product":
            result["products"].append(ent["name"])
        elif ent["type"] == "amount":
            result["amounts"].append(ent["value"])
        elif ent["type"] == "percentage":
            result["percentages"].append(ent["value"])
        elif ent["type"] == "clause":
            result["clauses"].append(ent["value"])
        elif ent["type"] == "date":
            result["dates"].append(ent["value"])
        elif ent["type"] == "period":
            result["periods"].append(ent["value"])
        elif ent["type"] == "metric":
            result["metrics"].append(ent["name"])

    # 提取关键词（改进版：使用公共工具）
    result["keywords"] = extract_chinese_keywords(question)

    # 去重
    for key in result:
        result[key] = list(set(result[key]))

    return result


if __name__ == "__main__":
    # 测试
    test_text = """
    根据《上市公司治理准则》第四十七条，公司为资产负债率超过百分之七十的担保对象提供的担保，
    须经股东会审议通过。比亚迪股份有限公司2025年营业收入3,847亿元，净利润186亿元。
    """
    entities = extract_entities(test_text)
    for ent in entities:
        print(f"  [{ent['type']}] {ent['name']}: {ent['value']}")
