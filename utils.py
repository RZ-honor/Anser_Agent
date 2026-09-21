"""
公共工具模块 - 统一的工具函数
"""
import re


def html_to_text(html: str) -> str:
    """HTML 表格转纯文本（保留行列对齐）"""
    text = re.sub(r'<tr[^>]*>', '\n', html)
    text = re.sub(r'<td[^>]*>', ' | ', text)
    text = re.sub(r'<th[^>]*>', ' | ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


def html_to_markdown(html: str) -> str:
    """HTML 表格转 Markdown 格式"""
    return html_to_text(html)  # 当前实现相同，可独立演进


def normalize_entity(name: str) -> str:
    """实体名称归一化：去除常见后缀，统一简称"""
    suffixes = [
        "股份有限公司", "有限责任公司", "有限公司", "集团公司", "集团",
        "股份", "银行", "证券", "保险", "基金",
    ]
    result = name
    for suffix in suffixes:
        if result.endswith(suffix) and len(result) > len(suffix):
            result = result[: -len(suffix)]
            break
    return result


def extract_chinese_keywords(text: str, min_len: int = 2, max_len: int = 8) -> list[str]:
    """提取中文关键词（改进版：去停用词、去重、按长度排序）"""
    stopwords = {
        "的", "了", "在", "是", "和", "与", "或", "及", "等", "中",
        "对", "为", "上", "下", "有", "被", "所", "其", "这", "那",
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
        "根据", "以下", "下列", "关于", "说法", "正确", "描述", "选项",
        "哪个", "哪些", "什么", "如何", "多少", "是否", "属于", "包括",
        "公司", "规定", "应当", "可以", "需要", "不得", "必须", "条款",
    }
    # 匹配连续中文字符
    keywords = re.findall(rf'[一-鿿]{{{min_len},{max_len}}}', text)
    # 去停用词、去重、按长度降序（长词优先）
    seen = set()
    result = []
    for kw in keywords:
        if kw not in stopwords and kw not in seen:
            seen.add(kw)
            result.append(kw)
    result.sort(key=len, reverse=True)
    return result
