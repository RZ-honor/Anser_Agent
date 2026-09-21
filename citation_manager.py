"""
引用管理与拒答检测（改进.md §3.1 强制引用 + §3.2 拒答机制）

职责：
1. 解析模型输出中的证据引用标记 [证据N]，校验编号是否指向真实证据
2. 检测模型的拒答信号（"文档中未提及"等短语）
3. 校验答案中的数值是否有证据支撑（strict 模式下触发重写）

与 config.py 的 CITATION_MODE / REFUSAL_MODE 配合：
- record 模式：仅记录引用校验结果，不干预答案
- strict 模式：答案中出现无证据支撑的数值时触发一次重写
"""
import re

# 引用标记：[证据1] / [证据 12]（与 _format_evidence 输出的编号格式对齐）
CITATION_PATTERN = re.compile(r"\[证据\s*(\d{1,2})\]")

# 拒答信号短语：模型明确表示文档中找不到依据（出现在分析文本中即视为拒答信号）
REFUSAL_PHRASES = [
    "文档中未提及", "文档未提及", "文档中没有提到", "文档中找不到",
    "未找到相关证据", "无法从文档", "证据不足，无法", "均无证据支持",
]

# 数值提取：千分位/小数/百分比（与检索层数值归一化口径一致）
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?")


def parse_citations(text: str) -> list[int]:
    """从模型输出中解析引用编号，返回去重排序后的编号列表"""
    if not text:
        return []
    return sorted({int(m.group(1)) for m in CITATION_PATTERN.finditer(text)})


def verify_citations(cited: list[int], pool_size: int) -> dict:
    """校验引用编号是否指向证据池中的真实条目

    Args:
        cited: parse_citations 的输出（编号从 1 开始）
        pool_size: 证据池大小（初始证据 + 工具追加证据）

    Returns:
        {"valid": [...], "invalid": [...], "coverage": 0.0~1.0}
        invalid 编号说明模型引用了不存在的证据（幻觉引用）
    """
    valid = [c for c in cited if 1 <= c <= pool_size]
    invalid = [c for c in cited if c < 1 or c > pool_size]
    return {
        "valid": valid,
        "invalid": invalid,
        "coverage": len(valid) / pool_size if pool_size else 0.0,
    }


def detect_refusal(text: str) -> bool:
    """检测模型输出中是否包含拒答信号"""
    if not text:
        return False
    return any(p in text for p in REFUSAL_PHRASES)


def extract_numbers(text: str) -> list[str]:
    """提取文本中的数值串（保留原始格式，含千分位）"""
    return _NUM_RE.findall(text or "")


def check_numeric_support(content: str, evidence_texts: list[str]) -> list[str]:
    """校验模型结论中的数值是否出现在证据原文

    归一化规则：去千分位后子串匹配（"12,132" 匹配 "12132"，与检索层数值索引口径一致）

    Returns:
        无证据支撑的数值列表（空列表 = 全部数值有支撑）
    """
    if not content or not evidence_texts:
        return []
    # 拼接证据原文并去千分位（一次归一化，供全部数值比对复用）
    evidence_norm = "".join(evidence_texts).replace(",", "")
    unsupported = []
    for num in extract_numbers(content):
        # 过滤短数值（答案字母序号、个位数字等噪音）与引用编号
        core = num.replace(",", "").rstrip(".")
        if len(core) < 2:
            continue
        if core not in evidence_norm:
            unsupported.append(num)
    return unsupported
