"""针对证据不足题目的补充全文检索

用法：python supplement_search.py <qid> <关键词1> [关键词2] ...
在所属领域所有文档全文中搜索关键词，返回匹配段落
"""
import os
import sys
import json
import re

BASE = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE, "submission", "processed_data")
EXTRA_REGULATORY_DIR = r"D:\PROJECT\tianci\1"

# 题目→领域映射（用于限定搜索范围）
QID_DOMAIN = {
    # financial_contracts
    "fc_b_001": "text01",  # 广晟控股
    "fc_b_006": "multi",   # 安克/普联/本川
    "fc_b_007": "multi",
    "fc_b_008": "multi",
    "fc_b_009": "multi",
    "fc_b_011": "text14",  # 西部证券
    "fc_b_014": "text14",
    "fc_b_015": "text11",  # 科源制药
    # financial_reports - 财报补充检索（修复前：0覆盖，导致中期分红等关键数据检索不到）
    "fin_b_001": "fr_multi_2025",  # 比亚迪2025年报
    "fin_b_002": "annual_byd_2025_report",
    "fin_b_003": "fr_multi_catl_midea",  # 宁德+美的
    "fin_b_004": "fr_multi_3comp",  # 比亚迪/宁德/美的
    "fin_b_005": "fr_multi_catl_midea_cmb_cscec",  # 4公司分红
    "fin_b_006": "annual_cmb_2025_report",  # 招行风险指标
    "fin_b_007": "annual_chinamobile_2025_report",  # 中国移动
    "fin_b_008": "annual_cscec_2025_report",  # 中国建筑
    "fin_b_009": "fr_multi_catl_midea",  # 宁德+美的研发
    "fin_b_010": "fr_multi_byd_catl",  # 比亚迪+宁德偿债
    "fin_b_011": "fr_multi_4comp",  # 多公司计算
    "fin_b_012": "annual_midea_2025_report",  # 美的合并/母公司
    "fin_b_013": "annual_byd_2025_report",  # 比亚迪境外收入
    "fin_b_014": "annual_byd_2025_report",  # 比亚迪净利率/现金流
    "fin_b_015": "fr_multi_catl_midea",  # 宁德+美的现金流率
    "fin_b_016": "fr_multi_catl_midea_cmb_cscec",  # 4公司全年分红
    "fin_b_017": "annual_chinamobile_2025_report",  # 中国移动EBITDA
    "fin_b_018": "annual_midea_2025_report",  # 美的权益乘数
    "fin_b_019": "fr_multi_3comp",  # 3公司权益乘数
    "fin_b_020": "annual_cscec_2025_report",  # 中国建筑分红
    # insurance - 保险补充检索（修复前：0覆盖，导致退保费用/给付比例等条款检索不到）
    "ins_b_001": "ins_multi_all",  # 4款产品退保
    "ins_b_002": "ins_multi_all",  # 4款产品身故保险金
    "ins_b_003": "ins_multi_all",  # 4款产品身故保险金
    "ins_b_004": "ins_multi_all",  # 4款产品保单贷款
    "ins_b_005": "ins_multi_all",  # 4款产品年龄错误
    "ins_b_006": "ins_multi_all",  # 恐怖活动免责
    "ins_b_007": "ins_multi_all",  # 交通肇事逃逸
    "ins_b_008": "ins_multi_all",  # 精神损害赔偿
    "ins_b_009": "ins_multi_all",  # 等待期意外
    "ins_b_010": "ins_multi_all",  # 未成年人限制
    "ins_b_011": "1",  # 平安智盈金生现金价值
    "ins_b_012": "ins_multi_all",  # 地震免责
    "ins_b_013": "ins_multi_all",  # 行政/司法行为
    "ins_b_014": "16",  # 平安富鸿金生减额交清
    "ins_b_015": "ins_multi_all",  # 诉讼时效
    "ins_b_016": "ins_multi_all",  # 2年内自杀
    "ins_b_017": "ins_multi_all",  # 核风险免责
    "ins_b_018": "1",  # 平安智盈金生身故保险金
    "ins_b_019": "ins_multi_all",  # 4款产品退保
    "ins_b_020": "ins_multi_all",  # 产品保障触发条件
    # regulatory - 监管法规补充检索
    "reg_b_001": "strict_v3_017_中华人民共和国反洗钱法",  # 反洗钱法
    "reg_b_003": "csrc_0023_att1",  # 重组60日
    "reg_b_004": "strict_v3_016_中国人民银行_国家金融监督管理总局令〔2025〕第2号（银行卡清算机构管理办法）",  # 银行卡清算90日
    "reg_b_005": "csrc_0027_att1",  # 分类评价减半扣分
    "reg_b_007": "csrc_0023_att1",  # 重组30日公告
    "reg_b_008": "csrc_0023_att1",  # 定期报告披露
    "reg_b_009": "strict_v3_009_中国人民银行_国家金融监督管理总局_中国证券监督管理委员会令〔2025〕第11号（金融机构客户尽职调查和客户身份资料及交易记录保存管理办法）",  # 第三方尽调责任
    "reg_b_013": "csrc_0027_att1",  # 自评结果上报
    "reg_b_014": "csrc_0027_att1",  # 警示函扣分
    "reg_b_015": "strict_v3_008_中国人民银行令〔2025〕第12号（金融机构客户受益所有人识别管理办法）",  # 受益所有人差异反馈
    "reg_b_016": "strict_v3_009_中国人民银行_国家金融监督管理总局_中国证券监督管理委员会令〔2025〕第11号（金融机构客户尽职调查和客户身份资料及交易记录保存管理办法）",  # 境外汇款核实
    "reg_b_017": "strict_v3_009_中国人民银行_国家金融监督管理总局_中国证券监督管理委员会令〔2025〕第11号（金融机构客户尽职调查和客户身份资料及交易记录保存管理办法）",  # 可疑交易核实
    "reg_b_018": "csrc_0262",  # 收费调整公示
    "reg_b_020": "csrc_0027_att1",  # 分类评价广告宣传
    "reg_b_021": "strict_v3_008_中国人民银行令〔2025〕第12号（金融机构客户受益所有人识别管理办法）",  # 简化豁免
    "reg_b_023": "strict_v3_017_中华人民共和国反洗钱法",  # 拒绝尽调
    "reg_b_024": "csrc_0027_att1",  # 重组中介责任
    "reg_b_025": "strict_v3_016_中国人民银行_国家金融监督管理总局令〔2025〕第2号（银行卡清算机构管理办法）",  # 银行卡清算机构
    "reg_b_026": "csrc_0023_att1",  # 非交易时段披露
    "reg_b_027": "csrc_0262",  # 收费调整公示
    # research - 研报补充检索
    "res_b_005": "pack2_text04",  # 单车带电量数据
    "res_b_007": "pack2_text09",  # 芯原新签订单
}

# 复合文档映射：将虚拟doc_id展开为实际多文档列表
COMPOSITE_DOC_MAP = {
    "fr_multi_2025": ["annual_byd_2025_report"],
    "fr_multi_catl_midea": ["annual_catl_2025_report", "annual_midea_2025_report"],
    "fr_multi_3comp": ["annual_byd_2025_report", "annual_catl_2025_report", "annual_midea_2025_report"],
    "fr_multi_4comp": ["annual_byd_2025_report", "annual_catl_2025_report", "annual_midea_2025_report",
                       "annual_chinamobile_2025_report", "annual_cmb_2025_report", "annual_cscec_2025_report"],
    "fr_multi_catl_midea_cmb_cscec": ["annual_catl_2025_report", "annual_midea_2025_report",
                                       "annual_cmb_2025_report", "annual_cscec_2025_report"],
    "fr_multi_byd_catl": ["annual_byd_2025_report", "annual_catl_2025_report"],
    "ins_multi_all": [str(i) for i in range(1, 17)],
}


def load_doc_text(doc_id):
    """加载文档全文"""
    for d in [PROCESSED_DIR, EXTRA_REGULATORY_DIR]:
        path = os.path.join(d, f"{doc_id}.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8-sig") as f:
                    data = json.load(f)
                # 合并 pages 文本 + chunks 文本
                parts = []
                for p in data.get("pages", []):
                    parts.append(p.get("text", ""))
                for c in data.get("chunks", []):
                    parts.append(c.get("text", ""))
                return "\n".join(parts)
            except Exception:
                pass
    return ""


def search(text, keywords, window=500, topn=8):
    """在文本中搜索关键词组合，返回包含所有/多数关键词的段落

    支持大小写不敏感匹配（如kWh/kwh/KWH均可匹配）
    """
    if not text:
        return []
    # 构建小写文本用于匹配
    text_lower = text.lower()
    # 找每个关键词的位置
    positions = []
    for kw in keywords:
        kw_lower = kw.lower()
        start = 0
        while True:
            idx = text_lower.find(kw_lower, start)
            if idx < 0:
                break
            positions.append((idx, kw))
            start = idx + len(kw)
    if not positions:
        return []
    positions.sort()
    # 聚类：相邻位置合并为段落
    segments = []
    used = set()
    for pos, kw in positions:
        if pos in used:
            continue
        seg_start = max(0, pos - window // 4)
        seg_end = min(len(text), pos + window)
        seg = text[seg_start:seg_end]
        for p in range(seg_start, seg_end):
            used.add(p)
        # 大小写不敏感匹配检查
        seg_lower = seg.lower()
        matched = [k for k in keywords if k.lower() in seg_lower]
        segments.append((len(matched), pos, seg, matched))
    segments.sort(key=lambda x: (-x[0], x[1]))
    return segments[:topn]


def resolve_doc_ids(doc_id):
    """将虚拟doc_id展开为实际文档列表

    支持三类映射：
    1. "multi" - 固定的金融合同多文档列表（向后兼容）
    2. COMPOSITE_DOC_MAP中的虚拟id - 如"ins_multi_all"展开为1-16全部保险产品
    3. 普通doc_id - 直接返回单元素列表
    """
    if doc_id == "multi":
        return ["text04", "text05", "text06", "text07", "text08", "text09",
                "text10", "text11", "text12", "text13", "text14"]
    # 复合文档映射（如ins_multi_all → [1,2,...,16]）
    if doc_id in COMPOSITE_DOC_MAP:
        return COMPOSITE_DOC_MAP[doc_id]
    return [doc_id]


def main():
    if len(sys.argv) < 3:
        print("用法: python supplement_search.py <qid> <关键词1> [关键词2] ...")
        sys.exit(1)
    qid = sys.argv[1]
    keywords = sys.argv[2:]

    doc_id = QID_DOMAIN.get(qid, "")
    if not doc_id:
        print(f"未配置 qid={qid} 的文档映射")
        return

    doc_ids = resolve_doc_ids(doc_id)

    for did in doc_ids:
        text = load_doc_text(did)
        if not text:
            continue
        results = search(text, keywords)
        if results:
            print(f"\n===== 文档 {did} (命中{len(results)}段) =====")
            for cnt, pos, seg, matched in results:
                print(f"\n--- 命中{cnt}个关键词: {matched} (位置{pos}) ---")
                print(seg[:500])


if __name__ == "__main__":
    main()
