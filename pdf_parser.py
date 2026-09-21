"""
分块模块 - 提供统一分块功能（block 感知 + 表格独立 + overlap）
含 PyMuPDF 降级解析器
"""
import os
import re

from utils import html_to_text


def parse_pdf(pdf_path: str) -> dict:
    """
    用 PyMuPDF 提取 PDF 文本（降级方案）

    Returns:
        {"path", "filename", "doc_id", "pages": [{"page_num", "text"}], "full_text", "page_count"}
    """
    import fitz  # PyMuPDF

    filename = os.path.basename(pdf_path)
    doc_id = os.path.splitext(filename)[0]

    doc = fitz.open(pdf_path)
    pages = []
    full_text_parts = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        pages.append({"page_num": page_num + 1, "text": text})
        full_text_parts.append(text)

    doc.close()

    return {
        "path": pdf_path,
        "filename": filename,
        "doc_id": doc_id,
        "pages": pages,
        "full_text": "\n\n".join(full_text_parts),
        "page_count": len(pages),
    }


def split_into_chunks(
    text: str,
    max_chars: int = 1500,
    overlap_chars: int = 200,
    min_length: int = 100,
) -> list[dict]:
    """
    将文本分块，按段落/句子边界切分

    Args:
        max_chars: 每个chunk最大字符数
        overlap_chars: chunk重叠字符数
        min_length: 最小段落长度

    Returns:
        [{"chunk_id": int, "text": str, "start_char": int, "end_char": int}]
    """
    if not text or len(text.strip()) < min_length:
        return []

    paragraphs = re.split(r'\n{2,}', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""
    chunk_id = 0
    char_offset = 0

    for para in paragraphs:
        if current_chunk and len(current_chunk) + len(para) < max_chars:
            current_chunk += "\n\n" + para
        else:
            if current_chunk and len(current_chunk) >= min_length:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk,
                    "start_char": char_offset,
                    "end_char": char_offset + len(current_chunk),
                })
                chunk_id += 1
                char_offset += len(current_chunk) + 2
            current_chunk = para

    if current_chunk and len(current_chunk) >= min_length:
        chunks.append({
            "chunk_id": chunk_id,
            "text": current_chunk,
            "start_char": char_offset,
            "end_char": char_offset + len(current_chunk),
        })

    return chunks


# 条款/章节边界正则：识别"第X条"、"第X章"、"第X节"、"X.Y.Z"编号等结构边界
CLAUSE_BOUNDARY_RE = re.compile(
    r'^(?:'
    r'第[一二三四五六七八九十百千\d]+\s*[章节条款]'  # 第X条/章/节/款
    r'|\d+(?:\.\d+){1,3}(?:\s*[|｜])?'              # 1.1.1 / 1.1.1 |
    r'|(?:[一二三四五六七八九十]+、)'                # 一、二、三、
    r'|(?:\([一二三四五六七八九十\d]+\))'            # (一) (1)
    r')',
    re.MULTILINE,
)


def _split_by_clause_boundary(text: str, max_chars: int) -> list[str]:
    """按条款边界切分文本，保留条款完整性

    优先在"第X条"/"X.Y.Z"等编号处切分，避免重要条款被截断。
    若单条款超过 max_chars，再按段落兜底切分。

    Args:
        text: 待切分文本
        max_chars: 每段最大字符数

    Returns:
        切分后的文本片段列表
    """
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    # 找出所有条款边界的起始位置
    boundaries = [m.start() for m in CLAUSE_BOUNDARY_RE.finditer(text)]
    if not boundaries or boundaries[0] > 0:
        boundaries.insert(0, 0)

    segments = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        segment = text[start:end].strip()
        if not segment:
            continue
        # 单条款仍超长时按段落兜底
        if len(segment) > max_chars:
            paras = re.split(r'\n{2,}', segment)
            buf = ""
            for p in paras:
                p = p.strip()
                if not p:
                    continue
                if buf and len(buf) + len(p) < max_chars:
                    buf += "\n\n" + p
                else:
                    if buf:
                        segments.append(buf)
                    buf = p
            if buf:
                segments.append(buf)
        else:
            segments.append(segment)

    # 合并过短片段（<100字的相邻片段合并）
    merged = []
    for seg in segments:
        if merged and len(merged[-1]) + len(seg) < max_chars:
            merged[-1] += "\n\n" + seg
        else:
            merged.append(seg)
    return merged


def chunk_blocks(
    pages: list[dict],
    doc_id: str,
    max_chars: int = None,
    overlap_chars: int = None,
    min_length: int = None,
) -> list[dict]:
    """统一的 block 感知分块函数（条款边界优化版）

    优化点：
    - 表格作为独立 chunk，保护行列对齐不被切断
    - 文本按条款/章节边界切分（第X条、X.Y.Z编号），保留条款完整性
    - 监管法规/保险条款类文档不再"每页1块"，而是"每条1块"
    - 超长文档（合同/财报）通过条款边界自然切分，避免BM25分数稀释

    Args:
        pages: [{"page_num": int, "blocks": [{"type": "text"|"table"|"formula",
                "lines":[{"text":...}], "html": str, "latex": str}], "text": str}]

    Returns:
        [{"chunk_id", "text", "has_table", "table_html", "page", "doc_id"}]
    """
    from config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS, CHUNK_MIN_LENGTH
    max_chars = max_chars or CHUNK_MAX_CHARS
    overlap_chars = overlap_chars if overlap_chars is not None else CHUNK_OVERLAP_CHARS
    min_length = min_length or CHUNK_MIN_LENGTH

    chunks = []
    chunk_id = 0

    def _flush(current_text, page_num):
        nonlocal chunk_id
        text = current_text.strip()
        if not text or len(text) < min_length:
            return ""
        chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "has_table": False,
            "table_html": "",
            "page": page_num,
            "doc_id": doc_id,
        })
        chunk_id += 1
        if overlap_chars > 0 and len(text) > overlap_chars:
            return text[-overlap_chars:]
        return ""

    for page_data in pages:
        page_num = page_data.get("page_num", 0)
        blocks = page_data.get("blocks")

        if not blocks:
            text = page_data.get("text", "")
            if not text or len(text.strip()) < min_length:
                continue
            # 按条款边界切分（替代旧的段落切分）
            for segment in _split_by_clause_boundary(text, max_chars):
                if len(segment.strip()) >= min_length:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": segment.strip(),
                        "has_table": False,
                        "table_html": "",
                        "page": page_num,
                        "doc_id": doc_id,
                    })
                    chunk_id += 1
            continue

        current_text = ""
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "text")

            if btype == "table":
                overlap = _flush(current_text, page_num)
                current_text = overlap
                table_md = block.get("markdown", "")
                table_html = block.get("html", "")
                if not table_md and table_html:
                    table_md = html_to_text(table_html)
                if table_md:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": f"[表格 - 第{page_num}页]\n{table_md}",
                        "has_table": True,
                        "table_html": table_html,
                        "page": page_num,
                        "doc_id": doc_id,
                    })
                    chunk_id += 1

            elif btype == "text":
                if "lines" in block:
                    block_text = "\n".join(
                        line.get("text", "").strip()
                        for line in block.get("lines", [])
                        if isinstance(line, dict) and line.get("text", "").strip()
                    )
                else:
                    block_text = block.get("text", "").strip()
                if not block_text:
                    continue
                # 累积文本，超长时按条款边界切分
                if len(current_text) + len(block_text) < max_chars:
                    current_text += "\n" + block_text if current_text else block_text
                else:
                    # 按条款边界切分当前累积文本
                    full_text = current_text + "\n" + block_text if current_text else block_text
                    segments = _split_by_clause_boundary(full_text, max_chars)
                    for seg in segments[:-1]:
                        if len(seg.strip()) >= min_length:
                            chunks.append({
                                "chunk_id": chunk_id,
                                "text": seg.strip(),
                                "has_table": False,
                                "table_html": "",
                                "page": page_num,
                                "doc_id": doc_id,
                            })
                            chunk_id += 1
                    current_text = segments[-1] if segments else ""

            elif btype == "formula":
                latex = block.get("latex", "")
                if latex:
                    current_text += f"\n[公式] {latex}"

        _flush(current_text, page_num)

    return chunks
