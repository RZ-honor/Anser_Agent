"""
交互式检索工具 - 供子agent调用

用法：
  1. 查看题目：python search_tool.py question --qid fc_a_001
  2. 基础检索：python search_tool.py search --qid fc_a_001 [--top_k 8]
  3. 自定义查询：python search_tool.py search --query "违约赔偿" --domain financial_contracts --doc_ids text01,text02 [--top_k 5]
  4. 选项级检索：python search_tool.py search_option --qid fc_a_001 --option A [--top_k 5]
  5. 查看文档原文：python search_tool.py doc --doc_id text01 --keyword "违约" [--max_chars 2000]
  6. 全文搜索：python search_tool.py fulltext --keyword "第十二条" --domain regulatory [--top_k 10]
  7. 列出所有文档：python search_tool.py list_docs [--domain financial_contracts]
"""
import os
import sys
import json
import pickle
import hashlib
import argparse
import re

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from new_retrieval_system import build_domain_indexes
from preparsed_loader import load_preparsed_documents
from pdf_parser import chunk_blocks

# 比赛题目目录
QUESTIONS_DIR = r"D:\PROJECT\tianci\public_dataset_a\public_dataset_upload\questions\group_a"
INDEX_CACHE_PATH = os.path.join(config.CACHE_DIR, "domain_indexes.pkl")
PREPARSED_DIR = r"D:\PROJECT\tianci\1"


def _compute_index_hash():
    """计算索引内容的哈希值（用于缓存校验）"""
    h = hashlib.md5()
    h.update(f"chunk={config.CHUNK_MAX_CHARS},{config.CHUNK_OVERLAP_CHARS}".encode())
    for f in sorted(os.listdir(PREPARSED_DIR)):
        if f.endswith('.json'):
            h.update(f.encode())
            h.update(str(os.path.getmtime(os.path.join(PREPARSED_DIR, f))).encode())
    return h.hexdigest()


# 全局缓存索引
_DOMAIN_SYSTEMS = None
_ALL_QUESTIONS = None
_PREPARSED_DOCS = None


def load_indexes():
    """加载BM25索引（带缓存）"""
    global _DOMAIN_SYSTEMS
    if _DOMAIN_SYSTEMS is not None:
        return _DOMAIN_SYSTEMS

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    current_hash = _compute_index_hash()

    if os.path.exists(INDEX_CACHE_PATH):
        try:
            with open(INDEX_CACHE_PATH, 'rb') as f:
                cache = pickle.load(f)
            if cache.get('hash') == current_hash:
                _DOMAIN_SYSTEMS = cache['domain_systems']
                return _DOMAIN_SYSTEMS
        except Exception:
            pass

    print("[索引] 构建中...", file=sys.stderr)
    docs = load_preparsed_documents()
    for domain, dd in docs.items():
        for doc_id, d in dd.items():
            pages = d.get('pages', [])
            if pages:
                d['chunks'] = chunk_blocks(pages, doc_id)

    _DOMAIN_SYSTEMS = build_domain_indexes(docs)

    with open(INDEX_CACHE_PATH, 'wb') as f:
        pickle.dump({'hash': current_hash, 'domain_systems': _DOMAIN_SYSTEMS}, f)
    return _DOMAIN_SYSTEMS


def load_questions():
    """加载所有题目"""
    global _ALL_QUESTIONS
    if _ALL_QUESTIONS is not None:
        return _ALL_QUESTIONS

    _ALL_QUESTIONS = {}
    for fname in sorted(os.listdir(QUESTIONS_DIR)):
        if not fname.endswith('_questions.json'):
            continue
        fpath = os.path.join(QUESTIONS_DIR, fname)
        with open(fpath, encoding='utf-8') as f:
            qs = json.load(f)
        for q in qs:
            _ALL_QUESTIONS[q['qid']] = q
    return _ALL_QUESTIONS


def load_preparsed():
    """加载预解析文档（用于查看原文）"""
    global _PREPARSED_DOCS
    if _PREPARSED_DOCS is not None:
        return _PREPARSED_DOCS

    _PREPARSED_DOCS = {}
    for fname in sorted(os.listdir(PREPARSED_DIR)):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(PREPARSED_DIR, fname)
        try:
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)
            doc_id = data.get("doc_id", os.path.splitext(fname)[0])
            _PREPARSED_DOCS[doc_id] = data
        except Exception:
            pass
    return _PREPARSED_DOCS


def _get_search_systems(domain, domain_systems):
    """获取可用的检索系统（含跨领域fallback）"""
    system = domain_systems.get(domain)
    search_systems = []
    if system:
        search_systems.append(system)
    # regulatory领域的strict_v3文档在unknown领域
    if domain == 'regulatory':
        unknown_system = domain_systems.get('unknown')
        if unknown_system and unknown_system not in search_systems:
            search_systems.append(unknown_system)
    if domain == 'research':
        unknown_system = domain_systems.get('unknown')
        if unknown_system and unknown_system not in search_systems:
            search_systems.append(unknown_system)
    if not search_systems:
        # fallback：搜索所有领域
        for sys_obj in domain_systems.values():
            search_systems.append(sys_obj)
    return search_systems


def cmd_question(args):
    """查看题目详情"""
    questions = load_questions()
    q = questions.get(args.qid)
    if not q:
        print(f"[错误] 未找到题目: {args.qid}")
        return

    print(f"QID: {q['qid']}")
    print(f"领域: {q['domain']}")
    print(f"题型: {q.get('type', '')} (answer_format={q.get('answer_format', '')})")
    print(f"文档ID: {q.get('doc_ids', [])}")
    print(f"\n问题: {q['question']}")
    print(f"\n选项:")
    for key, text in q.get('options', {}).items():
        print(f"  {key}. {text}")


def cmd_search(args):
    """执行检索"""
    domain_systems = load_indexes()
    questions = load_questions()

    # 确定查询参数
    if args.qid:
        q = questions.get(args.qid)
        if not q:
            print(f"[错误] 未找到题目: {args.qid}")
            return
        question = q['question']
        options = q.get('options', {})
        doc_ids = q.get('doc_ids', [])
        domain = q['domain']
    else:
        question = args.query or ""
        options = {}
        doc_ids = args.doc_ids.split(',') if args.doc_ids else None
        domain = args.domain or ""

    # 获取检索系统
    search_systems = _get_search_systems(domain, domain_systems)
    if not search_systems:
        print(f"[错误] 未找到领域检索系统: {domain}")
        return

    top_k = args.top_k or 8

    # 执行检索
    all_results = []
    seen = set()
    for sys_obj in search_systems:
        try:
            results = sys_obj.search(question, options, doc_ids=doc_ids, top_k=top_k)
            for r in results:
                key = (r.get('doc_id', ''), r.get('chunk_id', ''))
                if key not in seen:
                    seen.add(key)
                    all_results.append(r)
        except Exception as e:
            print(f"[警告] 检索失败: {e}", file=sys.stderr)

    # 按分数排序
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    all_results = all_results[:top_k]

    # 输出结果
    print(f"检索查询: {question[:80]}...")
    print(f"领域: {domain}, 文档: {doc_ids}, 检索系统: {len(search_systems)}个")
    print(f"结果数: {len(all_results)}\n")
    print("=" * 80)

    for i, r in enumerate(all_results, 1):
        text = r.get('text', '')
        # 截断显示
        if len(text) > 1500:
            text = text[:1500] + "..."
        print(f"\n--- 证据 {i}/{len(all_results)} ---")
        print(f"doc_id: {r.get('doc_id', '')}  chunk_id: {r.get('chunk_id', '')}  "
              f"score: {r.get('score', 0):.2f}  has_table: {r.get('has_table', False)}")
        print(f"内容:\n{text}")


def cmd_search_option(args):
    """针对单个选项检索"""
    domain_systems = load_indexes()
    questions = load_questions()

    q = questions.get(args.qid)
    if not q:
        print(f"[错误] 未找到题目: {args.qid}")
        return

    option_key = args.option.upper()
    options = q.get('options', {})
    if option_key not in options:
        print(f"[错误] 选项 {option_key} 不存在，可用选项: {list(options.keys())}")
        return

    option_text = options[option_key]
    question = q['question']
    doc_ids = q.get('doc_ids', [])
    domain = q['domain']

    # 构造针对该选项的查询
    # 提取问题关键词 + 选项文本
    q_keywords = [w for w in re.findall(r'[\u4e00-\u9fa5]{2,}', question) if len(w) >= 2][:4]
    query = " ".join(q_keywords[:2]) + " " + option_text

    # 获取检索系统
    search_systems = _get_search_systems(domain, domain_systems)
    top_k = args.top_k or 5

    all_results = []
    seen = set()
    for sys_obj in search_systems:
        try:
            # 使用选项文本作为查询
            results = sys_obj.search(query, {option_key: option_text}, doc_ids=doc_ids, top_k=top_k)
            for r in results:
                key = (r.get('doc_id', ''), r.get('chunk_id', ''))
                if key not in seen:
                    seen.add(key)
                    all_results.append(r)
        except Exception as e:
            print(f"[警告] 检索失败: {e}", file=sys.stderr)

    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    all_results = all_results[:top_k]

    print(f"选项 {option_key} 检索: {option_text[:80]}...")
    print(f"查询词: {query[:80]}...")
    print(f"结果数: {len(all_results)}\n")
    print("=" * 80)

    for i, r in enumerate(all_results, 1):
        text = r.get('text', '')
        if len(text) > 1500:
            text = text[:1500] + "..."
        print(f"\n--- 证据 {i}/{len(all_results)} ---")
        print(f"doc_id: {r.get('doc_id', '')}  score: {r.get('score', 0):.2f}")
        print(f"内容:\n{text}")


def cmd_doc(args):
    """查看文档原文（含关键词高亮）"""
    preparsed = load_preparsed()
    doc_data = preparsed.get(args.doc_id)
    if not doc_data:
        print(f"[错误] 未找到文档: {args.doc_id}")
        print(f"可用文档: {list(preparsed.keys())[:20]}...")
        return

    full_text = doc_data.get('full_text', '')
    if not full_text:
        # 拼接pages
        pages = doc_data.get('pages', [])
        full_text = '\n'.join(p.get('text', '') for p in pages)

    max_chars = args.max_chars or 3000
    keyword = args.keyword or ''

    if keyword:
        # 查找关键词位置，输出上下文
        positions = [m.start() for m in re.finditer(re.escape(keyword), full_text)]
        if not positions:
            print(f"文档 {args.doc_id} 中未找到关键词: {keyword}")
            print(f"文档总长度: {len(full_text)} 字符")
            return

        print(f"文档 {args.doc_id} 中找到 {len(positions)} 处关键词 '{keyword}'")
        print(f"显示前 {min(5, len(positions))} 处上下文:\n")
        print("=" * 80)

        for i, pos in enumerate(positions[:5], 1):
            start = max(0, pos - max_chars // 3)
            end = min(len(full_text), pos + max_chars * 2 // 3)
            snippet = full_text[start:end]
            print(f"\n--- 出现 {i}/{min(5, len(positions))} (位置 {pos}) ---")
            print(snippet)
    else:
        # 输出文档开头
        print(f"文档 {args.doc_id} (总长度: {len(full_text)} 字符)")
        print(f"页数: {doc_data.get('page_count', 0)}")
        print("=" * 80)
        print(full_text[:max_chars])


def cmd_fulltext(args):
    """全文搜索关键词"""
    domain_systems = load_indexes()
    keyword = args.keyword
    domain = args.domain
    top_k = args.top_k or 10

    # 确定搜索范围
    search_systems = []
    if domain:
        search_systems = _get_search_systems(domain, domain_systems)
    else:
        search_systems = list(domain_systems.values())

    results = []
    for sys_obj in search_systems:
        for chunk_id, data in sys_obj.chunk_data.items():
            text = data['text']
            if keyword in text:
                # 找到关键词位置
                pos = text.find(keyword)
                start = max(0, pos - 200)
                end = min(len(text), pos + 800)
                snippet = text[start:end]
                results.append({
                    'doc_id': data['doc_id'],
                    'chunk_id': chunk_id,
                    'position': pos,
                    'snippet': snippet,
                    'text_length': len(text),
                })

    # 去重（同一文档只保留前几个）
    seen_docs = {}
    for r in results:
        did = r['doc_id']
        if did not in seen_docs:
            seen_docs[did] = []
        seen_docs[did].append(r)

    print(f"全文搜索 '{keyword}'")
    print(f"匹配数: {len(results)}, 文档数: {len(seen_docs)}")
    print("=" * 80)

    count = 0
    for doc_id, items in seen_docs.items():
        if count >= top_k:
            break
        print(f"\n--- 文档 {doc_id} ({len(items)}处匹配) ---")
        for item in items[:2]:  # 每文档最多显示2处
            print(f"位置 {item['position']} (chunk {item['chunk_id']}):")
            print(item['snippet'])
            print()
        count += 1


def cmd_list_docs(args):
    """列出所有文档"""
    preparsed = load_preparsed()
    domain_filter = args.domain

    if domain_filter:
        # 按领域过滤
        domain_systems = load_indexes()
        system = domain_systems.get(domain_filter)
        if system:
            docs_in_domain = set()
            for chunk_id, data in system.chunk_data.items():
                docs_in_domain.add(data['doc_id'])
            print(f"领域 {domain_filter} 的文档 ({len(docs_in_domain)}个):")
            for doc_id in sorted(docs_in_domain):
                doc = preparsed.get(doc_id, {})
                page_count = doc.get('page_count', '?')
                print(f"  {doc_id}: {page_count}页, {doc.get('filename', '')}")
        else:
            print(f"未找到领域: {domain_filter}")
            print(f"可用领域: {list(domain_systems.keys())}")
    else:
        print(f"所有文档 ({len(preparsed)}个):")
        for doc_id in sorted(preparsed.keys()):
            doc = preparsed[doc_id]
            page_count = doc.get('page_count', '?')
            print(f"  {doc_id}: {page_count}页, {doc.get('filename', '')}")


def main():
    parser = argparse.ArgumentParser(description='检索工具 - 供子agent调用')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # question: 查看题目
    p_q = subparsers.add_parser('question', help='查看题目详情')
    p_q.add_argument('--qid', required=True, help='题目ID')

    # search: 基础检索
    p_s = subparsers.add_parser('search', help='执行BM25检索')
    p_s.add_argument('--qid', help='题目ID（自动填充问题和文档）')
    p_s.add_argument('--query', help='自定义查询词')
    p_s.add_argument('--domain', help='领域')
    p_s.add_argument('--doc_ids', help='文档ID（逗号分隔）')
    p_s.add_argument('--top_k', type=int, default=8, help='返回结果数')

    # search_option: 选项级检索
    p_so = subparsers.add_parser('search_option', help='针对单个选项检索')
    p_so.add_argument('--qid', required=True, help='题目ID')
    p_so.add_argument('--option', required=True, help='选项字母（A/B/C/D）')
    p_so.add_argument('--top_k', type=int, default=5, help='返回结果数')

    # doc: 查看文档原文
    p_d = subparsers.add_parser('doc', help='查看文档原文')
    p_d.add_argument('--doc_id', required=True, help='文档ID')
    p_d.add_argument('--keyword', help='关键词（高亮上下文）')
    p_d.add_argument('--max_chars', type=int, default=3000, help='最大显示字符数')

    # fulltext: 全文搜索
    p_f = subparsers.add_parser('fulltext', help='全文搜索关键词')
    p_f.add_argument('--keyword', required=True, help='搜索关键词')
    p_f.add_argument('--domain', help='限定领域')
    p_f.add_argument('--top_k', type=int, default=10, help='返回结果数')

    # list_docs: 列出文档
    p_l = subparsers.add_parser('list_docs', help='列出所有文档')
    p_l.add_argument('--domain', help='限定领域')

    args = parser.parse_args()

    if args.command == 'question':
        cmd_question(args)
    elif args.command == 'search':
        cmd_search(args)
    elif args.command == 'search_option':
        cmd_search_option(args)
    elif args.command == 'doc':
        cmd_doc(args)
    elif args.command == 'fulltext':
        cmd_fulltext(args)
    elif args.command == 'list_docs':
        cmd_list_docs(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
