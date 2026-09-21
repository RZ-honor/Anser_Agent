"""
重建领域索引缓存脚本
====================
修复内容：
1. P0：补全 entity_index 构建（domain_retrieval.py 已修改）
2. P0：修复 phrase_index 上限不一致（new_retrieval_system.py 已修改）
3. P1：增加条款号原文索引（new_retrieval_system.py 已修改）

执行流程：
1. 删除旧缓存 domain_indexes_v2.pkl
2. 调用 build_all_domain_indexes 重建索引
3. 保存新缓存到 cache/domain_indexes_v2.pkl
4. 输出每个领域的索引统计（chunks/numeric/phrase/entity 数量）

执行环境：conda Audio
"""
import os
import sys
import pickle
from pathlib import Path

# 强制 UTF-8 输出（避免 Windows 控制台编码问题）
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_FILE = CACHE_DIR / "domain_indexes_v2.pkl"


def main():
    print("=" * 70)
    print("重建领域索引缓存（P0+P1 修复后）")
    print("=" * 70)

    # 1. 删除旧缓存（强制重建）
    if CACHE_FILE.exists():
        print(f"\n删除旧缓存: {CACHE_FILE}")
        os.remove(CACHE_FILE)
    else:
        print(f"\n旧缓存不存在: {CACHE_FILE}")

    # 2. 重建索引
    print("\n开始重建索引...")
    from domain_retrieval import build_all_domain_indexes
    systems = build_all_domain_indexes()

    # 3. 输出统计
    print("\n" + "=" * 70)
    print("索引构建统计")
    print("=" * 70)
    total_chunks = 0
    total_numeric = 0
    total_phrase = 0
    total_entity = 0
    for domain, system in systems.items():
        n_chunks = len(system.chunk_data)
        n_numeric = len(system.numeric_index)
        n_phrase = len(system.phrase_index)
        n_entity = len(system.entity_index)
        total_chunks += n_chunks
        total_numeric += n_numeric
        total_phrase += n_phrase
        total_entity += n_entity
        print(f"  {domain:25s}: chunks={n_chunks:6d}, numeric={n_numeric:7d}, "
              f"phrase={n_phrase:8d}, entity={n_entity:6d}")
    print(f"  {'总计':25s}: chunks={total_chunks:6d}, numeric={total_numeric:7d}, "
          f"phrase={total_phrase:8d}, entity={total_entity:6d}")

    # 4. 保存新缓存
    print(f"\n保存新缓存: {CACHE_FILE}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "domain_systems": systems,
        "build_info": {
            "total_chunks": total_chunks,
            "total_numeric": total_numeric,
            "total_phrase": total_phrase,
            "total_entity": total_entity,
        },
    }
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache_data, f)
    file_size = CACHE_FILE.stat().st_size / (1024 * 1024)
    print(f"缓存大小: {file_size:.2f} MB")
    print("\n重建完成！")


if __name__ == "__main__":
    main()
