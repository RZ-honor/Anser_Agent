# 四基线批量运行器：统一模型串行跑完 bare_llm → symbolic → naive_vector → full_system
#
# 用途：保证四个基线使用同一问答模型（对比才有效），失败题可再用
# run_eval.py --mode X --retry-failed 兜底。
# 用法：python eval/run_all_baselines.py
import os
import subprocess
import sys

PY = sys.executable
EVAL_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_eval.py")


def run_mode(mode: str, vector_enabled: str) -> int:
    """运行单个基线模式；vector_enabled 控制向量兜底开关（naive_vector 必须开）"""
    env = dict(os.environ, VECTOR_RECALL_ENABLED=vector_enabled)
    print(f"\n{'=' * 20} 开始 {mode}（VECTOR_RECALL_ENABLED={vector_enabled}）{'=' * 20}",
          flush=True)
    r = subprocess.run([PY, "-u", EVAL_RUNNER, "--mode", mode, "--max", "50"], env=env)
    print(f"{'=' * 20} {mode} 结束 exit={r.returncode} {'=' * 20}\n", flush=True)
    return r.returncode


if __name__ == "__main__":
    total_rc = 0
    # 顺序：快的先跑，full_system 最慢放最后
    total_rc |= run_mode("bare_llm", "0")
    total_rc |= run_mode("symbolic_single_turn", "0")
    total_rc |= run_mode("naive_vector", "1")
    total_rc |= run_mode("full_system", "0")
    sys.exit(total_rc)
