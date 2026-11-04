# SPDX-License-Identifier: Apache-2.0
"""bench.py --synthetic 的合成请求。"""
import importlib.util
import sys
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "vllm" / "p2p" / "bench.py"
_spec = importlib.util.spec_from_file_location("p2p_bench", _P)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)


def test_synthetic_cases_shape():
    tasks = ["gsm8k", "mbpp", "no_robots"]
    cases = bench.synthetic_cases(tasks, 5, 128, seed=1)
    assert len(cases) == 15
    assert [u for u, _ in cases] == [u for u in tasks for _ in range(5)]
    prompts = [p for _, p in cases]
    assert len(set(prompts)) == 15                         # 各条不同：不命中前缀缓存
    assert all(100 <= len(p.split()) <= 140 for p in prompts)


def test_synthetic_cases_deterministic():
    assert bench.synthetic_cases(["x"], 3, 64, seed=7) == bench.synthetic_cases(["x"], 3, 64, seed=7)
    assert bench.synthetic_cases(["x"], 3, 64, seed=7) != bench.synthetic_cases(["x"], 3, 64, seed=8)
