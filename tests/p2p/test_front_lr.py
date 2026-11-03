# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""前段本机 LR（LR 选择器在含第 L₀ 层的前段卡上）：协调器直接用前段交来的判定。不要 GPU、不起引擎。"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from vllm.p2p.tiered import Engine, TieredCoordinator  # noqa: E402


def _co(**kw):
    fronts = [Engine("front-N09", "front", client=object())]
    backs = [Engine("back-gsm8k-0", "back", client=object(), task="gsm8k"),
             Engine("back-mbpp-0", "back", client=object(), task="mbpp")]
    return TieredCoordinator(fronts, backs, **kw)


def _req(feat, true_task="mbpp"):
    return SimpleNamespace(rid="r1", true_task=true_task, task=true_task, n_prompt=10,
                           front=SimpleNamespace(name="front-N09"), body=[[2, 8]], feat=feat,
                           info={})


def test_route_lr_needs_classify_or_front_lr():
    with pytest.raises(ValueError, match="front_lr"):
        _co(route="lr")
    co = _co(route="lr", front_lr=True)
    assert co.classify is None and co.front_lr


def test_uses_front_verdict_without_gateway_classifier():
    co = _co(route="lr", front_lr=True)
    r = _req({"verdict": {"task": "gsm8k", "conf": 0.93, "lr_ms_front": 0.4}})
    co._identify(r)
    v = r.info["verdict"]
    assert r.task == "gsm8k" and v["lr_at"] == "front" and v["conf"] == 0.93
    assert v["n_prompt"] == 10 and v["n_body"] == 6 and v["lr_ms_front"] == 0.4


def test_shadow_keeps_true_task():
    co = _co(route="task", front_lr=True)
    r = _req({"verdict": {"task": "gsm8k"}}, true_task="mbpp")
    co._identify(r)
    assert r.task == "mbpp" and r.info["verdict"]["task"] == "gsm8k"


def test_stats_without_any_classifier_is_an_error_not_a_hang():
    co = _co(route="lr", front_lr=True)
    r = _req({"1": {"counts": [1, 2]}})          # 前段没配 p2p_lr，交来的是统计
    co._identify(r)
    assert r.task is None and "p2p_lr" in r.info["verdict"]["error"]


def test_gateway_classifier_still_works():
    seen = {}

    def classify(feat, info):
        seen["feat"], seen["info"] = feat, info
        return {"task": "mbpp", "conf": 0.7}

    co = _co(route="lr", classify=classify)
    r = _req({"1": {"counts": [1, 2]}}, true_task="gsm8k")
    co._identify(r)
    assert r.task == "mbpp" and "lr_at" not in r.info["verdict"]
    assert seen["info"]["n_prompt"] == 10
