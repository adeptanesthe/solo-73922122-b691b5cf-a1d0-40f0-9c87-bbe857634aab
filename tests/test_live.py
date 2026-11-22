"""可选的真实 HTTP 端到端测试。

默认跳过；设置 SEGMENTER_BASE_URL 后运行（例如 docker compose 的 tests 服务）：
    SEGMENTER_BASE_URL=http://127.0.0.1:8000 python -m pytest tests/test_live.py -q
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("SEGMENTER_BASE_URL")

pytestmark = pytest.mark.skipif(
    not BASE, reason="set SEGMENTER_BASE_URL to run live HTTP tests"
)


def _post(path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _get(path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=15) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _frame(i: int, n: int = 2, hi: float = 0.95) -> list[float]:
    row = [round((1.0 - hi) / (n - 1), 6)] * n
    row[i] = hi
    return row


def test_live_end_to_end():
    status, _ = _get("/healthz")
    assert status == 200

    status, session = _post(
        "/sessions",
        {
            "languages": ["zh", "en"],
            "frame_ms": 20,
            "switch_frames": 3,
            "posterior_threshold": 0.55,
        },
    )
    assert status == 201, session
    sid = session["session_id"]

    chunk = {"start_seq": 0, "posteriors": [ _frame(0) for _ in range(5)]}
    status, first = _post(f"/sessions/{sid}/chunks", chunk)
    assert status == 200, first
    assert first["next_seq"] == 5

    # 重复块返回原结果且不推进。
    status, dup = _post(f"/sessions/{sid}/chunks", chunk)
    assert status == 200 and dup["duplicate"] is True
    assert dup["next_seq"] == 5

    # 迟到块被拒且不改变会话。
    status, late = _post(
        f"/sessions/{sid}/chunks",
        {"start_seq": 2, "posteriors": [_frame(1)]},
    )
    assert status == 409 and late["error"]["code"] == "late_chunk"
    status, state = _get(f"/sessions/{sid}")
    assert state["next_seq"] == 5 and state["current_language"] == "zh"

    # 切到 en：3 帧达阈值。
    status, switched = _post(
        f"/sessions/{sid}/chunks",
        {"start_seq": 5, "posteriors": [_frame(1) for _ in range(3)], "final": True},
    )
    assert status == 200, switched
    assert switched["sealed"] is True
    assert switched["current_language"] == "en"
    assert [(s["language"], s["start_frame"], s["end_frame"]) for s in switched["segments"]] == [
        ("zh", 0, 4),
        ("en", 5, 7),
    ]

    # 封存后新块 409。
    status, after = _post(
        f"/sessions/{sid}/chunks",
        {"start_seq": 8, "posteriors": [_frame(0)]},
    )
    assert status == 409 and after["error"]["code"] == "session_sealed"

    # 封存后的最终区间完整覆盖全部帧。
    status, final = _get(f"/sessions/{sid}/segments")
    assert status == 200
    assert [(s["language"], s["start_frame"], s["end_frame"]) for s in final["segments"]] == [
        ("zh", 0, 4),
        ("en", 5, 7),
    ]


def test_live_final_chunk_resend_and_normal_chunk_resend():
    status, _ = _get("/healthz")
    assert status == 200

    status, session = _post(
        "/sessions",
        {
            "languages": ["zh", "en"],
            "frame_ms": 20,
            "switch_frames": 3,
            "posterior_threshold": 0.55,
        },
    )
    assert status == 201, session
    sid = session["session_id"]

    first_chunk = {"start_seq": 0, "posteriors": [_frame(0) for _ in range(5)]}
    status, first = _post(f"/sessions/{sid}/chunks", first_chunk)
    assert status == 200, first

    last = {"start_seq": 5, "posteriors": [_frame(1) for _ in range(5)], "final": True}
    status, first_final = _post(f"/sessions/{sid}/chunks", last)
    assert status == 200, first_final
    assert first_final["sealed"] is True and first_final["open_segment"] is None
    expected = [("zh", 0, 4), ("en", 5, 9)]
    assert [(s["language"], s["start_frame"], s["end_frame"]) for s in first_final["segments"]] == expected

    # 末块原样重送：与首次回执、封存查询都一致，open_segment 为 null。
    status, dup_final = _post(f"/sessions/{sid}/chunks", last)
    assert status == 200, dup_final
    assert dup_final["duplicate"] is True and dup_final["sealed"] is True
    assert dup_final["open_segment"] is None
    for key in ("segments", "start_seq", "end_seq", "next_seq"):
        assert dup_final[key] == first_final[key], key
    status, sealed = _get(f"/sessions/{sid}/segments")
    assert sealed["segments"] == dup_final["segments"]

    # 同序号仅改 final → 409，封存状态与区间不变。
    status, conflict = _post(f"/sessions/{sid}/chunks", {**last, "final": False})
    assert status == 409 and conflict["error"]["code"] == "chunk_conflict"
    status, sealed_again = _get(f"/sessions/{sid}/segments")
    assert sealed_again == sealed

    # 封存后重送普通块：返回当时的回执，sealed 报告当前状态。
    status, dup_first = _post(f"/sessions/{sid}/chunks", first_chunk)
    assert status == 200 and dup_first["duplicate"] is True
    assert dup_first["sealed"] is True
    assert dup_first["segments"] == first["segments"] == []
    assert dup_first["open_segment"] == first["open_segment"]

    # 另起会话验证普通块重送 + 后续切点回退不改写旧回执。
    status, session2 = _post(
        "/sessions",
        {"languages": ["zh", "en"], "frame_ms": 20, "switch_frames": 3},
    )
    sid2 = session2["session_id"]
    c0 = {"start_seq": 0, "posteriors": [_frame(0) for _ in range(5)]}
    c1 = {"start_seq": 5, "posteriors": [_frame(1)]}
    c2 = {"start_seq": 6, "posteriors": [_frame(1), _frame(1)]}
    _, r0 = _post(f"/sessions/{sid2}/chunks", c0)
    _, r1 = _post(f"/sessions/{sid2}/chunks", c1)
    _, r2 = _post(f"/sessions/{sid2}/chunks", c2)
    assert r1["open_segment"]["end_frame"] == 5
    assert r2["segments"][-1]["end_frame"] == 4  # 切点回退

    status, d1 = _post(f"/sessions/{sid2}/chunks", c1)
    assert status == 200 and d1["duplicate"] is True
    assert d1["segments"] == r1["segments"] == []
    assert d1["open_segment"] == r1["open_segment"]
    assert d1["pending_language"] == "en" and d1["pending_frames"] == 1
    assert d1["next_seq"] == 6

    status, d0 = _post(f"/sessions/{sid2}/chunks", c0)
    assert status == 200 and d0["duplicate"] is True
    assert d0["segments"] == r0["segments"] == []
    assert d0["open_segment"] == r0["open_segment"]
