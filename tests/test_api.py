"""自测：核心状态机 + API 行为（重复、迟到、越界、切换边界、低置信、封存）。"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, _sessions
from app.smoother import LanguageSmoother, SessionError


def strong(i: int, n: int, hi: float = 0.97) -> list[float]:
    v = [round((1.0 - hi) / (n - 1), 6)] * n
    v[i] = hi
    return v


@pytest.fixture(autouse=True)
def _clear_sessions():
    _sessions.clear()
    yield
    _sessions.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def make_session(client: TestClient, **overrides) -> dict:
    body = {
        "languages": ["zh", "en"],
        "frame_ms": 20,
        "switch_frames": 3,
        "posterior_threshold": 0.55,
        "switch_penalty": 2.0,
    }
    body.update(overrides)
    r = client.post("/sessions", json=body)
    assert r.status_code == 201, r.text
    return r.json()


# ------------------------------------------------------------------ 基本分段
def test_basic_build_switch_seal(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    rows = [strong(0, 2) for _ in range(5)] + [strong(1, 2) for _ in range(5)]
    r = client.post(f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": rows})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] is True
    assert body["duplicate"] is False
    assert body["current_language"] == "en"
    assert body["next_seq"] == 10
    # 已关闭区间：zh 0..4；en 段仍在延伸。
    assert body["segments"] == [
        {
            "language": "zh",
            "start_frame": 0,
            "end_frame": 4,
            "start_ms": 0,
            "end_ms": 100,
            "duration_ms": 100,
        }
    ]
    assert body["open_segment"]["language"] == "en"
    assert body["open_segment"]["start_frame"] == 5
    assert body["open_segment"]["end_frame"] == 9

    seal = client.post(f"/sessions/{sid}/seal")
    assert seal.status_code == 200
    segs = seal.json()["segments"]
    assert [(x["language"], x["start_frame"], x["end_frame"]) for x in segs] == [
        ("zh", 0, 4),
        ("en", 5, 9),
    ]
    # 封存后新块被拒。
    late = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 10, "posteriors": [strong(0, 2)]}
    )
    assert late.status_code == 409
    assert late.json()["error"]["code"] == "session_sealed"


def test_final_flag_seals_and_segments_endpoint(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    r = client.post(
        f"/sessions/{sid}/chunks",
        json={
            "start_seq": 0,
            "posteriors": [strong(0, 2) for _ in range(4)],
            "final": True,
        },
    )
    assert r.status_code == 200
    assert r.json()["sealed"] is True
    assert r.json()["open_segment"] is None
    assert [(x["language"], x["start_frame"], x["end_frame"]) for x in r.json()["segments"]] == [
        ("zh", 0, 3)
    ]
    segs = client.get(f"/sessions/{sid}/segments").json()
    assert segs["sealed"] is True
    assert [(x["language"], x["start_frame"], x["end_frame"]) for x in segs["segments"]] == [
        ("zh", 0, 3)
    ]


# ------------------------------------------------------------- 切换边界
def test_switch_needs_exactly_threshold_consecutive_frames(client: TestClient):
    s = make_session(client, switch_frames=3)
    sid = s["session_id"]
    rows = [strong(0, 2) for _ in range(4)] + [strong(1, 2) for _ in range(2)]
    r = client.post(f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": rows})
    body = r.json()
    # 只有 2 帧 en：未达阈值，不切段，仍归 zh。
    assert body["current_language"] == "zh"
    assert body["pending_language"] == "en"
    assert body["pending_frames"] == 2
    assert body["segments"] == []
    assert body["open_segment"]["language"] == "zh"
    assert body["open_segment"]["end_frame"] == 5

    # 第 3 帧 en 到达：切段，切点回退到 en 首帧（4）。
    r2 = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 6, "posteriors": [strong(1, 2)]}
    )
    body2 = r2.json()
    assert body2["current_language"] == "en"
    assert body2["segments"][-1] == {
        "language": "zh",
        "start_frame": 0,
        "end_frame": 3,
        "start_ms": 0,
        "end_ms": 80,
        "duration_ms": 80,
    }
    assert body2["open_segment"]["language"] == "en"
    assert body2["open_segment"]["start_frame"] == 4
    assert body2["open_segment"]["end_frame"] == 6


def test_single_frame_jitter_does_not_switch(client: TestClient):
    s = make_session(client, switch_frames=3)
    sid = s["session_id"]
    rows = (
        [strong(0, 2) for _ in range(4)]
        + [strong(1, 2)]
        + [strong(0, 2) for _ in range(2)]
        + [strong(1, 2)]
        + [strong(0, 2) for _ in range(2)]
    )
    body = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": rows}
    ).json()
    assert body["current_language"] == "zh"
    assert body["segments"] == []
    assert body["open_segment"]["language"] == "zh"
    assert body["open_segment"]["end_frame"] == len(rows) - 1


def test_low_confidence_frames_stay_in_previous_state(client: TestClient):
    s = make_session(client, switch_frames=3, posterior_threshold=0.8)
    sid = s["session_id"]
    # 0.6/0.4 的最大后验 0.6 < 0.8：低置信。
    low = [0.6, 0.4]
    rows = [strong(0, 2) for _ in range(4)] + [low, low, low, low]
    body = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": rows}
    ).json()
    assert body["current_language"] == "zh"
    assert body["pending_language"] is None
    assert body["segments"] == []  # 尚未发生切换
    assert body["open_segment"]["end_frame"] == 7  # 低置信帧并入当前段


def test_low_confidence_breaks_consecutive_run(client: TestClient):
    s = make_session(client, switch_frames=3, posterior_threshold=0.8)
    sid = s["session_id"]
    low = [0.3, 0.7]  # max 0.7 < 0.8 -> 低置信，且原始 argmax 是 en
    rows = [strong(0, 2) for _ in range(4)] + [strong(1, 2), strong(1, 2), low, strong(1, 2)]
    body = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": rows}
    ).json()
    # 两帧 en + 低置信打断 + 一帧 en：未连续达 3 帧。
    assert body["current_language"] == "zh"
    assert body["pending_frames"] == 1


# ------------------------------------------------------------------ 重复块
def test_duplicate_chunk_returns_original_result_and_keeps_state(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    chunk = {"start_seq": 0, "posteriors": [strong(0, 2) for _ in range(5)]}
    first = client.post(f"/sessions/{sid}/chunks", json=chunk).json()
    state_after_first = client.get(f"/sessions/{sid}").json()

    dup = client.post(f"/sessions/{sid}/chunks", json=chunk)
    assert dup.status_code == 200
    d = dup.json()
    assert d["duplicate"] is True
    assert d["next_seq"] == first["next_seq"]
    assert d["segments"] == first["segments"]
    assert d["open_segment"] == first["open_segment"]
    # 会话状态未被改动。
    assert client.get(f"/sessions/{sid}").json() == state_after_first

    # 后续块仍从正确的序号继续。
    nxt = client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 5, "posteriors": [strong(1, 2)]},
    )
    assert nxt.status_code == 200
    assert nxt.json()["next_seq"] == 6


def test_duplicate_with_different_content_conflicts(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 0, "posteriors": [strong(0, 2) for _ in range(3)]},
    )
    before = client.get(f"/sessions/{sid}").json()
    bad = client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 0, "posteriors": [strong(1, 2) for _ in range(3)]},
    )
    assert bad.status_code == 409
    assert bad.json()["error"]["code"] == "chunk_conflict"
    assert client.get(f"/sessions/{sid}").json() == before


def test_duplicate_after_seal_still_returns_original(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    chunk = {"start_seq": 0, "posteriors": [strong(0, 2) for _ in range(4)]}
    first = client.post(f"/sessions/{sid}/chunks", json=chunk).json()
    client.post(f"/sessions/{sid}/seal")
    dup = client.post(f"/sessions/{sid}/chunks", json=chunk)
    assert dup.status_code == 200
    assert dup.json()["duplicate"] is True
    assert dup.json()["sealed"] is True
    assert dup.json()["segments"] == first["segments"]


# ------------------------------------------------- 带 final 的末块重送（回归）
def _final_last_chunk_session(client: TestClient) -> tuple[str, dict, dict]:
    """两段会话：zh 0..4，en 5..9；末块带 final 封存。返回 (sid, 末块, 首次回执)。"""
    s = make_session(client)
    sid = s["session_id"]
    client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 0, "posteriors": [strong(0, 2) for _ in range(5)]},
    )
    last = {
        "start_seq": 5,
        "posteriors": [strong(1, 2) for _ in range(5)],
        "final": True,
    }
    first = client.post(f"/sessions/{sid}/chunks", json=last)
    assert first.status_code == 200, first.text
    assert first.json()["sealed"] is True
    return sid, last, first.json()


def test_final_last_chunk_resend_matches_first_receipt_and_seal_query(client: TestClient):
    sid, last, first = _final_last_chunk_session(client)
    # 首次回执：末段已封存进 segments，open_segment 为 null。
    assert first["duplicate"] is False
    assert first["open_segment"] is None
    assert [(x["language"], x["start_frame"], x["end_frame"]) for x in first["segments"]] == [
        ("zh", 0, 4),
        ("en", 5, 9),
    ]

    dup = client.post(f"/sessions/{sid}/chunks", json=last)
    assert dup.status_code == 200, dup.text
    d = dup.json()
    assert d["duplicate"] is True
    assert d["sealed"] is True
    # 重送回执与首次回执逐字段一致，末块区间不会退化为临时尾段。
    for key in ("start_seq", "end_seq", "segments", "open_segment", "next_seq"):
        assert d[key] == first[key], key
    assert d["open_segment"] is None

    # 与封存查询（GET /segments）完全一致。
    sealed = client.get(f"/sessions/{sid}/segments").json()
    assert d["segments"] == sealed["segments"]
    assert sealed["sealed"] is True

    # 会话状态没有被重送改变。
    again = client.get(f"/sessions/{sid}/segments").json()
    assert again == sealed


def test_same_seq_only_final_changed_returns_409_and_keeps_state(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    chunk = {"start_seq": 0, "posteriors": [strong(0, 2) for _ in range(4)]}
    client.post(f"/sessions/{sid}/chunks", json=chunk)
    before = client.get(f"/sessions/{sid}").json()

    conflict = client.post(f"/sessions/{sid}/chunks", json={**chunk, "final": True})
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "chunk_conflict"
    after = client.get(f"/sessions/{sid}").json()
    assert after == before
    assert after["sealed"] is False  # 冲突请求没有顺手封存

    # 反向：同一 final 块的 final 改为 false 也是冲突，且封存状态不变。
    sid2, last, _ = _final_last_chunk_session(client)
    sealed_before = client.get(f"/sessions/{sid2}/segments").json()
    conflict2 = client.post(f"/sessions/{sid2}/chunks", json={**last, "final": False})
    assert conflict2.status_code == 409
    assert conflict2.json()["error"]["code"] == "chunk_conflict"
    assert client.get(f"/sessions/{sid2}/segments").json() == sealed_before


def test_normal_chunk_resend_returns_receipt_as_of_then(client: TestClient):
    s = make_session(client, switch_frames=3)
    sid = s["session_id"]
    chunk0 = {"start_seq": 0, "posteriors": [strong(0, 2) for _ in range(5)]}
    chunk1 = {"start_seq": 5, "posteriors": [strong(1, 2)]}
    chunk2 = {"start_seq": 6, "posteriors": [strong(1, 2) for _ in range(2)]}
    first0 = client.post(f"/sessions/{sid}/chunks", json=chunk0).json()
    first1 = client.post(f"/sessions/{sid}/chunks", json=chunk1).json()
    client.post(f"/sessions/{sid}/chunks", json=chunk2)

    for chunk, first in ((chunk0, first0), (chunk1, first1)):
        dup = client.post(f"/sessions/{sid}/chunks", json=chunk)
        assert dup.status_code == 200, dup.text
        d = dup.json()
        assert d["duplicate"] is True
        assert d["segments"] == first["segments"]
        assert d["open_segment"] == first["open_segment"]
        assert d["current_language"] == first["current_language"]
        assert d["pending_language"] == first["pending_language"]
        assert d["pending_frames"] == first["pending_frames"]
        assert d["next_seq"] == first["next_seq"]

    # 当时 chunk0 还没有任何已关闭区间、尾段是 zh 0..4；chunk1 时尾段延伸到 5。
    assert first0["segments"] == [] and first0["open_segment"]["end_frame"] == 4
    assert first1["open_segment"]["end_frame"] == 5 and first1["next_seq"] == 6
    # 但重送时当前已关闭区间是 zh 0..4（切点回退到 4），旧回执不应被改写。
    state = client.get(f"/sessions/{sid}").json()
    assert [(x["language"], x["start_frame"], x["end_frame"]) for x in state["segments"]] == [
        ("zh", 0, 4)
    ]


def test_old_receipt_keeps_its_tail_when_later_switch_regresses_cut(client: TestClient):
    """确认切换让旧段切点回退（5 帧 jitter 后的 en 只回溯 3 帧），旧回执仍保留尾巴。"""
    s = make_session(client, switch_frames=3)
    sid = s["session_id"]
    chunks = [
        {"start_seq": 0, "posteriors": [strong(0, 2) for _ in range(5)]},
        {"start_seq": 5, "posteriors": [strong(1, 2)]},
        {"start_seq": 6, "posteriors": [strong(1, 2) for _ in range(2)]},
    ]
    first1 = client.post(f"/sessions/{sid}/chunks", json=chunks[0]).json()
    first1 = client.post(f"/sessions/{sid}/chunks", json=chunks[1]).json()
    # 当时 en 还是候选、zh 尾段延伸到第 5 帧。
    assert first1["pending_language"] == "en" and first1["pending_frames"] == 1
    assert first1["open_segment"]["language"] == "zh"
    assert first1["open_segment"]["end_frame"] == 5

    after = client.post(f"/sessions/{sid}/chunks", json=chunks[2]).json()
    # 确认切换：zh 段在候选起点前一帧关闭（4 < 旧回执的 5），切点回退了。
    assert [(x["language"], x["start_frame"], x["end_frame"]) for x in after["segments"]] == [
        ("zh", 0, 4)
    ]
    assert after["open_segment"]["language"] == "en"
    assert after["open_segment"]["start_frame"] == 5

    dup = client.post(f"/sessions/{sid}/chunks", json=chunks[1]).json()
    assert dup["duplicate"] is True
    assert dup["open_segment"] == first1["open_segment"]
    assert dup["open_segment"]["end_frame"] == 5  # 不被后来回退后的 4 覆盖
    assert dup["segments"] == first1["segments"] == []
    assert dup["pending_frames"] == 1
    assert dup["next_seq"] == 6


def test_resend_does_not_rerun_hmm(client: TestClient, monkeypatch):
    s = make_session(client)
    sid = s["session_id"]
    chunk = {"start_seq": 0, "posteriors": [strong(0, 2) for _ in range(5)]}
    client.post(f"/sessions/{sid}/chunks", json=chunk)
    smoother = _sessions[sid]
    calls = []
    original = smoother._forward
    monkeypatch.setattr(
        smoother, "_forward", lambda row: (calls.append(1), original(row))[1]
    )
    alpha_before = smoother._log_alpha.copy()

    dup = client.post(f"/sessions/{sid}/chunks", json=chunk)
    assert dup.status_code == 200 and dup.json()["duplicate"] is True
    assert calls == []  # 重送走缓存，不重跑前向滤波
    assert np.array_equal(smoother._log_alpha, alpha_before)
    assert smoother.next_seq == 5


def test_session_memory_grows_linearly_with_chunks_plus_closed_segments():
    """单会话为分段与回执保留的内存随 (已收块数 + 已关闭区间数) 线性增长。

    既统计可达的独立 Segment 对象数，也统计容器里指向 Segment 的**引用槽**数：
    若每条回执都复制/引用整份区间列表，引用槽会随两者之积（平方级）增长。
    """
    import gc

    from app.smoother import Segment

    def segment_stats(obj) -> tuple[int, int]:
        seen, stack = set(), [obj]
        unique, slots = 0, 0
        while stack:
            cur = stack.pop()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            if type(cur) is Segment:
                unique += 1
                continue
            if isinstance(cur, (list, tuple)):
                slots += sum(1 for item in cur if type(item) is Segment)
            stack.extend(gc.get_referents(cur))
        return unique, slots

    num_chunks = 40
    params = dict(frame_ms=20, switch_frames=2, posterior_threshold=0.55)
    sm = LanguageSmoother("mem", ["zh", "en"], **params)
    # 每块 4 帧、方向每块翻转：块与块之间都发生一次确认切换 → 已关闭区间 ~ 块数。
    for i in range(num_chunks):
        rows = np.array([strong(i % 2, 2) for _ in range(4)])
        sm.submit_chunk(i * 4, rows)
    closed = len(sm._segments)
    assert closed >= num_chunks // 2  # 确实有大量已关闭区间
    assert len(sm._chunk_cache) == num_chunks

    unique, slots = segment_stats(sm)
    assert unique <= closed + num_chunks + 3, (unique, closed, num_chunks)
    assert slots <= closed + num_chunks + 3, (slots, closed, num_chunks)
    # 朴素实现每条回执保存整份列表时，引用槽约为 sum(i) ≈ 780；这里必须远小于。
    assert slots < num_chunks * (closed / 2)


# ------------------------------------------------------------- 迟到 / 越界
def test_late_chunk_rejected_without_state_change(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 0, "posteriors": [strong(0, 2) for _ in range(4)]},
    )
    before = client.get(f"/sessions/{sid}").json()
    late = client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 2, "posteriors": [strong(1, 2) for _ in range(2)]},
    )
    assert late.status_code == 409
    assert late.json()["error"]["code"] == "late_chunk"
    assert client.get(f"/sessions/{sid}").json() == before


def test_out_of_order_chunk_rejected_without_state_change(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    before = client.get(f"/sessions/{sid}").json()
    gap = client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 3, "posteriors": [strong(0, 2)]},
    )
    assert gap.status_code == 409
    assert gap.json()["error"]["code"] == "out_of_order"
    assert client.get(f"/sessions/{sid}").json() == before

    # 缺口被拒后，正确序号仍可提交。
    ok = client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 0, "posteriors": [strong(0, 2)]},
    )
    assert ok.status_code == 200


# ------------------------------------------------------------- 增量一致性
def test_frame_by_frame_equals_single_chunk():
    rows = [
        strong(0, 3),
        strong(0, 3),
        strong(1, 3),
        strong(0, 3),
        strong(1, 3),
        strong(1, 3),
        strong(1, 3),
        strong(2, 3),
        strong(2, 3),
        strong(2, 3),
        strong(2, 3),
    ]
    params = dict(frame_ms=20, switch_frames=3, posterior_threshold=0.55, switch_penalty=2.0)
    one = LanguageSmoother("a", ["zh", "en", "ja"], **params)
    one.submit_chunk(0, np.array(rows))
    one_segs = [s.to_dict() for s in one.seal()]

    many = LanguageSmoother("b", ["zh", "en", "ja"], **params)
    for i, row in enumerate(rows):
        many.submit_chunk(i, np.array([row]))
    many_segs = [s.to_dict() for s in many.seal()]
    assert one_segs == many_segs


def test_chunk_splits_do_not_change_boundaries():
    rows = [strong(0, 2) for _ in range(4)] + [strong(1, 2) for _ in range(6)]
    params = dict(frame_ms=20, switch_frames=3, posterior_threshold=0.55, switch_penalty=2.0)
    one = LanguageSmoother("a", ["zh", "en"], **params)
    one.submit_chunk(0, np.array(rows))
    one_segs = [s.to_dict() for s in one.seal()]

    many = LanguageSmoother("b", ["zh", "en"], **params)
    many.submit_chunk(0, np.array(rows[:2]))
    many.submit_chunk(2, np.array(rows[2:5]))
    many.submit_chunk(5, np.array(rows[5:]))
    many_segs = [s.to_dict() for s in many.seal()]
    assert one_segs == many_segs


# ------------------------------------------------------------- 校验与错误
def test_session_parameter_validation(client: TestClient):
    one = client.post("/sessions", json={"languages": ["zh"]})
    assert one.status_code == 422
    five = client.post("/sessions", json={"languages": ["a", "b", "c", "d", "e"]})
    assert five.status_code == 422
    dup = client.post("/sessions", json={"languages": ["zh", "zh"]})
    assert dup.status_code == 422
    bad_thr = client.post("/sessions", json={"languages": ["zh", "en"], "posterior_threshold": 0})
    assert bad_thr.status_code == 422
    bad_frames = client.post("/sessions", json={"languages": ["zh", "en"], "switch_frames": 0})
    assert bad_frames.status_code == 422


def test_chunk_validation(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    wrong_width = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": [[0.5, 0.3, 0.2]]}
    )
    assert wrong_width.status_code == 422
    assert wrong_width.json()["error"]["code"] == "dimension_mismatch"

    neg = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": [[1.2, -0.2]]}
    )
    assert neg.status_code == 422
    assert neg.json()["error"]["code"] == "negative_probability"

    zero = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": [[0.0, 0.0]]}
    )
    assert zero.status_code == 422
    assert zero.json()["error"]["code"] == "invalid_distribution"

    empty = client.post(f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": []})
    assert empty.status_code == 422

    # 失败的请求没有推进会话。
    assert client.get(f"/sessions/{sid}").json()["next_seq"] == 0


def test_unknown_session(client: TestClient):
    r = client.post("/sessions/nope/chunks", json={"start_seq": 0, "posteriors": [[0.9, 0.1]]})
    assert r.status_code == 404
    assert client.get("/sessions/nope").status_code == 404


def test_direct_smoother_rejects_bad_seq():
    sm = LanguageSmoother("x", ["zh", "en"], 20, 3, 0.55)
    with pytest.raises(SessionError) as e:
        sm.submit_chunk(-1, np.array([[0.9, 0.1]]))
    assert e.value.code == "invalid_seq"
    with pytest.raises(SessionError) as e2:
        sm.submit_chunk(1, np.array([[0.9, 0.1]]))
    assert e2.value.code == "out_of_order"
