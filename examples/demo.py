"""端到端调用示例：创建会话 → 分块提交 → 重复块 → 迟到块 → 封存 → 重送语义。

先启动服务（二选一）：
    docker compose up --build
    # 或本地：
    uvicorn app.main:app --port 8000

再运行：
    python examples/demo.py [base_url]

只依赖标准库，便于在容器外直接执行。末尾 A/B 两段分别演示带 final 的末块
原样重送（与首次回执、封存查询一致，open_segment 为 null；仅改 final 返回
409）和普通块重送（返回当时的已关闭区间、临时尾段与切换进度，后续切点回退
不改写旧回执）。
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
LANGUAGES = ["zh", "en", "ja"]


def post(path: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def get(path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def frame(lang_idx: int, hi: float = 0.95) -> list[float]:
    n = len(LANGUAGES)
    row = [round((1.0 - hi) / (n - 1), 6)] * n
    row[lang_idx] = hi
    return row


def frame_n(lang_idx: int, n: int, hi: float = 0.95) -> list[float]:
    row = [round((1.0 - hi) / (n - 1), 6)] * n
    row[lang_idx] = hi
    return row


def show(label: str, status: int, body: dict) -> None:
    summary = {
        k: body.get(k)
        for k in (
            "session_id",
            "languages",
            "frame_ms",
            "switch_frames",
            "posterior_threshold",
            "switch_penalty",
            "accepted",
            "duplicate",
            "sealed",
            "current_language",
            "pending_language",
            "pending_frames",
            "next_seq",
            "segments",
            "open_segment",
            "error",
        )
        if k in body
    }
    print(f"\n[{label}] HTTP {status}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    status, session = post(
        "/sessions",
        {
            "languages": LANGUAGES,
            "frame_ms": 20,
            "switch_frames": 3,
            "posterior_threshold": 0.55,
            "switch_penalty": 2.0,
        },
    )
    show("create session", status, session)
    sid = session["session_id"]

    # 块 0：5 帧 zh，随后 1 帧 en 抖动，再 2 帧 zh —— 抖动不应切段。
    chunk0 = {
        "start_seq": 0,
        "posteriors": [frame(0) for _ in range(5)]
        + [frame(1)]
        + [frame(0) for _ in range(2)],
    }
    status, body = post(f"/sessions/{sid}/chunks", chunk0)
    show("chunk 0 (zh, jitter, zh)", status, body)

    # 重复提交块 0：应返回原结果（duplicate=true），会话不变。
    status, dup = post(f"/sessions/{sid}/chunks", chunk0)
    show("duplicate chunk 0", status, dup)

    # 块 1：3 帧 en（达到切换阈值）+ 4 帧 en。
    status, body = post(
        f"/sessions/{sid}/chunks",
        {"start_seq": 8, "posteriors": [frame(1) for _ in range(7)]},
    )
    show("chunk 1 (en x7)", status, body)

    # 迟到块：起始序号 6 < 期望序号 15，应被拒且不改变会话。
    status, late = post(
        f"/sessions/{sid}/chunks",
        {"start_seq": 6, "posteriors": [frame(0) for _ in range(2)]},
    )
    show("late chunk (start 6)", status, late)

    # 越界块：起始序号 20 > 期望序号 15，同样被拒。
    status, gap = post(
        f"/sessions/{sid}/chunks",
        {"start_seq": 20, "posteriors": [frame(0)]},
    )
    show("out-of-order chunk (start 20)", status, gap)

    # 块 2：5 帧 ja，后 2 帧低置信（0.6/0.2/0.2 最大后验 0.6 达标；这里用 0.45 低置信）。
    low = [0.45, 0.30, 0.25]
    status, body = post(
        f"/sessions/{sid}/chunks",
        {
            "start_seq": 15,
            "posteriors": [frame(2) for _ in range(5)] + [low, low],
        },
    )
    show("chunk 2 (ja x5 + low-confidence x2)", status, body)

    # 封存。
    status, sealed = post(f"/sessions/{sid}/seal")
    show("seal", status, sealed)

    # 封存后提交新块：应 409。
    status, after = post(
        f"/sessions/{sid}/chunks",
        {"start_seq": 22, "posteriors": [frame(0)]},
    )
    show("submit after seal", status, after)

    status, final = get(f"/sessions/{sid}/segments")
    print("\n[final segments]")
    print(json.dumps(final, ensure_ascii=False, indent=2))

    demo_final_resend()
    demo_normal_chunk_resend()


def _show_receipt(label: str, status: int, body: dict) -> None:
    """只打印回执里与重送语义相关的字段，便于对照。"""
    print(f"\n[{label}] HTTP {status}")
    print(
        json.dumps(
            {
                k: body.get(k)
                for k in (
                    "start_seq",
                    "end_seq",
                    "accepted",
                    "duplicate",
                    "sealed",
                    "segments",
                    "open_segment",
                    "current_language",
                    "pending_language",
                    "pending_frames",
                    "next_seq",
                    "error",
                )
                if k in body
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def demo_final_resend() -> None:
    """带 final 的末块重送：回执、封存查询三者一致，且不把末段标成临时尾段。"""
    print("\n" + "=" * 70)
    print("A. 带 final 的末块原样重送")
    print("=" * 70)
    status, session = post(
        "/sessions",
        {
            "languages": ["zh", "en"],
            "frame_ms": 20,
            "switch_frames": 3,
            "posterior_threshold": 0.55,
        },
    )
    sid = session["session_id"]
    chunk_zh = {"start_seq": 0, "posteriors": [frame_n(0, 2) for _ in range(5)]}
    status, first_zh = post(f"/sessions/{sid}/chunks", chunk_zh)
    _show_receipt("普通块 chunk 0（zh x5）", status, first_zh)

    last = {"start_seq": 5, "posteriors": [frame_n(1, 2) for _ in range(5)], "final": True}
    status, first_final = post(f"/sessions/{sid}/chunks", last)
    _show_receipt("末块 chunk 1（en x5, final=true）首次", status, first_final)

    status, dup_final = post(f"/sessions/{sid}/chunks", last)
    _show_receipt("末块 chunk 1 原样重送", status, dup_final)
    assert status == 200 and dup_final["duplicate"] is True
    assert dup_final["segments"] == first_final["segments"]
    assert dup_final["open_segment"] is None
    print("  -> 重送回执与首次回执一致，open_segment 为 null")

    status, sealed = get(f"/sessions/{sid}/segments")
    _show_receipt("GET /segments 封存查询", status, sealed)
    assert dup_final["segments"] == sealed["segments"]
    print("  -> 区间与封存查询一致")

    status, conflict = post(
        f"/sessions/{sid}/chunks", {**last, "final": False}
    )
    _show_receipt("同序号仅把 final 改为 false", status, conflict)
    assert status == 409 and conflict["error"]["code"] == "chunk_conflict"
    status, still_sealed = get(f"/sessions/{sid}/segments")
    assert still_sealed["segments"] == sealed["segments"] and still_sealed["sealed"] is True
    print("  -> 409，会话状态（含封存状态）不变")

    status, old_normal = post(f"/sessions/{sid}/chunks", chunk_zh)
    _show_receipt("封存后重送普通块 chunk 0", status, old_normal)
    assert status == 200 and old_normal["duplicate"] is True
    assert old_normal["sealed"] is True
    assert old_normal["segments"] == first_zh["segments"] == []
    assert old_normal["open_segment"] == first_zh["open_segment"]
    print("  -> 返回当时的回执（无已关闭区间、尾段 zh 0..4），sealed 报告当前状态")


def demo_normal_chunk_resend() -> None:
    """普通块重送：返回当时的已关闭区间、临时尾段、切换进度；切点回退不改旧回执。"""
    print("\n" + "=" * 70)
    print("B. 普通块重送（含后续切换让切点回退）")
    print("=" * 70)
    status, session = post(
        "/sessions",
        {
            "languages": ["zh", "en"],
            "frame_ms": 20,
            "switch_frames": 3,
            "posterior_threshold": 0.55,
        },
    )
    sid = session["session_id"]
    chunk0 = {"start_seq": 0, "posteriors": [frame_n(0, 2) for _ in range(5)]}
    chunk1 = {"start_seq": 5, "posteriors": [frame_n(1, 2)]}
    chunk2 = {"start_seq": 6, "posteriors": [frame_n(1, 2) for _ in range(2)]}
    _, r0 = post(f"/sessions/{sid}/chunks", chunk0)
    _, r1 = post(f"/sessions/{sid}/chunks", chunk1)
    _show_receipt("chunk 1（en x1，候选切换进行中）首次", 200, r1)
    _, r2 = post(f"/sessions/{sid}/chunks", chunk2)
    _show_receipt("chunk 2（en x2，累计达阈值后切段）", 200, r2)
    assert r2["segments"][-1]["end_frame"] == 4
    assert r1["open_segment"]["end_frame"] == 5
    print("  -> 确认切换让 zh 段切点回退到 4（旧回执当时是 5）")

    status, dup1 = post(f"/sessions/{sid}/chunks", chunk1)
    _show_receipt("重送 chunk 1", status, dup1)
    assert dup1["duplicate"] is True
    assert dup1["segments"] == r1["segments"] == []
    assert dup1["open_segment"] == r1["open_segment"]
    assert dup1["pending_frames"] == r1["pending_frames"] == 1
    assert dup1["next_seq"] == r1["next_seq"] == 6
    print("  -> 旧回执保留当时的尾段 zh 0..5 与切换进度，未被回退后的切点改写")

    status, dup0 = post(f"/sessions/{sid}/chunks", chunk0)
    _show_receipt("重送 chunk 0", status, dup0)
    assert dup0["duplicate"] is True and dup0["segments"] == []
    assert dup0["open_segment"] == r0["open_segment"]
    print("  -> 同样返回当时的回执，而非当前已关闭的 zh 0..4")


if __name__ == "__main__":
    main()
