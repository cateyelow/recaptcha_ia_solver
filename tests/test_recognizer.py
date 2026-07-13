"""Unit tests for the VLM recognizer and its solver integration.

All network is mocked: these verify parsing, range-filtering, retry/fallback,
and the solver's recognizer-mode + dead-driver routing — no live API or
browser required.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


# ----------------------------- recognizer core ----------------------------- #

def _fake_response(payload_text, status=200):
    return SimpleNamespace(
        status_code=status,
        text=payload_text,
        json=lambda: {
            "candidates": [{"content": {"parts": [{"text": payload_text}]}}]
        },
    )


def test_parse_cells_json_plain_and_fenced():
    from recaptcha_ia_solver.recognizer import _parse_cells_json

    assert _parse_cells_json('{"cells":[1,2,3]}') == {"cells": [1, 2, 3]}
    fenced = "```json\n{\"cells\": [4, 5]}\n```"
    assert _parse_cells_json(fenced) == {"cells": [4, 5]}
    trailing = 'Here you go: {"cells":[7]} hope that helps'
    assert _parse_cells_json(trailing) == {"cells": [7]}


def test_overlay_grid_numbers_preserves_size():
    from PIL import Image
    from recaptcha_ia_solver.recognizer import overlay_grid_numbers

    import numpy as np

    img = Image.new("RGB", (300, 300), (10, 20, 30))
    out = overlay_grid_numbers(img, 3)
    assert out.size == (300, 300)
    # overlay must actually draw something (yellow numerals) -> pixels change
    assert not np.array_equal(np.asarray(out), np.asarray(img))


def test_overlay_grid_numbers_draws_visible_4x4_boundaries():
    from PIL import Image
    from recaptcha_ia_solver.recognizer import overlay_grid_numbers

    img = Image.new("RGB", (400, 400), (10, 20, 30))
    out = overlay_grid_numbers(img, 4)

    # A single-photo 4x4 challenge has no visible gutters.  Explicit boundary
    # lines keep the VLM from assigning an object sliver to the adjacent cell.
    assert out.getpixel((100, 60)) != img.getpixel((100, 60))
    assert out.getpixel((260, 200)) != img.getpixel((260, 200))


def test_recognize_cells_happy_path(monkeypatch):
    from recaptcha_ia_solver import recognizer
    from PIL import Image

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        recognizer.requests, "post",
        lambda *a, **k: _fake_response('{"cells":[1,4,9]}'),
    )
    cells = recognizer.recognize_cells(Image.new("RGB", (300, 300)), "bus", 3)
    assert cells == [1, 4, 9]


def test_recognize_cells_filters_out_of_range(monkeypatch):
    from recaptcha_ia_solver import recognizer
    from PIL import Image

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    # 0 and 99 are out of the 1..9 range and must be dropped; dupes collapsed.
    monkeypatch.setattr(
        recognizer.requests, "post",
        lambda *a, **k: _fake_response('{"cells":[0,5,5,99,7]}'),
    )
    cells = recognizer.recognize_cells(Image.new("RGB", (300, 300)), "car", 3)
    assert cells == [5, 7]


def test_recognize_cells_4x4_range(monkeypatch):
    from recaptcha_ia_solver import recognizer
    from PIL import Image

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        recognizer.requests, "post",
        lambda *a, **k: _fake_response('{"cells":[1,16,17]}'),
    )
    cells = recognizer.recognize_cells(Image.new("RGB", (400, 400)), "bus", 4)
    assert cells == [1, 16]  # 17 is out of range for a 16-cell grid


def test_recognize_cells_empty_is_valid_answer(monkeypatch):
    from recaptcha_ia_solver import recognizer
    from PIL import Image

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        recognizer.requests, "post",
        lambda *a, **k: _fake_response('{"cells":[]}'),
    )
    cells = recognizer.recognize_cells(Image.new("RGB", (300, 300)), "boat", 3)
    assert cells == []  # empty list, NOT None — "nothing matches" is an answer


def test_recognize_cells_no_key_returns_none(monkeypatch):
    from recaptcha_ia_solver import recognizer
    from PIL import Image

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("RECAPTCHA_VLM_API_KEY", raising=False)
    assert not recognizer.vlm_enabled()
    cells = recognizer.recognize_cells(Image.new("RGB", (300, 300)), "bus", 3)
    assert cells is None  # None -> caller falls back to local YOLO


def test_recognize_cells_persistent_error_returns_none(monkeypatch):
    from recaptcha_ia_solver import recognizer
    from PIL import Image

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("RECAPTCHA_VLM_RETRIES", "1")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(recognizer.requests, "post", boom)
    cells = recognizer.recognize_cells(Image.new("RGB", (300, 300)), "bus", 3)
    assert cells is None


def test_recognize_cells_temperature_env_flows_into_request(monkeypatch):
    # RECAPTCHA_VLM_TEMPERATURE must reach the request body so a self-consistency
    # caller can make repeated passes vary; default stays deterministic (0).
    from recaptcha_ia_solver import recognizer
    from PIL import Image

    captured = {}

    def fake_post(url, json=None, **k):
        captured["body"] = json
        return _fake_response('{"cells":[1]}')

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(recognizer.requests, "post", fake_post)

    monkeypatch.delenv("RECAPTCHA_VLM_TEMPERATURE", raising=False)
    recognizer.recognize_cells(Image.new("RGB", (300, 300)), "bus", 3)
    assert captured["body"]["generationConfig"]["temperature"] == 0.0

    monkeypatch.setenv("RECAPTCHA_VLM_TEMPERATURE", "0.7")
    recognizer.recognize_cells(Image.new("RGB", (300, 300)), "bus", 3)
    assert captured["body"]["generationConfig"]["temperature"] == 0.7

    # a garbage value must not crash recognition — it falls back to 0.
    monkeypatch.setenv("RECAPTCHA_VLM_TEMPERATURE", "not-a-number")
    recognizer.recognize_cells(Image.new("RGB", (300, 300)), "bus", 3)
    assert captured["body"]["generationConfig"]["temperature"] == 0.0


def _threadsafe_post(payloads):
    """A requests.post stand-in that hands out queued payloads under a lock,
    so the parallel self-consistency passes each get a distinct response."""
    import threading
    from collections import deque

    q = deque(payloads)
    lock = threading.Lock()

    def fake_post(*a, **k):
        with lock:
            payload = q.popleft()
        return _fake_response(payload)

    return fake_post


def test_recognize_cells_self_consistency_unanimous(monkeypatch):
    # RECAPTCHA_VLM_SAMPLES>1 votes; default ratio 1.0 keeps only cells EVERY
    # pass agreed on. 5 appears in all 3 passes; 6 and 7 in only 2 -> {5}.
    from recaptcha_ia_solver import recognizer
    from PIL import Image

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("RECAPTCHA_VLM_SAMPLES", "3")
    monkeypatch.delenv("RECAPTCHA_VLM_VOTE_RATIO", raising=False)
    monkeypatch.setattr(
        recognizer.requests, "post",
        _threadsafe_post(['{"cells":[5,6]}', '{"cells":[5,6,7]}', '{"cells":[5,7]}']),
    )
    cells = recognizer.recognize_cells(Image.new("RGB", (300, 300)), "bus", 3)
    assert cells == [5]


def test_recognize_cells_self_consistency_majority(monkeypatch):
    # Same votes (5:3, 6:2, 7:2) but ratio 0.5 -> threshold ceil(3*0.5)=2 -> all.
    from recaptcha_ia_solver import recognizer
    from PIL import Image

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("RECAPTCHA_VLM_SAMPLES", "3")
    monkeypatch.setenv("RECAPTCHA_VLM_VOTE_RATIO", "0.5")
    monkeypatch.setattr(
        recognizer.requests, "post",
        _threadsafe_post(['{"cells":[5,6]}', '{"cells":[5,6,7]}', '{"cells":[5,7]}']),
    )
    cells = recognizer.recognize_cells(Image.new("RGB", (300, 300)), "bus", 3)
    assert cells == [5, 6, 7]


def test_recognize_cells_self_consistency_all_fail_returns_none(monkeypatch):
    # If every sampled pass dies, the vote has nothing -> None (YOLO fallback).
    from recaptcha_ia_solver import recognizer
    from PIL import Image

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("RECAPTCHA_VLM_SAMPLES", "3")
    monkeypatch.setenv("RECAPTCHA_VLM_RETRIES", "0")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(recognizer.requests, "post", boom)
    cells = recognizer.recognize_cells(Image.new("RGB", (300, 300)), "bus", 3)
    assert cells is None


def test_sampled_recognizer_does_not_wait_for_blocked_workers_after_hard_deadline(
    monkeypatch,
):
    """A caller's hard deadline must escape without joining VLM workers."""
    import os
    import signal
    import threading
    import time

    from PIL import Image
    from recaptcha_ia_solver import recognizer

    if not hasattr(signal, "SIGALRM"):
        pytest.skip("SIGALRM is required to reproduce the POSIX hard deadline")

    class HardDeadline(BaseException):
        pass

    samples = 5
    started_count = 0
    started_lock = threading.Lock()
    all_started = threading.Event()
    release_workers = threading.Event()

    def blocked_call(*args, **kwargs):
        nonlocal started_count
        with started_lock:
            started_count += 1
            if started_count == samples:
                all_started.set()
        release_workers.wait(timeout=2)
        return {"cells": [1]}

    def raise_deadline_once_workers_are_blocked():
        if all_started.wait(timeout=2):
            os.kill(os.getpid(), signal.SIGALRM)

    def deadline_handler(_signum, _frame):
        raise HardDeadline()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("RECAPTCHA_VLM_SAMPLES", str(samples))
    monkeypatch.setenv("RECAPTCHA_VLM_RETRIES", "0")
    monkeypatch.setattr(recognizer, "_call_gemini", blocked_call)

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, deadline_handler)
    trigger = threading.Thread(
        target=raise_deadline_once_workers_are_blocked,
        daemon=True,
    )
    # Keep RED bounded: the buggy executor waits for these workers, then the
    # elapsed-time assertion below proves that it violated the hard deadline.
    watchdog = threading.Timer(0.75, release_workers.set)
    started = time.monotonic()
    try:
        trigger.start()
        watchdog.start()
        with pytest.raises(HardDeadline):
            recognizer.recognize_cells(
                Image.new("RGB", (400, 400)), "bus", 4, verbose=False
            )
        elapsed = time.monotonic() - started
    finally:
        signal.signal(signal.SIGALRM, previous_handler)
        release_workers.set()
        trigger.join(timeout=1)
        watchdog.cancel()
        watchdog.join(timeout=1)

    assert all_started.is_set()
    assert elapsed < 0.35


def test_bus_prompt_explicitly_excludes_recreational_vehicles():
    from recaptcha_ia_solver import recognizer

    prompt = recognizer._build_prompt("버스", 3).lower()

    assert "motorhome" in prompt
    assert "rv" in prompt
    assert "camper" in prompt
    assert "not a bus" in prompt


@pytest.mark.parametrize("target", ["자동차", "car"])
def test_car_prompt_explicitly_excludes_non_passenger_vehicles(target):
    from recaptcha_ia_solver import recognizer

    prompt = recognizer._build_prompt(target, 3).lower()

    assert "passenger car" in prompt
    for excluded in ("bus", "coach", "truck", "van", "motorcycle", "scooter"):
        assert excluded in prompt
    assert "not a car" in prompt


def test_recognize_cells_self_consistency_quorum(monkeypatch):
    # Fewer than half the requested samples surviving -> untrustworthy vote, so
    # return None. A lone survivor must NOT clear the unanimous threshold (=1)
    # and masquerade as consensus — that would also block the YOLO fallback.
    import threading
    from collections import deque

    from recaptcha_ia_solver import recognizer
    from PIL import Image

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("RECAPTCHA_VLM_SAMPLES", "3")
    monkeypatch.setenv("RECAPTCHA_VLM_RETRIES", "0")
    monkeypatch.delenv("RECAPTCHA_VLM_VOTE_RATIO", raising=False)

    # 2 of 3 passes die, only 1 returns an over-selected answer -> quorum
    # (majority of 3 = 2) not met -> None.
    q = deque(['{"cells":[1,2,3,4]}', "BOOM", "BOOM"])
    lock = threading.Lock()

    def fake_post(*a, **k):
        with lock:
            p = q.popleft()
        if p == "BOOM":
            raise RuntimeError("network down")
        return _fake_response(p)

    monkeypatch.setattr(recognizer.requests, "post", fake_post)
    cells = recognizer.recognize_cells(Image.new("RGB", (300, 300)), "bus", 3)
    assert cells is None


@pytest.mark.parametrize(
    ("samples", "survivors", "expected"),
    [
        pytest.param(2, 1, None, id="one-of-two-is-not-a-majority"),
        pytest.param(4, 2, None, id="two-of-four-is-not-a-majority"),
        pytest.param(4, 3, [2], id="three-of-four-is-a-majority"),
    ],
)
def test_recognize_cells_requires_strict_sample_majority(
    monkeypatch, samples, survivors, expected
):
    import threading
    from collections import deque

    from PIL import Image
    from recaptcha_ia_solver import recognizer

    outcomes = deque([[2]] * survivors + [None] * (samples - survivors))
    lock = threading.Lock()

    def fake_recognize_single(*_args, **_kwargs):
        with lock:
            return outcomes.popleft()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("RECAPTCHA_VLM_SAMPLES", str(samples))
    monkeypatch.setenv("RECAPTCHA_VLM_RETRIES", "0")
    monkeypatch.setattr(recognizer, "_recognize_single", fake_recognize_single)

    cells = recognizer.recognize_cells(
        Image.new("RGB", (300, 300)), "bus", 3
    )

    assert cells == expected


@pytest.mark.parametrize(
    "payload",
    [
        '{"cells":"10"}',          # string, not a list (would iterate to '1','0')
        '{"cells":{"10":true}}',   # dict (would iterate its keys)
        '{"selected":[1,2]}',      # missing the cells/answers key entirely
    ],
)
def test_recognize_cells_malformed_json_falls_back(monkeypatch, payload):
    # A non-list / missing `cells` must NOT be silently turned into a bogus or
    # empty answer; it has to fail to None so the caller falls back to local
    # YOLO. (An empty *list* stays a valid "none match" answer — tested above.)
    from recaptcha_ia_solver import recognizer
    from PIL import Image

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("RECAPTCHA_VLM_RETRIES", "0")
    monkeypatch.setattr(
        recognizer.requests, "post", lambda *a, **k: _fake_response(payload)
    )
    cells = recognizer.recognize_cells(Image.new("RGB", (400, 400)), "bus", 4)
    assert cells is None


# --------------------------- solver integration ---------------------------- #

def test_recognizer_mode_parsing(monkeypatch):
    from recaptcha_ia_solver import solver

    for val, expect in [("vlm", "vlm"), ("LOCAL", "local"), ("hybrid", "hybrid"),
                        ("garbage", "hybrid"), ("", "hybrid")]:
        if val:
            monkeypatch.setenv("RECAPTCHA_RECOGNIZER", val)
        else:
            monkeypatch.delenv("RECAPTCHA_RECOGNIZER", raising=False)
        assert solver._recognizer_mode() == expect


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("MaxRetryError: Max retries exceeded with url", True),
        ("NewConnectionError: Connection refused", True),
        ("WebDriverException: chrome not reachable", True),
        ("InvalidSessionIdException: invalid session id", True),
        ("StaleElementReferenceException: stale element", False),
        ("TimeoutException: timed out waiting", False),
    ],
)
def test_is_dead_driver_error(msg, expected):
    from recaptcha_ia_solver import solver

    assert solver._is_dead_driver_error(Exception(msg)) is expected


def test_vlm_answers_local_mode_returns_none(monkeypatch):
    from recaptcha_ia_solver import solver

    monkeypatch.setenv("RECAPTCHA_RECOGNIZER", "local")
    # local mode must never consult the VLM, regardless of key presence
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert solver._vlm_answers("bus", 3, verbose=False) is None
