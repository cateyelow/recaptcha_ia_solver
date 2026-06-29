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
