"""VLM-based grid recognizer (Gemini Flash).

Why this exists
---------------
The bundled `yolov8s-cls` classifier answers "what is the single dominant
class in this 100x100 tile?". reCAPTCHA actually asks "does *any* part of a
{target} appear in this tile?", over cluttered real-world street scenes where
one tile routinely holds a road + a car + a distant bus. A single-label top-1
classifier structurally under-recalls on that distribution (measured: it calls
a clear bus tile `stair`@0.95 and a bridge tile `car`@0.91), and it only knows
14 classes — anything else (boat/stairs/parking meter/...) reloads forever.

A vision-language model sidesteps all three problems at once: it reasons about
*presence* (not dominance), it understands the challenge phrase in any locale
(Korean "버스" included) so no phrase->class table is needed, and it handles the
4x4 "one photo cut into 16 squares" mode by selecting every square an object
overlaps. This module is the primary recognizer; the YOLO path stays as an
offline fallback (see RECAPTCHA_RECOGNIZER in solver.py).

Public entry point
------------------
`recognize_cells(image, target_phrase, grid_n, verbose=False) -> list[int] | None`
returns the 1-indexed cells to click, or None on unrecoverable VLM failure so
the caller can fall back to the local model.
"""

from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from time import sleep
from typing import List, Optional, Union

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

ImageLike = Union[str, "np.ndarray", Image.Image]

# Endpoint + defaults. gemini-2.5-flash balances accuracy and ~2s latency;
# flash-lite is ~0.5s faster but weaker on hard recognition. Override with
# RECAPTCHA_VLM_MODEL. The key falls back to the standard GEMINI_API_KEY.
_GENAI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
DEFAULT_VLM_MODEL = "gemini-2.5-flash"

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def vlm_api_key() -> str:
    """API key for the VLM. RECAPTCHA_VLM_API_KEY wins, else GEMINI_API_KEY."""
    return os.environ.get("RECAPTCHA_VLM_API_KEY") or os.environ.get("GEMINI_API_KEY", "")


def vlm_enabled() -> bool:
    """True when a key is present so the recognizer can actually run."""
    return bool(vlm_api_key())


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _to_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    return Image.fromarray(np.asarray(image)).convert("RGB")


def overlay_grid_numbers(img: Image.Image, grid_n: int) -> Image.Image:
    """Stamp each cell's 1-indexed number in its top-left corner.

    VLMs map "cell 7" to a screen region far more reliably when the number is
    drawn *into* the region than when it has to infer row/col arithmetic from a
    prose description. The number sits on a solid dark chip so it stays legible
    over any tile content.
    """
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cw, ch = w / grid_n, h / grid_n
    font = _load_font(max(14, int(min(cw, ch) * 0.28)))
    n = 1
    for r in range(grid_n):
        for c in range(grid_n):
            x, y = c * cw, r * ch
            label = str(n)
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                tw, th = 10 * len(label), 14
            pad = 3
            draw.rectangle(
                [x + 1, y + 1, x + tw + 2 * pad + 1, y + th + 2 * pad + 1],
                fill=(0, 0, 0),
            )
            draw.text((x + pad + 1, y + pad), label, fill=(255, 255, 0), font=font)
            n += 1
    return img


def _build_prompt(target_phrase: str, grid_n: int) -> str:
    target = (target_phrase or "").strip()
    total = grid_n * grid_n
    if grid_n >= 4:
        kind = (
            f"This image is ONE photo cut into {total} equal squares arranged "
            f"{grid_n}x{grid_n}. Select EVERY square that contains ANY part of "
            "the target object, even a small sliver at a square's edge."
        )
    else:
        kind = (
            f"This is a {grid_n}x{grid_n} grid of {total} independent photos. "
            "Select every cell whose photo contains ANY part of the target "
            "object, even small, distant, or partially visible ones."
        )
    return (
        "You are solving a reCAPTCHA image challenge. "
        f"{kind} "
        f"Cells are numbered 1..{total}; the yellow number printed in the "
        "top-left corner of each cell is its id. "
        f'The challenge asks to select all images matching: "{target}". '
        "Interpret the target in any language. Include a cell when you can "
        "actually see the target in it (count even small, distant, or partially "
        "visible instances, including a sliver at a cell's edge), but do NOT "
        "select a cell for a merely related scene that lacks the object itself "
        "(a plain road is not a crosswalk; a truck or van is not a car or bus) "
        "or on a guess. "
        'Respond with ONLY strict JSON: {"cells": [ids]} where ids is the list '
        f"of matching cell numbers (1..{total}), empty list if none match."
    )


def _call_gemini(image: Image.Image, prompt: str, model: str, timeout: float) -> dict:
    """Single Gemini generateContent call; returns the parsed JSON object.

    Raises on transport error, safety block, or unparseable output so the
    retry wrapper can decide whether to try again or surrender to the fallback.
    """
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": b64}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "response_mime_type": "application/json",
            "maxOutputTokens": 4096,
        },
    }
    url = _GENAI_URL.format(model=model) + f"?key={vlm_api_key()}"
    resp = requests.post(
        url, json=body, headers={"Content-Type": "application/json"}, timeout=timeout
    )
    if resp.status_code != 200:
        raise RuntimeError(f"gemini HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"gemini no candidates: {json.dumps(data)[:200]}")
    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise RuntimeError("gemini empty text")
    return _parse_cells_json(text)


def _parse_cells_json(text: str) -> dict:
    """Tolerant JSON extraction — strips ```json fences and trailing prose."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"unparseable VLM output: {text[:200]}")


def recognize_cells(
    image: ImageLike,
    target_phrase: str,
    grid_n: int,
    verbose: bool = False,
) -> Optional[List[int]]:
    """Return the 1-indexed cells to click for `target_phrase`, or None.

    None means the VLM could not produce an answer (no key, repeated transport
    errors, or garbage output) — the caller should fall back to the local YOLO
    path. An empty list is a *valid* answer ("nothing matches this image").
    """
    if not vlm_enabled():
        if verbose:
            print("recognizer: no VLM key, skipping")
        return None

    model = _env("RECAPTCHA_VLM_MODEL", DEFAULT_VLM_MODEL)
    try:
        timeout = float(_env("RECAPTCHA_VLM_TIMEOUT", "30"))
    except ValueError:
        timeout = 30.0
    try:
        retries = int(_env("RECAPTCHA_VLM_RETRIES", "2"))
    except ValueError:
        retries = 2

    pil = _to_pil(image)
    numbered = overlay_grid_numbers(pil, grid_n)
    prompt = _build_prompt(target_phrase, grid_n)
    total = grid_n * grid_n

    last_err = None
    for attempt in range(retries + 1):
        try:
            obj = _call_gemini(numbered, prompt, model, timeout)
            # Contract is {"cells": [int, ...]}. A missing key or a non-list
            # value is malformed: {"cells": "10"} would iterate to ['1','0'] and
            # a dict would iterate its keys, both yielding a bogus answer. Raise
            # so the retry/fallback path runs instead of silently returning a
            # wrong/empty list. An empty list IS a valid answer ("none match").
            if "cells" in obj:
                raw = obj["cells"]
            elif "answers" in obj:
                raw = obj["answers"]
            else:
                raise ValueError(f"VLM JSON has no 'cells' key: {obj!r}")
            if not isinstance(raw, list):
                raise ValueError(f"VLM 'cells' is not a list: {raw!r}")
            cells = sorted(
                {int(x) for x in raw if isinstance(x, (int, float, str))
                 and str(x).strip().lstrip("-").isdigit()
                 and 1 <= int(x) <= total}
            )
            if verbose:
                print(
                    f"recognizer[{model}]: target={target_phrase!r} "
                    f"grid={grid_n}x{grid_n} -> {cells} (raw={raw})"
                )
            return cells
        except Exception as exc:  # noqa: BLE001 - transport/parse/safety
            last_err = exc
            if verbose:
                print(f"recognizer attempt {attempt + 1} failed: {exc!r}")
            if attempt < retries:
                sleep(0.6 * (attempt + 1))
    if verbose:
        print(f"recognizer: all attempts failed ({last_err!r}); falling back")
    return None
