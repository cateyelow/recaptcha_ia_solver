# recaptcha_ia_solver

Image-grid reCAPTCHA solver. The recognizer is pluggable (`RECAPTCHA_RECOGNIZER`):

1. **Primary — VLM (`hybrid`/`vlm` mode, default).** A vision-language model
   (Gemini Flash) answers *"which cells contain the target?"* straight from the
   composite. It reasons about object **presence** in cluttered multi-object
   tiles (not the single dominant class), understands the challenge phrase in
   **any locale** (Korean `버스` included — no phrase→class table), covers **every**
   category, and handles the 4×4 "one photo cut into 16 squares" mode by picking
   every square an object overlaps. This is what the per-cell classifier
   structurally could not do.
2. **Fallback — local YOLO (`local` mode, or automatic on VLM failure).** The
   bundled fine-tuned `yolov8s-cls` (14 classes) plus the Open Images V7
   `yolov8x-oiv7.pt` detector. Runs fully offline; used when no VLM key is set,
   when `RECAPTCHA_RECOGNIZER=local`, or when a VLM call fails after retries.

Why the change, measured on **20 real reCAPTCHA challenges** captured from the
live demo and hand-labeled (a grid passes only when **every** cell is right —
the exact condition reCAPTCHA gates on):

| recognizer       | 3×3 grid-exact | 4×4 grid-exact | cell recall |
| ---------------- | -------------- | -------------- | ----------- |
| per-cell YOLO    | 40 %           | 0 %            | 0.90 / 0.39 |
| VLM (2.5-flash)  | **60 %**       | **20 %**       | 0.95 / 0.92 |

The single-label classifier answers *"dominant class of this tile"*, so it drops
true matches in multi-object cells and is **structurally blind to the 4×4
"one photo cut into 16 squares" mode (0 %)** — each square holds only a fragment
of the object. The VLM answers *"does the target appear in this cell"* and lifts
every mode. Notably a *stronger* VLM did **not** help (3.5-flash and 2.5-pro tied
or trailed 2.5-flash on this set), so the default is the fast, cheap 2.5-flash;
a precision-tuned prompt trims its only real weakness, over-selection.

> **Single-pass recognition alone cannot reach 99 %.** ~60 % is the per-grid
> ceiling on real street scenes for *every* recognizer tested — the task is
> genuinely ambiguous (a truck vs. a bus, a faint distant crosswalk). The
> remaining gap is closed by two non-recognition levers: a high **trust score**
> (reCAPTCHA then serves easy 3×3s or skips the challenge, instead of escalating
> to the 4×4 / dynamic loops where even the VLM is at 20 %) and **bounded
> reloads** (an independent fresh grid after a miss: with p≈0.6 per grid, a few
> retries compound toward a high end-to-end rate *as long as trust score holds*).
> Recognition maximizes per-grid odds; trust score and retry turn that into the
> pass rate.

> **No-LLM mode** is still first-class: set `RECAPTCHA_RECOGNIZER=local` (or simply
> leave `GEMINI_API_KEY` unset) and the solver runs the offline YOLO path exactly
> as before.

## Install

```bash
pip install -e .
# optional: harness for the demo-page example
pip install -e .[runtime]
```

Selenium needs a Chrome/Chromium binary on `PATH`. The bundled example uses
`undetected_chromedriver`; if you want to use plain selenium it works the
same — just pass any `selenium.webdriver` instance to `solve_recaptcha`.

## Use it

```python
from recaptcha_ia_solver import solve_recaptcha

# `driver` is any selenium WebDriver pointed at a page that has a v2 reCAPTCHA
# challenge embedded. `solve_recaptcha` clicks the checkbox, walks any image
# challenges, and returns when the checkbox shows verified (or after a
# 120-second wall-clock cap if it can't).
solve_recaptcha(driver, verbose=True)
```

The library is driver-agnostic. Provide your own webdriver — typically with
stealth tooling (`undetected_chromedriver`, etc.) since reCAPTCHA aggressively
fingerprints headless / wired browsers and degrades to NoScript fallback when
it suspects a bot.

End-to-end example: `examples/solve_demo_page.py` runs N attempts against
`https://www.google.com/recaptcha/api2/demo?hl=en` and reports a pass score:

```bash
DISPLAY=:0 python3 examples/solve_demo_page.py 10
```

## Environment overrides

| variable                  | default                          | meaning                                                              |
| ------------------------- | -------------------------------- | -------------------------------------------------------------------- |
| `RECAPTCHA_RECOGNIZER`    | `hybrid`                         | `vlm` (VLM only) · `local` (YOLO only) · `hybrid` (VLM, YOLO on fail) |
| `GEMINI_API_KEY`          | —                                | Gemini key for the VLM; unset ⇒ behaves like `local`                 |
| `RECAPTCHA_VLM_API_KEY`   | (falls back to `GEMINI_API_KEY`) | dedicated VLM key override                                           |
| `RECAPTCHA_VLM_MODEL`     | `gemini-2.5-flash`               | VLM model id (`gemini-2.5-flash-lite` is ~0.5 s faster)              |
| `RECAPTCHA_VLM_TIMEOUT`   | `30`                             | per-call VLM timeout (seconds)                                       |
| `RECAPTCHA_VLM_RETRIES`   | `2`                              | VLM retries before falling back to local YOLO                        |
| `RECAPTCHA_MAX_RELOADS`   | `12`                             | reload budget before giving up (bounds reCAPTCHA's suspicion)        |
| `RECAPTCHA_SOLVER_DEADLINE_SEC` | `120`                      | hard wall-clock cap on one `solve_recaptcha` call                    |
| `RECAPTCHA_YOLO_MODEL`    | `models/recaptcha_classifier.pt` | local-fallback classifier weights                                    |
| `RECAPTCHA_YOLO_FALLBACK` | `models/yolov8x-oiv7.pt`         | local-fallback OIV7 detector; empty string disables it               |
| `RECAPTCHA_YOLO_MIN_CONF` | `0.35`                           | reject local classifier predictions below this top-1 confidence      |

The library is driver-agnostic but **trust score is the dominant lever on the
real-world pass rate**: a high-trust session (real logged-in Chrome profile,
residential IP, human-like behaviour) is served easy challenges or none at all,
while a flagged automated session is escalated into the hardest 4×4 / dynamic
loops that no recognizer clears reliably. `solve_recaptcha` now approaches the
checkbox and every cell with a short curved mouse path + jittered timing to
keep that score up; supply a persistent, logged-in webdriver for best results.

## Retrain on your own data

```bash
python3 scripts/train_classifier.py \
  --source /path/to/dataset/<class>/<images> \
  --epochs 40 --imgsz 128 --batch 256 \
  --out models/recaptcha_classifier.pt
```

The script materializes an 80/20 train/val split from a flat class-folder
layout, fine-tunes `yolov8s-cls`, and copies `best.pt` to `--out`. Add
`--export-onnx` for ONNX too.

`scripts/benchmark.py` reports per-class precision/recall for the local
classifier. `scripts/bench_recognizer.py` is the recognizer A/B: it builds
clean labeled grids and reports **grid-level exact-match** pass rate (the metric
reCAPTCHA actually gates on) for `classifier` vs `vlm`:

```bash
python3 scripts/bench_recognizer.py --grids-per-class 8 --backends classifier,vlm
```

## What makes the realtime path actually pass

1. **VLM recognition of object _presence_.** The classifier answered "dominant
   class of this tile"; reCAPTCHA asks "does any part of the target appear
   here". The VLM answers the right question, in any locale, for any category,
   and for the 4×4 cross-tile mode — measured 40 %→60 % (3×3) and 0 %→20 % (4×4)
   grid-exact on real captured challenges.
2. **Cell numbers burned into the image.** Each cell is stamped with its id
   before the VLM sees it, so "cell 7" maps to an exact region instead of
   relying on row/col arithmetic — a big reliability win for grid prompts.
3. **Human-like cursor + timing.** Checkbox and every cell are clicked via a
   short curved `ActionChains` approach with jittered dwell, because reCAPTCHA
   scores cursor telemetry and a teleport-click drives the trust score into the
   hardest challenge loops.
4. **Bounded reloads + fast dead-driver abort.** Empty answers reload only up to
   `RECAPTCHA_MAX_RELOADS` (excessive reloads themselves raise suspicion), and a
   crashed/disconnected browser aborts immediately instead of burning the whole
   deadline re-throwing the same connection error.
5. **iframe selectors keyed on `@src`**, not the localized `@title`; **stale-
   element resilience** (re-anchor instead of silent looks-solved giveup); and
   the demo harness submits via **`requestSubmit()`** to beat reCAPTCHA's
   lingering z-index-2-billion transparent overlay.

## Tests

```bash
pip install -e .[test]
pytest -q
```

14 unit tests cover phrase→class resolution (OIV7 + classifier), aliases,
multi-class umbrella terms, model-path resolution, and the per-cell
classifier code path with a confidence-floor check.

## License

MIT.
