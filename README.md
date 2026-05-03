# recaptcha_ia_solver

Image-grid reCAPTCHA solver. Two-stage architecture, no LLMs:

1. **Primary** — fine-tuned `yolov8s-cls` (14 reCAPTCHA-specific classes:
   bicycle, bridge, bus, car, chimney, crosswalk, hydrant, motorcycle,
   mountain, other, palm, stair, tractor, traffic light). Trained on the
   merged `verytuffcat/recaptcha-dataset` + `DannyLuna/recaptcha-57k-images-dataset`
   (~57 k images). Bundled at `models/recaptcha_classifier.pt`.
2. **Fallback** — Open Images V7 pretrained `yolov8x-oiv7.pt` detector,
   lazy-loaded only when the challenge phrase resolves to a class the
   primary classifier wasn't trained on (e.g. `boat`, `truck`, `taxi`,
   `parking meter`, `stop sign`, `tower`, `vehicle`).

Real-world score on Google's official demo page (`/recaptcha/api2/demo`):
**9–10 / 10** across multiple 10-attempt runs. Each attempt = fresh browser,
fresh session.

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

| variable                  | default                                  | meaning                                                           |
| ------------------------- | ---------------------------------------- | ----------------------------------------------------------------- |
| `RECAPTCHA_YOLO_MODEL`    | `models/recaptcha_classifier.pt`         | primary classifier weights                                        |
| `RECAPTCHA_YOLO_FALLBACK` | `models/yolov8x-oiv7.pt`                 | OIV7 detector weights; empty string disables fallback             |
| `RECAPTCHA_YOLO_MIN_CONF` | `0.35`                                   | reject classifier predictions below this top-1 confidence         |

Raising `MIN_CONF` trades recall for precision (fewer false clicks, more reloads).

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

`scripts/benchmark.py` (assumes the verytuffcat + DannyLuna datasets are
extracted under `/tmp/`) reports per-class precision/recall and overall F1
across both distributions.

## What makes the realtime path actually pass

1. **iframe selectors keyed on `@src`**, not `@title` (titles are localized:
   English "challenge" vs. Korean "챌린지" etc.).
2. **Bounded inner loops**. The dynamic-captcha "wait for refreshed
   thumbnails" poll, the "paste new tile into the working image" retry, and
   the outer solve loop all carry monotonic-clock deadlines so a single
   stuck attempt can't hang forever.
3. **Stale-element resilience**. iframe re-renders mid-click used to take
   the solver into a silent giveup that returned a "looks-solved" status
   without verification — now we re-anchor and retry until the deadline.
4. **Confidence floor on cell predictions**. False clicks cost much more
   than misses (a wrong click forces reCAPTCHA to surface the next, harder
   challenge); we drop top-1 predictions below `RECAPTCHA_YOLO_MIN_CONF`.
5. **Form submit via `requestSubmit()`**. Plain `.click()` on the page
   submit button races reCAPTCHA's z-index 2-billion transparent overlay
   that lingers ~1–2 s after the checkbox flips green.

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
