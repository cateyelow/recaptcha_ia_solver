"""Grid-level recognizer benchmark: classifier vs VLM on clean labeled grids.

reCAPTCHA only accepts a challenge when EVERY cell is correct, so the metric
that predicts pass rate is grid-level exact-match accuracy, not per-cell F1.
This builds clean 3x3 grids from a labeled tile dataset (default: the
verytuffcat tiles the classifier itself trained on — a conservative, in-domain
test that favors the classifier) and reports, per backend:

  grid pass%  : fraction of grids where predicted cells == ground-truth cells
  cell prec/rec

Usage:
  python3 scripts/bench_recognizer.py --grids-per-class 6 --backends classifier,vlm
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))

DEFAULT_ROOT = "/tmp/recaptcha_ds/verytuffcat/data/train"
PHRASE = {
    "bicycle": "bicycle", "bridge": "bridge", "bus": "bus", "car": "car",
    "chimney": "chimney", "crosswalk": "crosswalk", "hydrant": "fire hydrant",
    "motorcycle": "motorcycle", "mountain": "mountain", "palm": "palm tree",
    "stair": "stairs", "tractor": "tractor", "traffic light": "traffic light",
}


def _tiles(root, cls):
    return [p for p in glob.glob(f"{root}/{cls}/*")
            if p.lower().endswith((".png", ".jpg", ".jpeg"))]


def build_grid(target_tiles, other_tiles, rng, save_to):
    n_pos = rng.randint(2, 5)
    pos = set(rng.sample(range(9), n_pos))
    cells = []
    for i in range(9):
        src = target_tiles if i in pos else other_tiles
        img = Image.open(rng.choice(src)).convert("RGB").resize((100, 100))
        cells.append(np.asarray(img))
    grid = np.zeros((300, 300, 3), dtype=np.uint8)
    for i, c in enumerate(cells):
        grid[(i // 3) * 100:(i // 3 + 1) * 100, (i % 3) * 100:(i % 3 + 1) * 100] = c
    Image.fromarray(grid).save(save_to)
    return {p + 1 for p in pos}


def classifier_predict(model, target_set):
    from recaptcha_ia_solver import solver as M
    return set(M.classify_grid_cells(target_set, 3, verbose=False, model=model))


def vlm_predict(phrase):
    from recaptcha_ia_solver import recognizer
    cells = recognizer.recognize_cells("recaptcha_images/0.png", phrase, 3, verbose=False)
    return set(cells or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--grids-per-class", type=int, default=6)
    ap.add_argument("--backends", default="classifier,vlm")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    os.makedirs("recaptcha_images", exist_ok=True)
    backends = args.backends.split(",")

    classes = sorted(d for d in os.listdir(args.root)
                     if os.path.isdir(f"{args.root}/{d}") and d in PHRASE)
    model = None
    if "classifier" in backends:
        from ultralytics import YOLO
        model = YOLO("models/recaptcha_classifier.pt")

    from recaptcha_ia_solver import solver as M
    rng = random.Random(args.seed)
    agg = {b: {"pass": 0, "n": 0, "tp": 0, "fp": 0, "fn": 0} for b in backends}

    print(f"{'class':>14} | " + " | ".join(f"{b:>18}" for b in backends))
    print("-" * (16 + 21 * len(backends)))
    for cls in classes:
        tgt = _tiles(args.root, cls)
        others_pool = []
        for o in classes:
            if o != cls:
                others_pool += _tiles(args.root, o)
        if len(tgt) < 3 or len(others_pool) < 6:
            continue
        target_set = set()
        if model is not None:
            target_set = M._resolve_target_classes(PHRASE[cls], model)
        row = {b: {"pass": 0, "n": 0} for b in backends}
        for _ in range(args.grids_per_class):
            gt = build_grid(tgt, others_pool, rng, "recaptcha_images/0.png")
            for b in backends:
                if b == "classifier":
                    pred = classifier_predict(model, target_set)
                else:
                    pred = vlm_predict(PHRASE[cls])
                agg[b]["n"] += 1
                row[b]["n"] += 1
                if pred == gt:
                    agg[b]["pass"] += 1
                    row[b]["pass"] += 1
                agg[b]["tp"] += len(pred & gt)
                agg[b]["fp"] += len(pred - gt)
                agg[b]["fn"] += len(gt - pred)
        cells = " | ".join(
            f"{row[b]['pass']:>2}/{row[b]['n']:<2} pass{'':>8}" for b in backends
        )
        print(f"{cls:>14} | " + cells)

    print("\n=== TOTAL (grid-level exact-match = reCAPTCHA accept condition) ===")
    for b in backends:
        a = agg[b]
        gp = 100 * a["pass"] / a["n"] if a["n"] else 0
        prec = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) else 0
        rec = a["tp"] / (a["tp"] + a["fn"]) if (a["tp"] + a["fn"]) else 0
        print(f"  {b:>12}: grid pass {a['pass']:>3}/{a['n']:<3} = {gp:5.1f}%   "
              f"cell prec={prec:.3f} rec={rec:.3f}")


if __name__ == "__main__":
    main()
