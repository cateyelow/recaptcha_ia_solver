"""Head-to-head: v1 vs v4 on TWO test sets to detect domain bias.

Test set A (verytuffcat val cells): favors v1 (its training distribution).
Test set B (DannyLuna val cells):   favors v4-flavor models (broader source).

For each set we build 20 synthetic 3x3 grids per class, run both classifiers
through `classify_grid_cells`, and report precision/recall.
"""
import os
import glob
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
os.makedirs("recaptcha_images", exist_ok=True)

CLASSES = [
    "bicycle", "bridge", "bus", "car", "chimney", "crosswalk",
    "hydrant", "motorcycle", "palm", "traffic light",
]
PHRASES = {
    "bicycle": "Select all images with bicycles",
    "bridge": "Select all images with bridges",
    "bus": "Select all images with buses",
    "car": "Select all images with cars",
    "chimney": "Select all images with chimneys",
    "crosswalk": "Select all images with crosswalks",
    "hydrant": "Select all images with fire hydrants",
    "motorcycle": "Select all images with motorcycles",
    "palm": "Select all images with palm trees",
    "traffic light": "Select all images with traffic lights",
}

# (label, root, has_train_val, class_alias_fn)
TEST_SETS = [
    ("verytuffcat (v1's home turf)",
     "/tmp/recaptcha_ds/verytuffcat/data/train",
     False,
     lambda c: c),
    ("dannyluna val (v4's home turf)",
     "/tmp/recaptcha_57k/dataset_cls_full_57k/val",
     True,
     lambda c: {"traffic light": "Traffic Light"}.get(c, c.title())),
]


def evaluate(model, label, ds_root, alias_fn):
    from recaptcha_ia_solver import solver as M

    others = sorted(glob.glob(f"{ds_root}/{alias_fn('other')}/*"))
    if not others:
        # verytuffcat keeps "other" lowercase in train; dannyluna in val uses "Other"
        for cand in ("other", "Other"):
            others = sorted(glob.glob(f"{ds_root}/{cand}/*"))
            if others:
                break
    print(f"\n=== {label} | {ds_root} ===")
    print(f"{'class':>15s}  {'TP':>4s} {'FP':>4s} {'FN':>4s}  {'prec':>6s} {'rec':>6s}")
    total_tp = total_fp = total_fn = 0

    for cls in CLASSES:
        subdir = alias_fn(cls)
        pool = sorted(glob.glob(f"{ds_root}/{subdir}/*"))
        # filter to images
        pool = [p for p in pool if p.lower().endswith((".png", ".jpg", ".jpeg"))]
        others_pool = [p for p in others if p.lower().endswith((".png", ".jpg", ".jpeg"))]
        if len(pool) < 5 or len(others_pool) < 5:
            print(f"{cls:>15s}  skip (pool={len(pool)}, others={len(others_pool)})")
            continue
        target_set = M._resolve_target_classes(PHRASES[cls], model)
        if not target_set:
            print(f"{cls:>15s}  no target_set!")
            continue
        rng = random.Random(hash(cls) & 0xffff)
        tp = fp = fn = 0
        for _ in range(20):
            n_t = rng.randint(2, 6)
            positions = set(rng.sample(range(9), n_t))
            cells = []
            for i in range(9):
                src = pool if i in positions else others_pool
                img = Image.open(rng.choice(src)).convert("RGB").resize((100, 100))
                cells.append(np.asarray(img))
            grid = np.zeros((300, 300, 3), dtype=np.uint8)
            for i, cell in enumerate(cells):
                r, c = i // 3, i % 3
                grid[r * 100:(r + 1) * 100, c * 100:(c + 1) * 100] = cell
            Image.fromarray(grid).save("recaptcha_images/0.png")
            expected = {p + 1 for p in positions}
            answers = set(M.classify_grid_cells(target_set, 3, verbose=False, model=model))
            tp += len(expected & answers)
            fp += len(answers - expected)
            fn += len(expected - answers)
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        print(f"{cls:>15s}  {tp:>4d} {fp:>4d} {fn:>4d}  {p:>6.2f} {r:>6.2f}")
        total_tp += tp
        total_fp += fp
        total_fn += fn
    P = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    R = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    F = 2 * P * R / (P + R) if (P + R) else 0
    print(f"{'TOTAL':>15s}  {total_tp:>4d} {total_fp:>4d} {total_fn:>4d}  {P:>6.3f} {R:>6.3f}  F1={F:.3f}")
    return P, R, F


if __name__ == "__main__":
    from ultralytics import YOLO

    v1 = YOLO("models/recaptcha_classifier.pt")
    v4 = YOLO("models/recaptcha_classifier_v4.pt")

    print(f"v1 classes: {v1.names}")
    print(f"v4 classes: {v4.names}")

    rows = []
    for label, root, _, alias_fn in TEST_SETS:
        for tag, model in (("v1", v1), ("v4", v4)):
            p, r, f = evaluate(model, f"{tag} on {label}", root, alias_fn)
            rows.append((label, tag, p, r, f))

    print("\n=== summary ===")
    print(f"{'test set':<35s}  {'model':>5s}  {'prec':>6s}  {'rec':>6s}  {'F1':>6s}")
    for label, tag, p, r, f in rows:
        print(f"{label:<35s}  {tag:>5s}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}")
    # winner = best avg F1 across both test sets
    by_model = {}
    for _, tag, _, _, f in rows:
        by_model.setdefault(tag, []).append(f)
    print("\n=== avg F1 across both test sets ===")
    for tag, fs in by_model.items():
        print(f"{tag}: {sum(fs)/len(fs):.3f}")
