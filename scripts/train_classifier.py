"""
Fine-tune a YOLOv8 cell-classifier on a folder-style reCAPTCHA dataset.

Layout expected (HF `verytuffcat/recaptcha-dataset` shape):

    <SOURCE>/data/train/<class>/*.png

Output:
    models/recaptcha_classifier.pt          (PyTorch weights for inference)
    models/recaptcha_classifier.onnx        (optional ONNX export)

The trained model is plug-compatible with `recaptcha_ia_solver.solver` once
RECAPTCHA_YOLO_MODEL points at the .pt file (the solver auto-detects the
classify task and switches to per-cell inference).
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

# Default split: 80% train / 20% val.
VAL_FRACTION = 0.2
RANDOM_SEED = 17


def split_dataset(src: Path, dst: Path) -> dict[str, dict[str, int]]:
    """Materialize a YOLO classification layout (train/val) from a flat class folder."""
    if dst.exists():
        shutil.rmtree(dst)
    rng = random.Random(RANDOM_SEED)
    counts: dict[str, dict[str, int]] = {}
    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        files = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not files:
            continue
        rng.shuffle(files)
        n_val = max(1, int(len(files) * VAL_FRACTION)) if len(files) > 1 else 0
        val_files = files[:n_val]
        train_files = files[n_val:] or files
        for split, items in (("train", train_files), ("val", val_files)):
            out = dst / split / class_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, out / f.name)
        counts[class_dir.name] = {"train": len(train_files), "val": len(val_files)}
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/tmp/recaptcha_ds/verytuffcat/data/train"))
    parser.add_argument(
        "--prebuilt",
        type=Path,
        default=None,
        help="Path to a dataset already split as <root>/train/<class> + <root>/val/<class>; "
             "if set, --source is ignored.",
    )
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/recaptcha_cls_ws"))
    parser.add_argument("--base-model", type=str, default="yolov8s-cls.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--out", type=Path, default=Path("models/recaptcha_classifier.pt"))
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    if args.prebuilt is not None:
        dataset_root = args.prebuilt.resolve()
        if not (dataset_root / "train").exists() or not (dataset_root / "val").exists():
            raise SystemExit(
                f"--prebuilt expects {dataset_root}/train and /val subfolders"
            )
        print(f"[1/3] using prebuilt split at {dataset_root}")
    else:
        src = args.source.resolve()
        if not src.exists():
            raise SystemExit(f"source does not exist: {src}")
        dataset_root = (args.workdir / "dataset").resolve()
        print(f"[1/3] splitting {src} -> {dataset_root}")
        counts = split_dataset(src, dataset_root)
        for k, v in counts.items():
            print(f"     {k:>15s} train={v['train']:>4d} val={v['val']:>4d}")

    # Lazy import keeps the script importable without ultralytics installed.
    from ultralytics import YOLO

    print(f"[2/3] training {args.base_model} for {args.epochs} epochs")
    model = YOLO(args.base_model)
    results = model.train(
        data=str(dataset_root),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.workdir / "runs"),
        name="recaptcha_cls",
        exist_ok=True,
        verbose=True,
        patience=15,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (args.workdir / "runs" / "recaptcha_cls")
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise SystemExit(f"training finished but best.pt not found at {best_pt}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, args.out)
    print(f"[3/3] saved best weights to {args.out}")

    if args.export_onnx:
        export_model = YOLO(str(args.out))
        onnx_path = export_model.export(format="onnx", imgsz=args.imgsz, dynamic=False)
        target = args.out.with_suffix(".onnx")
        if Path(onnx_path) != target:
            shutil.copy2(onnx_path, target)
        print(f"     ONNX -> {target}")

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
