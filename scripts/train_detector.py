"""Train the YOLO26 road-sign detector on the converted GTSDB dataset, then evaluate it.

Prerequisite: python scripts/prepare_gtsdb.py   (creates data/processed/gtsdb_yolo_super/data.yaml)

Key settings and why:
  --imgsz 1024   signs are small (16-128 px in 1360x800 images); 640 would shrink them too much
  fliplr=0.0     horizontal flips change a sign's meaning (left/right) and are disabled
  --batch 4      fits in 4 GB of VRAM at imgsz 1024 (GTX 1650); use 8-16 on the RTX 4050 / Colab
  --amp off      GTX 16xx cards can produce NaN losses with mixed precision; enable on other GPUs

After training, the best checkpoint is evaluated ONCE on the official GTSDB test images,
and the results are copied to results/detector/<name>/ (committed) while the weights
go to models/weights/detector_<name>.pt (git-ignored).

Usage:
    python scripts/train_detector.py --epochs 3 --name smoke          # 5-minute check
    python scripts/train_detector.py --name v1                        # full run (GTX 1650)
    python scripts/train_detector.py --name v1 --batch 16 --amp       # RTX 4050 / Colab
    python scripts/train_detector.py --name v1 --resume               # continue an interrupted run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("train_detector")

PLOTS_TO_KEEP = ("results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png",
                 "labels.jpg", "val_batch0_pred.jpg", "val_batch0_labels.jpg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=REPO_ROOT / "data/processed/gtsdb_yolo_super/data.yaml")
    p.add_argument("--model", default=str(REPO_ROOT / "models/pretrained/yolo26n.pt"),
                   help="starting weights (COCO-pretrained: transfer learning)")
    p.add_argument("--name", default="v1", help="run name, e.g. v1, smoke, s_1280")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=30, help="stop if val mAP does not improve for N epochs")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="auto", help="'auto', 'cpu', '0'")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", help="mixed precision (not on GTX 16xx)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="resume runs/detect/<name>/weights/last.pt")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if not args.data.is_file():
        logger.error("Dataset config not found: %s. Run: python scripts/prepare_gtsdb.py", args.data)
        return 1

    import torch
    from ultralytics import YOLO

    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    project = REPO_ROOT / "runs" / "detect"
    run_dir = project / args.name
    logger.info("Device: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device != "cpu" else "")

    start = time.perf_counter()
    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.is_file():
            logger.error("Nothing to resume: %s not found", last)
            return 1
        YOLO(str(last)).train(resume=True)
    else:
        if run_dir.exists():
            logger.error("Run '%s' already exists in %s. Use another --name, or --resume.", args.name, run_dir)
            return 1
        YOLO(args.model).train(
            data=str(args.data), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
            batch=args.batch, device=device, workers=args.workers, amp=args.amp, seed=args.seed,
            deterministic=True, project=str(project), name=args.name, exist_ok=False,
            fliplr=0.0,  # never mirror road signs
            plots=True, verbose=True,
        )
    train_minutes = (time.perf_counter() - start) / 60

    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        logger.error("Training finished without %s", best)
        return 1

    # Single final evaluation on the held-out official test images.
    model = YOLO(str(best))
    m = model.val(data=str(args.data), split="test", imgsz=args.imgsz, batch=args.batch, device=device,
                  project=str(project), name=f"{args.name}_test", exist_ok=True, plots=True, verbose=False)
    names = model.names
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                    "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)}
    speed = getattr(m, "speed", {}) or {}

    metrics = {
        "description": "YOLO26 detector fine-tuned by the team on GTSDB, evaluated on the official test images.",
        "trained_by_team": True,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": args.name, "start_weights": Path(args.model).name, "data": str(args.data.name),
        "classes": list(names.values()), "imgsz": args.imgsz, "batch": args.batch,
        "epochs_requested": args.epochs, "device": device, "amp": args.amp,
        "test": {"precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
                 "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4)},
        "test_per_class": per_class,
        "inference_ms_per_image": round(float(speed.get("inference", 0.0)), 1),
        "training_minutes": round(train_minutes, 1),
    }

    out = REPO_ROOT / "results" / "detector" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for src_dir, prefix in ((run_dir, "train_"), (project / f"{args.name}_test", "test_")):
        for fname in PLOTS_TO_KEEP:
            if (src_dir / fname).is_file():
                shutil.copy2(src_dir / fname, out / f"{prefix}{fname}")
    if (run_dir / "results.csv").is_file():
        shutil.copy2(run_dir / "results.csv", out / "train_history.csv")
    weights_out = REPO_ROOT / "models" / "weights" / f"detector_{args.name}.pt"
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, weights_out)

    t = metrics["test"]
    print(f"\n=== YOLO detector '{args.name}' on GTSDB official test images ===")
    print(f"Precision {t['precision']:.3f} | Recall {t['recall']:.3f} | mAP50 {t['mAP50']:.3f} | "
          f"mAP50-95 {t['mAP50_95']:.3f}")
    for cls, v in per_class.items():
        print(f"  {cls:<12} P {v['precision']:.3f}  R {v['recall']:.3f}  mAP50 {v['mAP50']:.3f}")
    print(f"Inference: {metrics['inference_ms_per_image']} ms/image | training: {metrics['training_minutes']} min")
    print(f"Saved: {out}  and  {weights_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
