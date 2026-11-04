"""여러 가중치를 같은 val 에서 CPU, batch 1 로 비교한다 (크기 / mAP50 / mAP50-95 / 추론 시간).

사용법 (.venv-tf):
    python sign/compare_int8.py runs/sign/v2/weights/best.pt weights/sign_v2_fp32.tflite weights/sign_v2_int8.tflite
"""
import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def bench(model, imgsz, n=50):
    """전처리를 뺀 순수 추론 시간(ms, 중앙값). 랜덤 입력."""
    img = (np.random.default_rng(0).random((imgsz, imgsz, 3)) * 255).astype(np.uint8)
    for _ in range(5):
        model.predict(img, imgsz=imgsz, device="cpu", verbose=False)
    ts = []
    for _ in range(n):
        r = model.predict(img, imgsz=imgsz, device="cpu", verbose=False)[0]
        ts.append(r.speed["inference"])
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+", type=Path)
    ap.add_argument("--data", type=Path, default=ROOT / "data/yolo_clean_v2/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    rows = []
    for w in args.weights:
        m = YOLO(str(w), task="detect")
        met = m.val(data=str(args.data), imgsz=args.imgsz, batch=1, device="cpu", split="val",
                    plots=False, verbose=False, project=str(ROOT / "runs/sign"), name=f"cmp_{w.stem}", exist_ok=True)
        per_cls = {met.names[int(c)]: met.box.class_result(i)[2] for i, c in enumerate(met.box.ap_class_index)}
        rows.append((w.name, w.stat().st_size / 1e6, met.box.map50, met.box.map, bench(m, args.imgsz), per_cls))

    print(f"\n{'weights':28s} {'MB':>6s} {'mAP50':>6s} {'50-95':>6s} {'ms':>6s}")
    for name, mb, m50, m, ms, _ in rows:
        print(f"{name:28s} {mb:6.2f} {m50:6.3f} {m:6.3f} {ms:6.1f}")
    print("\nclass mAP50: " + " | ".join(r[0] for r in rows))
    for c in rows[0][5]:
        print(f"{c:12s} " + " ".join(f"{r[5].get(c, float('nan')):6.3f}" for r in rows))


if __name__ == "__main__":
    main()
