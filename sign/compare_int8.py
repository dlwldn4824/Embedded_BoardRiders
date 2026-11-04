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


def recall_at(model, data_yaml, imgsz, conf=0.5, iou_thr=0.5):
    """conf 이상 검출로 정답 박스(같은 클래스, IoU > iou_thr)를 찾은 비율. 주행 코드 기준(SIGN_MIN_CONF)과 맞춘다."""
    import cv2
    import yaml

    d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    val = Path(d["val"]) if Path(d["val"]).is_absolute() else Path(data_yaml).parent / d.get("path", ".") / d["val"]
    hit, tot = {}, {}
    for p in sorted(val.iterdir()):
        lab = val.parent.parent / "labels" / val.name / (p.stem + ".txt")
        if not lab.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        r = model.predict(img, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
        det = list(zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()))
        for line in lab.read_text().splitlines():
            c, x, y, bw, bh = map(float, line.split())
            g = ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)
            n = model.names[int(c)]
            tot[n] = tot.get(n, 0) + 1
            for dc, b in det:
                ix = max(0, min(g[2], b[2]) - max(g[0], b[0])) * max(0, min(g[3], b[3]) - max(g[1], b[1]))
                u = (g[2] - g[0]) * (g[3] - g[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
                if int(dc) == int(c) and ix / (u + 1e-9) > iou_thr:
                    hit[n] = hit.get(n, 0) + 1
                    break
    return sum(hit.values()) / sum(tot.values()), {k: hit.get(k, 0) / v for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+", type=Path)
    ap.add_argument("--data", type=Path, default=ROOT / "data/yolo_clean_v2/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5, help="재현율을 잴 기준 (drive/config.py SIGN_MIN_CONF)")
    args = ap.parse_args()

    from ultralytics import YOLO

    rows = []
    for w in args.weights:
        m = YOLO(str(w), task="detect")
        met = m.val(data=str(args.data), imgsz=args.imgsz, batch=1, device="cpu", split="val",
                    plots=False, verbose=False, project=str(ROOT / "runs/sign"), name=f"cmp_{w.stem}", exist_ok=True)
        per_cls = {met.names[int(c)]: met.box.class_result(i)[2] for i, c in enumerate(met.box.ap_class_index)}
        rec, rec_cls = recall_at(m, args.data, args.imgsz, conf=args.conf)
        rows.append((w.name, w.stat().st_size / 1e6, met.box.map50, met.box.map, bench(m, args.imgsz), per_cls,
                     rec, rec_cls))

    print(f"\n{'weights':28s} {'MB':>6s} {'mAP50':>6s} {'50-95':>6s} {'ms':>6s}  "
          f"recall@{args.conf} (전체 / crosswalk / red_line)")
    for name, mb, m50, m, ms, _, rec, rc in rows:
        print(f"{name:28s} {mb:6.2f} {m50:6.3f} {m:6.3f} {ms:6.1f}  "
              f"{rec:.3f} / {rc.get('crosswalk', 0):.2f} / {rc.get('red_line', 0):.2f}")
    print("\nclass mAP50: " + " | ".join(r[0] for r in rows))
    for c in rows[0][5]:
        print(f"{c:12s} " + " ".join(f"{r[5].get(c, float('nan')):6.3f}" for r in rows))


if __name__ == "__main__":
    main()
