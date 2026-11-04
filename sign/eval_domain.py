"""실차 카메라 run 에서의 오검출 세기 (라벨 없음).

run_20261028_191445 는 학습에 넣지 않은 실차 카메라 주행이다.
- 이 run 에는 멀리 보이는 횡단보도 말고 다른 표지판이 없다 (사진을 보고 확인함).
- 그래서 crosswalk 가 아닌 검출은 모두 오검출로 센다.
- crosswalk 검출 수는 참고용으로만 낸다 (진짜일 수 있다).

사용법:
    python sign/eval_domain.py runs/sign/v2/weights/best.pt weights/sign_v2_int8.tflite
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+")
    ap.add_argument("--run", type=Path, default=ROOT / "data/raw/run_20261028_191445")
    ap.add_argument("--conf", type=float, default=0.5, help="drive/config.py 의 SIGN_MIN_CONF")
    ap.add_argument("--imgsz", type=int, default=0, help="0 이면 모델 메타데이터 값")
    args = ap.parse_args()

    from ultralytics import YOLO

    frames = sorted(args.run.glob("*.png"))
    imgs = [cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR) for f in frames]
    print(f"{'weights':40s} {'FP imgs':>8s} {'FP boxes':>8s} {'crosswalk':>9s}  FP 클래스")
    for w in args.weights:
        m = YOLO(w, task="detect")
        kw = {"imgsz": args.imgsz} if args.imgsz else {}
        fp_img = fp_box = cw = 0
        cls = {}
        for img in imgs:
            r = m.predict(img, conf=args.conf, verbose=False, **kw)[0]
            names = [m.names[int(c)] for c in r.boxes.cls.tolist()]
            fps = [n for n in names if n != "crosswalk"]
            cw += len(names) - len(fps)
            fp_box += len(fps)
            fp_img += bool(fps)
            for n in fps:
                cls[n] = cls.get(n, 0) + 1
        print(f"{w:40s} {fp_img:8d} {fp_box:8d} {cw:9d}  {cls}  (/{len(frames)}장)")


if __name__ == "__main__":
    main()
