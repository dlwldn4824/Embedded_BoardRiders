"""Pi 에서 표지판 모델 속도 재기. 데이터 없이 돈다.

ultralytics predict 한 번(전처리 + 추론 + NMS)에 걸리는 시간을 잰다. 주행 코드가 부르는 것과 같은 호출이다.
시뮬레이터 결과(docs/sim-signs.md): 표지판 검출 한 번이 0.3초 이하여야 미션이 통과한다.

사용법 (레포 루트에서):
    python3 sign/pi_bench.py weights/sign_v3_8n_int8.tflite weights/sign_v3_26n_320_int8.tflite
"""
import argparse
import json
import statistics
import time
import zipfile
from pathlib import Path

import numpy as np

LIMIT_MS = 300   # docs/sim-signs.md


def model_imgsz(path):
    """tflite 뒤에 붙여 둔 metadata.json 에서 입력 크기를 읽는다 (sign/export_int8.py 가 붙임)."""
    try:
        with zipfile.ZipFile(path) as z:
            return int(json.loads(z.read("metadata.json"))["imgsz"][0])
    except Exception:
        return 640


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+", type=Path)
    ap.add_argument("--n", type=int, default=30, help="잴 횟수")
    args = ap.parse_args()

    from ultralytics import YOLO

    rng = np.random.default_rng(0)
    frame = (rng.random((480, 640, 3)) * 255).astype(np.uint8)   # 카메라와 같은 640x480
    rows = []
    for w in args.weights:
        imgsz = model_imgsz(w)
        m = YOLO(str(w), task="detect")
        for _ in range(3):                                     # 첫 몇 번은 느리다
            m.predict(frame, imgsz=imgsz, verbose=False)
        ts = []
        for _ in range(args.n):
            t0 = time.perf_counter()
            m.predict(frame, imgsz=imgsz, verbose=False)
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        med, p90 = statistics.median(ts), ts[int(len(ts) * 0.9) - 1]
        rows.append((w.name, imgsz, med, p90))

    print(f"\n{'weights':32s} {'imgsz':>5s} {'중앙값 ms':>9s} {'p90 ms':>7s}  판정 (기준 {LIMIT_MS}ms)")
    for name, imgsz, med, p90 in rows:
        verdict = "OK" if p90 <= LIMIT_MS else ("아슬아슬" if med <= LIMIT_MS else "느림 → 320 모델 검토")
        print(f"{name:32s} {imgsz:5d} {med:9.0f} {p90:7.0f}  {verdict}")
    print("\n주행 중에는 차선 처리와 CPU 를 나눠 쓰므로 실제로는 조금 더 느리다. "
          "drive.sign_check 로 주행 코드 경로의 시간도 확인한다.")


if __name__ == "__main__":
    main()
