"""표지판 촬영 (Raspberry Pi). 차량 카메라로, 차선 수집과 같은 카메라 고정값으로 찍는다.

파일 이름은 기존 데이터와 같은 {class}_{번호}.jpg 이고, 번호는 이미 있는 파일 다음부터 이어진다.
clean_dataset.py 가 이 번호로 연속 프레임 블록을 만들어서 train/val 을 나누므로, 장면을 바꿀 때마다
번호를 크게 건너뛰는 --gap 을 쓰면 블록이 장면 단위로 나뉜다.

사용법:
    python sign/capture.py --cls right            # 초당 3장, Ctrl+C 로 끝
    python sign/capture.py --cls right --fake --seconds 2
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy"))
from hw import Camera  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True, help="찍는 표지판 클래스 (예: right)")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=ROOT / "data/sign_capture")
    ap.add_argument("--gap", type=int, default=40, help="시작 번호를 이만큼 건너뛰어 이전 장면과 블록을 분리")
    ap.add_argument("--config", type=Path, default=ROOT / "deploy/config.yaml")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--seconds", type=float)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if not args.fake and any(cfg["camera"].get(k) is None for k in ("exposure_time", "analogue_gain", "colour_gains")):
        print("경고: config.yaml 에 카메라 고정값이 비어 있습니다. 차선 수집 run 의 meta.json 값을 먼저 적으세요.")
    args.out.mkdir(parents=True, exist_ok=True)
    nums = [int(p.stem.rsplit("_", 1)[1]) for p in args.out.glob(f"{args.cls}_*.jpg")]
    n = (max(nums) + 1 + args.gap) if nums else 0

    cam = Camera(cfg["camera"], fake=args.fake)
    print(f"{args.cls}_{n:04d} 부터 저장. 차를 조금씩 움직이며 거리·각도·위치를 바꿔 주세요 (Ctrl+C 종료)")
    t_start, saved = time.time(), 0
    try:
        while not (args.seconds and time.time() - t_start > args.seconds):
            t0 = time.time()
            cv2.imwrite(str(args.out / f"{args.cls}_{n:04d}.jpg"), cam.read(), [cv2.IMWRITE_JPEG_QUALITY, 92])
            n, saved = n + 1, saved + 1
            print(f"\r저장 {saved}장", end="", flush=True)
            time.sleep(max(0.0, 1 / args.fps - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
    print(f"\n{args.out} 에 {saved}장 저장. labelImg 로 라벨링 후 clean_dataset.py --extra 로 합치세요.")


if __name__ == "__main__":
    main()
