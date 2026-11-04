"""A_THR, 스캔 띠 y좌표 (그리고 MIN_RUN, 커널 크기)를 트랙바로 바꿔 보는 도구.

  python  -m drive.tune --input run_20261028_190928   # 이미지 폴더 (n / b: 다음 / 이전 이미지)
  python  -m drive.tune --input run.mp4               # 동영상 (space: 멈춤)
  python3 -m drive.tune --camera                      # Pi 카메라 실시간

파일 입력은 이미 바로 세운 것으로 본다. 카메라 원본(거꾸로)이면 --flip.

마우스를 올리면 그 위치의 LAB a* 값이 보인다.
w: 현재 값을 config.py 형식으로 출력, q: 종료. config.py는 직접 고치지 않는다.
"""
from __future__ import annotations

import argparse
import copy
import sys
import types
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from drive import config
from drive import vision
from drive.main import IMAGE_EXTS, CameraSource, _text, draw_mask, draw_overlay, imread

WIN = "tune"


def snapshot(mod):
    return types.SimpleNamespace(**{k: copy.deepcopy(getattr(mod, k)) for k in dir(mod) if k.isupper()})


def make_trackbars(cfg):
    limit = cfg.PROC_SIZE[1] - cfg.BOTTOM_EXCLUDE
    noop = lambda v: None  # noqa: E731
    cv2.namedWindow(WIN)
    cv2.createTrackbar("A_THR", WIN, cfg.A_THR, 255, noop)
    cv2.createTrackbar("MIN_RUN", WIN, cfg.MIN_RUN, 40, noop)
    cv2.createTrackbar("OPEN_K", WIN, cfg.OPEN_KERNEL, 21, noop)
    cv2.createTrackbar("CLOSE_K", WIN, cfg.CLOSE_KERNEL, 41, noop)
    for b in cfg.BANDS:
        cv2.createTrackbar(f"Y_{b['name']}", WIN, b["y"], limit - b["h"], noop)


def read_trackbars(cfg):
    cfg.A_THR = cv2.getTrackbarPos("A_THR", WIN)
    cfg.MIN_RUN = max(1, cv2.getTrackbarPos("MIN_RUN", WIN))
    cfg.OPEN_KERNEL = cv2.getTrackbarPos("OPEN_K", WIN)
    cfg.CLOSE_KERNEL = cv2.getTrackbarPos("CLOSE_K", WIN)
    for b in cfg.BANDS:
        b["y"] = cv2.getTrackbarPos(f"Y_{b['name']}", WIN)


def print_config(cfg, det):
    print("\n# ---- tune.py 결과 (config.py에 붙여 넣기) ----")
    print(f"A_THR = {cfg.A_THR}")
    print(f"MIN_RUN = {cfg.MIN_RUN}")
    print(f"OPEN_KERNEL = {cfg.OPEN_KERNEL}")
    print(f"CLOSE_KERNEL = {cfg.CLOSE_KERNEL}")
    print("BANDS = [")
    for b in cfg.BANDS:
        print(f'    {{"name": "{b["name"]}", "y": {b["y"]}, "h": {b["h"]}, "weight": {b["weight"]}}},')
    print("]")
    widths = [None if w is None else int(round(w)) for w in det.widths]
    print(f"ROAD_WIDTH_INIT = {widths}")


def main(argv=None):
    p = argparse.ArgumentParser(description="녹색 마스크 / 스캔 띠 튜닝")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="이미지 폴더, 이미지 파일 또는 동영상")
    g.add_argument("--camera", action="store_true", help="Pi 카메라 사용")
    p.add_argument("--flip", action="store_true", help="파일 입력이 카메라 원본(거꾸로)일 때")
    args = p.parse_args(argv)

    cfg = snapshot(config)
    det = vision.LaneDetector(cfg, flip=args.camera or args.flip)
    files, cap, cam = [], None, None
    if args.camera:
        cam = CameraSource(cfg)
    else:
        src = Path(args.input)
        if src.is_dir():
            files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        elif src.suffix.lower() in IMAGE_EXTS:
            files = [src]
        else:
            cap = cv2.VideoCapture(str(src))
            if not cap.isOpened():
                raise SystemExit(f"열 수 없는 입력: {src}")
        if cap is None and not files:
            raise SystemExit(f"이미지가 없다: {src}")

    make_trackbars(cfg)
    mouse = {"xy": None}

    def on_mouse(event, x, y, flags, param):
        mouse["xy"] = (x // cfg.DISPLAY_SCALE, y // cfg.DISPLAY_SCALE)

    cv2.setMouseCallback(WIN, on_mouse)

    idx, frame, paused = 0, None, False
    try:
        while True:
            if files:
                if frame is None:
                    frame = imread(files[idx])
                title = f"{files[idx].name} ({idx + 1}/{len(files)})"
            elif cap is not None:
                if not paused or frame is None:
                    ok, nxt = cap.read()
                    if not ok:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        det.reset()
                        continue
                    frame = nxt
                title = f"frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}" + (" (paused)" if paused else "")
            else:
                f = cam.read()
                if f is None:
                    if cv2.waitKey(30) in config.KEY_QUIT:
                        break
                    continue
                frame, title = f.image, "camera"

            read_trackbars(cfg)
            if files:
                det.reset()  # 이미지마다 새로 판단 (도로 폭도 그 이미지에서 학습)
            res = det.process(frame)
            view = draw_overlay(res, cfg, extra=[title])
            if mouse["xy"] is not None:
                x, y = mouse["xy"]
                if 0 <= x < res.image.shape[1] and 0 <= y < res.image.shape[0]:
                    lab = cv2.cvtColor(res.image[y:y + 1, x:x + 1], cv2.COLOR_BGR2LAB)[0, 0]
                    s = cfg.DISPLAY_SCALE
                    cv2.drawMarker(view, (x * s, y * s), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
                    _text(view, f"({x},{y}) L={lab[0]} a*={lab[1]} b*={lab[2]}",
                          (6, view.shape[0] - 10), (0, 255, 255))
            cv2.imshow(WIN, np.hstack([view, draw_mask(res, cfg)]))

            key = cv2.waitKey(30)
            if key in config.KEY_QUIT:
                break
            if key == ord("w"):
                print_config(cfg, det)
            elif files and key == ord("n"):
                idx, frame = (idx + 1) % len(files), None
            elif files and key == ord("b"):
                idx, frame = (idx - 1) % len(files), None
            elif cap is not None and key == ord(" "):
                paused = not paused
    finally:
        if cap is not None:
            cap.release()
        if cam is not None:
            cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
