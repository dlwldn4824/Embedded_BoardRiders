"""v4 학습 목록: v3 방식 + "신호등 아래쪽(빨강 등)만 보이는 장면" + 실제 신호등 사진.

왜: 차가 정지선 근처에 서면 카메라에 신호등 위쪽(초록 등)이 안 보이고 아래쪽 빨강 등만 보인다.
그때 꺼진 빨강 등을 켜진 빨강(red_light)으로 잘못 봤다. 위쪽을 안 봐도 빨강 등이 켜졌는지 꺼졌는지로
구분하도록 가르친다. 클래스는 그대로다 (빨강 등 꺼짐 = 초록 신호 = green_light).

1. 신호등 아래쪽 자르기: green_light / red_light 상자가 있는 사진에서, 신호등 상자의 위쪽 CUT(45~60%)를
   잘라 낸 사진을 만든다. 라벨은 남은 신호등 부분(같은 클래스). 다른 물체는 60% 넘게 보이면 잘라서 남긴다.
   - green_light 사진을 자르면 꺼진 빨강 등만 보인다 → green_light
   - red_light 사진을 자르면 켜진 빨강 등만 보인다 → red_light
   신호등은 위부터 초록·노랑·빨강 순서다 (실습 신호등 사진).
2. 실제 신호등 사진 (data/raw/lights_20261110/*.jpg, 라벨 같은 이름 .txt) 을 --photo-repeat 번 넣는다.
3. 나머지는 v3 와 같다: yolo_clean_v2 train, red_line 오버샘플링, (선택) 실차 카메라 배경.
4. val 은 v2/v3 와 같다. 따로 "아래쪽만 자른 신호등" val (light_crop) 을 만들어 꺼짐/켜짐 구분을 잰다.

사용법:
    python sign/make_v4.py                              # 배경 없이
    python sign/make_v4.py --ref runs/sign/v2/weights/best.pt   # v3 처럼 실차 배경도
    → data/yolo_v4/{data.yaml, data_lightcrop.yaml, train.txt, crops/...}
학습 (Pi 기본 모델과 같은 320, v3 가중치에서 이어서):
    python sign/train.py --data data/yolo_v4/data.yaml --model weights/sign_v3_26n.pt --imgsz 320 \\
        --name v4_26n_320 --epochs 30 --device 0 --hsv-v 0.6
    python sign/export_int8.py runs/sign/v4_26n_320/weights/best.pt --imgsz 320 --out weights/sign_v4_26n_320_int8.tflite
꺼짐/켜짐 구분 확인:
    yolo val model=weights/sign_v4_26n_320_int8.tflite data=data/yolo_v4/data_lightcrop.yaml imgsz=320
"""
import argparse
import os
import random
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
GREEN, RED, RED_LINE = 1, 4, 5


def read_labels(p):
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        v = line.split()
        if len(v) >= 5:
            out.append([int(v[0])] + [float(x) for x in v[1:5]])
    return out


def crop_lower(img, labels, light, cut, rng):
    """light 상자의 위쪽 cut 비율을 잘라 낸 사진과 라벨. 가로는 그대로 두고 세로만 자른다."""
    h, w = img.shape[:2]
    c, cx, cy, bw, bh = light
    y1, y2 = (cy - bh / 2) * h, (cy + bh / 2) * h
    top = int(round(y1 + cut * (y2 - y1)))
    # 신호등이 화면 위 끝에 걸리게 (실차에서 위쪽이 화면 밖으로 나간 모습). 아래는 원래 사진 끝까지
    if h - top < 0.25 * h:
        return None, None
    crop = img[top:]
    ch = crop.shape[0]
    out = []
    for lab in labels:
        k, lx, ly, lw, lh = lab
        ly1, ly2 = (ly - lh / 2) * h - top, (ly + lh / 2) * h - top
        vis1, vis2 = max(0.0, ly1), min(ch, ly2)
        if vis2 - vis1 <= 0:
            continue
        frac = (vis2 - vis1) / max(ly2 - ly1, 1e-6)
        if lab is not light and frac < 0.6:
            continue
        out.append(f"{k} {lx:.6f} {(vis1 + vis2) / 2 / ch:.6f} {lw:.6f} {(vis2 - vis1) / ch:.6f}")
    return crop, out


def make_crops(src, split, out_dir, rng, per_light):
    img_dir, lab_dir = out_dir / "images" / split, out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    made = {GREEN: 0, RED: 0}
    for img_p in sorted((src / "images" / split).iterdir()):
        labels = read_labels(src / "labels" / split / (img_p.stem + ".txt"))
        lights = [l for l in labels if l[0] in (GREEN, RED) and l[4] > 0.05]
        if not lights:
            continue
        img = cv2.imdecode(np.fromfile(str(img_p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for li, light in enumerate(lights):
            for j in range(per_light):
                cut = rng.uniform(0.45, 0.6)
                crop, lab = crop_lower(img, labels, light, cut, rng)
                if crop is None:
                    continue
                name = f"{img_p.stem}_low{li}{j}"
                cv2.imencode(".jpg", crop)[1].tofile(str(img_dir / f"{name}.jpg"))
                (lab_dir / f"{name}.txt").write_text("\n".join(lab) + "\n")
                made[light[0]] += 1
    return img_dir, made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data/yolo_clean_v2")
    ap.add_argument("--photos", type=Path, default=ROOT / "data/raw/lights_20261110")
    ap.add_argument("--photo-repeat", type=int, default=3)
    ap.add_argument("--crop-per-light", type=int, default=2, help="신호등 하나당 아래쪽 자른 사진 수 (자르는 높이를 바꿔 가며)")
    ap.add_argument("--red-line-repeat", type=int, default=3)
    ap.add_argument("--bg-run", type=Path, default=ROOT / "data/raw/run_20261028_191415")
    ap.add_argument("--ref", type=Path, default=None, help="주면 v3 처럼 실차 배경도 넣는다 (배경 고를 기준 모델)")
    ap.add_argument("--out", type=Path, default=ROOT / "data/yolo_v4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    lines, n_red = [], 0
    for img in sorted((args.src / "images/train").iterdir()):
        lab = args.src / "labels/train" / (img.stem + ".txt")
        has_red = any(l[0] == RED_LINE for l in read_labels(lab))
        lines += [img] * (args.red_line_repeat if has_red else 1)
        n_red += has_red

    crops = out / "crops"
    tr_dir, tr_made = make_crops(args.src, "train", crops, rng, args.crop_per_light)
    va_dir, va_made = make_crops(args.src, "val", crops, rng, 1)
    lines += sorted(tr_dir.glob("*.jpg"))

    photos = sorted(args.photos.glob("*.jpg")) if args.photos.exists() else []
    photos = [p for p in photos if p.with_suffix(".txt").exists()]
    lines += photos * args.photo_repeat

    bg = []
    if args.ref:
        from ultralytics import YOLO
        m = YOLO(args.ref)
        frames = sorted(args.bg_run.glob("*.png"))
        idx = lambda p: int(re.search(r"_(\d+)_", p.name).group(1))  # noqa: E731
        empty, cw = [], []
        for f in frames:
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            r = m.predict(img, imgsz=640, conf=0.05, verbose=False)[0]
            if not len(r.boxes):
                empty.append(f)
            if any(m.names[int(c)] == "crosswalk" and cf >= 0.25 for c, cf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())):
                cw.append(idx(f))
        bg = [f for f in empty if all(abs(idx(f) - c) > 10 for c in cw)]
        lines += bg

    rel = lambda p: "./" + Path(os.path.relpath(p, out)).as_posix()  # noqa: E731
    (out / "train.txt").write_text("\n".join(rel(p) for p in lines) + "\n", encoding="utf-8")
    src_yaml = yaml.safe_load((args.src / "data.yaml").read_text(encoding="utf-8"))
    base = {"path": ".", "nc": src_yaml["nc"], "names": src_yaml["names"]}
    (out / "data.yaml").write_text(yaml.safe_dump(
        {**base, "train": str((out / "train.txt").resolve()), "val": str((args.src / "images/val").resolve())},
        allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "data_lightcrop.yaml").write_text(yaml.safe_dump(
        {**base, "train": str(va_dir.resolve()), "val": str(va_dir.resolve())},
        allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"red_line 이미지 {n_red}장 x {args.red_line_repeat}")
    print(f"신호등 아래쪽 자른 사진 train: 꺼진 빨강(green_light) {tr_made[GREEN]}장, 켜진 빨강(red_light) {tr_made[RED]}장")
    print(f"신호등 아래쪽 자른 사진 val (light_crop): 꺼진 빨강 {va_made[GREEN]}장, 켜진 빨강 {va_made[RED]}장")
    print(f"실제 신호등 사진 {len(photos)}장 x {args.photo_repeat}, 실차 배경 {len(bg)}장")
    print(f"train 목록 {len(lines)}줄 → {out / 'train.txt'}")


if __name__ == "__main__":
    main()
