"""v3 학습 목록 만들기: yolo_clean_v2 train + 실차 카메라 배경 이미지 + red_line 오버샘플링.

- 배경: data/raw/run_20261028_191415 (실차 카메라, 어두운 도로) 중에서 고른다.
  - 기준 모델이 아무것도 못 잡은 프레임(conf 0.05)만 쓴다.
  - 그중에서도 crosswalk 가 잡힌 프레임에서 ±10 프레임 안은 뺀다 (멀리 진짜 횡단보도가 있을 수 있어서).
  - 라벨 없는 배경으로 넣어, 어두운 도로·의자·문틀 같은 데서 나는 오검출을 줄인다.
- run_20261028_191445 는 넣지 않는다. 오검출 평가용으로 남긴다 (sign/eval_domain.py).
- red_line 이 있는 train 이미지는 목록에 --red-line-repeat 번 넣는다.
- val 은 v2 와 같다 (비교가 공정하도록).
- data.yaml 에 절대 경로가 들어가므로 git 에 올리지 않는다. 다른 PC 에서는 이 스크립트를 다시 돌린다.

사용법:
    python sign/make_v3.py --ref runs/sign/v2/weights/best.pt
    → data/yolo_v3/{train.txt, data.yaml}
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
RED_LINE = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data/yolo_clean_v2")
    ap.add_argument("--bg-run", type=Path, default=ROOT / "data/raw/run_20261028_191415")
    ap.add_argument("--ref", type=Path, required=True, help="배경 고를 때 쓰는 기준 모델")
    ap.add_argument("--red-line-repeat", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "data/yolo_v3")
    args = ap.parse_args()

    from ultralytics import YOLO

    m = YOLO(args.ref)
    frames = sorted(args.bg_run.glob("*.png"))
    idx = lambda p: int(re.search(r"_(\d+)_", p.name).group(1))
    empty, cw = [], []
    for f in frames:
        img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
        r = m.predict(img, imgsz=640, conf=0.05, verbose=False)[0]
        if not len(r.boxes):
            empty.append(f)
        if any(m.names[int(c)] == "crosswalk" and cf >= 0.25 for c, cf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())):
            cw.append(idx(f))
    bg = [f for f in empty if all(abs(idx(f) - c) > 10 for c in cw)]

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    lines, n_red = [], 0
    for img in sorted((args.src / "images/train").iterdir()):
        lab = args.src / "labels/train" / (img.stem + ".txt")
        has_red = lab.exists() and any(l.split()[:1] == [str(RED_LINE)] for l in lab.read_text().splitlines())
        rep = args.red_line_repeat if has_red else 1
        n_red += has_red
        lines += [img] * rep
    lines += bg
    # txt 안의 "./" 경로는 txt 파일이 있는 폴더 기준이다.
    import os
    (out / "train.txt").write_text("\n".join("./" + Path(os.path.relpath(p, out)).as_posix() for p in lines) + "\n",
                                   encoding="utf-8")
    src_yaml = yaml.safe_load((args.src / "data.yaml").read_text(encoding="utf-8"))
    data = {"path": ".", "train": str((out / "train.txt").resolve()), "val": str((args.src / "images/val").resolve()),
            "nc": src_yaml["nc"], "names": src_yaml["names"]}
    (out / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"배경 후보 {len(empty)} / {len(frames)} 중 crosswalk 근처 제외 → {len(bg)}장")
    print(f"red_line 이미지 {n_red}장 × {args.red_line_repeat}")
    print(f"train 목록 {len(lines)}줄 → {out / 'train.txt'}")


if __name__ == "__main__":
    main()
