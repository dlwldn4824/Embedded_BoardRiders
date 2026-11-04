"""left 표지판 이미지를 좌우 반전해서 right 학습 데이터를 합성한다 (train 만).

좌회전 표지판을 거울에 비추면 우회전 표지판이 된다. 그래서 반전한 이미지에서는 left 라벨을 right 로 바꾼다.
반대로 글자가 있는 표지판(speed_20, stop)은 반전하면 글자가 뒤집히므로, 그런 표지판이 함께 찍힌 이미지는 건너뛴다.
val 은 실제 촬영 이미지만 남겨서 평가가 합성 데이터에 휘둘리지 않게 한다.

사용법:
    python sign/flip_left_to_right.py data/yolo_clean
"""
import argparse
from pathlib import Path

import cv2

SWAP = {"left": "right", "right": "left"}
NO_FLIP = {"speed_20", "stop"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()

    names = [l.strip()[2:] for l in (args.root / "data.yaml").read_text().splitlines() if l.startswith("  - ")]
    idx = {n: i for i, n in enumerate(names)}
    lab_dir, img_dir = args.root / "labels/train", args.root / "images/train"

    made = skipped = 0
    for lab in sorted(lab_dir.glob("*.txt")):
        if lab.stem.endswith("_flipR"):
            continue
        rows = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        classes = {names[int(r[0])] for r in rows}
        if "left" not in classes:
            continue
        if classes & NO_FLIP:
            skipped += 1
            continue
        img_path = next(img_dir.glob(f"{lab.stem}.*"))
        img = cv2.flip(cv2.imread(str(img_path)), 1)
        out = []
        for c, x, y, w, h in rows:
            n = names[int(c)]
            out.append(f"{idx[SWAP.get(n, n)]} {1 - float(x):.6f} {y} {w} {h}")
        stem = f"{lab.stem}_flipR"
        cv2.imwrite(str(img_dir / f"{stem}{img_path.suffix}"), img)
        (lab_dir / f"{stem}.txt").write_text("\n".join(out) + "\n")
        made += 1
    print(f"반전 합성 {made}장 (글자 표지판이 같이 있어 건너뜀 {skipped}장)")


if __name__ == "__main__":
    main()
