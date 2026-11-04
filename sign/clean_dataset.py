"""yolo_dataset 정리: 중복 제거, 라벨 없는 이미지 분리, 연속 프레임 블록 단위로 train/val 재분할.

원본 문제
- val 이미지 대부분이 train 이미지와 바이트 단위로 같음 → val mAP가 의미 없음
- 표지판이 찍혀 있는데 라벨 파일이 없는 이미지 → 그대로 학습하면 배경으로 학습됨
- 파일 이름이 {class}_{frame}.jpg 라서 이웃 프레임이 train/val에 섞임

사용법:
    python sign/clean_dataset.py data/raw/yolo_dataset data/yolo_clean
    python sign/clean_dataset.py data/raw/yolo_dataset data/yolo_clean --extra data/to_label_review data/sign_capture
    python sign/flip_left_to_right.py data/yolo_clean   # 재분할 후 다시 실행
"""
import argparse
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--block", type=int, default=40, help="연속 프레임을 묶는 블록 크기")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", type=Path, nargs="*", default=[],
                    help="라벨링을 마친 추가 폴더 (이미지와 txt 가 같은 폴더)")
    args = ap.parse_args()

    names = (args.src / "labels/train/classes.txt").read_text().split()

    # 1) 모든 이미지를 모아서 바이트 중복 제거 (라벨은 train 쪽 우선)
    items, seen = {}, set()
    unlabeled = {}   # md5 -> 이미지. 나중에 --extra 에서 라벨이 생기면 빠진다
    for split in ("train", "val"):
        for img in sorted((args.src / "images" / split).iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            h = md5(img)
            if h in seen or h in unlabeled:
                continue
            lab = args.src / "labels" / split / f"{img.stem}.txt"
            stem = img.stem if img.stem not in items else f"{img.stem}_{split}"
            if img.stem.startswith("none"):
                items[stem] = (img, None)  # 진짜 배경 이미지
            elif lab.exists():
                items[stem] = (img, lab)
            else:
                unlabeled[h] = img
                continue
            seen.add(h)

    # 1-1) 추가 데이터: 이미지와 txt 가 같은 폴더에 있는 형식 (labelImg 결과, sign/capture.py 촬영분)
    for extra in args.extra:
        for img in sorted(extra.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            h = md5(img)
            if h in seen:
                continue
            lab = img.with_suffix(".txt")
            if not lab.exists():
                unlabeled.setdefault(h, img)
                continue
            seen.add(h)
            unlabeled.pop(h, None)   # 원본에서 라벨이 없던 이미지를 여기서 라벨링한 경우
            stem = img.stem if img.stem not in items else f"{img.stem}_{extra.name}"
            items[stem] = (img, lab)

    # 2) {class}_{frame} 을 블록으로 묶어서 블록 단위로 val 배정
    blocks = defaultdict(list)
    for stem in items:
        prefix, _, num = stem.rpartition("_")
        if not num.isdigit():
            prefix, _, num = stem.rsplit("_", 2)[0], "", stem.rsplit("_", 2)[1]
        blocks[(prefix, int(num) // args.block)].append(stem)
    keys = sorted(blocks)
    rng = random.Random(args.seed)
    by_class = defaultdict(list)
    for k in keys:
        by_class[k[0]].append(k)
    val_keys = set()
    for cls_keys in by_class.values():
        rng.shuffle(cls_keys)
        n = max(1, round(len(cls_keys) * args.val_ratio))
        val_keys.update(cls_keys[:n])

    # 3) 출력
    if args.dst.exists():
        shutil.rmtree(args.dst)
    count = defaultdict(int)
    for k in keys:
        split = "val" if k in val_keys else "train"
        for stem in blocks[k]:
            img, lab = items[stem]
            (args.dst / "images" / split).mkdir(parents=True, exist_ok=True)
            (args.dst / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, args.dst / "images" / split / f"{stem}{img.suffix}")
            dst_lab = args.dst / "labels" / split / f"{stem}.txt"
            if lab:
                shutil.copy2(lab, dst_lab)
            else:
                dst_lab.write_text("")
            count[split] += 1

    todo = args.dst / "to_label"
    todo.mkdir(parents=True, exist_ok=True)
    for img in unlabeled.values():
        shutil.copy2(img, todo / img.name)

    # path 를 쓰지 않으면 ultralytics 가 data.yaml 이 있는 폴더를 기준으로 삼는다 (다른 PC 로 옮겨도 동작)
    (args.dst / "data.yaml").write_text(
        f"train: images/train\nval: images/val\n\nnc: {len(names)}\nnames:\n"
        + "".join(f"  - {n}\n" for n in names)
    )
    print(f"고유 이미지 {len(items)}장 → train {count['train']}, val {count['val']}")
    print(f"라벨 없음(to_label/로 분리) {len(unlabeled)}장")


if __name__ == "__main__":
    main()
