"""라벨 없는 이미지를 학습한 모델로 미리 라벨링한다. 사람은 labelImg 에서 고치기만 하면 된다.

출력 폴더에 이미지 + YOLO txt + classes.txt 를 같이 둔다 (labelImg 가 바로 여는 형식).
review.csv 에는 검토 우선순위를 적는다: 파일 이름의 클래스(예: horn_0687 → horn)가 검출되지 않은 이미지가 먼저.

사용법:
    python sign/prelabel.py runs/sign/baseline/weights/best.pt data/yolo_clean/to_label data/to_label_review
    labelImg data/to_label_review data/to_label_review/classes.txt data/to_label_review
"""
import argparse
import csv
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = model.names
    args.dst.mkdir(parents=True, exist_ok=True)
    (args.dst / "classes.txt").write_text("\n".join(names[i] for i in range(len(names))) + "\n")

    rows = []
    imgs = sorted(p for p in args.src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for res in model.predict([str(p) for p in imgs], conf=args.conf, imgsz=args.imgsz, device=args.device,
                             stream=True, verbose=False):
        src = Path(res.path)
        shutil.copy2(src, args.dst / src.name)
        lines, found = [], []
        for c, xywhn, cf in zip(res.boxes.cls.tolist(), res.boxes.xywhn.tolist(), res.boxes.conf.tolist()):
            lines.append(f"{int(c)} " + " ".join(f"{v:.6f}" for v in xywhn))
            found.append((names[int(c)], cf))
        (args.dst / f"{src.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        expected = src.stem.rsplit("_", 1)[0]
        hit = [cf for n, cf in found if n == expected]
        rows.append({
            "image": src.name, "expected": expected,
            "expected_found": int(bool(hit)), "expected_conf": f"{max(hit):.2f}" if hit else "",
            "detected": " ".join(f"{n}:{cf:.2f}" for n, cf in found),
        })

    rows.sort(key=lambda r: (r["expected_found"], r["expected_conf"]))
    with open(args.dst / "review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    miss = sum(1 for r in rows if not r["expected_found"])
    print(f"{len(rows)}장 사전 라벨링 → {args.dst}")
    print(f"파일 이름의 클래스를 못 찾은 이미지 {miss}장 (review.csv 위쪽부터 확인)")


if __name__ == "__main__":
    main()
