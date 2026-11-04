"""주행 데이터 점검: run 폴더별 각도 분포, 프레임 연속성, 이미지 크기, 각도별 샘플 몽타주.

파일 이름 형식: train_{frame:05d}_{angle:03d}.png

사용법:
    python lane/inspect_data.py data/raw
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

NAME_RE = re.compile(r"train_(\d+)_(\d+)\.png$")


def load_run(run_dir):
    frames = []
    for p in sorted(run_dir.glob("*.png")):
        m = NAME_RE.search(p.name)
        if m:
            frames.append((int(m.group(1)), int(m.group(2)), p))
    frames.sort()
    return frames


def segments(angles):
    """같은 각도가 연속되는 구간 길이 목록."""
    out = []
    for a in angles:
        if out and out[-1][0] == a:
            out[-1][1] += 1
        else:
            out.append([a, 1])
    return out


def montage(frames, out_path, per_angle=6, width=240):
    by_angle = {}
    for _, a, p in frames:
        by_angle.setdefault(a, []).append(p)
    rows = []
    for a in sorted(by_angle):
        paths = by_angle[a]
        idx = np.linspace(0, len(paths) - 1, min(per_angle, len(paths))).astype(int)
        tiles = []
        for i in idx:
            img = cv2.imread(str(paths[i]))
            h = int(img.shape[0] * width / img.shape[1])
            img = cv2.resize(img, (width, h))
            cv2.putText(img, f"{a}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            tiles.append(img)
        while len(tiles) < per_angle:
            tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(out_path), np.vstack(rows))


def inspect_new_runs(runs):
    """deploy/collect.py 형식 (labels.csv + meta.json) 점검."""
    import csv
    import json

    for run in runs:
        rows = list(csv.DictReader(open(run / "labels.csv")))
        meta = json.loads((run / "meta.json").read_text())
        bad = [(float(a), float(b)) for a, b in list(csv.reader(open(run / "bad.csv")))[1:]]
        t = np.array([float(r["t"]) for r in rows])
        s = np.array([float(r["steer"]) for r in rows])
        is_bad = np.zeros(len(rows), bool)
        for a, b in bad:
            is_bad |= (t >= a) & (t <= b)
        hist, _ = np.histogram(s[~is_bad], bins=[-1, -0.6, -0.2, 0.2, 0.6, 1.0001])
        print(f"\n== {run.name}  방향 {meta['direction']}  메모 '{meta['note']}'")
        print(f"  {len(rows)}장, 무효 {is_bad.sum()}장, 복구 태그 {np.mean([r['recovery'] == '1' for r in rows]):.0%}, "
              f"실제 fps {len(rows) / max(1e-6, t[-1] - t[0]):.1f}" if rows else "  프레임 없음")
        print("  조향 분포 (강좌/약좌/직진/약우/강우):", hist.tolist())
        print("  카메라 고정값:", meta.get("camera_locked"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/data_report"))
    args = ap.parse_args()

    new_runs = sorted(p.parent for p in args.root.rglob("labels.csv"))
    if new_runs:
        inspect_new_runs(new_runs)
    runs = sorted({p.parent for p in args.root.rglob("train_*.png")})
    if not runs:
        if new_runs:
            return
        sys.exit(f"{args.root} 아래에 train_*.png 나 labels.csv 가 없습니다")
    args.out.mkdir(parents=True, exist_ok=True)

    total = Counter()
    for run in runs:
        frames = load_run(run)
        nums = [f for f, _, _ in frames]
        angles = [a for _, a, _ in frames]
        cnt = Counter(angles)
        total.update(cnt)
        gaps = [b - a for a, b in zip(nums, nums[1:]) if b - a != 1]
        shapes = Counter(cv2.imread(str(p)).shape for _, _, p in frames[:: max(1, len(frames) // 20)])
        segs = segments(angles)

        print(f"\n== {run.name}  ({len(frames)}장, 프레임 {nums[0]}~{nums[-1]})")
        print("  각도 분포:", {a: f"{n} ({n / len(frames):.0%})" for a, n in sorted(cnt.items())})
        print(f"  프레임 번호 끊김: {len(gaps)}곳" + (f" (예: {gaps[:5]})" if gaps else ""))
        print("  이미지 크기:", dict(shapes))
        print(f"  같은 각도 연속 구간: {len(segs)}개, 길이 중앙값 {int(np.median([n for _, n in segs]))}")
        print("  구간 순서(앞 20개):", " ".join(f"{a}x{n}" for a, n in segs[:20]))
        montage(frames, args.out / f"{run.name}_montage.jpg")

    n = sum(total.values())
    print(f"\n== 전체 {n}장:", {a: f"{c} ({c / n:.0%})" for a, c in sorted(total.items())})
    print(f"몽타주 저장: {args.out}/")


if __name__ == "__main__":
    main()
