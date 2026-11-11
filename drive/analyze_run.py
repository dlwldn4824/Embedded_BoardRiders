"""녹화한 주행을 프레임마다 다시 돌려, 초록 경계가 언제 안 보이고 왜 길을 잃는지 본다.

입력: --record 로 저장한 폴더 (000123.jpg, 번호 = CSV 의 fid) 또는 사람이 운전한 사진 폴더 (train_00012_090.png)
     같은 실행의 CSV (--csv) 를 주면 그때 실제 동작(action)·예약 방향(pending)·표지판/회전 이유(note)를 같이 붙인다.
출력 (--out 폴더):
  analysis.csv  프레임마다 상태·오차·띠별 경계 위치·도로 폭(px, cm 추정)·회전할 쪽이 진짜 열림/모름/닫힘·실제 동작
  timeline.txt  회전할 쪽 초록 경계가 사라진 때 / 다시 보인 때 / 상태가 바뀐 때 / 표지판·회전 이유
  overlay.mp4   차선 인식 화면 + 위 정보 (프레임 번호, 동작, 회전할 쪽 경계 상태)

사용법:
    python -m drive.analyze_run logs/rec_right --csv logs/run_20261110_160000_real.csv --turn RIGHT --out logs/ana_right
    python -m drive.analyze_run data/raw/real_20261110/side_move_4 --turn RIGHT --out logs/ana_side4

도로 폭 cm 는 가까운/중간/먼 띠에서 가운데 세로 도로(21cm, 2026-11-11 실측)가 264/206/143px 로 보인다는 값(실차 사진)으로 환산한 추정이다.
참고 실측 폭: 바깥 28cm, 가운데 세로 도로 21cm, 가운데 가로 도로 18cm.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drive import config, vision  # noqa: E402
from drive.controller import band_edges, real_openings  # noqa: E402
from drive.main import draw_overlay, _text  # noqa: E402

REF_ROAD_CM = 21.0
PX_PER_REF = (264.0, 206.0, 143.0)   # 가까운·중간·먼 띠에서 가운데 세로 도로(21cm)의 픽셀 폭 (처리 해상도 320x240 기준, 실차 사진)


def frame_index(p: Path):
    m = re.search(r"_(\d+)_", p.name) or re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def load_csv(path):
    rows = {}
    if path:
        with open(path, errors="ignore") as f:
            for r in csv.DictReader(line.replace("\0", "") for line in f):
                if r.get("fid", "").isdigit():
                    rows[int(r["fid"])] = r
    return rows


def side_status(opens, turn):
    """회전할 쪽: 'open'(진짜 열림), 'unknown'(화면 밖이라 모름), 'closed'(경계 보임), '-'(띠 정보 없음)."""
    if not opens or turn not in ("LEFT", "RIGHT"):
        return "-"
    e = opens[0]
    if turn == "RIGHT":
        return {"L-": "open", "L?": "unknown"}.get(e, "closed" if e and e[1:2] == "R" else "open" if e == "--" else "-")
    return {"-R": "open", "?R": "unknown"}.get(e, "closed" if e and e[:1] == "L" else "open" if e == "--" else "-")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--turn", choices=["LEFT", "RIGHT"], default=None, help="볼 쪽. 없으면 CSV 의 pending 을 쓴다")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=10.0, help="overlay.mp4 재생 속도")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args(argv)

    files = sorted([p for p in args.frames.iterdir() if p.suffix.lower() in (".jpg", ".png")], key=frame_index)
    if not files:
        sys.exit(f"프레임이 없다: {args.frames}")
    rows = load_csv(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)
    det = vision.LaneDetector(config, flip=False)
    out_csv = open(args.out / "analysis.csv", "w", newline="")
    w = csv.writer(out_csv)
    head = ["fid", "label", "state", "error", "ahead_green", "edges", "opens", "turn_side"]
    for name in ("near", "mid", "far"):
        head += [f"{name}_left", f"{name}_right", f"{name}_width_px", f"{name}_width_cm", f"{name}_learned_px"]
    head += ["action", "pending", "note"]
    w.writerow(head)
    vw = None
    timeline, prev_side, prev_state = [], None, None
    for p in files:
        fid = frame_index(p)
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, tuple(config.CAMERA_SIZE))
        res = det.process(img)
        r = rows.get(fid, {})
        turn = args.turn or (r.get("pending", "").split()[0] if r.get("pending") else None)
        opens = real_openings(res, config)
        side = side_status(opens, turn)
        label = p.stem.rsplit("_", 1)[-1] if p.stem.startswith("train_") else ""
        line = [fid, label, res.state, "" if res.error is None else f"{res.error:.3f}", f"{res.ahead_green:.2f}",
                "/".join(band_edges(res)), "/".join(opens), side]
        for i, b in enumerate(res.bands[:3]):
            wpx = b.width if b.width is not None else (b.right - b.left if b.left is not None and b.right is not None else None)
            wcm = "" if wpx is None else f"{wpx * REF_ROAD_CM / PX_PER_REF[min(i, 2)] * (config.PROC_SIZE[0] / 320.0):.1f}"
            line += [b.left, b.right, wpx, wcm, "" if b.learned is None else f"{b.learned:.0f}"]
        line += [r.get("action", ""), r.get("pending", ""), r.get("note", "")]
        w.writerow(line)
        if side != prev_side and turn:
            what = {"open": "회전할 쪽 초록 경계 사라짐 (진짜 열림)", "unknown": "회전할 쪽 경계가 화면 밖 (모름)",
                    "closed": "회전할 쪽 초록 경계 보임"}.get(side, side)
            timeline.append(f"{fid:6d}  {what}  [{line[5]}]")
            prev_side = side
        if res.state != prev_state:
            timeline.append(f"{fid:6d}  상태 {prev_state} -> {res.state}  [{line[5]}]")
            prev_state = res.state
        if r.get("note"):
            timeline.append(f"{fid:6d}  {r['note']}  (동작 {r.get('action', '')})")
        if not args.no_video:
            ov = draw_overlay(res, config, None, (0.0, 0.0), 0.0, 0.0, True, None, [])
            if ov.shape[0] != 480:
                ov = cv2.resize(ov, (int(ov.shape[1] * 480 / ov.shape[0]), 480))
            col = {"open": (0, 200, 255), "unknown": (0, 0, 255), "closed": (0, 220, 0)}.get(side, (255, 255, 255))
            _text(ov, f"fid {fid} {label}  {turn or ''} side: {side}", (10, ov.shape[0] - 60), col, 0.6)
            _text(ov, f"{r.get('action', '')}  pend {r.get('pending', '')}  {r.get('note', '')[:40]}",
                  (10, ov.shape[0] - 35), (255, 255, 255), 0.5)
            if vw is None:
                vw = cv2.VideoWriter(str(args.out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                     (ov.shape[1], ov.shape[0]))
            vw.write(ov)
    out_csv.close()
    if vw is not None:
        vw.release()
    (args.out / "timeline.txt").write_text("\n".join(timeline) + "\n", encoding="utf-8")
    print("\n".join(timeline[:80]))
    print(f"\n→ {args.out}/analysis.csv, timeline.txt" + ("" if args.no_video else ", overlay.mp4"))


if __name__ == "__main__":
    main()
