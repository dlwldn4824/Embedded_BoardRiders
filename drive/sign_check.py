"""실차 표지판 점검. 모터 없이 카메라 + 표지판 모델만 돌린다.

주행 코드와 같은 경로를 쓴다 (CameraSource, 180° 뒤집기, YoloSignDetector, SignInterpreter).
- 매 검출마다 클래스·신뢰도·박스 높이와, 주행 코드 필터를 통과하는지(OK / 왜 안 되는지)를 찍는다.
- 필터를 모두 통과해 실제 주행 이벤트가 나면 EVENT 줄을 찍는다.
- 검출이 있는 프레임은 박스를 그려 <out>/det/ 에 저장한다 → 무엇을 잡고 놓쳤는지 나중에 본다.
- --raw-every N 프레임마다 원본을 <out>/raw/ 에 저장한다 → 놓친 표지판을 라벨링해 재학습에 쓴다.
- 끝나면 (Ctrl+C 또는 --frames) 검출 시간·클래스별 최대 박스 높이·이벤트를 <out>/summary.txt 에 남긴다.

사용법:
    python3 -m drive.sign_check --signs weights/sign_v3_8n_int8.tflite              # Pi, 카메라
    python  -m drive.sign_check --signs weights/sign_v3_8n_int8.tflite --input rec_signs   # PC, 저장한 사진
"""
from __future__ import annotations

import argparse
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2

from drive import config
from drive.main import ROOT, OfflineSource, imwrite
from drive.signs import SignInterpreter, YoloSignDetector, covers_center

LIMIT_MS = 300   # docs/sim-signs.md: 검출 한 번이 이보다 길면 미션이 실패하기 시작한다


def why_not(d, cfg):
    """주행 코드 필터(SignInterpreter._rule)로 보면 이 검출이 왜 무시되는지. 통과하면 'OK'."""
    if d["cls"] not in cfg.SIGN_CLASS_EVENTS:
        return "이벤트 없는 클래스"
    rule = {"min_conf": cfg.SIGN_MIN_CONF, "min_h": cfg.SIGN_MIN_BOX_H, "min_y2": 0}
    rule.update(getattr(cfg, "SIGN_CLASS_RULES", {}).get(d["cls"], {}))
    if d["conf"] < rule["min_conf"]:
        return f"신뢰도 낮음 (<{rule['min_conf']})"
    if d["box_h"] < rule["min_h"]:
        return f"멀다 (박스 <{rule['min_h']}px)"
    if d["box_y2"] < rule["min_y2"]:
        return f"아직 위쪽 (y2 <{rule['min_y2']})"
    if rule.get("center") and not covers_center(d, cfg.CAMERA_SIZE[0]):
        return "화면 가운데를 안 덮음 (옆길 것)"
    return "OK"


def draw(img, dets, cfg, text):
    out = img.copy()
    for d in dets:
        ok = why_not(d, cfg) == "OK"
        x1, x2 = int(d["box_cx"] - d["box_w"] / 2), int(d["box_cx"] + d["box_w"] / 2)
        y2 = int(d["box_y2"]); y1 = int(y2 - d["box_h"])
        col = (0, 220, 0) if ok else (0, 200, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{d['cls']} {d['conf']:.2f} h{d['box_h']:.0f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.putText(out, text, (6, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="실차 표지판 점검 (모터 없음)")
    p.add_argument("--signs", required=True, metavar="WEIGHTS")
    p.add_argument("--input", help="카메라 대신 이미지 폴더/동영상 (바로 세운 사진)")
    p.add_argument("--out", default=None, help="저장 폴더 (기본 logs/sign_check_<시각>)")
    p.add_argument("--frames", type=int, default=0, help="이만큼 처리하고 끝낸다 (0 = Ctrl+C 까지)")
    p.add_argument("--raw-every", type=int, default=10, help="N 프레임마다 원본 저장 (0 = 안 함)")
    p.add_argument("--imgsz", type=int, default=0, help="0 이면 config.SIGN_IMGSZ")
    args = p.parse_args(argv)

    cfg = config
    out = Path(args.out) if args.out else ROOT / cfg.LOG_DIR / f"sign_check_{time.strftime('%Y%m%d_%H%M%S')}"
    (out / "det").mkdir(parents=True, exist_ok=True)
    if args.raw_every:
        (out / "raw").mkdir(exist_ok=True)

    det = YoloSignDetector(args.signs, imgsz=args.imgsz or cfg.SIGN_IMGSZ)
    interp = SignInterpreter(cfg)
    if args.input:
        source, flip = OfflineSource(args.input), False
    else:
        from drive.main import CameraSource
        source, flip = CameraSource(cfg), True
    print(f"표지판 점검: {args.signs} → {out}  (Ctrl+C 로 끝)")

    ms, max_h, max_y2, n_det, events = [], defaultdict(float), defaultdict(float), defaultdict(int), []
    fid, last_cam = 0, None
    t_start = time.perf_counter()
    try:
        while not args.frames or fid < args.frames:
            if args.input:
                frame = source.read()
                if frame is None or source.finished:
                    break
            else:
                frame = source.read()
                if frame is None or frame.fid == last_cam:
                    time.sleep(0.005)
                    continue
                last_cam = frame.fid
            fid += 1
            img = cv2.flip(frame.image, cfg.FLIP_CODE) if flip else frame.image
            t0 = time.perf_counter()
            dets = det.detect(img)
            ms.append((time.perf_counter() - t0) * 1000)
            for d in dets:
                d["frame_id"] = fid
                n_det[d["cls"]] += 1
                max_h[d["cls"]] = max(max_h[d["cls"]], d["box_h"])
                max_y2[d["cls"]] = max(max_y2[d["cls"]], d["box_y2"])
                print(f"#{fid:05d} {d['cls']:12s} conf {d['conf']:.2f}  박스 h {d['box_h']:5.0f}px  "
                      f"y2 {d['box_y2']:5.0f}  → {why_not(d, cfg)}")
            for ev in interp(dets):
                events.append((fid, ev))
                print(f"#{fid:05d} *** EVENT {ev.kind} {ev.value or ''} ({ev.cls}) — 주행 중이면 여기서 동작한다")
            if dets:
                imwrite(out / "det" / f"{fid:06d}.jpg", draw(img, dets, cfg, f"#{fid} {ms[-1]:.0f}ms"))
            if args.raw_every and fid % args.raw_every == 0:
                imwrite(out / "raw" / f"{fid:06d}.jpg", img)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.input:
            source.release()

    wall = time.perf_counter() - t_start
    lines = [f"가중치: {args.signs}", f"프레임: {fid}  ({fid / max(wall, 1e-6):.1f} fps)"]
    if ms:
        s = sorted(ms)
        med, p90 = statistics.median(s), s[max(0, int(len(s) * 0.9) - 1)]
        verdict = "OK" if p90 <= LIMIT_MS else ("아슬아슬" if med <= LIMIT_MS else "느림 → 320 모델 검토")
        lines.append(f"검출 시간: 중앙값 {med:.0f}ms, p90 {p90:.0f}ms → {verdict} (기준 {LIMIT_MS}ms)")
    lines.append("클래스별: 검출 프레임 수 / 최대 박스 높이 / 최대 y2 (반응 기준 SIGN_MIN_BOX_H "
                 f"{cfg.SIGN_MIN_BOX_H}px, 횡단보도는 y2 {cfg.SIGN_CLASS_RULES.get('crosswalk', {}).get('min_y2', 0)})")
    for c in sorted(n_det):
        lines.append(f"  {c:12s} {n_det[c]:5d}  {max_h[c]:5.0f}px  y2 {max_y2[c]:4.0f}")
    lines.append(f"이벤트 {len(events)}개: " + ", ".join(f"#{f}:{e.cls}" for f, e in events))
    text = "\n".join(lines)
    print("\n" + text)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\n저장: {out}/det (검출 표시), {out}/raw (원본), summary.txt")


if __name__ == "__main__":
    main()
