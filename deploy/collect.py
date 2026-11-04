"""차선 주행 데이터 수집 (Raspberry Pi).

사람이 조향하는 동안 카메라 프레임과 연속 조향값을 같이 저장한다.

    data/runs/{run_id}/
        images/000000.jpg ...
        labels.csv   frame, t, segment, steer, throttle, left, right, recovery
        bad.csv      t_start, t_end   ("방금 실수" 버튼으로 무효 처리한 구간)
        meta.json    방향, 메모, 카메라 고정값, 설정 전체

사용 예:
    python deploy/collect.py --direction ccw --note "형광등, 배터리 완충"
    python deploy/collect.py --input keyboard --direction cw
    python deploy/collect.py --test-input              # 게임패드 버튼 이름 확인
    python deploy/collect.py --fake --input script --seconds 5   # 하드웨어 없이 시험
"""
import argparse
import csv
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from hw import Camera, Car
from inputs import make_input

ROOT = Path(__file__).resolve().parent.parent


class Writer(threading.Thread):
    """디스크 쓰기를 메인 루프와 분리해서 프레임 간격이 흔들리지 않게 한다."""

    FIELDS = ["frame", "t", "segment", "steer", "throttle", "left", "right", "recovery"]

    def __init__(self, run_dir, jpeg_quality):
        super().__init__(daemon=True)
        self.img_dir = run_dir / "images"
        self.img_dir.mkdir(parents=True)
        self.f = open(run_dir / "labels.csv", "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        self.w.writeheader()
        self.q = queue.Queue(maxsize=200)
        self.params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        self.count = 0

    def run(self):
        while (item := self.q.get()) is not None:
            frame, row = item
            cv2.imwrite(str(self.img_dir / f"{row['frame']:06d}.jpg"), frame, self.params)
            self.w.writerow(row)
            self.f.flush()
            self.count += 1

    def close(self):
        self.q.put(None)
        self.join()
        self.f.close()


def test_input(kind, cfg):
    inp = make_input(kind, cfg)
    print("입력 테스트 (Ctrl+C 종료)")
    if kind == "gamepad":
        import evdev

        print(f"장치: {inp.dev.name}")
        for ev in inp.dev.read_loop():
            if ev.type in (evdev.ecodes.EV_KEY, evdev.ecodes.EV_ABS):
                print(evdev.categorize(ev))
    while True:
        steer, ev = inp.poll()
        inp.status(f"steer {steer:+.2f} {ev}")
        time.sleep(0.05)


def summarize(run_dir):
    rows = list(csv.DictReader(open(run_dir / "labels.csv")))
    if not rows:
        return "저장된 프레임 없음"
    s = np.array([float(r["steer"]) for r in rows])
    hist, edges = np.histogram(s, bins=[-1, -0.6, -0.2, 0.2, 0.6, 1.0001])
    names = ["강한 좌", "약한 좌", "직진", "약한 우", "강한 우"]
    rec = np.mean([r["recovery"] == "1" for r in rows])
    lines = [f"{len(rows)}장, segment {rows[-1]['segment']}개, 복구 태그 {rec:.0%}"]
    lines += [f"  {n:6s} {c:5d} ({c / len(rows):.0%})" for n, c in zip(names, hist)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "deploy/config.yaml")
    ap.add_argument("--input", choices=["gamepad", "keyboard", "script"], default="gamepad")
    ap.add_argument("--direction", choices=["ccw", "cw"], help="트랙 진행 방향 (반시계/시계)")
    ap.add_argument("--note", default="", help="조명, 배터리, 트랙 배치 등 메모")
    ap.add_argument("--out", type=Path, default=ROOT / "data/runs")
    ap.add_argument("--fake", action="store_true", help="카메라·모터 없이 실행")
    ap.add_argument("--seconds", type=float, help="이 시간 후 자동 종료 (테스트용)")
    ap.add_argument("--test-input", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ccfg = cfg["collect"]
    if args.test_input:
        return test_input(args.input, ccfg)
    if not args.direction:
        ap.error("--direction ccw|cw 를 지정하세요 (진행 방향은 학습 데이터 균형에 필요)")

    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / run_id
    cam = Camera(cfg["camera"], fake=args.fake)
    car = Car(cfg["drive"], fake=args.fake)
    writer = Writer(run_dir, ccfg["jpeg_quality"])
    writer.start()
    meta = {
        "run_id": run_id, "direction": args.direction, "note": args.note, "input": args.input,
        "fake": args.fake, "camera_locked": cam.locked, "config": cfg,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    bad_f = open(run_dir / "bad.csv", "w", newline="")
    bad_w = csv.writer(bad_f)
    bad_w.writerow(["t_start", "t_end"])

    inp = make_input(args.input, ccfg)
    period = 1.0 / ccfg["fps"]
    cruise = ccfg["cruise_throttle"]
    recording = driving = recovery = False
    segment, frame_idx, dropped = 0, 0, 0
    t_start = time.time()
    try:
        while True:
            t0 = time.time()
            t = t0 - t_start
            frame = cam.read()
            steer, events = inp.poll()

            for e in events:
                if e == "record":
                    recording = not recording
                    segment += recording
                elif e == "drive":
                    driving = not driving
                elif e == "recovery":
                    recovery = not recovery
                elif e == "bad":
                    bad_w.writerow([f"{max(0.0, t - ccfg['bad_mark_seconds']):.3f}", f"{t:.3f}"])
                    bad_f.flush()
                elif e == "faster":
                    cruise = min(1.0, cruise + 0.05)
                elif e == "slower":
                    cruise = max(0.0, cruise - 0.05)
                elif e == "estop":
                    driving = recording = False
                elif e == "quit":
                    raise KeyboardInterrupt

            throttle = cruise if driving else 0.0
            left, right = car.drive(steer, throttle)

            if recording and driving:  # 정지 상태 프레임은 저장하지 않는다
                row = {
                    "frame": frame_idx, "t": f"{t:.3f}", "segment": segment, "steer": f"{steer:.4f}",
                    "throttle": f"{throttle:.3f}", "left": f"{left:.3f}", "right": f"{right:.3f}",
                    "recovery": int(recovery),
                }
                try:
                    writer.q.put_nowait((frame, row))
                    frame_idx += 1
                except queue.Full:
                    dropped += 1

            bar = "".join("|" if i == int((steer + 1) * 10) else "-" for i in range(21))
            inp.status(
                f"{'● REC' if recording else '  rec'} {'DRIVE' if driving else 'stop '} "
                f"{'RECOV' if recovery else '     '} [{bar}] {steer:+.2f} 속도 {cruise:.2f} 저장 {frame_idx} 누락 {dropped}"
            )
            if args.seconds and t > args.seconds:
                break
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        car.close()
        inp.close()
        writer.close()
        bad_f.close()
        cam.close()
        meta.update(ended=datetime.now().isoformat(timespec="seconds"), frames=writer.count, dropped=dropped)
        (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"\n저장: {run_dir}\n{summarize(run_dir)}")


if __name__ == "__main__":
    main()
