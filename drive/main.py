"""룰베이스 주행 실행 파일.

  python3 -m drive.main --mode real                        # Pi: 카메라 + 모터
  python3 -m drive.main --mode dry                         # Pi: 카메라만, 모터 없음
  python  -m drive.main --mode offline --input docs/samples --repeat 5
  python  -m drive.main --mode offline --input run_20261028_190928
  python3 -m drive.main --mode real --signs runs/sign/baseline/weights/best.pt   # 표지판 포함

저장된 이미지/동영상은 이미 바로 세운 것으로 본다 (수집 코드와 --record 모두 flip 후 저장).
카메라 원본(거꾸로)을 offline으로 볼 때만 --flip을 붙인다.

키 (OpenCV 창): s 시작/일시정지, 아래 방향키·space 비상 정지, m 마스크 창, q 종료
               offline: p 재생 멈춤/재개, n 멈춘 상태에서 한 프레임
"""
from __future__ import annotations

import argparse
import csv
import queue
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from drive import config
from drive import vision
from drive.controller import Command, Controller
from drive.signs import NullSignDetector, SignInterpreter, SignWorker

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MOVING_ACTIONS = ("PID", "CROSS", "TURN")


# ================================================================ 프레임 입력
def imread(path):
    """cv2.imread는 Windows에서 한글 경로를 못 읽으므로 바이트로 읽어 디코딩한다."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, img):
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if ok:
        buf.tofile(str(path))
    return ok


@dataclass
class Frame:
    image: np.ndarray
    fid: int
    stamp: float
    name: str = ""
    last: bool = True   # offline 이미지 폴더: 같은 이미지 반복 중 마지막


class CameraSource:
    """MyPiCamera를 별도 스레드에서 계속 읽고 최신 프레임만 보관한다.
    read()는 기다리지 않으므로 카메라가 멈춰도 주행 루프는 돌고, 프레임 나이로 정지를 판단한다."""

    def __init__(self, cfg):
        try:
            from drive.mycamera import MyPiCamera
        except ImportError as e:
            raise SystemExit(f"picamera2를 불러올 수 없다 ({e}). Pi에서 실행하거나 --mode offline을 쓴다.")

        self.cam = MyPiCamera(*cfg.CAMERA_SIZE)
        if not self.cam.isOpened():
            raise RuntimeError("카메라를 열 수 없다")
        self._lock = threading.Lock()
        self._latest = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        fid = 0
        while self._running:
            try:
                ok, img = self.cam.read()
            except Exception as e:
                print(f"[camera] read error: {e}")
                return
            if not ok:
                time.sleep(0.01)
                continue
            fid += 1
            with self._lock:
                self._latest = Frame(img, fid, time.monotonic())

    def read(self):
        with self._lock:
            return self._latest

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cam.release()


class OfflineSource:
    """이미지 폴더(이름순), 이미지 한 장, 또는 동영상. read()는 호출마다 다음 프레임."""

    def __init__(self, path, repeat=1):
        p = Path(path)
        self.cap = None
        if p.is_dir():
            self.files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        elif p.suffix.lower() in IMAGE_EXTS:
            self.files = [p]
        else:
            self.files = []
            self.cap = cv2.VideoCapture(str(p))
            if not self.cap.isOpened():
                raise SystemExit(f"열 수 없는 입력: {p}")
        if self.cap is None and not self.files:
            raise SystemExit(f"이미지가 없다: {p}")
        self.is_images = self.cap is None
        self.repeat = max(1, repeat)
        self.finished = False
        self._i = 0
        self._rep = 0
        self._fid = 0
        self._img = None

    def read(self):
        if self.is_images:
            while self._rep == 0:
                if self._i >= len(self.files):
                    self.finished = True
                    return None
                self._img = imread(self.files[self._i])
                if self._img is not None:
                    break
                print(f"[offline] 읽기 실패: {self.files[self._i]}")
                self._i += 1
            name = self.files[self._i].name
            self._rep += 1
            last = self._rep >= self.repeat
            if last:
                self._i += 1
                self._rep = 0
            img = self._img
        else:
            ok, img = self.cap.read()
            if not ok:
                self.finished = True
                return None
            name, last = f"frame{self._fid + 1}", True
        self._fid += 1
        return Frame(img, self._fid, time.monotonic(), name, last)

    def release(self):
        if self.cap is not None:
            self.cap.release()


class Recorder(threading.Thread):
    """flip한 전체 해상도 프레임을 JPG로 저장 (offline 튜닝용). 밀리면 버린다."""

    def __init__(self, out_dir):
        super().__init__(daemon=True)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.q = queue.Queue(maxsize=60)
        self.dropped = 0
        self.start()

    def put(self, fid, img):
        try:
            self.q.put_nowait((fid, img))
        except queue.Full:
            self.dropped += 1

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            imwrite(self.out / f"{item[0]:06d}.jpg", item[1])

    def close(self):
        self.q.put(None)
        self.join(timeout=3.0)
        if self.dropped:
            print(f"[record] 저장 못 한 프레임 {self.dropped}개")


# ================================================================ 로그
class CsvLog:
    # fid: --record 로 저장한 이미지 파일 이름({fid:06d}.jpg)과 같은 번호. 이미지와 조향값을 짝짓는 데 쓴다.
    FIELDS = ["fid", "t", "state", "error", "steering", "throttle", "left", "right", "edges_found",
              "raw_state", "action"]

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(self.FIELDS)
        self.n = 0

    def write(self, t, res, cmd, wheels, fid=None):
        f3 = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
        self.w.writerow(["" if fid is None else fid, f"{t:.3f}", res.state or "", f3(res.error), f3(cmd.steering),
                         f3(cmd.throttle), f3(wheels[0]), f3(wheels[1]), res.edges_found,
                         res.candidate or "", cmd.action])
        self.n += 1
        if self.n % 20 == 0:
            self.f.flush()

    def close(self):
        self.f.close()


# ================================================================ 오버레이
STATE_COLORS = {
    vision.LANE: (0, 220, 0),
    vision.ONE_SIDE: (0, 220, 220),
    vision.INTERSECTION: (255, 170, 0),
    vision.OFF_ROAD: (0, 0, 255),
    vision.LOST: (0, 0, 255),
}


def _text(img, s, org, color=(255, 255, 255), scale=0.5):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_overlay(res, cfg, cmd=None, wheels=(0.0, 0.0), fps=0.0, proc_ms=0.0,
                 running=False, banner=None, extra=None):
    """스캔 띠, 좌(빨강)/우(파랑) 경계, 띠 중앙(노랑), 도로 중앙(자홍), 화면 중앙선(흰색), 수치."""
    s = cfg.DISPLAY_SCALE
    h, w = res.image.shape[:2]
    img = cv2.resize(res.image, (w * s, h * s), interpolation=cv2.INTER_NEAREST)
    P = lambda v: int(round(v * s))  # noqa: E731

    ex = h - cfg.BOTTOM_EXCLUDE
    img[P(ex):] = (img[P(ex):] * 0.4).astype(np.uint8)
    cv2.line(img, (P(w / 2), 0), (P(w / 2), P(h)), (255, 255, 255), 1)

    for b in res.bands:
        cv2.rectangle(img, (0, P(b.y)), (P(w) - 1, P(b.y + b.h)), (160, 160, 160), 1)
        ym = P(b.y + b.h / 2)
        cv2.drawMarker(img, (P(b.start), ym), (200, 200, 200), cv2.MARKER_TILTED_CROSS, 8, 1)
        if b.left is not None:
            cv2.circle(img, (P(b.left), ym), 5, (0, 0, 255), -1)
        if b.right is not None:
            cv2.circle(img, (P(b.right), ym), 5, (255, 80, 0), -1)
        if b.center is not None:
            cv2.drawMarker(img, (P(b.center), ym), (0, 255, 255), cv2.MARKER_DIAMOND, 10, 2)
        lw = "-" if b.learned is None else f"{b.learned:.0f}"
        info = f"{b.name} {b.edges} w={b.width if b.width is not None else '-'}/{lw} {b.note}"
        _text(img, info, (4, P(b.y) - 3), (220, 220, 220), 0.4)

    x0, x1 = P(w / 2 - cfg.OFFROAD_HALF_W), P(w / 2 + cfg.OFFROAD_HALF_W)
    near = max(res.bands, key=lambda b: b.y)
    cv2.rectangle(img, (x0, P(near.y)), (x1, P(near.y + near.h)), (255, 0, 255), 1)
    if res.center is not None:
        cv2.line(img, (P(res.center), P(ex * 0.45)), (P(res.center), P(ex)), (255, 0, 255), 2)

    cmd = cmd or Command()
    color = STATE_COLORS.get(res.state, (200, 200, 200))
    err = "  -  " if res.error is None else f"{res.error:+.2f}"
    lines = [
        (f"{res.state or '-'}  (raw {res.candidate or '-'})  {'RUN' if running else 'PAUSED'}", color),
        (f"err {err}  steer {cmd.steering:+.2f}  thr {cmd.throttle:.2f}  {cmd.action}", None),
        (f"PWM L {wheels[0]:+.2f}  R {wheels[1]:+.2f}   offroad {res.offroad_ratio:.2f}", None),
        (f"FPS {fps:4.1f}  proc {proc_ms:4.1f} ms", None),
    ]
    for extra_line in extra or []:
        lines.append((extra_line, (0, 255, 255)))
    for i, (t, c) in enumerate(lines):
        _text(img, t, (6, 18 + 18 * i), c or (255, 255, 255), 0.5)
    if banner:
        _text(img, banner, (P(w / 2) - 70, P(h / 2)), (0, 0, 255), 0.9)
    return img


def draw_mask(res, cfg):
    s = cfg.DISPLAY_SCALE
    h, w = res.mask.shape[:2]
    m = cv2.cvtColor(res.mask, cv2.COLOR_GRAY2BGR)
    for b in res.bands:
        cv2.rectangle(m, (0, b.y), (w - 1, b.y + b.h), (0, 0, 255), 1)
    return cv2.resize(m, (w * s, h * s), interpolation=cv2.INTER_NEAREST)


# ================================================================ 실행
def make_car(mode, cfg):
    from drive.motors import Car

    if mode == "real":
        return Car(cfg, backend="gpio")
    if mode == "offline":
        try:
            from drive.motors import use_mock_pins

            use_mock_pins()
            return Car(cfg, backend="gpio")
        except ImportError:
            print("[offline] gpiozero가 없어 핀 없이 실행한다 (pip install gpiozero 로 mock 사용)")
    return Car(cfg, backend="none")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="녹색 경계 도로 중앙 주행")
    p.add_argument("--mode", choices=["real", "dry", "offline"], required=True)
    p.add_argument("--input", help="offline: 이미지 폴더, 이미지 파일 또는 동영상")
    p.add_argument("--repeat", type=int, default=1,
                   help="offline 이미지: 각 이미지를 몇 프레임 반복할지 (상태 확인용)")
    p.add_argument("--flip", action="store_true",
                   help="offline: 카메라 원본(거꾸로)인 입력일 때 flip 적용 (real/dry는 항상 적용)")
    p.add_argument("--no-gui", action="store_true", help="창 없이 실행 (dry/offline)")
    p.add_argument("--show-mask", action="store_true", help="녹색 마스크 창도 띄움")
    p.add_argument("--record", metavar="DIR", help="real/dry: flip한 카메라 프레임 저장")
    p.add_argument("--log", metavar="CSV", help="로그 경로 (기본 logs/run_<시각>_<모드>.csv)")
    p.add_argument("--signs", metavar="WEIGHTS",
                   help="표지판 YOLO 가중치 (.pt/.onnx/.tflite). 없으면 표지판을 보지 않는다")
    a = p.parse_args(argv)
    if a.mode == "offline" and not a.input:
        p.error("offline 모드는 --input이 필요하다")
    if a.mode == "real" and a.no_gui:
        p.error("real 모드는 s 키로 출발하므로 창이 필요하다")
    return a


class Runner:
    def __init__(self, args, cfg, car, source, signs, recorder, log):
        self.args, self.cfg = args, cfg
        self.car, self.source, self.signs, self.recorder, self.log = car, source, signs, recorder, log
        self.offline = args.mode == "offline"
        self.gui = not args.no_gui
        self.detector = vision.LaneDetector(cfg, flip=not self.offline or args.flip)
        self.ctrl = Controller(cfg)
        self.show_mask = args.show_mask
        self.playback_paused = False
        self.step = False
        self.res = None
        self.cmd = Command()
        self.wheels = (0.0, 0.0)
        self.last_fid = None
        self.fps = 0.0
        self.proc_ms = 0.0
        self.banner = None
        self.states = Counter()

    # ------------------------------------------------------------ 한 프레임 처리
    def _process(self, frame, t):
        cfg, ctrl, car = self.cfg, self.ctrl, self.car
        self.last_fid = frame.fid
        self.banner = None
        t_proc = time.perf_counter()
        res = self.res = self.detector.process(frame.image)
        self.proc_ms = (time.perf_counter() - t_proc) * 1000.0

        upright = cv2.flip(frame.image, cfg.FLIP_CODE) if self.detector.flip else frame.image
        self.signs.submit(upright, frame.fid)
        for ev in self.signs.poll():
            print(f"[sign] {ev}")
            ctrl.on_sign(ev)

        self.cmd = ctrl.update(res.state, res.error)
        if self.cmd.action in MOVING_ACTIONS:
            self.wheels = car.drive(self.cmd.throttle, self.cmd.steering)
        else:
            self.wheels = car.stop()
        car.horn(self.cmd.horn)

        self.log.write(t, res, self.cmd, self.wheels, frame.fid)
        self.states[res.state] += 1
        if self.recorder:
            self.recorder.put(frame.fid, upright)
        if self.offline and self.source.is_images and frame.last:
            err = "None" if res.error is None else f"{res.error:+.3f}"
            print(f"{frame.name}: state={res.state} raw={res.candidate} "
                  f"edges={res.edges_found} error={err} offroad={res.offroad_ratio:.2f}")

    def _no_frame(self):
        if self.cmd.action != "NO_FRAME":
            print("[camera] 프레임 없음 -> 정지")
        self.ctrl.reset()
        self.cmd = Command(action="NO_FRAME")
        self.wheels = self.car.stop()
        self.car.horn(False)
        self.banner = "NO FRAME"

    # ------------------------------------------------------------ 화면과 키
    def _show(self):
        cfg, ctrl = self.cfg, self.ctrl
        if self.res is None:
            img = np.zeros((cfg.PROC_SIZE[1] * cfg.DISPLAY_SCALE,
                            cfg.PROC_SIZE[0] * cfg.DISPLAY_SCALE, 3), np.uint8)
            _text(img, "waiting for camera...", (20, 40))
            cv2.imshow("drive", img)
            return
        extra = []
        if ctrl.pending_turn:
            extra.append(f"next turn: {ctrl.pending_turn}")
        if ctrl._turn:
            extra.append(f"turn {ctrl._turn[0]} {ctrl._turn[1]}")
        if ctrl._wait:
            extra.append(f"wait {ctrl._wait[0]} {ctrl._wait[1]}")
        if ctrl.speed_scale != 1.0:
            extra.append(f"speed x{ctrl.speed_scale:.2f}")
        if self.offline and self.playback_paused:
            extra.append("playback paused (p / n)")
        cv2.imshow("drive", draw_overlay(self.res, cfg, self.cmd, self.wheels, self.fps,
                                         self.proc_ms, ctrl.running, self.banner, extra))
        if self.show_mask:
            cv2.imshow("mask", draw_mask(self.res, cfg))

    def _key(self, key):
        """False를 돌려주면 종료."""
        cfg, ctrl = self.cfg, self.ctrl
        if key in cfg.KEY_QUIT:
            return False
        if key in cfg.KEY_ESTOP:
            ctrl.set_running(False)
            self.wheels = self.car.stop()
            print("비상 정지")
        elif key in cfg.KEY_TOGGLE_RUN:
            ctrl.set_running(not ctrl.running)
            if not ctrl.running:
                self.wheels = self.car.stop()
            print("주행 시작" if ctrl.running else "일시정지")
        elif key in cfg.KEY_TOGGLE_MASK:
            self.show_mask = not self.show_mask
            if not self.show_mask:
                cv2.destroyWindow("mask")
        elif self.offline and key in cfg.KEY_PLAYBACK_PAUSE:
            self.playback_paused = not self.playback_paused
        elif self.offline and key in cfg.KEY_PLAYBACK_STEP:
            self.step = True
        return True

    # ------------------------------------------------------------ 고정 주기 루프
    def run(self):
        cfg = self.cfg
        period = 1.0 / cfg.LOOP_HZ
        pace = not self.offline or self.gui   # offline 창 없음: 최대 속도로 처리
        t0 = next_t = time.monotonic()
        last_tick = None
        tick = 0
        while True:
            now = time.monotonic()
            if last_tick is not None and now > last_tick:
                inst = 1.0 / (now - last_tick)
                self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
            last_tick = now
            tick += 1

            frame = None
            if not self.playback_paused or self.step:
                frame = self.source.read()
                self.step = False
                if frame is None and self.offline and self.source.finished:
                    return
            if not self.offline and (frame is None or now - frame.stamp > cfg.FRAME_TIMEOUT_S):
                self._no_frame()  # 카메라 프레임이 끊겼다 -> 정지
            elif frame is not None and frame.fid != self.last_fid:
                self._process(frame, now - t0)

            if self.gui:
                if tick % cfg.DISPLAY_EVERY == 0:
                    self._show()
                key = cv2.waitKey(1)
                if key != -1 and not self._key(key):
                    return

            if pace:
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()  # 밀렸으면 따라잡지 않고 다시 맞춘다


def main(argv=None):
    args = parse_args(argv)
    cfg = config
    print(f"Python {sys.version.split()[0]}  OpenCV {cv2.__version__}  mode={args.mode}")
    if sys.version_info < (3, 9):
        print("경고: Python 3.9 이상에서 확인했다")

    car = source = signs = recorder = log = runner = None
    try:
        car = make_car(args.mode, cfg)
        if args.mode == "offline":
            source = OfflineSource(args.input, args.repeat)
        else:
            source = CameraSource(cfg)
        detector = NullSignDetector()
        if args.signs:
            from drive.signs import YoloSignDetector

            detector = YoloSignDetector(args.signs, imgsz=cfg.SIGN_IMGSZ)
            print(f"표지판 검출기: {args.signs}")
        signs = SignWorker(detector, SignInterpreter(cfg))
        signs.start()
        if args.record and args.mode != "offline":
            recorder = Recorder(args.record)
        log_path = Path(args.log) if args.log else \
            ROOT / cfg.LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{args.mode}.csv"
        log = CsvLog(log_path)
        runner = Runner(args, cfg, car, source, signs, recorder, log)
        runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        # 어떤 오류로 끝나도 바퀴부터 멈춘다
        if car is not None:
            car.stop()
            car.horn(False)
            car.close()
        if signs is not None:
            signs.stop()
        if source is not None:
            source.release()
        if recorder is not None:
            recorder.close()
        if log is not None:
            log.close()
            summary = ", ".join(f"{k}={v}" for k, v in runner.states.items()) if runner else ""
            print(f"로그: {log.path}  ({log.n} 프레임 {summary})")
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
