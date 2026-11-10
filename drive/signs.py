"""표지판(YOLO) 검출과 이벤트 변환.

흐름:  SignWorker(스레드) 또는 SignProcess(별도 프로세스)  --detect(frame)-->  SignInterpreter  --SignEvent-->  Controller.on_sign()
Pi 에서는 SignProcess 를 쓴다. 스레드로 돌리면 YOLO 의 파이썬 전·후처리가 GIL 을 잡아 주행 루프가 20 Hz 를 못 지키고,
프레임 수로 센 동작(교차로 회전 등)이 그만큼 늘어진다.
주행 루프는 submit()으로 최신 프레임만 넘기고 poll()로 쌓인 이벤트만 가져가므로
검출기가 느려도 루프가 막히지 않는다. 처리 중에 들어온 프레임은 최신 것만 남는다.

검출기에는 바로 세운 전체 해상도(640x480) 프레임이 들어온다. 박스 좌표와 크기도 이 해상도 기준이다.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


class SignDetector:
    """detect(frame) -> [{"cls", "conf", "box_w", "box_h", "box_cx", "box_y2", "frame_id"}]"""

    def detect(self, frame):
        raise NotImplementedError


class NullSignDetector(SignDetector):
    def detect(self, frame):
        return []


class YoloSignDetector(SignDetector):
    """ultralytics YOLO. .pt / .onnx / .tflite 가중치를 모두 받는다 (Pi 에서는 INT8 .tflite 권장)."""

    def __init__(self, weights, imgsz=640, conf=0.25, ref_size=(640, 480)):
        from ultralytics import YOLO

        self.ref_w = ref_size[0]   # 박스 좌표·크기는 항상 이 폭(640x480) 기준으로 돌려준다 (SIGN_* 기준값이 이 단위)

        self.model = YOLO(weights, task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.names = self.model.names

    def detect(self, frame):
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        k = self.ref_w / frame.shape[1]   # 320x240 으로 찍어도 640x480 기준 좌표로 맞춘다
        out = []
        for c, cf, box in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
            x1, y1, x2, y2 = (v * k for v in box)
            out.append({"cls": self.names[int(c)], "conf": float(cf), "box_w": x2 - x1, "box_h": y2 - y1,
                        "box_cx": (x1 + x2) / 2, "box_y2": y2})
        return out


def covers_center(d, ref_w):
    """상자가 화면 가운데 세로선을 덮는지 (차 바로 앞, 지금 가는 길 위에 있는지). 상자 정보가 없으면 True."""
    if "box_cx" not in d or "box_w" not in d:
        return True
    return d["box_cx"] - d["box_w"] / 2 <= ref_w / 2 <= d["box_cx"] + d["box_w"] / 2


@dataclass
class SignEvent:
    # "TURN" (value: LEFT/STRAIGHT/RIGHT), "HORN", "SPEED" (value: 속도 배율), "STOP",
    # "LIGHT" (value: RED/GREEN/OFF, OFF = 빨간불이 꺼짐), "CROSSWALK" (차 바로 앞 횡단보도), "FINISH"
    kind: str
    value: object
    cls: str
    conf: float
    frame_id: int
    box: tuple = field(default=None, repr=False)   # (x1, y1, x2, y2), 640x480 기준. 무엇을 보고 이벤트가 났는지 저장용


class SignInterpreter:
    """검출 결과를 이벤트로 바꾼다. 신뢰도, 박스 크기/위치, 연속 검출, 같은 클래스 반복을 거른다."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._last_fired = {}
        self._hits = defaultdict(deque)   # 클래스별 조건을 통과한 frame_id
        self._last_horn = None            # 마지막 경적 검출 (우회전으로 헷갈린 것 거르기)
        self._red_fired = False           # 빨간불 이벤트를 냈고 아직 꺼짐(OFF)을 알리지 않았다
        self._red_missing = 0             # 빨간불이 하나도 안 보인 연속 검출 횟수
        self._red_cut = False             # 마지막 빨간불 상자가 화면 끝에 닿았다 (사라지면 꺼진 건지 화면 밖으로 나간 건지 모름)
        self._last_fid = None

    def _rule(self, cls):
        rule = {"min_conf": self.cfg.SIGN_MIN_CONF, "min_h": self.cfg.SIGN_MIN_BOX_H, "min_y2": 0}
        rule.update(getattr(self.cfg, "SIGN_CLASS_RULES", {}).get(cls, {}))
        return rule

    def __call__(self, detections):
        cfg = self.cfg
        hits_needed = getattr(cfg, "SIGN_CONFIRM_HITS", 1)
        window = getattr(cfg, "SIGN_CONFIRM_WINDOW", 0)
        events = []
        seen = set()
        # 빨간불이 꺼졌는지 보려고, 거르기 전에 이번 검출에 빨간불이 있었는지 본다 (크기·신뢰도 상관없이)
        reds = [d for d in detections if d["cls"] == "red_light"]
        for d in detections:
            if d.get("frame_id") is not None:
                self._last_fid = d["frame_id"]
        # 화면 왼쪽·오른쪽 끝에 걸려 잘린 회전 표지판은 방향을 잘못 읽는다 (실차: 왼쪽 끝에 반쯤 걸린 경적을
        # right 로 봄). 상자가 끝에서 SIGN_EDGE_MARGIN px 안에 닿으면 버린다. 기준 폭은 검출기 기준 화면(640)
        ref_w = cfg.CAMERA_SIZE[0]
        m = getattr(cfg, "SIGN_EDGE_MARGIN", 4)

        def cut(d):
            if "box_cx" not in d or "box_w" not in d:
                return False
            return d["box_cx"] - d["box_w"] / 2 <= m or d["box_cx"] + d["box_w"] / 2 >= ref_w - m

        detections = [d for d in detections
                      if not (cfg.SIGN_CLASS_EVENTS.get(d["cls"], ("",))[0] == "TURN" and cut(d))]
        # 비스듬히 본 표지판은 상자가 가늘다. 직진(↑) 표지판을 옆에서 보면 ↗ 처럼 보여 right 로 읽었다 (실차: 출발점
        # 오른쪽 앞 직진 표지판). 학습 데이터의 좌·우 표지판 상자는 95% 가 가로/세로 0.73 이상이다
        min_asp = getattr(cfg, "SIGN_TURN_MIN_ASPECT", 0.7)
        detections = [d for d in detections
                      if not (d["cls"] in ("left", "right") and "box_w" in d
                              and d["box_w"] / max(d["box_h"], 1e-6) < min_asp)]
        # 회전 표지판이 한 화면에 여럿 보이면 가장 큰 것(가장 가까운 것)만 센다. 먼 교차로의 표지판이
        # 가까운 표지판보다 먼저 이벤트가 되어 방향이 바뀌는 것을 막는다
        turn = [d for d in detections if cfg.SIGN_CLASS_EVENTS.get(d["cls"], ("",))[0] == "TURN"
                and d["conf"] >= self._rule(d["cls"])["min_conf"]]
        if len(turn) > 1:
            # 같은 표지판에 straight 와 right 가 겹쳐 나오면 신뢰도가 높은 것을 고른다. 예전에는 상자가 조금 큰 쪽을
            # 골라 신뢰도가 낮은 right 가 뽑혔다 (실차: 출발할 때 straight 를 right 로). 서로 다른 표지판이면 가까운 것
            def same(a, b):
                return abs(a.get("box_cx", 0) - b.get("box_cx", 0)) < max(a.get("box_w", 0), b.get("box_w", 0), 1) / 2 \
                    and abs(a.get("box_y2", 0) - b.get("box_y2", 0)) < max(a["box_h"], b["box_h"], 1) / 2
            groups = []
            for d in sorted(turn, key=lambda d: -d["conf"]):
                for g in groups:
                    if same(g[0], d):
                        g.append(d)
                        break
                else:
                    groups.append([d])
            nearest = max((g[0] for g in groups), key=lambda d: d["box_h"])
            detections = [d for d in detections if d not in turn or d is nearest]
        # 경적(파란 원 + 나팔)을 가끔 우회전(파란 원 + 화살표)으로 본다. 경적이 HORN_CONFUSE_FRAMES 안에 보였던 자리에서
        # 비슷한 크기의 회전 표지판이 보이면 같은 표지판으로 보고 버린다
        for d in detections:
            if d["cls"] == "horn" and d["conf"] >= self._rule("horn")["min_conf"]:
                self._last_horn = d
        h = self._last_horn
        if h is not None and h.get("frame_id") is not None:
            keep = []
            for d in detections:
                if cfg.SIGN_CLASS_EVENTS.get(d["cls"], ("",))[0] == "TURN" and d.get("frame_id") is not None \
                        and 0 <= d["frame_id"] - h["frame_id"] <= getattr(cfg, "HORN_CONFUSE_FRAMES", 30) \
                        and abs(d.get("box_cx", 0) - h.get("box_cx", 0)) < max(d.get("box_w", 0), h.get("box_w", 0), 1) \
                        and 0.6 <= d["box_h"] / max(h["box_h"], 1) <= 1.7:
                    continue
                keep.append(d)
            detections = keep
        for d in sorted(detections, key=lambda d: -d["conf"]):
            spec = cfg.SIGN_CLASS_EVENTS.get(d["cls"])
            if spec is None or d["cls"] in seen:
                continue
            r = self._rule(d["cls"])
            if d["conf"] < r["min_conf"] or d["box_h"] < r["min_h"] or d.get("box_y2", 0) < r["min_y2"]:
                continue
            if r.get("center") and not covers_center(d, ref_w):
                continue
            seen.add(d["cls"])
            fid = d["frame_id"]
            hits = self._hits[d["cls"]]
            hits.append(fid)
            while hits and fid - hits[0] > window:
                hits.popleft()
            if len(hits) < hits_needed:
                continue  # 한두 프레임 오검출로 동작하지 않도록 연속으로 보일 때만
            last = self._last_fired.get(d["cls"])
            if last is not None and fid - last < cfg.SIGN_COOLDOWN_FRAMES:
                continue
            self._last_fired[d["cls"]] = fid
            hits.clear()
            box = None
            if "box_cx" in d and "box_w" in d:
                box = (d["box_cx"] - d["box_w"] / 2, d.get("box_y2", 0) - d["box_h"], d["box_cx"] + d["box_w"] / 2,
                       d.get("box_y2", 0))
            events.append(SignEvent(spec[0], spec[1], d["cls"], d["conf"], fid, box))
        # 빨간불이 꺼짐: 빨간불 이벤트를 낸 뒤 LIGHT_OFF_RUNS 번 연속 검출에서 빨간불이 하나도 안 보이면 OFF 를 한 번 낸다.
        # 초록불을 못 잡아도(늦은 검출) 빨간불이 꺼진 것으로 출발할 수 있다. 검출 횟수로 세므로 YOLO 가 쉬는 동안은 안 센다.
        # 마지막 빨간불 상자가 화면 끝(좌·우·위) 가까이 있었으면 꺼진 게 아니라 화면 밖으로 나간 것일 수 있어 세지 않는다
        # (sim: 횡단보도 옆 신호등이 정지 위치에서 화면 오른쪽 밖으로 빠져 OFF 로 보고 빨간불에 출발했다)
        if reds:
            self._red_missing = 0
            b = max(reds, key=lambda d: d["box_h"])
            em = getattr(cfg, "LIGHT_EDGE_MARGIN", 64)
            self._red_cut = ("box_cx" in b and "box_w" in b and (b["box_cx"] - b["box_w"] / 2 <= em
                                                                  or b["box_cx"] + b["box_w"] / 2 >= ref_w - em)) \
                or ("box_y2" in b and b["box_y2"] - b["box_h"] <= em)
        elif not self._red_cut:
            self._red_missing += 1
        for ev in events:
            if ev.kind == "LIGHT":
                self._red_fired = ev.value == "RED"
        if self._red_fired and self._red_missing >= getattr(cfg, "LIGHT_OFF_RUNS", 4):
            self._red_fired = False
            self._last_fired.pop("red_light", None)   # 빨간불이 다시 켜지면 쿨다운 없이 바로 알린다
            events.append(SignEvent("LIGHT", "OFF", "red_light_off", 0.0, self._last_fid))
        return events


class SignWorker(threading.Thread):
    def __init__(self, detector, interpreter, interval_s=0.0, cpu_cores=None):
        super().__init__(daemon=True)
        self.detector = detector
        self.interpreter = interpreter
        self.interval_s = interval_s     # 검출 한 번이 이보다 빨리 끝나면 남은 시간만큼 쉰다
        self.cpu_cores = cpu_cores       # 이 스레드(와 검출기가 만드는 스레드)가 쓸 CPU 코어
        self._cond = threading.Condition()
        self._slot = None
        self._stopped = False
        self._events = queue.Queue()

    def submit(self, frame, frame_id):
        with self._cond:
            self._slot = (frame, frame_id)
            self._cond.notify()

    def poll(self):
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def run(self):
        if self.cpu_cores and hasattr(os, "sched_setaffinity"):   # Linux(Pi)에서만
            try:
                os.sched_setaffinity(0, set(self.cpu_cores))
            except OSError as e:
                print(f"[signs] CPU 코어 제한 실패: {e}")
        while True:
            with self._cond:
                while self._slot is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                frame, frame_id = self._slot
                self._slot = None
            t0 = time.monotonic()
            try:
                detections = self.detector.detect(frame)
            except Exception as e:  # 검출기 오류로 주행을 멈추지 않는다
                print(f"[signs] detector error: {e}")
                continue
            rest = self.interval_s - (time.monotonic() - t0)
            for d in detections:
                d.setdefault("frame_id", frame_id)
            for ev in self.interpreter(detections):
                self._events.put(ev)
            if rest > 0:
                time.sleep(rest)

    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify()


def _sign_process_main(weights, imgsz, interval_s, cpu_cores, frames, results):
    """SignProcess 의 자식 프로세스. 최신 프레임만 받아 검출하고 결과를 돌려준다."""
    if cpu_cores and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(cpu_cores))
        except OSError as e:
            print(f"[signs] CPU 코어 제한 실패: {e}")
    import numpy as np

    detector = YoloSignDetector(weights, imgsz=imgsz)
    t0 = time.monotonic()
    detector.detect(np.zeros((480, 640, 3), np.uint8))   # 첫 추론은 오래 걸리므로 미리
    results.put(("ready", (time.monotonic() - t0) * 1000.0))
    while True:
        item = frames.get()
        if item is None:
            return
        frame, frame_id = item
        t0 = time.monotonic()
        try:
            detections = detector.detect(frame)
        except Exception as e:
            print(f"[signs] detector error: {e}")
            continue
        for d in detections:
            d.setdefault("frame_id", frame_id)
        results.put(("det", detections))
        rest = interval_s - (time.monotonic() - t0)
        if rest > 0:
            time.sleep(rest)


class SignProcess:
    """SignWorker 와 같은 사용법(start / submit / poll / stop)이지만 검출을 별도 프로세스에서 돌린다."""

    def __init__(self, weights, imgsz, interpreter, interval_s=0.0, cpu_cores=None):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")   # 카메라·GPIO 를 연 부모를 fork 하지 않는다
        self.interpreter = interpreter
        self._frames = ctx.Queue(maxsize=1)
        self._results = ctx.Queue()
        self._proc = ctx.Process(target=_sign_process_main, daemon=True,
                                 args=(weights, imgsz, interval_s, cpu_cores, self._frames, self._results))
        self.ready_ms = None

    def start(self, timeout=120.0):
        self._proc.start()
        kind, ms = self._results.get(timeout=timeout)   # 모델 준비가 끝날 때까지 기다린다
        self.ready_ms = ms

    def submit(self, frame, frame_id):
        try:
            self._frames.put_nowait((frame, frame_id))
        except queue.Full:   # 검출기가 아직 바쁘면 이전 프레임을 버리고 최신 것만 남긴다
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait((frame, frame_id))
            except queue.Full:
                pass

    def poll(self):
        out = []
        while True:
            try:
                kind, detections = self._results.get_nowait()
            except queue.Empty:
                return out
            if kind == "det":
                out.extend(self.interpreter(detections))

    def stop(self):
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            pass
        self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()

