"""표지판(YOLO) 연결 지점. 지금은 아무것도 검출하지 않는다.

흐름:  SignWorker(스레드)  --detect(frame)-->  SignInterpreter  --SignEvent-->  Controller.on_sign()
주행 루프는 submit()으로 최신 프레임만 넘기고 poll()로 쌓인 이벤트만 가져가므로
검출기가 느려도 루프가 막히지 않는다. 처리 중에 들어온 프레임은 최신 것만 남는다.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass


class SignDetector:
    """detect(frame) -> [{"cls", "conf", "box_w", "box_h", "frame_id"}]"""

    def detect(self, frame):
        raise NotImplementedError


class NullSignDetector(SignDetector):
    def detect(self, frame):
        return []


@dataclass
class SignEvent:
    kind: str          # "TURN" (value: LEFT/STRAIGHT/RIGHT), "HORN", "SPEED" (value: 속도 배율)
    value: object
    cls: str
    conf: float
    frame_id: int


class SignInterpreter:
    """검출 결과를 이벤트로 바꾼다. 신뢰도, 박스 크기, 같은 클래스 반복을 거른다."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._last_fired = {}

    def __call__(self, detections):
        events = []
        for d in detections:
            spec = self.cfg.SIGN_CLASS_EVENTS.get(d["cls"])
            if spec is None:
                continue
            if d["conf"] < self.cfg.SIGN_MIN_CONF or d["box_h"] < self.cfg.SIGN_MIN_BOX_H:
                continue
            last = self._last_fired.get(d["cls"])
            if last is not None and d["frame_id"] - last < self.cfg.SIGN_COOLDOWN_FRAMES:
                continue
            self._last_fired[d["cls"]] = d["frame_id"]
            events.append(SignEvent(spec[0], spec[1], d["cls"], d["conf"], d["frame_id"]))
        return events


class SignWorker(threading.Thread):
    def __init__(self, detector, interpreter):
        super().__init__(daemon=True)
        self.detector = detector
        self.interpreter = interpreter
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
        while True:
            with self._cond:
                while self._slot is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                frame, frame_id = self._slot
                self._slot = None
            try:
                detections = self.detector.detect(frame)
            except Exception as e:  # 검출기 오류로 주행을 멈추지 않는다
                print(f"[signs] detector error: {e}")
                continue
            for d in detections:
                d.setdefault("frame_id", frame_id)
            for ev in self.interpreter(detections):
                self._events.put(ev)

    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify()
