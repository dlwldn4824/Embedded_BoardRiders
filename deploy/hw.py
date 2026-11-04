"""카메라와 차량 제어. 수집과 실주행 추론에서 같이 쓴다.

차량: 좌우 바퀴 속도 차로 도는 차동 구동 (TB6612 계열, PWM + IN1/IN2).
핀 번호, 전진 방향, pivot 규칙은 팀 주행 코드 drive/motors.py 와 같다.

steer: -1(최대 좌) ~ +1(최대 우), throttle: 0 ~ 1
fake=True 면 하드웨어 없이 (Mac 등에서) 동작한다.
"""
import threading
import time

import cv2
import numpy as np


class Camera:
    def __init__(self, cfg, fake=False):
        self.cfg = cfg
        self.fake = fake
        self.locked = {}
        self._t = 0
        if fake:
            return
        from libcamera import Transform
        from picamera2 import Picamera2

        self.cam = Picamera2()
        conf = self.cam.create_video_configuration(
            main={"size": (cfg["width"], cfg["height"]), "format": "RGB888"},  # 메모리 순서는 BGR → OpenCV 그대로 사용
            transform=Transform(hflip=cfg.get("hflip", False), vflip=cfg.get("vflip", False)),
        )
        self.cam.configure(conf)
        self.cam.start()
        if cfg.get("lock_exposure", True):
            self._lock()

    def _lock(self):
        fixed = {
            "ExposureTime": self.cfg.get("exposure_time"),
            "AnalogueGain": self.cfg.get("analogue_gain"),
            "ColourGains": self.cfg.get("colour_gains"),
        }
        if any(v is None for v in fixed.values()):
            time.sleep(2.0)  # 자동 노출/AWB 수렴 대기
            md = self.cam.capture_metadata()
            for k in fixed:
                if fixed[k] is None:
                    fixed[k] = md[k]
        fixed["ColourGains"] = tuple(fixed["ColourGains"])
        self.cam.set_controls({"AeEnable": False, "AwbEnable": False, **fixed})
        self.locked = {k: (list(v) if isinstance(v, tuple) else v) for k, v in fixed.items()}

    def read(self):
        if self.fake:
            self._t += 1
            img = np.full((self.cfg["height"], self.cfg["width"], 3), 60, np.uint8)
            x = int(self.cfg["width"] / 2 + 120 * np.sin(self._t / 20))
            cv2.line(img, (x, self.cfg["height"]), (self.cfg["width"] // 2, self.cfg["height"] // 2), (0, 220, 255), 12)
            return img
        return self.cam.capture_array("main")

    def close(self):
        if not self.fake:
            self.cam.stop()


def mix(throttle, steer, mode="pivot"):
    """(throttle, steer) → (left, right). drive/motors.py 의 mix 와 같은 규칙.

    pivot: 안쪽 바퀴만 (1-|steer|)배로 감속. steer=-1 이면 왼쪽 바퀴가 멈춘다.
    spin : 안쪽 바퀴를 (1-2|steer|)배로. |steer|>0.5 면 역회전.
    """
    throttle = float(np.clip(throttle, -1, 1))
    steer = float(np.clip(steer, -1, 1))
    inner = throttle * (1 - 2 * abs(steer) if mode == "spin" else 1 - abs(steer))
    left = right = throttle
    if steer < 0:
        left = inner
    elif steer > 0:
        right = inner
    return left, right


class Car:
    def __init__(self, cfg, fake=False):
        self.cfg = cfg
        self.fake = fake
        self.wheels = (0.0, 0.0)
        self._last_cmd = time.monotonic()
        self._lock = threading.Lock()
        self._closed = False
        if fake:
            return
        from gpiozero import DigitalOutputDevice, PWMOutputDevice

        f = cfg["pwm_frequency"]
        self.motors = []
        for side in ("left", "right"):
            p = cfg[side]
            self.motors.append((
                PWMOutputDevice(p["pwm"], frequency=f),
                DigitalOutputDevice(p["in1"]),
                DigitalOutputDevice(p["in2"]),
            ))
        self.stop()
        threading.Thread(target=self._watch, daemon=True).start()

    def _wheel(self, motor, value):
        pwm, in1, in2 = motor
        # 전진: IN1=0, IN2=1 (교안 motor_go 와 같음)
        in1.value, in2.value = (0, 1) if value >= 0 else (1, 0)
        pwm.value = min(1.0, abs(value))

    def drive(self, steer, throttle):
        left, right = mix(throttle, steer, self.cfg["turn_mode"])
        with self._lock:
            if not self.fake and not self._closed:
                self._wheel(self.motors[0], left)
                self._wheel(self.motors[1], right)
            self.wheels = (left, right)
            self._last_cmd = time.monotonic()
        return self.wheels

    def stop(self):
        with self._lock:
            if not self.fake and not self._closed:
                for pwm, _, _ in self.motors:
                    pwm.value = 0.0
            self.wheels = (0.0, 0.0)
            self._last_cmd = time.monotonic()

    def _watch(self):
        """메인 루프가 멈춰도 바퀴가 계속 돌지 않게 한다."""
        while not self._closed:
            time.sleep(0.05)
            if self.wheels != (0.0, 0.0) and time.monotonic() - self._last_cmd > self.cfg["watchdog_s"]:
                self.stop()

    def close(self):
        self.stop()
        with self._lock:
            self._closed = True
            if not self.fake:
                for dev in self.motors:
                    for d in dev:
                        d.close()
