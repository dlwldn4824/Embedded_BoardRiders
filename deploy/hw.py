"""카메라와 차량 제어. 수집과 실주행 추론에서 같이 쓴다.

steer: -1(최대 좌) ~ +1(최대 우), throttle: 0 ~ 1
fake=True 면 하드웨어 없이 (Mac 등에서) 동작한다.
"""
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


class Car:
    def __init__(self, cfg, fake=False):
        self.cfg = cfg
        self.fake = fake
        self.type = cfg["type"]
        self.last_angle = None
        if fake:
            return
        from gpiozero import AngularServo, Motor

        def motor(m):
            return Motor(forward=m["forward"], backward=m["backward"], enable=m.get("enable"), pwm=True)

        if self.type == "servo":
            s = cfg["servo"]
            self.servo = AngularServo(
                s["pin"], min_angle=0, max_angle=180,
                min_pulse_width=s["min_pulse_width"], max_pulse_width=s["max_pulse_width"],
            )
            self.motor = motor(cfg["motor"])
        elif self.type == "differential":
            self.left = motor(cfg["left_motor"])
            self.right = motor(cfg["right_motor"])
        else:
            raise ValueError(f"drive.type 은 servo 또는 differential: {self.type}")

    def servo_angle(self, steer):
        s = self.cfg["servo"]
        sign = -1 if s.get("invert") else 1
        return float(np.clip(s["center"] + sign * steer * s["range"], 0, 180))

    def drive(self, steer, throttle):
        steer = float(np.clip(steer, -1, 1))
        throttle = float(np.clip(throttle, 0, 1))
        if self.cfg.get("invert_throttle"):
            throttle = -throttle
        if self.type == "servo":
            self.last_angle = self.servo_angle(steer)
            if not self.fake:
                self.servo.angle = self.last_angle
                self._set(self.motor, throttle)
        else:
            g = self.cfg["differential_gain"]
            l = float(np.clip(throttle * (1 + g * steer), -1, 1))
            r = float(np.clip(throttle * (1 - g * steer), -1, 1))
            if not self.fake:
                self._set(self.left, l)
                self._set(self.right, r)

    @staticmethod
    def _set(m, v):
        if v > 0:
            m.forward(v)
        elif v < 0:
            m.backward(-v)
        else:
            m.stop()

    def stop(self):
        if self.fake:
            return
        if self.type == "servo":
            self.motor.stop()
            self.servo.angle = self.cfg["servo"]["center"]
        else:
            self.left.stop()
            self.right.stop()
