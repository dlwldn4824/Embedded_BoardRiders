"""차동 구동 모터와 부저.

drive()의 기본(pivot) 규칙은 Donkeycar TwoWheelSteeringThrottle
(donkeycar/parts/actuator.py, https://github.com/autorope/donkeycar)에서 가져왔다.
Copyright (c) 2017 Will Roscoe, MIT License. 전문은 drive/LICENSE-donkeycar 참고.

backend:
  'gpio' : gpiozero로 실제 핀 제어 (offline에서는 GPIOZERO_PIN_FACTORY=mock)
  'none' : 핀을 만들지 않음 (dry 모드). 계산된 좌우 값만 돌려준다.

단독 실행 (바퀴를 띄운 상태에서):
  python3 -m drive.motors --test    # drive(0.3,0) -> drive(0.3,-1) -> drive(0.3,1) -> 경적
  python3 -m drive.motors --sweep   # PWM을 조금씩 올려 MIN_THROTTLE 실측
"""
from __future__ import annotations

import os
import threading
import time


def use_mock_pins():
    """GPIO 없는 PC에서 gpiozero를 mock 핀으로 쓴다 (PWM, TonalBuzzer 지원 핀)."""
    os.environ["GPIOZERO_PIN_FACTORY"] = "mock"
    os.environ["GPIOZERO_MOCK_PIN_CLASS"] = "mockpwmpin"
    from gpiozero import Device
    from gpiozero.pins.mock import MockFactory, MockPWMPin

    if not isinstance(Device.pin_factory, MockFactory):
        Device.pin_factory = MockFactory(pin_class=MockPWMPin)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def mix(throttle, steering, mode="pivot"):
    """throttle(-1~1), steering(-1 좌 ~ +1 우) -> (left, right), 각 -1~1.

    pivot: 안쪽 바퀴만 (1-|steering|)배로 감속 (TwoWheelSteeringThrottle과 같음).
           steering=-1이면 왼쪽 바퀴가 멈춘다.
    spin : 안쪽 바퀴를 (1-2|steering|)배로. |steering|=0.5에서 멈추고
           그 이상이면 역회전, -1/+1에서는 제자리 회전.
    """
    throttle = clamp(throttle, -1.0, 1.0)
    steering = clamp(steering, -1.0, 1.0)
    if mode == "spin":
        inner = throttle * (1.0 - 2.0 * abs(steering))
    else:
        inner = throttle * (1.0 - abs(steering))
    left = right = throttle
    if steering < 0:
        left = inner
    elif steering > 0:
        right = inner
    return left, right


class Car:
    def __init__(self, cfg, backend="gpio"):
        self.cfg = cfg
        self.backend = backend
        self.turn_mode = cfg.TURN_MODE
        self._lock = threading.Lock()
        self._last_cmd = time.monotonic()
        self._moving = False
        self._horn_on = False
        self._closed = False
        self.wheels = (0.0, 0.0)

        if backend == "gpio":
            from gpiozero import DigitalOutputDevice, PWMOutputDevice, TonalBuzzer

            f = cfg.PWM_FREQUENCY
            self.PWMA = PWMOutputDevice(cfg.PWMA, frequency=f)
            self.AIN1 = DigitalOutputDevice(cfg.AIN1)
            self.AIN2 = DigitalOutputDevice(cfg.AIN2)
            self.PWMB = PWMOutputDevice(cfg.PWMB, frequency=f)
            self.BIN1 = DigitalOutputDevice(cfg.BIN1)
            self.BIN2 = DigitalOutputDevice(cfg.BIN2)
            self.BUZZER = TonalBuzzer(cfg.BUZZER_PIN)
            self.stop()
            # 주행 루프가 멈춰도 바퀴가 계속 돌지 않도록 감시
            self._watchdog = threading.Thread(target=self._watch, daemon=True)
            self._watchdog.start()
        elif backend != "none":
            raise ValueError(f"unknown backend: {backend}")

    # ------------------------------------------------------------ 바퀴
    def _wheel(self, pwm, in1, in2, value):
        # 전진: IN1=0, IN2=1 (교안 motor_go와 같음), 후진: IN1=1, IN2=0
        if value >= 0:
            in1.value, in2.value = 0, 1
        else:
            in1.value, in2.value = 1, 0
        pwm.value = abs(value)

    def set_wheels(self, left, right):
        """좌우 바퀴 PWM. 0~1 범위 밖은 자른다 (spin 모드의 역회전만 음수 사용)."""
        left = clamp(left, -1.0, 1.0)
        right = clamp(right, -1.0, 1.0)
        with self._lock:
            if self.backend == "gpio" and not self._closed:
                self._wheel(self.PWMA, self.AIN1, self.AIN2, left)
                self._wheel(self.PWMB, self.BIN1, self.BIN2, right)
            self.wheels = (left, right)
            self._moving = left != 0 or right != 0
            self._last_cmd = time.monotonic()
        return self.wheels

    def drive(self, throttle, steering):
        left, right = mix(throttle, steering, self.turn_mode)
        # 좌우 모터 세기 차 보정. 교차로 직진/회전은 시간 기반이라 차가 한쪽으로 휘면 실패한다.
        return self.set_wheels(left, right * getattr(self.cfg, "RIGHT_WHEEL_TRIM", 1.0))

    def stop(self):
        with self._lock:
            if self.backend == "gpio" and not self._closed:
                self.PWMA.value = 0.0
                self.PWMB.value = 0.0
            self.wheels = (0.0, 0.0)
            self._moving = False
            self._last_cmd = time.monotonic()
        return self.wheels

    # ------------------------------------------------------------ 부저
    def horn(self, on):
        if on == self._horn_on:
            return
        self._horn_on = on
        if self.backend == "gpio" and not self._closed:
            if on:
                self.BUZZER.play(self.cfg.HORN_TONE_HZ)
            else:
                self.BUZZER.stop()

    # ------------------------------------------------------------ 안전
    def _watch(self):
        while not self._closed:
            time.sleep(0.05)
            if self._moving and time.monotonic() - self._last_cmd > self.cfg.MOTOR_WATCHDOG_S:
                print("[motors] watchdog: no command, stopping")
                self.stop()

    def close(self):
        self.stop()
        self.horn(False)
        with self._lock:
            self._closed = True
            if self.backend == "gpio":
                for dev in (self.PWMA, self.AIN1, self.AIN2, self.PWMB, self.BIN1, self.BIN2, self.BUZZER):
                    dev.close()


def _bench(args):
    from drive import config

    car = Car(config, backend="gpio")
    try:
        if args.test:
            steps = [
                ("drive(0.3, 0)  : 양쪽 전진", lambda: car.drive(0.3, 0)),
                ("stop", car.stop),
                ("drive(0.3, -1) : 왼쪽 바퀴 정지, 오른쪽만 전진", lambda: car.drive(0.3, -1)),
                ("stop", car.stop),
                ("drive(0.3, 1)  : 오른쪽 바퀴 정지, 왼쪽만 전진", lambda: car.drive(0.3, 1)),
                ("stop", car.stop),
            ]
            for name, fn in steps:
                lr = fn()
                print(f"{name:42s} left={lr[0]:+.2f} right={lr[1]:+.2f}")
                t_end = time.monotonic() + (1.0 if name == "stop" else 2.0)
                while time.monotonic() < t_end:  # 워치독에 걸리지 않도록 명령 반복
                    fn()
                    time.sleep(0.1)
            print("horn")
            car.horn(True)
            time.sleep(0.3)
            car.horn(False)
        if args.sweep:
            print("바퀴가 처음 돌기 시작하는 값을 config.MIN_THROTTLE에 넣는다.")
            pwm = 0.05
            while pwm <= 0.6 + 1e-9:
                print(f"PWM {pwm:.2f}")
                t_end = time.monotonic() + 1.5
                while time.monotonic() < t_end:
                    car.set_wheels(pwm, pwm)
                    time.sleep(0.1)
                pwm += 0.05
    except KeyboardInterrupt:
        pass
    finally:
        car.close()


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    p = argparse.ArgumentParser(description="모터/부저 점검 (바퀴를 띄우고 실행)")
    p.add_argument("--test", action="store_true", help="drive() 동작 확인 순서 실행")
    p.add_argument("--sweep", action="store_true", help="MIN_THROTTLE 실측용 PWM 증가")
    a = p.parse_args()
    if not (a.test or a.sweep):
        p.error("--test 또는 --sweep을 지정한다")
    _bench(a)
