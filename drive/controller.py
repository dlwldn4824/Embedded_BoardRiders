"""PID 조향, 속도 계산, 상태별 동작, 표지판 이벤트 받는 자리.

PID 사용 방식은 Donkeycar LineFollower (donkeycar/parts/line_follower.py,
https://github.com/autorope/donkeycar)를 참고했다.
Copyright (c) 2017 Will Roscoe, MIT License. 전문은 drive/LICENSE-donkeycar 참고.

원본과 다른 점:
  - 목표는 항상 화면 중앙(오차 0). 첫 프레임 위치를 목표로 삼는 TARGET_PIXEL=None 방식은 쓰지 않는다.
  - 도로를 못 찾으면 직전 조향/속도를 유지하지 않고 정지한다.
  - dt를 1/LOOP_HZ로 고정해 넘긴다 (offline에서도 같은 D항).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from simple_pid import PID

from drive.vision import INTERSECTION, LANE, ONE_SIDE

PID_STATES = (LANE, ONE_SIDE)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass
class Command:
    throttle: float = 0.0
    steering: float = 0.0
    action: str = "STOP"   # PID / CROSS / STOP / PAUSE
    horn: bool = False


class Controller:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = 1.0 / cfg.LOOP_HZ
        # simple_pid의 오차는 (setpoint - input). 도로가 오른쪽(error>0)이면 오른쪽(+)으로
        # 돌아야 하므로 setpoint=0, input=-error로 넣어 출력 = Kp * error 가 되게 한다.
        self.pid = PID(cfg.PID_P, cfg.PID_I, cfg.PID_D, setpoint=0.0,
                       sample_time=None, output_limits=(-1.0, 1.0))
        self.running = False        # 시작은 항상 정지 상태
        self.pending_turn = None    # 표지판으로 받은, 다음 교차로에서 실행할 방향
        self.speed_scale = 1.0      # 서행 표지판 속도 배율
        self._horn_frames = 0
        self._cross_frames = 0
        self._last_action = None

    # ------------------------------------------------------------ 표지판 이벤트
    def on_sign(self, ev):
        if ev.kind == "TURN":
            self.pending_turn = ev.value
        elif ev.kind == "HORN":
            self._horn_frames = self.cfg.HORN_FRAMES
        elif ev.kind == "SPEED":
            self.speed_scale = float(ev.value)

    # ------------------------------------------------------------ 주행 시작/정지
    def set_running(self, on):
        self.running = on
        self.pid.reset()

    def reset(self):
        self.pid.reset()
        self._cross_frames = 0

    # ------------------------------------------------------------ 매 프레임
    def throttle_for(self, steering):
        cfg = self.cfg
        t = cfg.BASE_THROTTLE * self.speed_scale * (1.0 - cfg.SLOW_K * abs(steering))
        return clamp(t, cfg.MIN_THROTTLE, cfg.MAX_THROTTLE)

    def _intersection(self):
        # TODO(YOLO): self.pending_turn(LEFT/STRAIGHT/RIGHT)에 따라 회전 동작을 넣는다.
        # 지금은 감속 직진만 하다가 INTERSECTION_MAX_FRAMES를 넘으면 정지한다.
        self._cross_frames += 1
        if self._cross_frames > self.cfg.INTERSECTION_MAX_FRAMES:
            return Command(action="STOP")
        t = clamp(self.cfg.INTERSECTION_THROTTLE * self.speed_scale,
                  self.cfg.MIN_THROTTLE, self.cfg.MAX_THROTTLE)
        return Command(throttle=t, steering=0.0, action="CROSS")

    def update(self, state: Optional[str], error: Optional[float]) -> Command:
        horn = self._horn_frames > 0
        if horn:
            self._horn_frames -= 1

        if not self.running:
            cmd = Command(action="PAUSE")
        elif state in PID_STATES and error is not None:
            if self._last_action != "PID":
                self.pid.reset()  # 정지/교차로에서 돌아올 때 이전 적분/미분값을 버린다
            steering = float(self.pid(-error, dt=self.dt))
            cmd = Command(throttle=self.throttle_for(steering), steering=steering, action="PID")
        elif state == INTERSECTION:
            cmd = self._intersection()
        else:
            # OFF_ROAD, LOST, 상태 없음, 또는 이번 프레임에 도로 중앙이 없음 -> 즉시 정지
            cmd = Command(action="STOP")

        if state != INTERSECTION:
            if self._cross_frames:
                self.pending_turn = None  # 교차로를 지나면 저장한 방향을 비운다
            self._cross_frames = 0
        self._last_action = cmd.action
        cmd.horn = horn
        return cmd
