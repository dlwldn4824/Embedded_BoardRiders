"""PID 조향, 속도 계산, 상태별 동작, 표지판 이벤트 처리 (교차로 회전, 정지, 신호등, 정지선).

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
    action: str = "STOP"   # PID / CROSS / TURN (움직임) | STOP / PAUSE / RED / FINISH (정지)
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
        self.finished = False       # 정지선 도착. set_running(True)로 다시 출발할 때까지 정지
        self._horn_frames = 0
        self._cross_frames = 0
        self._last_action = None
        self._run_frames = 0        # 출발 후 프레임 수 (출발 지점 정지선 무시용)
        self._speed_frames = 0
        self._wait = None           # (종류, 남은 프레임): "STOP" 정지 표지판, "RED" 빨간불
        self._turn = None           # 교차로 회전 진행: [방향, 단계, 단계 프레임]
        self._finish_frames = None  # 정지선을 본 뒤 남은 주행 프레임
        self._stop_ignore = 0       # 정지 표지판 대기를 마친 뒤 같은 표지판을 무시할 남은 프레임
        self._pending_seen = None   # 나이를 세는 중인 pending_turn
        self._pending_age = 0       # pending_turn 을 받은 뒤 지난 프레임
        self._side_open = 0         # 회전할 쪽이 연속으로 열려 보인 프레임
        self._side_closed = 0       # 그 전에 양쪽 경계가 다 보인 프레임 (옆길은 닫혀 있다가 열린다)

    # ------------------------------------------------------------ 표지판 이벤트
    def on_sign(self, ev):
        cfg = self.cfg
        if ev.kind == "TURN":
            self.pending_turn = ev.value
        elif ev.kind == "HORN":
            self._horn_frames = cfg.HORN_FRAMES
        elif ev.kind == "SPEED":
            self.speed_scale = float(ev.value)
            self._speed_frames = cfg.SPEED_FRAMES
        elif ev.kind == "STOP":
            # 이미 서 있거나 막 출발했으면 같은 표지판을 다시 본 것이다 (차 앞에 그대로 보인다)
            if not (self._wait and self._wait[0] == "STOP") and self._stop_ignore == 0:
                self._wait = ["STOP", cfg.STOP_WAIT_FRAMES]
        elif ev.kind == "LIGHT":
            if ev.value == "RED":
                self._wait = ["RED", cfg.LIGHT_MAX_WAIT_FRAMES]
            elif self._wait and self._wait[0] == "RED":
                self._wait = None   # 초록불: 바로 출발
        elif ev.kind == "FINISH":
            if self._run_frames >= cfg.FINISH_ARM_FRAMES and self._finish_frames is None:
                self._finish_frames = cfg.FINISH_DELAY_FRAMES

    # ------------------------------------------------------------ 주행 시작/정지
    def set_running(self, on):
        self.running = on
        self.pid.reset()
        if on and self.finished:
            self.finished = False
            self._finish_frames = None
            self._run_frames = 0

    def reset(self):
        self.pid.reset()
        self._cross_frames = 0
        self._turn = None

    def sign_paused(self):
        """YOLO 를 쉬어야 하는지. 회전(TURN) 중과 출발 직후는 모터 전류가 가장 커서, YOLO 의 CPU 부하가 겹치면
        배터리 전압이 떨어져 Pi 가 꺼진다. 이 구간에는 표지판도 필요 없다 (회전 방향은 이미 정해져 있다)."""
        if not getattr(self.cfg, "SIGN_PAUSE_IN_TURN", True):
            return False
        return self._turn is not None or (self.running and self._run_frames < getattr(self.cfg, "SIGN_PAUSE_START_FRAMES", 0))

    # ------------------------------------------------------------ 매 프레임
    def throttle_for(self, steering):
        cfg = self.cfg
        t = cfg.BASE_THROTTLE * self.speed_scale * (1.0 - cfg.SLOW_K * abs(steering))
        return clamp(t, cfg.MIN_THROTTLE, cfg.MAX_THROTTLE)

    def _slow(self, throttle):
        return clamp(throttle * self.speed_scale, self.cfg.MIN_THROTTLE, self.cfg.MAX_THROTTLE)

    def _start_turn(self, direction, entry_frames=None):
        self._turn = [direction, "ENTRY", 0, entry_frames or self.cfg.TURN_ENTRY_FRAMES]
        self.pending_turn = None

    def _side_branch(self, edges):
        """T자·옆길 교차로: 한쪽 경계는 그대로인데 회전할 쪽만 열리면 비전은 ONE_SIDE 로 본다.
        회전 표지판을 본 직후(SIDE_TURN_WINDOW_FRAMES 안)에 그쪽이 SIDE_OPEN_FRAMES 연속 열리면 교차로로 본다.
        edges: 띠별 "LR"/"-R"/"L-"/"--" (가까운 띠 먼저)."""
        cfg = self.cfg
        if self.pending_turn != self._pending_seen:
            self._pending_seen, self._pending_age, self._side_open = self.pending_turn, 0, 0
            self._side_closed = 0
        self._pending_age += 1
        if edges and edges[0] == "LR":
            self._side_closed += 1
        if self.pending_turn not in ("LEFT", "RIGHT") or not edges or \
                self._pending_age > getattr(cfg, "SIDE_TURN_WINDOW_FRAMES", 0):
            self._side_open = 0
            return False
        open_pat = "-R" if self.pending_turn == "LEFT" else "L-"
        n = getattr(cfg, "SIDE_OPEN_BANDS", 2)
        # 교차로를 막 빠져나올 때도 한쪽이 열려 보이므로, 양쪽 경계를 먼저 충분히 본 뒤에 열린 것만 센다
        if self._side_closed < getattr(cfg, "SIDE_CLOSED_FRAMES", 5):
            self._side_open = 0
            return False
        self._side_open = self._side_open + 1 if all(e == open_pat for e in edges[:n]) else 0
        return self._side_open >= getattr(cfg, "SIDE_OPEN_FRAMES", 3)

    def _turning(self, state):
        """교차로 회전. 비전이 교차로 안에서 믿을 수 없으므로 시간 기반으로 돈다.
        ENTRY: 교차로 안쪽까지 직진 -> SPIN: 한쪽 바퀴 위주로 회전 -> EXIT: 차선이 다시 보일 때까지 서행 직진."""
        cfg = self.cfg
        direction, phase, n, entry = self._turn
        self._turn[2] += 1
        if phase == "ENTRY":
            if n + 1 >= entry:
                self._turn[1:3] = ["SPIN", 0]
            return Command(throttle=self._slow(cfg.INTERSECTION_THROTTLE), steering=0.0, action="TURN")
        if phase == "SPIN":
            if n + 1 >= cfg.TURN_FRAMES:
                self._turn[1:3] = ["EXIT", 0]
            s = cfg.TURN_STEERING if direction == "RIGHT" else -cfg.TURN_STEERING
            return Command(throttle=self._slow(cfg.TURN_THROTTLE), steering=s, action="TURN")
        # EXIT
        if state in PID_STATES:
            self._turn = None
            self._side_closed = 0
            return None   # 차선을 다시 잡았다 -> 이번 프레임부터 PID
        if n >= cfg.TURN_EXIT_MAX_FRAMES:
            self._turn = None
            return Command(action="STOP")
        return Command(throttle=self._slow(cfg.INTERSECTION_THROTTLE), steering=0.0, action="TURN")

    def _intersection(self):
        if self.pending_turn in ("LEFT", "RIGHT"):
            self._start_turn(self.pending_turn)
            return self._turning(INTERSECTION)
        # 직진 표지판 또는 표지판 없음: 감속 직진, INTERSECTION_MAX_FRAMES를 넘으면 정지
        self._cross_frames += 1
        if self._cross_frames > self.cfg.INTERSECTION_MAX_FRAMES:
            return Command(action="STOP")
        return Command(throttle=self._slow(self.cfg.INTERSECTION_THROTTLE), steering=0.0, action="CROSS")

    def _tick_timers(self):
        if self._speed_frames:
            self._speed_frames -= 1
            if self._speed_frames == 0:
                self.speed_scale = 1.0
        if self._stop_ignore:
            self._stop_ignore -= 1
        if self._wait:
            self._wait[1] -= 1
            if self._wait[1] <= 0:
                if self._wait[0] == "STOP":
                    self._stop_ignore = getattr(self.cfg, "STOP_IGNORE_FRAMES", 0)
                self._wait = None

    def update(self, state: Optional[str], error: Optional[float], edges=None) -> Command:
        """edges: 비전 띠별 경계 문자열 목록 (가까운 띠 먼저). 주면 T자·옆길 교차로 회전도 한다."""
        horn = self._horn_frames > 0
        if horn:
            self._horn_frames -= 1
        cmd = None
        if not self.running:
            cmd = Command(action="PAUSE")
        elif self.finished:
            cmd = Command(action="FINISH")
        elif self._wait:
            self._tick_timers()
            cmd = Command(action=self._wait[0] if self._wait else "STOP")
        else:
            self._run_frames += 1
            self._tick_timers()
            if self._finish_frames is not None:
                self._finish_frames -= 1
                if self._finish_frames < 0:
                    self.finished = True
                    cmd = Command(action="FINISH")
            if cmd is None and self._turn is None and state in PID_STATES and self._side_branch(edges):
                self._start_turn(self.pending_turn, getattr(self.cfg, "SIDE_TURN_ENTRY_FRAMES", None))
            if cmd is None and self._turn is not None:
                cmd = self._turning(state)
            if cmd is None:
                if state in PID_STATES and error is not None:
                    if self._last_action != "PID":
                        self.pid.reset()  # 정지/교차로에서 돌아올 때 이전 적분/미분값을 버린다
                    steering = float(self.pid(-error, dt=self.dt))
                    cmd = Command(throttle=self.throttle_for(steering), steering=steering, action="PID")
                elif state == INTERSECTION:
                    cmd = self._intersection()
                else:
                    # OFF_ROAD, LOST, 상태 없음, 또는 이번 프레임에 도로 중앙이 없음 -> 즉시 정지
                    cmd = Command(action="STOP")
        if state != INTERSECTION and self._turn is None:
            if self._cross_frames:
                self.pending_turn = None  # 교차로를 지나면 저장한 방향을 비운다
            self._cross_frames = 0
        self._last_action = cmd.action
        cmd.horn = horn
        return cmd
