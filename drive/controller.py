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

from drive.vision import INTERSECTION, LANE, LOST, OFF_ROAD, ONE_SIDE

PID_STATES = (LANE, ONE_SIDE)


def near_error(res, cfg):
    """차가 도로 가운데에서 얼마나 벗어났는지 추정 (-1~1, 가까운 띠 화면 폭 기준). 없으면 None.
    카메라는 앞만 본다. 차가 가운데 쪽으로 비스듬히 향하면 가까운 띠의 도로 중앙은 화면 가운데에 와도 차는 아직
    치우쳐 있다. 그래서 가까운 띠와 중간 띠의 도로 중앙을 이은 선을 차 쪽으로 CENTER_EXTRAP 배 연장한다."""
    if not res.bands or res.bands[0].center is None:
        return None
    half = cfg.PROC_SIZE[0] / 2.0
    e_near = (res.bands[0].center - half) / half
    e = e_near
    if len(res.bands) > 1 and res.bands[1].center is not None:
        e_mid = (res.bands[1].center - half) / half
        e = e_near + getattr(cfg, "CENTER_EXTRAP", 1.0) * (e_near - e_mid)
    return max(-1.0, min(1.0, e))


def band_edges(res):
    """컨트롤러에 넘기는 띠별 경계. 양쪽 경계를 찾았지만 폭이 평소 도로 폭과 안 맞아 비전이 버린 띠(center 없음)는
    "lr" 로 바꾼다. 실차: 왼쪽으로 30~40도 돌았을 때 블록 양옆 좁은 틈을 "LR" 로 넘겨, 새 길로 보고 회전을 끝냈다."""
    # 초록 위에서 시작한 띠(앞이 트랙 테두리·블록으로 막힘)는 "GG"
    return ["GG" if getattr(b, "note", "") == "start on green" else
            "lr" if b.edges == "LR" and b.center is None else b.edges for b in res.bands]


def real_openings(res, cfg):
    """띠마다 경계가 없는 쪽이 '진짜 열린 길'인지. 반대쪽 경계 + 평소 도로 폭으로 없는 경계가 있어야 할 자리를 구해,
    그 자리가 화면 안쪽(SIDE_OPEN_MARGIN px 이상)인데 경계가 없으면 열린 길이다. 화면 밖이면 차가 비스듬하거나
    한쪽에 붙어서 안 보이는 것이라 알 수 없다. 결과는 edges 와 같은 꼴: "L-" 은 오른쪽이 진짜 열림, "L?" 는 모름."""
    w = cfg.PROC_SIZE[0]
    margins = getattr(cfg, "SIDE_OPEN_MARGIN", (10, 30, 30))
    out = []
    for i, b in enumerate(res.bands):
        m = margins[min(i, len(margins) - 1)]
        e = b.edges
        if e == "L-" and b.left is not None and b.learned:
            e = "L-" if b.left + b.learned <= w - m else "L?"
        elif e == "-R" and b.right is not None and b.learned:
            e = "-R" if b.right - b.learned >= m else "?R"
        out.append(e)
    return out


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
        self._red_on = False        # 빨간불을 봤고 아직 안 꺼졌다. 멀리서 봐도 바로 서지 않고 횡단보도 앞에서 선다
        self._crosswalk_age = None  # 차 바로 앞 횡단보도(CROSSWALK 이벤트)를 본 뒤 지난 프레임
        self._turn = None           # 교차로 회전 진행: [방향, 단계, 단계 프레임]
        self._finish_frames = None  # 정지선을 본 뒤 남은 주행 프레임
        self._stop_ignore = 0       # 정지 표지판 대기를 마친 뒤 같은 표지판을 무시할 남은 프레임
        self._pending_seen = None   # 나이를 세는 중인 pending_turn
        self._pending_age = 0       # pending_turn 을 받은 뒤 지난 프레임
        self._side_open = 0         # 회전할 쪽이 연속으로 열려 보인 프레임
        self._side_closed = 0       # 그 전에 양쪽 경계가 다 보인 프레임 (옆길은 닫혀 있다가 열린다)
        self.turn_started = None    # 회전을 막 시작했으면 (방향, 이유, 띠 경계)
        self._centering = False     # 도로 가운데 찾기 모드
        self._post_dir = None       # 막 끝낸 회전 방향
        self._turn_dir_now = None   # 지금 하고 있는 회전 방향
        self._last_turn_dir = None  # 마지막으로 끝난 회전 방향
        self._since_turn_end = 10 ** 6
        self._ahead = None          # 비전 ahead_green (앞이 막혔는지)
        self._tend = False          # 지금 회전이 막힌 T자 끝에서 빠져나가는 것인지
        self._gg_seen = 0
        self._stuck = 0             # 회전 중 화면이 그대로인 프레임 (차가 안 돎)
        self._kick = 0              # 세게 미는 남은 프레임
        self._last_view = None
        self.notice = None          # main.py 가 로그로 찍고 지우는 알림
        self._since_cross = 0       # 마지막 교차로·회전 뒤 지난 프레임
        self._straight_open = 0     # 직진 예약 중 옆길이 열려 보인 프레임
        self._open_skip = 0         # 옆길을 알아볼 때 건너뛴 가까운 띠 수
        self._side_runs = {"-R": 0, "L-": 0}   # 그쪽이 열려 보인 프레임 수 (표지판과 상관없이)
        self._side_ages = {"-R": 0, "L-": 0}   # 열리기 시작한 뒤 지난 프레임
        self._side_late = 0         # 표지판보다 먼저 열려 있던 프레임 (그만큼 덜 직진)
        self._side_real = {"-R": 0, "L-": 0}   # 이번 열림 동안 진짜 열림을 본 프레임
        self._unknown_runs = {"-R": 0, "L-": 0}   # 모든 띠에서 그쪽 경계가 안 보인 연속 프레임
        self._last_error = None
        self._sign_h = None         # 마지막 회전 표지판 상자 높이 (640x480 기준)
        self._turn_ambiguous = False  # 지금 옆길 회전이 경계로 판단이 안 되는 상태에서 정해졌는지 (표지판 크기로 시점 보정)
        self._late_entry = None     # 교차로에 들어선 뒤 늦게 온 회전 표지판: 남은 진입 프레임
        from collections import deque
        self._err_hist = deque(maxlen=getattr(self.cfg, "CURVE_HIST_FRAMES", 20))   # 커브 판단용 최근 오차
        self._in_curve_now = False
        self._curve_dir = 0         # +1 오른쪽 커브, -1 왼쪽 커브
        self._curve_exit = 0        # 커브가 끝난 뒤 안쪽으로 붙을 남은 프레임
        self._seek = False          # 지금 옆길 회전이 "살짝 꺾으며 초록 찾기" 방식인지
        self._done_run = 0          # 회전 끝 조건이 연속으로 맞은 프레임
        self._last_pid = None       # 마지막으로 도로를 보고 낸 PID 명령 (잠깐 초록만 보일 때 유지)
        self._lost_hold = 0
        self._deferred_turn = None  # 앞 표지판 방향을 쓰기 전에 본 다른 방향 (다음 교차로용)
        self._pending_stale = 0     # pending_turn 표지판을 마지막으로 본 뒤 지난 프레임
        self._post_turn = False  # 회전을 막 끝내 아직 교차로 안 (비전은 INTERSECTION 이지만 새 길을 따라간다)
        self._post_lane = self._post_frames = 0
        self._open_run = 0          # 가까운 띠가 연속으로 양쪽 경계가 아닌 프레임

    # ------------------------------------------------------------ 표지판 이벤트
    def on_sign(self, ev):
        cfg = self.cfg
        if ev.kind == "TURN":
            # 교차로를 지나는 중(직진·회전)에는 방향을 바꾸지 않는다. 교차로 건너편 길의 표지판이 보여
            # 직진 중에 우회전으로 바뀌어 선을 가로지른 일이 있었다
            if self._turn is None and ev.value == self._last_turn_dir and \
                    self._since_turn_end < getattr(cfg, "POST_TURN_SAME_SIGN_FRAMES", 60):
                # 방금 돈 방향과 같은 표지판을 회전 직후 다시 봤다 = 방금 그 표지판 (실차 run_060120: 우회전 직후 같은 표지판을
                # 다시 보고 우회전이 또 예약됨)
                return
            if self._turn is not None:
                # 옆길 회전을 정하고 직진(ENTRY) 중인데 같은 방향 표지판이 충분히 커졌다(가깝다) -> 직진을 줄이고 곧 돈다
                box = getattr(ev, "box", None)
                if box and ev.value == self._turn[0] and self._turn[1] == "ENTRY" and not self._tend \
                        and self._turn_ambiguous \
                        and box[3] - box[1] >= getattr(cfg, "SIGN_NEAR_TURN_H", 60):
                    self._turn[3] = min(self._turn[3], self._turn[2] + getattr(cfg, "SIGN_NEAR_ENTRY_FRAMES", 5))
                    self.notice = f"표지판 가까움 ({box[3] - box[1]:.0f}px) -> 직진 줄임"
                return
            if self._cross_frames and self.pending_turn is None and ev.value in ("LEFT", "RIGHT") \
                    and self._cross_frames <= getattr(cfg, "CROSS_LATE_SIGN_FRAMES", 20):
                # 정한 방향 없이 교차로에 막 들어섰는데 회전 표지판이 늦게 확정됐다 (표지판이 옆길 바로 앞에 있음).
                # 직진 표지판을 보고 들어간 경우는 아래처럼 직진을 지킨다
                self.pending_turn = ev.value
                self._late_entry = max(getattr(cfg, "SIDE_MIN_ENTRY_FRAMES", 5),
                                       cfg.TURN_ENTRY_FRAMES - self._cross_frames)
                self._cross_frames = 0
                return
            if self._cross_frames:
                # 교차로를 직진하는 중에는 방향을 바꾸지 않는다 (가운데 교차로에서 건너편 길의 우회전 표지판을 보고
                # 대각선으로 꺾은 일). 다만 버리지 않고 받아 둔다: 앞이 막힌 T자면 직진할 수 없으니 이 방향으로 돈다.
                # 실차: T자 건너편 우회전 표지판이 교차로에 들어선 뒤에야 커져서 무시됐고, 막힌 쪽으로 직진하다 LOST
                if ev.value in ("LEFT", "RIGHT"):
                    self._deferred_turn = ev.value
                return
            if self.pending_turn is not None and ev.value != self.pending_turn and \
                    self._pending_stale < getattr(cfg, "PENDING_STALE_FRAMES", 200):
                # 앞 표지판 방향을 아직 쓰지 않았다 (그 교차로·옆길을 아직 안 지남). 뒤 표지판은 다음 교차로 것이므로
                # 받아 두었다가 앞 방향을 쓴 뒤에 적용한다. 앞 표지판은 교차로 약 30cm 전에 화면 밖으로 나가므로
                # "안 보이면 바로 바꾸기"는 하지 않는다
                self._deferred_turn = ev.value
                return
            if ev.value != self.pending_turn:
                self._deferred_turn = None
            self._pending_stale = 0              # 이 방향 표지판을 방금 봤다
            self.pending_turn = ev.value
            box = getattr(ev, "box", None)
            self._sign_h = (box[3] - box[1]) if box else None   # 표지판이 가까울수록 크다 (옆길 회전 시점)
            # 같은 표지판을 다시 보면 옆길 대기 시간을 처음부터 센다. 실차는 느려서(약 5cm/s) 표지판을 처음 본 뒤
            # 옆길까지 대기 시간보다 오래 걸린 일이 있었다 (우회전 표지판을 7번 보고도 옆길을 지나침)
            self._pending_age = 0
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
            # 신호등은 커서 멀리서도 빨간불이 확정된다. 보자마자 서면 횡단보도 한참 앞에서 섰다 (실차).
            # 그래서 빨간불은 기억만 해 두고, 차 바로 앞 횡단보도(CROSSWALK)가 같이 보일 때 선다
            if ev.value == "RED":
                if not self._before_cross():
                    return      # 이미 교차로 안: 신호가 바뀌어도 서지 않고 빠져나간다
                self._red_on = True
                if self._wait and self._wait[0] == "RED":
                    self._wait[1] = cfg.LIGHT_MAX_WAIT_FRAMES   # 아직 빨간불: 최대 대기를 처음부터 센다
                elif self._crosswalk_age is not None and \
                        self._crosswalk_age <= getattr(cfg, "RED_CROSSWALK_HOLD_FRAMES", 30):
                    self._stop_for_red("횡단보도 앞에서 빨간불 확정")
            else:   # GREEN, 또는 OFF (빨간불이 LIGHT_OFF_RUNS 번 연속 안 보임 = 꺼짐)
                self._red_on = False
                if self._wait and self._wait[0] == "RED":
                    self._wait = None   # 바로 출발
                    self.notice = "초록불 -> 출발" if ev.value == "GREEN" else "빨간불 꺼짐 -> 출발"
        elif ev.kind == "CROSSWALK":
            self._crosswalk_age = 0
            if self._red_on and self._before_cross() and not (self._wait and self._wait[0] == "RED"):
                self._stop_for_red("횡단보도 앞")
        elif ev.kind == "FINISH":
            if self._run_frames >= cfg.FINISH_ARM_FRAMES and self._finish_frames is None:
                self._finish_frames = cfg.FINISH_DELAY_FRAMES

    def _before_cross(self):
        """아직 교차로에 들어서지 않았는지 (빨간불에 설 수 있는 곳). 교차로를 직진·회전 중이거나 회전을 막 끝냈으면
        이미 지나온 것이다. 회전을 정하고 직진(ENTRY)하는 중은 아직 앞으로 본다 (옆길 회전은 옆길 전부터 ENTRY 다)."""
        return not self._cross_frames and not self._post_turn and (self._turn is None or self._turn[1] == "ENTRY")

    def _stop_for_red(self, why):
        self._wait = ["RED", self.cfg.LIGHT_MAX_WAIT_FRAMES]
        self.notice = f"빨간불: {why} -> 정지 (꺼질 때까지)"

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
        self._post_turn = False
        self._deferred_turn = None

    def sign_paused(self):
        """YOLO 를 쉬어야 하는지. 회전(TURN) 중과 출발 직후는 모터 전류가 가장 커서, YOLO 의 CPU 부하가 겹치면
        배터리 전압이 떨어져 Pi 가 꺼진다. 이 구간에는 표지판도 필요 없다 (회전 방향은 이미 정해져 있다)."""
        if not getattr(self.cfg, "SIGN_PAUSE_IN_TURN", True):
            return False
        if self._wait:
            return False    # 서 있으면 모터 전류가 없다. 빨간불에 서 있는 동안(회전 진입 중이라도) 신호가 바뀌는 것을 봐야 한다
        return self._turn is not None or (self.running and self._run_frames < getattr(self.cfg, "SIGN_PAUSE_START_FRAMES", 0))

    # ------------------------------------------------------------ 매 프레임
    def throttle_for(self, steering):
        cfg = self.cfg
        t = cfg.BASE_THROTTLE * self.speed_scale * (1.0 - cfg.SLOW_K * abs(steering))
        return clamp(t, cfg.MIN_THROTTLE, cfg.MAX_THROTTLE)

    def _center_first(self, state, error, edges):
        """곧은 길(중간·먼 띠 양쪽 경계)에서 오차가 CENTER_START_ERROR 를 넘으면 가운데 찾기 모드를 켜고,
        CENTER_DONE_ERROR 아래로 들어오면 끈다. 커브·교차로·회전 직후에는 쓰지 않는다 (커브는 바깥 바퀴를 밀어야 한다)."""
        cfg = self.cfg
        # 가장자리에 붙으면 가까운 띠에는 한쪽 경계가 화면 밖이다 (ONE_SIDE). 앞쪽(중간·먼 띠)에 양쪽 경계가
        # 보이면 곧은 길로 본다
        straight = (state in PID_STATES and not self._post_turn and edges and len(edges) >= 3
                    and edges[1] == "LR" and edges[2] == "LR")
        if not straight or not getattr(cfg, "CENTER_FIRST", True):
            self._centering = False
        elif self._centering:
            self._centering = abs(error) >= getattr(cfg, "CENTER_DONE_ERROR", 0.06)
        else:
            self._centering = abs(error) > getattr(cfg, "CENTER_START_ERROR", 0.15)
        return self._centering

    def lane_offset(self):
        """차선 안에서 설 자리 (+ 오른쪽, - 왼쪽, 오차 단위).
        - 기본 LANE_OFFSET: 표지판은 오른쪽 길가에 있다. 왼쪽으로 치우치면 표지판·교차로를 놓쳤다 (실차)
        - 회전이 정해지면 그쪽으로 TURN_PREP_OFFSET 만큼 붙는다 (왼쪽으로 돌 땐 왼쪽에 붙어야 잘 돈다)
        - 회전 직후에는 0 (새 길 가운데를 먼저 잡는다)"""
        cfg = self.cfg
        if self._post_turn:
            return 0.0
        if self.pending_turn == "LEFT":
            base = -getattr(cfg, "TURN_PREP_OFFSET", 0.0)
        elif self.pending_turn == "RIGHT":
            base = getattr(cfg, "TURN_PREP_OFFSET", 0.0)
        else:
            base = getattr(cfg, "LANE_OFFSET", 0.0)
        if self._curve_exit > 0:
            # 커브를 크게 돌고 나오면 바깥 라인에 붙어 다음 옆길을 못 찾았다 (실차: 왼쪽 위 커브 뒤 오른쪽 옆길).
            # 커브가 끝난 뒤 한동안 커브 안쪽(돈 쪽)으로 붙는다. 회전 예약 쪽과 같으면 더 큰 쪽을 쓴다
            add = self._curve_dir * getattr(cfg, "CURVE_EXIT_OFFSET", 0.0)
            base = max(base, add) if add > 0 else min(base, add)
        return base

    def _track_curve(self, state, error):
        """최근 오차 평균이 한쪽으로 CURVE_ERR 이상이면 커브를 도는 중이다 (실차 커브: 오차 -0.4~-0.8 이 계속).
        커브가 끝나면(평균이 CURVE_END_ERR 아래) CURVE_EXIT_FRAMES 동안 커브 안쪽으로 붙는다 (lane_offset)."""
        if self._curve_exit > 0:
            self._curve_exit -= 1
        if state not in PID_STATES or error is None or self._turn is not None or self._post_turn:
            self._err_hist.clear()
            self._in_curve_now = False
            return
        self._err_hist.append(error)
        if len(self._err_hist) < self._err_hist.maxlen:
            return
        m = sum(self._err_hist) / len(self._err_hist)
        if abs(m) >= getattr(self.cfg, "CURVE_ERR", 0.25) and self._curve_exit == 0:
            # (안쪽으로 붙는 동안에는 다시 판단하지 않는다: 설 자리를 옮길 때 오차가 출렁여 커브로 다시 보였다)
            self._in_curve_now, self._curve_dir = True, (1 if m > 0 else -1)
        elif self._in_curve_now and abs(m) < getattr(self.cfg, "CURVE_END_ERR", 0.1):
            self._in_curve_now = False
            self._curve_exit = getattr(self.cfg, "CURVE_EXIT_FRAMES", 120)
        if self._in_curve_now or self._curve_exit == getattr(self.cfg, "CURVE_EXIT_FRAMES", 120):
            # 커브 중·커브가 막 끝날 때는 교차로를 막 나온 것처럼 다룬다. 커브 안쪽 둥근 모서리가 물러나 보이는 것을
            # 옆길 열림으로 세어, 안쪽으로 붙자마자 진짜 옆길 전에 돌았다 (실차). 회전할 쪽 경계를 다시 양쪽 경계로 본 뒤
            # 열리는 것만 옆길로 센다. 그동안은 표지판 방향을 기억한 채 선을 따라간다
            self._since_cross = 0
            self._side_closed = 0
            for pat in ("-R", "L-"):
                self._side_runs[pat] = self._side_ages[pat] = self._side_real[pat] = self._unknown_runs[pat] = 0

    def _spin_throttle(self, error, edges, n=0):
        """제자리 회전 세기. 한쪽 바퀴를 세우고 도는 4륜 차는 바퀴를 바닥에 끌어야 해서, 0.32 로는 도중에 멈출 수 있다
        (실차: 45도쯤에서 못 돎). 화면(오차·띠 경계)이 TURN_STUCK_FRAMES 동안 그대로면 멈춘 것으로 보고
        TURN_KICK_FRAMES 동안 TURN_KICK_THROTTLE 로 민다. 세게 미는 시간을 짧게 두어 전압 강하를 줄인다."""
        cfg = self.cfg
        view = (None if error is None else round(error, 2), tuple(edges or ()))
        self._stuck = self._stuck + 1 if view == self._last_view else 0
        self._last_view = view
        if self._kick == 0 and self._stuck >= getattr(cfg, "TURN_STUCK_FRAMES", 8):
            self._kick = getattr(cfg, "TURN_KICK_FRAMES", 6)
            self._stuck = 0
            self.notice = f"회전 중 화면이 {getattr(cfg, 'TURN_STUCK_FRAMES', 8)} 프레임 그대로 -> 세게 밀기"
        if self._kick > 0:
            self._kick -= 1
            return self._slow(getattr(cfg, "TURN_KICK_THROTTLE", 0.42))
        # 오래 돌고 있으면(실차: 0.32 로 3초에 30도) 세기를 조금씩 올린다. 잘 도는 회전은 그 전에 끝나 영향이 없다
        boost = max(0, n - getattr(cfg, "TURN_BOOST_AFTER", 35)) * getattr(cfg, "TURN_BOOST_PER_FRAME", 0.005)
        return self._slow(min(cfg.TURN_THROTTLE + boost, getattr(cfg, "TURN_THROTTLE_MAX", 0.42)))

    def _slow(self, throttle):
        return clamp(throttle * self.speed_scale, self.cfg.MIN_THROTTLE, self.cfg.MAX_THROTTLE)

    def _start_turn(self, direction, entry_frames=None, why="교차로", edges=None):
        self._done_run = 0
        self._seek = False
        self._turn_ambiguous = False
        self._stuck = self._kick = 0
        self._last_view = None
        self._turn = [direction, "ENTRY", 0, entry_frames or self.cfg.TURN_ENTRY_FRAMES]
        self.pending_turn = None
        self.turn_started = (direction, why, "/".join(edges) if edges else "")   # main.py 가 로그로 찍고 지운다

    def _side_is_open(self, seen, open_pat, n=2):
        """회전할 쪽이 열렸는지. 가까운 n 개 띠가 모두 열림(open_pat)이면 열림.
        차가 반대쪽 선에 붙으면 경계가 있어야 할 자리가 화면 밖이라 "모름"("L?"/"?R")이 된다. 화면 밖으로 나가는 것은
        늘 가까운 띠부터다 (sim: 바깥 라인에 3cm 넘게 붙으면 오른쪽 옆길을 100% 놓침, 실제 띠: L?/L?/L-, L?/L-/L-).
        그래서 가까운 쪽에서부터 이어진 "모름"만 건너뛰고, 그 다음 띠부터 n 개가 열림이면(띠가 모자라면 남은 것 모두) 열림.
        커브는 가까운 띠가 열림인데 중간 띠가 "모름"(-R/?R/-R)이라 여기에 걸리지 않는다.
        건너뛴 띠 수를 self._open_skip 에 남긴다 (먼 띠로 알아봤으면 옆길이 그만큼 멀다)."""
        unknown = "L?" if open_pat == "L-" else "?R"
        look = list(seen[:n + 1])
        k = 0
        while k < len(look) and k < n and look[k] == unknown:
            k += 1
        rest = look[k:k + n] if k else look[:n]
        ok = bool(rest) and all(e == open_pat for e in rest)
        if ok:
            self._open_skip = k
        return ok

    def _recent_cross_now(self):
        return self._since_cross < getattr(self.cfg, "SIDE_AFTER_CROSS_FRAMES", 60)

    def _side_branch(self, edges, opens=None):
        """T자·옆길 교차로: 한쪽 경계는 그대로인데 회전할 쪽만 열리면 비전은 ONE_SIDE 로 본다.
        회전 표지판을 본 직후(SIDE_TURN_WINDOW_FRAMES 안)에 그쪽이 SIDE_OPEN_FRAMES 연속 열리면 교차로로 본다.
        edges: 띠별 "LR"/"-R"/"L-"/"--" (가까운 띠 먼저)."""
        cfg = self.cfg
        if self.pending_turn != self._pending_seen:
            self._pending_seen, self._pending_age, self._side_open = self.pending_turn, 0, 0
        self._pending_age += 1
        # 양쪽 경계를 본 기록은 표지판을 보기 전부터 센다. 표지판을 옆길 바로 앞에 두면, 표지판을 알아볼 때쯤
        # 이미 옆길이 열려 있다 (실차에서 표지판을 뒤로 옮겨야 돌던 원인). 다만 열린 지 오래됐으면 기록을 지운다.
        # 차가 회전할 쪽 반대편 경계에 붙어 가면 가까운 띠에는 회전할 쪽 경계가 화면 밖이라 늘 열려 보인다
        # (실차: 우회전 옆길에서 왼쪽 선에 붙어 있으면 옆길을 못 알아봄). 그래서 중간 띠의 양쪽 경계도 기록으로 센다
        if edges and (edges[0] == "LR" or (len(edges) > 1 and edges[1] == "LR")):
            self._side_closed += 1
            self._open_run = 0
        else:
            self._open_run += 1
            if self._open_run > getattr(cfg, "SIDE_OPEN_MAX_AGE", 40):
                self._side_closed = 0
        # 양쪽 옆길이 열려 보이는 것은 표지판이 오기 전부터 센다. 표지판이 옆길 바로 앞에 있으면 확정될 때쯤 차는 이미
        # 옆길 옆이라 먼 띠는 그 너머 블록(LR)을 본다 (sim·실차: 위 바깥 도로에서 오른쪽 옆길을 지나침)
        seen_all = opens if opens is not None else edges
        if edges:
            self._track_sides(seen_all, getattr(cfg, "SIDE_OPEN_BANDS", 2))
        if self.pending_turn == "STRAIGHT" and edges and not self._recent_cross_now():
            # 직진 표지판을 보고 옆길(진짜 열린 길)을 지나친다 -> 직진을 쓴 것. 예전에는 십자 교차로에서만 지워져서,
            # 뒤에 본 회전 표지판이 다음 교차로용으로 오래 보류되어 우회전 옆길을 지나쳤다
            seen = opens if opens is not None else edges
            n = getattr(cfg, "SIDE_OPEN_BANDS", 2)
            opened = all(e == "-R" for e in seen[:n]) or all(e == "L-" for e in seen[:n])
            self._straight_open = self._straight_open + 1 if opened else 0
            if self._straight_open >= getattr(cfg, "SIDE_OPEN_FRAMES", 10):
                self.pending_turn, self._straight_open = None, 0
                self._since_cross, self._side_closed = 0, 0    # 이 옆길에서 바로 다음 방향으로 돌지 않게
                return False
        if self.pending_turn not in ("LEFT", "RIGHT") or not edges or \
                self._pending_age > getattr(cfg, "SIDE_TURN_WINDOW_FRAMES", 0):
            self._side_open = 0
            return False
        open_pat = "-R" if self.pending_turn == "LEFT" else "L-"
        n = getattr(cfg, "SIDE_OPEN_BANDS", 2)
        # 교차로를 막 빠져나올 때도 한쪽이 열려 보이므로, 양쪽 경계를 먼저 충분히 본 뒤에 열린 것만 센다
        # 교차로나 회전을 마친 지 SIDE_AFTER_CROSS_FRAMES 가 지났으면 그 기록이 없어도 된다. 실차는 느려서
        # 표지판이 확정되기 전에 열린 구간이 40 프레임을 넘거나, 왼쪽에 붙어 와서 기록이 아예 없을 수 있다
        # (실차 사진: 오른쪽 옆길이 계속 L- 로 보이는데 직진)
        recent_cross = self._recent_cross_now()
        if self._side_closed < getattr(cfg, "SIDE_CLOSED_FRAMES", 5) and recent_cross:
            self._side_open = 0
            return False
        # opens: real_openings() 결과. 경계가 화면 밖이라 안 보이는 것("L?")은 열린 길로 세지 않는다. 경적을 우회전으로
        # 잘못 봤을 때 오른쪽에 길이 없는데도 돌던 것을 막는다
        self._side_open = self._side_runs[open_pat]
        if len(self._err_hist) >= 5 and abs(sum(self._err_hist) / len(self._err_hist)) >= getattr(cfg, "CURVE_ERR", 0.25):
            return False        # 커브를 도는 중 (커브 판정 20 프레임이 차기 전이라도): 옆길로 돌지 않는다
        if self._side_open >= getattr(cfg, "SIDE_OPEN_FRAMES", 3):
            # 표지판보다 먼저 열려 있었으면(age 가 길면) 옆길에 그만큼 가깝다 -> 직진을 줄인다
            self._side_late = max(0, self._side_ages[open_pat] - getattr(cfg, "SIDE_OPEN_FRAMES", 3))
            return True
        return False

    def _side_ambiguous(self, opens):
        """이번 열림이 진짜 열림을 거의 못 보고 "모름"만으로 쌓였으면 경계로는 옆길 위치를 알 수 없다.
        (실차 run_041701: 280 프레임 동안 오른쪽이 모두 "모름". sim 바깥 붙은 경우는 먼 띠에서 진짜 열림을 여러 번 본다)"""
        pat = "L-" if self.pending_turn == "RIGHT" else "-R"
        return self._side_real[pat] < getattr(self.cfg, "SIDE_REAL_MIN_FRAMES", 3)

    def _track_sides(self, seen, n):
        """양쪽(-R 왼쪽 열림, L- 오른쪽 열림)이 열려 보인 횟수와 열리기 시작한 뒤 지난 프레임을 센다.
        가까운 띠들이 모두 "모름"인 프레임은 지우지 않고 넘어간다 (바깥 라인에 붙으면 자주 나온다)."""
        for pat in ("-R", "L-"):
            unknown = "L?" if pat == "L-" else "?R"
            look = list(seen[:n + 1])
            if any(e in ("LR", "lr") for e in look[:n]):
                state = False
            elif self._side_is_open(seen, pat, n):
                state = True
                self._side_real[pat] += 1      # 진짜 열림(경계가 있어야 할 자리가 화면 안인데 없음)을 본 프레임
            elif look[:n] and all(e == unknown for e in look[:n]):
                state = None
            else:
                state = False
            straight = self._last_error is not None and abs(self._last_error) < getattr(self.cfg, "SIDE_ALL_UNKNOWN_MAX_ERR", 0.25)
            if state is None and straight and all(e in (unknown, pat) for e in look):
                # (곧게 갈 때만. 커브를 돌 때도 바깥 경계가 오래 화면 밖이지만 그때는 오차가 크다)
                # 모든 띠에서 그쪽 경계가 계속 안 보이면 열린 길이다 (실차 run_041701: 우회전 예약 중 L-/L-/L- 가 120 프레임
                # 넘게 이어졌는데 모두 "모름"이라 옆길로 세지 않고 지나침)
                self._unknown_runs[pat] += 1
                if self._unknown_runs[pat] >= getattr(self.cfg, "SIDE_ALL_UNKNOWN_FRAMES", 25):
                    state = True
            else:
                self._unknown_runs[pat] = 0
            if state is False:
                self._side_runs[pat] = self._side_ages[pat] = self._side_real[pat] = 0
                continue
            if state:
                self._side_runs[pat] += 1
            if self._side_runs[pat]:
                self._side_ages[pat] += 1

    def _turning(self, state, error=None, edges=None):
        """교차로 회전. ENTRY: 교차로 안쪽까지 직진 -> SPIN: 한쪽 바퀴 위주로 회전 -> EXIT: 차선이 보일 때까지 서행 직진.
        SPIN 은 TURN_FRAMES 를 채운 뒤에도, 새 길이 양쪽 경계와 함께 화면 가운데(|오차| < TURN_DONE_ERROR)에
        보일 때까지 계속 돈다 (최대 TURN_SPIN_MAX_FRAMES). 바닥에서 바퀴가 미끄러져 덜 돌았는데 한쪽 경계만 보고
        PID 로 넘어가면, 원래 길을 따라가 버린다 (실차에서 좌회전을 "하다 만" 원인)."""
        cfg = self.cfg
        direction, phase, n, entry = self._turn
        self._turn[2] += 1
        if phase == "ENTRY":
            if self._tend and edges and edges[-1] == "GG":
                # 막힌 T자 끝에서 빠져나갈 때: 먼 띠가 테두리에 닿은 뒤 TEND_EXTRA_FRAMES 더 가서 돈다 (사람 운전과 같은 시점).
                # 안으로 들어가는 회전보다 늦게 돈다
                self._gg_seen += 1
                if self._gg_seen >= getattr(cfg, "TEND_EXTRA_FRAMES", 4):
                    entry = n + 1
            if n + 1 >= entry:
                self._turn[1:3] = ["ARC" if self._seek else "SPIN", 0]
            return Command(throttle=self._slow(cfg.INTERSECTION_THROTTLE), steering=0.0, action="TURN")
        if phase == "ARC":
            # 옆으로 살짝 꺾으며 전진한다. 회전할 쪽 초록 경계가 가까운 띠에 다시 보이고 앞쪽(중간 띠)에 새 길의 양쪽
            # 경계가 보이면 라인 트레이싱(PID)으로 넘어간다
            side = "R" if direction == "RIGHT" else "L"
            found = (n + 1 >= getattr(cfg, "SEEK_MIN_FRAMES", 10) and edges and len(edges) >= 2
                     and side in edges[0] and edges[0] != "lr" and edges[1] == "LR" and state in PID_STATES)
            if found or n + 1 >= getattr(cfg, "SEEK_MAX_FRAMES", 200):
                self._turn = None
                self._post_turn, self._post_dir = True, direction
                self._post_lane = self._post_frames = 0
                return None
            st = getattr(cfg, "SEEK_STEER", 0.3) * (1 if direction == "RIGHT" else -1)
            return Command(throttle=self._slow(getattr(cfg, "SEEK_THROTTLE", 0.36)), steering=st, action="TURN")
        if phase == "SPIN":
            if n + 1 >= cfg.TURN_FRAMES:
                # 차는 아직 교차로 안이라 가까운 띠는 열려 있다. 앞쪽(중간·먼) 띠에 새 길의 양쪽 경계가 보이고
                # 도로 중앙이 화면 가운데 근처면 회전을 끝낸다 (상태값은 3프레임 확인 때문에 늦어서 띠를 직접 본다)
                # 비전이 OFF_ROAD/LOST 가 아닐 때만 (교차로 안이라 INTERSECTION 은 괜찮다). 45도쯤 블록을 정면으로
                # 보면 띠 경계는 "LR" 인데(블록 가운데를 좁은 길로 읽음) 상태는 OFF_ROAD 다. 그걸 새 길로 보고 끝내면
                # 반대로 조향하다 멈췄다
                done = (state in PID_STATES + (INTERSECTION,) and error is not None and edges is not None and len(edges) >= 3
                        and edges[1] == "LR" and edges[2] == "LR" and abs(error) < getattr(cfg, "TURN_DONE_ERROR", 0.35))
                # 45도 근처에서는 교차로 대각선 너머 블록 두 개 사이가 잠깐 길처럼 보일 수 있다. 연속으로 보여야 끝낸다
                self._done_run = self._done_run + 1 if done else 0
                if self._done_run >= getattr(cfg, "TURN_DONE_CONFIRM", 3):
                    self._done_run = 0
                    self._turn = None
                    self._side_closed = 0
                    self._post_turn = True   # 새 차선에 들어설 때까지는 새 회전을 시작하지 않고 PID 로 간다
                    self._post_dir = direction
                    self._post_lane = self._post_frames = 0
                    return None   # 새 길이 정면에 보인다 -> 이번 프레임부터 PID
                if n + 1 >= getattr(cfg, "TURN_SPIN_MAX_FRAMES", cfg.TURN_FRAMES):
                    # 도는 도중(약 45도)에는 화면이 블록의 초록만 보인다. 그때 직진(EXIT)으로 넘어가면 블록으로
                    # 들어가 멈춘다 (실차). 도로가 하나도 안 보이면 TURN_SPIN_HARD_MAX_FRAMES 까지 계속 돈다
                    # 새 길이 대충이라도 정면에 오기 전에는 직진으로 넘어가지 않는다. 실차: 0.32 로 30도밖에 못 돈 채
                    # 60 프레임이 지나 직진 -> PID 가 원래 길(오른쪽, 오차 +1.0)을 따라 오른쪽으로 꺾었다
                    near_aligned = (state in PID_STATES + (INTERSECTION,) and error is not None
                                    and abs(error) < getattr(cfg, "TURN_EXIT_ERROR", 0.5)
                                    and bool(edges) and any(e == "LR" for e in edges[1:3]))
                    if near_aligned or n + 1 >= getattr(cfg, "TURN_SPIN_HARD_MAX_FRAMES", 200):
                        self._turn[1:3] = ["EXIT", 0]
            s = cfg.TURN_STEERING if direction == "RIGHT" else -cfg.TURN_STEERING
            return Command(throttle=self._spin_throttle(error, edges, n), steering=s, action="TURN")
        # EXIT
        if state in PID_STATES:
            self._turn = None
            self._side_closed = 0
            return None   # 차선을 다시 잡았다 -> 이번 프레임부터 PID
        if n >= cfg.TURN_EXIT_MAX_FRAMES:
            self._turn = None
            return Command(action="STOP")
        return Command(throttle=self._slow(cfg.INTERSECTION_THROTTLE), steering=0.0, action="TURN")

    def _intersection(self, edges=None):
        if self.pending_turn in (None, "STRAIGHT") and self._deferred_turn in ("LEFT", "RIGHT") \
                and edges and edges[-1] == "GG" and not self._cross_frames:
            # 앞이 막힌 T자라 직진할 수 없다. 받아 둔 다음 방향이 이 교차로 것이다
            self.pending_turn, self._deferred_turn = self._deferred_turn, None
        if self.pending_turn in ("LEFT", "RIGHT"):
            tend = self._ahead is not None and self._ahead >= getattr(self.cfg, "TEND_AHEAD_GREEN", 0.75)
            late, self._late_entry = self._late_entry, None
            entry = getattr(self.cfg, "TURN_ENTRY_TEND_MAX", 90) if tend else late
            why = "막힌 T자 끝" if tend else ("교차로 (표지판 늦게 확정)" if late else "교차로")
            self._start_turn(self.pending_turn, entry, why, edges)
            self._tend, self._gg_seen = tend, 0
            return self._turning(INTERSECTION, edges=edges)
        # 직진 표지판 또는 표지판 없음: 감속 직진, INTERSECTION_MAX_FRAMES를 넘으면 정지
        self._cross_frames += 1
        if self._cross_frames > self.cfg.INTERSECTION_MAX_FRAMES:
            if self._deferred_turn in ("LEFT", "RIGHT"):
                # 교차로를 한참 직진했는데도 끝나지 않는다 (T자). 직진 중 받아 둔 회전 방향으로 돈다
                # (실차 run_060243: 우회전 표지판이 직진 중에 확정돼 보류됐고, 교차로 안에서 정지)
                d, self._deferred_turn, self._cross_frames = self._deferred_turn, None, 0
                self._start_turn(d, getattr(self.cfg, "TURN_BLOCKED_ENTRY_FRAMES", 3), "직진이 안 끝나 받아 둔 방향으로", edges)
                return self._turning(INTERSECTION, edges=edges)
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
                elif self._wait[0] == "RED":
                    self._red_on = False    # 빨간불을 오래 다시 못 봤다: 이 빨간불은 놓고 간다 (교차로 입구에서 또 서지 않게)
                self._wait = None

    def update(self, state: Optional[str], error: Optional[float], edges=None, near=None, opens=None,
               ahead=None) -> Command:
        """edges: 비전 띠별 경계 문자열 목록 (가까운 띠 먼저). 주면 T자·옆길 교차로 회전도 한다."""
        self._ahead = ahead
        self._last_error = error
        if self._crosswalk_age is not None:
            self._crosswalk_age += 1
        # 커브 판단은 설 자리를 반영한 오차(PID 가 맞추려는 값)로 한다. 곧은 길에서 한쪽에 서 있으면 원래 오차는 남지만 이 값은 0 근처다
        self._track_curve(state, None if error is None else error + self.lane_offset())
        horn = self._horn_frames > 0
        if horn:
            self._horn_frames -= 1
        cmd = None
        if self.running and not self.finished and not self._wait and self._red_on and state == INTERSECTION \
                and self._before_cross():
            # 빨간불인데 횡단보도를 못 잡은 채 교차로 입구까지 왔다 (실차 카메라는 횡단보도를 놓칠 수 있다).
            # 횡단보도 바로 뒤라 조금 늦지만, 신호를 어기고 교차로에 들어가지 않게 여기서 선다
            self._stop_for_red("횡단보도는 못 봤지만 교차로 입구")
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
            if self._post_turn:
                # 회전을 끝낸 차는 아직 교차로 안에 있다 (실차는 느려서 오래 머문다). 그때 다음 길의 회전 표지판을
                # 보고 바로 회전을 시작하면 교차로 안에서 엉뚱한 쪽으로 돈다. 새 차선(가까운 띠 양쪽 경계)을
                # POST_TURN_LANE_FRAMES 연속 볼 때까지는 회전을 시작하지 않는다. 표지판 방향은 받아 둔다
                self._post_frames += 1
                lane = state == LANE and bool(edges) and edges[0] == "LR"
                self._post_lane = self._post_lane + 1 if lane else 0
                if self._post_lane >= getattr(self.cfg, "POST_TURN_LANE_FRAMES", 8) or \
                        self._post_frames > getattr(self.cfg, "POST_TURN_MAX_FRAMES", 150):
                    self._post_turn = False
                    self._side_closed = 0
            if cmd is None and self._turn is None and not self._post_turn and state in PID_STATES \
                    and self._side_branch(edges, opens):
                # 먼 띠로 옆길을 알아봤으면(가까운 띠 "모름") 옆길이 그만큼 멀다 -> 더 직진 (SIDE_SKIP_EXTRA_FRAMES).
                # 표지판보다 옆길이 먼저 열려 있었으면 그만큼 옆길에 가깝다 -> 덜 직진
                amb = self._side_ambiguous(opens)
                late = self._side_late if self._side_late <= getattr(self.cfg, "SIDE_LATE_MAX_FRAMES", 40) else 0
                entry = getattr(self.cfg, "SIDE_TURN_ENTRY_FRAMES", 53) + \
                    self._open_skip * getattr(self.cfg, "SIDE_SKIP_EXTRA_FRAMES", 0) - late
                entry = max(getattr(self.cfg, "SIDE_MIN_ENTRY_FRAMES", 5), entry)
                if amb:
                    # 가까운·중간 띠 모두 "모름": 경계로는 옆길이 어디인지 모른다 (실차 run_041701: 표지판 전 280 프레임 동안
                    # 오른쪽이 안 보였다). 표지판이 가까워질 때(SIGN_NEAR_TURN_H) 직진을 줄여 돈다. 안 커지면 이만큼 뒤에 돈다
                    entry = getattr(self.cfg, "SIDE_AMBIG_ENTRY_FRAMES", 120)
                why = "옆길" + (f" (먼 띠 {self._open_skip}칸)" if self._open_skip else "") + \
                    (f" (표지판 전부터 열림 {late}프레임)" if late else "") + (" (경계 모름: 표지판 크기로)" if amb else "")
                if getattr(self.cfg, "SIDE_SEEK", False) and not amb:
                    # 살짝 꺾으며 초록 찾기 방식: 초록이 없어진 뒤 카메라가 보는 거리만큼 직진 (열림 확인에 쓴 프레임은 뺀다)
                    entry = max(1, getattr(self.cfg, "SEEK_STRAIGHT_FRAMES", 60) - getattr(self.cfg, "SIDE_OPEN_FRAMES", 10))
                    why += " (살짝 꺾으며 초록 찾기)"
                self._start_turn(self.pending_turn, entry, why, edges)
                self._turn_ambiguous = amb
                self._seek = getattr(self.cfg, "SIDE_SEEK", False) and not amb
            if cmd is None and self._turn is None and self._cross_frames and self._deferred_turn in ("LEFT", "RIGHT") \
                    and (state in (OFF_ROAD, LOST) or (edges and "GG" in edges)):
                # 직진하다 앞이 막혔다 (T자 끝) -> 받아 둔 방향으로 바로 돈다
                d, self._deferred_turn, self._cross_frames = self._deferred_turn, None, 0
                self.pending_turn = None
                self._start_turn(d, getattr(self.cfg, "TURN_BLOCKED_ENTRY_FRAMES", 3), "막힌 T자", edges)
            if cmd is None and self._turn is None and not self._post_turn and not self._cross_frames \
                    and self.pending_turn in ("LEFT", "RIGHT") and state in (OFF_ROAD, LOST):
                # 회전이 정해졌는데 앞이 막혔다 (섬·블록 정면, 실차 IMG_2399: 우회전 표지판 섬 앞에서 정지).
                # 그쪽 옆길이 최근 열려 보였으면 멈추지 않고 그쪽으로 돈다
                pat = "-R" if self.pending_turn == "LEFT" else "L-"
                if self._side_ages[pat] > 0 or self._side_runs[pat] > 0:
                    d = self.pending_turn
                    self._start_turn(d, getattr(self.cfg, "TURN_BLOCKED_ENTRY_FRAMES", 3), "앞이 막혀 정한 방향으로", edges)
            if cmd is None and self._turn is not None:
                cmd = self._turning(state, error, edges)
            if state == INTERSECTION or self._turn is not None or self._post_turn:
                self._side_closed = 0   # 교차로를 나올 때 한쪽이 열려 보이는 것을 옆길로 보지 않게
                self._since_cross = 0
            else:
                self._since_cross += 1
            if cmd is None:
                if self._post_turn and state not in PID_STATES + (INTERSECTION,) \
                        and self._post_frames <= getattr(self.cfg, "POST_TURN_MAX_FRAMES", 150):
                    # 회전을 끝낸 직후 다시 헷갈리는 화면(OFF_ROAD/LOST)이 나오면, 도로가 보일 때까지 하던 방향으로 계속 돈다
                    s = self.cfg.TURN_STEERING if self._post_dir == "RIGHT" else -self.cfg.TURN_STEERING
                    cmd = Command(throttle=self._slow(self.cfg.TURN_THROTTLE), steering=s, action="TURN")
                elif (state in PID_STATES or self._post_turn) and error is not None:
                    if self._last_action != "PID":
                        self.pid.reset()  # 정지/교차로에서 돌아올 때 이전 적분/미분값을 버린다
                    # 차선 안에서 설 자리. 오차에 더하면 차가 그만큼 오른쪽(+)/왼쪽(-)에 선다
                    offset = self.lane_offset()
                    error = error + offset
                    if near is not None:
                        near = near + offset
                    steering = float(self.pid(-error, dt=self.dt))
                    if (state == ONE_SIDE and not self._post_turn and edges and len(edges) >= 3
                            and edges[1] == "--" and edges[2] == "--" and getattr(self.cfg, "HOLD_BEFORE_CROSS", True)):
                        # 교차로가 바로 앞(중간·먼 띠 모두 열림)이면 가까운 띠의 한쪽 경계만 남는다. 블록 모서리가
                        # 둥글어서 그 경계가 바깥으로 물러나 보이고, 그걸 따라 조향하면 10~15도 틀어진 채 교차로에
                        # 들어가 직진 중 선을 넘는다. 그동안은 핸들을 곧게 둔다
                        steering = 0.0
                    k = getattr(self.cfg, "CENTER_K", 0.0)
                    if k and near is not None and state in PID_STATES and not self._post_turn:
                        # 내다본 오차만 쓰면 차가 이미 그쪽을 향할 때 치우쳐 있어도 "가운데"로 나온다. 커브에서 안쪽으로
                        # 파고들거나(sim) 바깥 테두리로 밀려(실차) 표지판·교차로를 놓쳤다. 차 위치의 치우침도 더한다
                        steering = clamp(steering + k * near, -1.0, 1.0)
                    throttle = self.throttle_for(steering)
                    if self._center_first(state, near if near is not None else error, edges):
                        # 도로 가운데에서 벗어나 있으면 (가장자리로 가면 표지판이 화면 밖으로 빠진다) 앞으로 가는 것보다
                        # 가운데로 돌아오는 것을 먼저 한다: 천천히 가면서 조향을 키운다. 가운데에 오면 원래 속도로
                        # 내다본 오차(error)는 차가 가운데 쪽으로 비스듬히 향하기만 해도 0 이 되어 조향을 멈춘다.
                        # 가까운 띠의 치우침(near)은 차가 실제로 도로 어디에 있는지라 그걸로 조향한다
                        e = near if near is not None else error
                        steering = clamp(getattr(self.cfg, "CENTER_GAIN", 1.5) * e, -1.0, 1.0)
                        throttle = self._slow(getattr(self.cfg, "CENTER_THROTTLE", self.cfg.MIN_THROTTLE))
                    cmd = Command(throttle=throttle, steering=steering, action="PID")
                elif state == INTERSECTION and self._post_turn:
                    cmd = Command(throttle=self._slow(self.cfg.INTERSECTION_THROTTLE), steering=0.0, action="PID")
                elif state == INTERSECTION:
                    cmd = self._intersection(edges)
                elif self._last_pid is not None and self._lost_hold < getattr(self.cfg, "LOST_HOLD_FRAMES", 8) \
                        and abs(self._last_pid.steering) >= getattr(self.cfg, "LOST_HOLD_MIN_STEER", 0.3):
                    # 커브를 꺾던 중 화면이 잠깐 초록만 보이면(블록을 정면으로 봄) 그 화면이 지나갈 때까지
                    # 직전 조향을 유지한다. LOST_HOLD_FRAMES 가 지나도 도로가 안 보이면 정지
                    self._lost_hold += 1
                    cmd = Command(throttle=self._last_pid.throttle, steering=self._last_pid.steering, action="PID")
                elif state in PID_STATES and edges and all(e == "--" for e in edges):
                    # 상태는 아직 차선인데 이번 프레임 띠가 모두 열렸다 = 교차로에 막 들어서는 중 (상태 확정은 3프레임 늦다).
                    # 여기서 한 프레임이라도 서면 모터 세기가 0부터 다시 올라간다. 그 사이 이런 프레임이 또 오면 정지 바퀴를 돌릴 세기에
                    # 끝내 못 닿아 교차로 입구에 영영 서 있었다 (sim: 빨간불에 섰다 출발한 뒤). 교차로 감속 직진처럼 곧게 간다
                    cmd = Command(throttle=self._slow(self.cfg.INTERSECTION_THROTTLE), steering=0.0, action="CROSS")
                else:
                    # OFF_ROAD, LOST, 상태 없음, 또는 이번 프레임에 도로 중앙이 없음 -> 즉시 정지
                    cmd = Command(action="STOP")
        if state != INTERSECTION and self._turn is None:
            if self._cross_frames:
                self.pending_turn = None  # 교차로를 지나면 저장한 방향을 비운다
            self._cross_frames = 0
        if self.pending_turn is not None:
            self._pending_stale += 1
            if self._deferred_turn and self._pending_stale >= getattr(self.cfg, "PENDING_STALE_FRAMES", 200):
                self.pending_turn, self._deferred_turn, self._pending_stale = self._deferred_turn, None, 0
        elif self._deferred_turn and not self._cross_frames:
            # 앞 방향을 썼다 (회전을 시작했거나 교차로를 지났다) -> 받아 둔 다음 교차로 방향을 적용
            self.pending_turn, self._deferred_turn, self._pending_stale = self._deferred_turn, None, 0
        if cmd.action != "PID":
            self._last_pid = None                       # 회전·직진·정지 등은 유지할 PID 가 없다
        elif (state in PID_STATES or self._post_turn) and error is not None:
            self._last_pid, self._lost_hold = cmd, 0    # 도로를 보고 낸 PID
        # 회전이 어떤 경로로 끝났든 (새 길 찾음 / EXIT / 정지) 끝난 뒤 지난 프레임을 센다 (같은 표지판 다시 보기 무시용)
        if self._turn is not None:
            self._turn_dir_now = self._turn[0]
        elif self._turn_dir_now is not None:
            self._last_turn_dir, self._turn_dir_now, self._since_turn_end = self._turn_dir_now, None, 0
        else:
            self._since_turn_end += 1
        self._last_action = cmd.action
        cmd.horn = horn
        return cmd
