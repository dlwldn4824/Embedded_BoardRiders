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
        self._stuck = 0             # 회전 중 화면이 그대로인 프레임 (차가 안 돎)
        self._kick = 0              # 세게 미는 남은 프레임
        self._last_view = None
        self.notice = None          # main.py 가 로그로 찍고 지우는 알림
        self._since_cross = 0       # 마지막 교차로·회전 뒤 지난 프레임
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
            if self._cross_frames or self._turn is not None:
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
        self._post_turn = False
        self._deferred_turn = None

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

    def _spin_throttle(self, error, edges):
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
        return self._slow(cfg.TURN_THROTTLE)

    def _slow(self, throttle):
        return clamp(throttle * self.speed_scale, self.cfg.MIN_THROTTLE, self.cfg.MAX_THROTTLE)

    def _start_turn(self, direction, entry_frames=None, why="교차로", edges=None):
        self._done_run = 0
        self._stuck = self._kick = 0
        self._last_view = None
        self._turn = [direction, "ENTRY", 0, entry_frames or self.cfg.TURN_ENTRY_FRAMES]
        self.pending_turn = None
        self.turn_started = (direction, why, "/".join(edges) if edges else "")   # main.py 가 로그로 찍고 지운다

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
        recent_cross = self._since_cross < getattr(cfg, "SIDE_AFTER_CROSS_FRAMES", 60)
        if self._side_closed < getattr(cfg, "SIDE_CLOSED_FRAMES", 5) and recent_cross:
            self._side_open = 0
            return False
        # opens: real_openings() 결과. 경계가 화면 밖이라 안 보이는 것("L?")은 열린 길로 세지 않는다. 경적을 우회전으로
        # 잘못 봤을 때 오른쪽에 길이 없는데도 돌던 것을 막는다
        seen = opens if opens is not None else edges
        self._side_open = self._side_open + 1 if all(e == open_pat for e in seen[:n]) else 0
        return self._side_open >= getattr(cfg, "SIDE_OPEN_FRAMES", 3)

    def _turning(self, state, error=None, edges=None):
        """교차로 회전. ENTRY: 교차로 안쪽까지 직진 -> SPIN: 한쪽 바퀴 위주로 회전 -> EXIT: 차선이 보일 때까지 서행 직진.
        SPIN 은 TURN_FRAMES 를 채운 뒤에도, 새 길이 양쪽 경계와 함께 화면 가운데(|오차| < TURN_DONE_ERROR)에
        보일 때까지 계속 돈다 (최대 TURN_SPIN_MAX_FRAMES). 바닥에서 바퀴가 미끄러져 덜 돌았는데 한쪽 경계만 보고
        PID 로 넘어가면, 원래 길을 따라가 버린다 (실차에서 좌회전을 "하다 만" 원인)."""
        cfg = self.cfg
        direction, phase, n, entry = self._turn
        self._turn[2] += 1
        if phase == "ENTRY":
            if n + 1 >= entry:
                self._turn[1:3] = ["SPIN", 0]
            return Command(throttle=self._slow(cfg.INTERSECTION_THROTTLE), steering=0.0, action="TURN")
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
                    no_road = not edges or all(e == "--" for e in edges) or state not in PID_STATES + (INTERSECTION,)
                    if not no_road or n + 1 >= getattr(cfg, "TURN_SPIN_HARD_MAX_FRAMES", 120):
                        self._turn[1:3] = ["EXIT", 0]
            s = cfg.TURN_STEERING if direction == "RIGHT" else -cfg.TURN_STEERING
            return Command(throttle=self._spin_throttle(error, edges), steering=s, action="TURN")
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

    def update(self, state: Optional[str], error: Optional[float], edges=None, near=None, opens=None) -> Command:
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
                self._start_turn(self.pending_turn, getattr(self.cfg, "SIDE_TURN_ENTRY_FRAMES", None), "옆길", edges)
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
                    steering = float(self.pid(-error, dt=self.dt))
                    if (state == ONE_SIDE and not self._post_turn and edges and len(edges) >= 3
                            and edges[1] == "--" and edges[2] == "--" and getattr(self.cfg, "HOLD_BEFORE_CROSS", True)):
                        # 교차로가 바로 앞(중간·먼 띠 모두 열림)이면 가까운 띠의 한쪽 경계만 남는다. 블록 모서리가
                        # 둥글어서 그 경계가 바깥으로 물러나 보이고, 그걸 따라 조향하면 10~15도 틀어진 채 교차로에
                        # 들어가 직진 중 선을 넘는다. 그동안은 핸들을 곧게 둔다
                        steering = 0.0
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
                    cmd = self._intersection()
                elif self._last_pid is not None and self._lost_hold < getattr(self.cfg, "LOST_HOLD_FRAMES", 8) \
                        and abs(self._last_pid.steering) >= getattr(self.cfg, "LOST_HOLD_MIN_STEER", 0.3):
                    # 커브를 꺾던 중 화면이 잠깐 초록만 보이면(블록을 정면으로 봄) 그 화면이 지나갈 때까지
                    # 직전 조향을 유지한다. LOST_HOLD_FRAMES 가 지나도 도로가 안 보이면 정지
                    self._lost_hold += 1
                    cmd = Command(throttle=self._last_pid.throttle, steering=self._last_pid.steering, action="PID")
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
        self._last_action = cmd.action
        cmd.horn = horn
        return cmd
