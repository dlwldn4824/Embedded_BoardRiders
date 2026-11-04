"""폐루프 시뮬레이터. 가상 트랙 위에서 실제 주행 코드(vision, controller, motors)를 그대로 돌린다.

  python -m drive.sim                          # 창: 왼쪽 가상 카메라 + 오버레이, 오른쪽 위에서 본 트랙
  python -m drive.sim --start cross            # 가운데 교차로로 향하는 위치에서 시작
  python -m drive.sim --no-gui --seconds 60    # 창 없이 시뮬레이션 시간 60초를 돌리고 요약만 출력

키: s 시작/일시정지, space 비상 정지, q 종료, m 마스크 창, p 시뮬레이션 멈춤, n 한 단계
    r 시작 위치로, 1~4 시작 위치 바꾸기, j / l 차를 왼쪽 / 오른쪽으로 10도 돌리기
    오른쪽 트랙 그림을 클릭하면 차를 그 자리로 옮긴다.

가상 카메라는 run_20261028_190928에서 잰 띠별 도로 폭(264/206/143px)이 나오도록
높이·기울기·초점거리를 맞췄고, 색도 그 사진에서 잰 값이다.
트랙 치수와 모터 특성은 추정값이라 아래 상수로 바꿔 가며 시험한다.
GPIO, 카메라 없이 노트북에서 돈다. 주행 설정은 drive/config.py를 그대로 쓴다.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter, deque
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from drive import config
from drive import vision
from drive.controller import Controller
from drive.main import ROOT, CsvLog, MOVING_ACTIONS, Recorder, _text, draw_mask, draw_overlay
from drive.motors import Car

# ---------------------------------------------------------------- 트랙 (cm)
# 도로 30cm, 녹색 블록 60cm로 잡았다. 바깥 테두리 안에 블록 4개, 길은 3x3 격자.
PX_PER_CM = 2
BORDER = 20            # 바깥 녹색 테두리 폭
ROAD = 30              # 도로 폭 (녹색 경계 사이, 노란 선 포함)
BLOCK = 60             # 안쪽 녹색 블록 한 변
CURVE_RADIUS = 25      # 바깥 커브 안쪽 반지름 (구석 블록의 바깥 모서리). 바깥쪽은 이 값 + ROAD
CORNER_RADIUS = 10     # 나머지 블록 모서리 반지름
LINE_W = 1.5           # 녹색 경계를 따라 도로 쪽에 그린 노란 선 폭
MARGIN = 60            # 트랙 밖 바닥
TRACK = 2 * BORDER + 3 * ROAD + 2 * BLOCK

# 색 (BGR, run_20261028_190928에서 잰 값)
GREEN = (75, 198, 15)
ROAD_COLOR = (54, 64, 44)
YELLOW = (100, 210, 195)
WHITE = (245, 250, 225)
BLUE = (250, 180, 10)
FLOOR = (140, 160, 140)
WALL = (55, 66, 50)
BODY = (42, 49, 49)
BODY_PART = (40, 200, 220)

# ---------------------------------------------------------------- 카메라 (640x480 기준)
CAM_W, CAM_H = config.CAMERA_SIZE
CAM_FOCAL = 320.0      # px. 수평 화각 약 90도
CAM_PITCH_DEG = 13.4   # 아래로 기울어진 각도. 소실점이 y=164(320x240 기준 82)에 오는 값
CAM_HEIGHT = 13.2      # cm. 도로 30cm가 가까운 띠에서 264px가 되는 값
CAM_FORWARD = 5.0      # 바퀴 축 중심에서 카메라까지 앞쪽 거리 cm
CAM_MAX_DIST = 400.0   # 이보다 먼 바닥은 벽으로 칠한다
NOISE_SIGMA = 5.0      # 픽셀 잡음 표준편차

# ---------------------------------------------------------------- 차량
WHEEL_BASE = 14.0      # 좌우 바퀴 간격 cm
V_MAX = 80.0           # PWM 1.0일 때 바퀴 속도 cm/s
PWM_DEADBAND = 0.15    # 이보다 작은 PWM에서는 바퀴가 돌지 않는다 (실차 MIN_THROTTLE과 비교)
MOTOR_TAU = 0.1        # 모터 응답 시간상수 s
LATENCY_FRAMES = 1     # 카메라 지연. 이만큼 이전 위치에서 찍은 프레임으로 판단한다
SUBSTEPS = 5

# 시작 위치: 이름 -> (x cm, y cm, 방향 도). 방향 0은 오른쪽(동), 90은 아래(남), -90은 위(북).
_MID = BORDER + ROAD + BLOCK + ROAD / 2            # 가운데 세로/가로 도로 중심선
_BOTTOM = TRACK - BORDER - ROAD / 2                 # 아래쪽 바깥 도로 중심선
STARTS = {
    "outer": (70.0, _BOTTOM, 0.0),                  # 아래 바깥 도로, 동쪽으로
    "cross": (_MID, _BOTTOM - 25, -90.0),           # 가운데 세로 도로, 교차로를 향해 북쪽으로
    "corner": (170.0, _BOTTOM, 0.0),                # 오른쪽 아래 커브 바로 앞
    "offset": (70.0, _BOTTOM + 8, -15.0),           # 테두리 쪽으로 8cm 치우치고 15도 틀어짐
}


def _px(cm):
    return int(round((cm + MARGIN) * PX_PER_CM))


def _rrect(shape, x0, y0, x1, y1, r):
    """트랙 좌표(cm)의 둥근 사각형 마스크. r은 숫자 하나 또는 (좌상, 우상, 좌하, 우하) 반지름."""
    m = np.zeros(shape[:2], np.uint8)
    X0, Y0, X1, Y1 = _px(x0), _px(y0), _px(x1), _px(y1)
    cv2.rectangle(m, (X0, Y0), (X1, Y1), 255, -1)
    radii = r if isinstance(r, tuple) else (r,) * 4
    for (cx, cy, sx, sy), rad in zip(((X0, Y0, 1, 1), (X1, Y0, -1, 1), (X0, Y1, 1, -1), (X1, Y1, -1, -1)),
                                     radii):
        R = int(round(rad * PX_PER_CM))
        if R <= 0:
            continue
        xa, xb = sorted((cx, cx + sx * R))
        ya, yb = sorted((cy, cy + sy * R))
        m[ya:yb + 1, xa:xb + 1] = 0
        cv2.circle(m, (cx + sx * R, cy + sy * R), R, 255, -1)
    return m


def build_track():
    """위에서 본 트랙 그림(BGR)과 도로 마스크를 만든다."""
    size = _px(TRACK + MARGIN)
    shape = (size, size)
    img = np.full((size, size, 3), FLOOR, np.uint8)
    r_out = CURVE_RADIUS + ROAD + BORDER
    img[_rrect(shape, -2, -2, TRACK + 2, TRACK + 2, r_out + 2) > 0] = WHITE   # 바깥 흰 테이프
    img[_rrect(shape, 0, 0, TRACK, TRACK, r_out) > 0] = GREEN

    road = _rrect(shape, BORDER, BORDER, TRACK - BORDER, TRACK - BORDER, CURVE_RADIUS + ROAD)
    lo, hi = BORDER + ROAD, BORDER + 2 * ROAD + BLOCK
    big, small = CURVE_RADIUS, CORNER_RADIUS
    blocks = {  # 구석 블록마다 트랙 바깥을 향한 모서리 하나만 크게 둥글다
        (lo, lo): (big, small, small, small),
        (hi, lo): (small, big, small, small),
        (lo, hi): (small, small, big, small),
        (hi, hi): (small, small, small, big),
    }
    for (bx, by), radii in blocks.items():
        road[_rrect(shape, bx, by, bx + BLOCK, by + BLOCK, radii) > 0] = 0
    img[road > 0] = ROAD_COLOR
    k = 2 * int(round(LINE_W * PX_PER_CM)) + 1
    inner = cv2.erode(road, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    img[(road > 0) & (inner == 0)] = YELLOW

    # 가운데 교차로로 들어가는 네 도로의 횡단보도. 들어오는 차의 오른쪽 블록에 파란 원
    c0, c1 = BORDER + ROAD + BLOCK, BORDER + 2 * ROAD + BLOCK       # 가운데 교차로 범위
    sw, gap, length, n = 3.5, 2.5, 12, 5
    first = c0 + (ROAD - (n * sw + (n - 1) * gap)) / 2
    crossings = {   # 방향: (횡단보도 시작, 세로 도로 위인지, 파란 원 위치)
        "N": (c0 - 8 - length, True, (c0 - 12, c0 - 8 - length / 2)),
        "S": (c1 + 8, True, (c1 + 12, c1 + 8 + length / 2)),
        "W": (c0 - 8 - length, False, (c0 - 8 - length / 2, c1 + 12)),
        "E": (c1 + 8, False, (c1 + 8 + length / 2, c0 - 12)),
    }
    for a0, vertical, (bx, by) in crossings.values():
        for i in range(n):
            b0 = first + i * (sw + gap)
            if vertical:
                p, q = (_px(b0), _px(a0)), (_px(b0 + sw), _px(a0 + length))
            else:
                p, q = (_px(a0), _px(b0)), (_px(a0 + length), _px(b0 + sw))
            cv2.rectangle(img, p, q, WHITE, -1)
        cv2.circle(img, (_px(bx), _px(by)), int(6 * PX_PER_CM), BLUE, -1, cv2.LINE_AA)
    return img, road


class Camera:
    """평평한 바닥을 보는 바늘구멍 카메라. 픽셀마다 바닥 위 (앞 거리, 오른쪽 거리)를 미리 계산해 둔다."""

    def __init__(self, track, light=1.0, seed=0):
        self.track = track
        self.light = light
        u, v = np.meshgrid(np.arange(CAM_W, dtype=np.float32), np.arange(CAM_H, dtype=np.float32))
        xc = (u - CAM_W / 2 + 0.5) / CAM_FOCAL
        yc = (v - CAM_H / 2 + 0.5) / CAM_FOCAL
        th = math.radians(CAM_PITCH_DEG)
        fwd = math.cos(th) - yc * math.sin(th)
        down = math.sin(th) + yc * math.cos(th)
        t = CAM_HEIGHT / np.maximum(down, 1e-3)
        self.d = (fwd * t).astype(np.float32)      # 앞 거리 cm
        self.r = (xc * t).astype(np.float32)       # 오른쪽 거리 cm
        sky = (down <= 1e-3) | (self.d > CAM_MAX_DIST)
        self.d[sky] = 0
        self.r[sky] = 0
        self._D = self.d * PX_PER_CM
        self._R = self.r * PX_PER_CM
        # 위쪽 벽과 화면 맨 아래 차체(범퍼, 노란 부품 두 개)를 덮어 그릴 그림
        self.cover = np.zeros((CAM_H, CAM_W, 3), np.uint8)
        self.cover_mask = np.zeros((CAM_H, CAM_W), np.uint8)
        self.cover[sky] = WALL
        self.cover_mask[sky] = 255
        y_body = int(CAM_H * 0.94)
        self.cover[y_body:] = BODY
        self.cover_mask[y_body:] = 255
        for x, ang in ((int(CAM_W * 0.11), 15), (int(CAM_W * 0.91), -15)):
            cv2.ellipse(self.cover, (x, CAM_H), (14, 44), ang, 0, 360, BODY_PART, -1)
            cv2.ellipse(self.cover_mask, (x, CAM_H), (14, 44), ang, 0, 360, 255, -1)
        rng = np.random.default_rng(seed)
        self._noise = []
        for _ in range(6):
            nz = rng.normal(0, NOISE_SIGMA, (CAM_H, CAM_W, 3))
            self._noise.append((np.clip(nz, 0, 255).astype(np.uint8), np.clip(-nz, 0, 255).astype(np.uint8)))
        self._i = 0

    def ground_to_world(self, x, y, psi, d, r):
        c, s = math.cos(psi), math.sin(psi)
        cx, cy = x + CAM_FORWARD * c, y + CAM_FORWARD * s
        return cx + d * c - r * s, cy + d * s + r * c

    def pixel_to_world(self, x, y, psi, u, v):
        u = int(min(max(u, 0), CAM_W - 1))
        v = int(min(max(v, 0), CAM_H - 1))
        return self.ground_to_world(x, y, psi, float(self.d[v, u]), float(self.r[v, u]))

    def render(self, x, y, psi):
        c, s = math.cos(psi), math.sin(psi)
        ox = (x + CAM_FORWARD * c + MARGIN) * PX_PER_CM
        oy = (y + CAM_FORWARD * s + MARGIN) * PX_PER_CM
        mx = cv2.addWeighted(self._D, c, self._R, -s, ox)
        my = cv2.addWeighted(self._D, s, self._R, c, oy)
        img = cv2.remap(self.track, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=FLOOR)
        img = cv2.copyTo(self.cover, self.cover_mask, img)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        pos, neg = self._noise[self._i]
        self._i = (self._i + 1) % len(self._noise)
        img = cv2.subtract(cv2.add(img, pos), neg)
        if self.light != 1.0:
            img = cv2.convertScaleAbs(img, alpha=self.light)
        return img


class Vehicle:
    """차동 구동 운동학. 좌우 PWM -> 바퀴 속도(데드밴드, 1차 지연) -> 위치와 방향."""

    def __init__(self, x, y, psi_deg, right_gain=1.0):
        self.right_gain = right_gain
        self.place(x, y, psi_deg)

    def place(self, x, y, psi_deg):
        self.x, self.y, self.psi = x, y, math.radians(psi_deg)
        self.vl = self.vr = 0.0

    @staticmethod
    def wheel_speed(pwm):
        mag = max(0.0, (abs(pwm) - PWM_DEADBAND) / (1.0 - PWM_DEADBAND))
        return math.copysign(V_MAX * mag, pwm)

    def step(self, left, right, dt):
        h = dt / SUBSTEPS
        a = 1.0 - math.exp(-h / MOTOR_TAU)
        tl, tr = self.wheel_speed(left), self.wheel_speed(right) * self.right_gain
        moved = 0.0
        for _ in range(SUBSTEPS):
            self.vl += (tl - self.vl) * a
            self.vr += (tr - self.vr) * a
            v = (self.vl + self.vr) / 2.0
            self.x += v * math.cos(self.psi) * h
            self.y += v * math.sin(self.psi) * h
            self.psi += (self.vl - self.vr) / WHEEL_BASE * h   # 왼쪽이 빠르면 오른쪽으로 돈다
            moved += abs(v) * h
        return moved

    @property
    def speed(self):
        return (self.vl + self.vr) / 2.0


class Sim:
    def __init__(self, args, cfg=config):
        self.cfg = cfg
        self.track, self.road = build_track()
        self.cam = Camera(self.track, args.light)
        self.start_name = args.start
        self.veh = Vehicle(*STARTS[args.start], right_gain=args.right_gain)
        self.det = vision.LaneDetector(cfg, flip=False)
        self.ctrl = Controller(cfg)
        self.car = Car(cfg, backend="none")
        self.frames = deque(maxlen=LATENCY_FRAMES + 1)
        self.t = 0.0
        self.odo = 0.0
        self.on_road = 0
        self.steps = 0
        self.stops = 0
        self.states = Counter()
        self.trail = []
        self.res = None
        self.cmd = None
        self.wheels = (0.0, 0.0)
        self.map_scale = 480 / self.track.shape[0]
        self.map_small = cv2.resize(self.track, (480, 480), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------ 시뮬레이션 한 단계
    def reset(self, name=None):
        if name:
            self.start_name = name
        self.place(*STARTS[self.start_name])
        self.det.reset()
        self.ctrl.reset()

    def place(self, x, y, psi_deg):
        self.veh.place(x, y, psi_deg)
        self.frames.clear()
        self.trail = []

    def is_on_road(self):
        px, py = _px(self.veh.x), _px(self.veh.y)
        h, w = self.road.shape
        return 0 <= px < w and 0 <= py < h and self.road[py, px] > 0

    def step(self, log=None, recorder=None):
        cfg, veh = self.cfg, self.veh
        dt = 1.0 / cfg.LOOP_HZ
        self.frames.append(self.cam.render(veh.x, veh.y, veh.psi))
        frame = self.frames[0]                     # LATENCY_FRAMES만큼 늦은 프레임
        self.res = self.det.process(frame)
        prev = self.cmd.action if self.cmd else None
        self.cmd = self.ctrl.update(self.res.state, self.res.error)
        if self.cmd.action in MOVING_ACTIONS:
            self.wheels = self.car.drive(self.cmd.throttle, self.cmd.steering)
        else:
            self.wheels = self.car.stop()
            if prev in MOVING_ACTIONS:
                self.stops += 1
        self.odo += veh.step(*self.wheels, dt)
        self.t += dt
        self.steps += 1
        self.on_road += self.is_on_road()
        self.states[self.res.state] += 1
        self.trail.append((veh.x, veh.y))
        del self.trail[:-3000]
        if log:
            log.write(self.t, self.res, self.cmd, self.wheels, self.steps)
        if recorder:
            recorder.put(self.steps, frame)

    def summary(self):
        pct = 100.0 * self.on_road / max(1, self.steps)
        st = ", ".join(f"{k}={v}" for k, v in self.states.items())
        return (f"sim {self.t:5.1f}s | 주행 거리 {self.odo / 100:5.2f} m | 도로 위 {pct:5.1f}% | "
                f"멈춘 횟수 {self.stops} | 상태 {st}")

    # ------------------------------------------------------------ 화면
    def _map_pt(self, x, y):
        k = PX_PER_CM * self.map_scale
        return int(round((x + MARGIN) * k)), int(round((y + MARGIN) * k))

    def draw_map(self):
        img = self.map_small.copy()
        veh = self.veh
        if len(self.trail) > 1:
            pts = ((np.array(self.trail) + MARGIN) * PX_PER_CM * self.map_scale).astype(np.int32)
            cv2.polylines(img, [pts], False, (0, 0, 255), 1, cv2.LINE_AA)
        if self.res is not None:   # 띠가 바닥에서 보는 선과 찾은 경계
            sx = CAM_W / self.cfg.PROC_SIZE[0]
            sy = CAM_H / self.cfg.PROC_SIZE[1]
            for b in self.res.bands:
                v = (b.y + b.h / 2) * sy
                a = self._map_pt(*self.cam.pixel_to_world(veh.x, veh.y, veh.psi, 0, v))
                z = self._map_pt(*self.cam.pixel_to_world(veh.x, veh.y, veh.psi, CAM_W - 1, v))
                cv2.line(img, a, z, (200, 200, 200), 1, cv2.LINE_AA)
                for xpx, col in ((b.left, (0, 0, 255)), (b.right, (255, 80, 0)), (b.center, (0, 255, 255))):
                    if xpx is not None and 0 <= xpx < self.cfg.PROC_SIZE[0]:
                        p = self._map_pt(*self.cam.pixel_to_world(veh.x, veh.y, veh.psi, xpx * sx, v))
                        cv2.circle(img, p, 3, col, -1, cv2.LINE_AA)
        c, s = math.cos(veh.psi), math.sin(veh.psi)
        body = [(-8, -8), (12, -8), (12, 8), (-8, 8)]   # 길이 20cm, 폭 16cm
        poly = np.array([self._map_pt(veh.x + f * c - r * s, veh.y + f * s + r * c) for f, r in body], np.int32)
        cv2.fillPoly(img, [poly], (0, 140, 255))
        cv2.line(img, self._map_pt(veh.x, veh.y), self._map_pt(veh.x + 14 * c, veh.y + 14 * s),
                 (0, 0, 0), 2, cv2.LINE_AA)
        pct = 100.0 * self.on_road / max(1, self.steps)
        _text(img, f"t {self.t:5.1f}s  v {veh.speed:5.1f} cm/s  {self.odo / 100:5.2f} m", (6, 18))
        _text(img, f"on road {pct:5.1f}%  stops {self.stops}  start {self.start_name}", (6, 36))
        return img

    def view(self, paused):
        if self.res is None:
            ov = np.zeros((480, 640, 3), np.uint8)
        else:
            extra = ["SIM paused (p / n)"] if paused else []
            ov = draw_overlay(self.res, self.cfg, self.cmd, self.wheels, self.cfg.LOOP_HZ, 0.0,
                              self.ctrl.running, None, extra)
            if ov.shape[0] != 480:
                ov = cv2.resize(ov, (int(ov.shape[1] * 480 / ov.shape[0]), 480))
        return np.hstack([ov, self.draw_map()])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="가상 트랙 폐루프 시뮬레이터")
    p.add_argument("--start", choices=list(STARTS), default="outer", help="시작 위치")
    p.add_argument("--no-gui", action="store_true", help="창 없이 돌리고 요약만 출력 (자동 출발)")
    p.add_argument("--seconds", type=float, default=60.0, help="--no-gui에서 돌릴 시뮬레이션 시간")
    p.add_argument("--autostart", action="store_true", help="창 모드에서도 s 없이 바로 출발")
    p.add_argument("--speed", type=float, default=1.0, help="창 모드 재생 배속")
    p.add_argument("--light", type=float, default=1.0, help="밝기 배율 (조명 변화 시험)")
    p.add_argument("--right-gain", type=float, default=1.0, help="오른쪽 모터 세기 비율 (좌우 불균형 시험)")
    p.add_argument("--show-mask", action="store_true", help="녹색 마스크 창도 띄움")
    p.add_argument("--record", metavar="DIR", help="가상 카메라 프레임 저장 (offline/tune으로 다시 보기)")
    p.add_argument("--log", metavar="CSV", help="로그 경로 (기본 logs/sim_<시각>.csv)")
    p.add_argument("--turn", choices=["LEFT", "RIGHT", "STRAIGHT"],
                   help="방향 표지판을 이미 본 것처럼 다음 교차로 방향을 정해 두고 출발")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = config
    sim = Sim(args, cfg)
    log_path = Path(args.log) if args.log else ROOT / cfg.LOG_DIR / f"sim_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    log = CsvLog(log_path)
    recorder = Recorder(args.record) if args.record else None
    if args.no_gui or args.autostart:
        sim.ctrl.set_running(True)
    if args.turn:
        sim.ctrl.pending_turn = args.turn

    try:
        if args.no_gui:
            while sim.t < args.seconds:
                sim.step(log, recorder)
            print(sim.summary())
            return sim

        win = "sim"
        cv2.namedWindow(win)

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and x >= 640:
                k = sim.map_scale * PX_PER_CM
                sim.place((x - 640) / k - MARGIN, y / k - MARGIN, math.degrees(sim.veh.psi))

        cv2.setMouseCallback(win, on_mouse)
        period = 1.0 / cfg.LOOP_HZ / max(args.speed, 0.05)
        paused, step, show_mask = False, False, args.show_mask
        next_t = time.monotonic()
        print("s: 출발/일시정지  space: 비상 정지  r: 처음 위치  1~4: 시작 위치  j/l: 돌리기  클릭: 옮기기  q: 종료")
        while True:
            if not paused or step:
                sim.step(log, recorder)
                step = False
            cv2.imshow(win, sim.view(paused))
            if show_mask and sim.res is not None:
                cv2.imshow("mask", draw_mask(sim.res, cfg))
            key = cv2.waitKey(1)
            if key != -1:
                if key in cfg.KEY_QUIT:
                    break
                elif key in cfg.KEY_ESTOP:
                    sim.ctrl.set_running(False)
                elif key in cfg.KEY_TOGGLE_RUN:
                    sim.ctrl.set_running(not sim.ctrl.running)
                elif key in cfg.KEY_TOGGLE_MASK:
                    show_mask = not show_mask
                    if not show_mask:
                        cv2.destroyWindow("mask")
                elif key in cfg.KEY_PLAYBACK_PAUSE:
                    paused = not paused
                elif key in cfg.KEY_PLAYBACK_STEP:
                    step = True
                elif key == ord("r"):
                    sim.reset()
                elif ord("1") <= key < ord("1") + len(STARTS):
                    sim.reset(list(STARTS)[key - ord("1")])
                elif key in (ord("j"), ord("l")):
                    sim.veh.psi += math.radians(-10 if key == ord("j") else 10)
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()
        print(sim.summary())
        return sim
    finally:
        if recorder:
            recorder.close()
        log.close()
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
