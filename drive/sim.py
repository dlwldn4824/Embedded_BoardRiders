"""폐루프 시뮬레이터. 가상 트랙 위에서 실제 주행 코드(vision, controller, motors)를 그대로 돌린다.

  python -m drive.sim                          # 창: 왼쪽 가상 카메라 + 오버레이, 오른쪽 위에서 본 트랙
  python -m drive.sim --start cross            # 가운데 교차로로 향하는 위치에서 시작
  python -m drive.sim --no-gui --seconds 60    # 창 없이 시뮬레이션 시간 60초를 돌리고 요약만 출력

키: s 시작/일시정지, space 비상 정지, q 종료, m 마스크 창, p 시뮬레이션 멈춤, n 한 단계
    r 시작 위치로, 1~4 시작 위치 바꾸기, j / l 차를 왼쪽 / 오른쪽으로 10도 돌리기
    오른쪽 트랙 그림을 클릭하면 차를 그 자리로 옮긴다.

가상 카메라는 run_20261028_190928에서 잰 띠별 도로 폭(264/206/143px)이 나오도록
높이·기울기·초점거리를 맞췄고, 색도 그 사진에서 잰 값이다.
트랙은 실습 트랙 도면, 차 크기·속도·회전 속도·루프 주기는 2026-11-09 실차 영상에서 잰 값이다.
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
from drive.controller import Controller, near_error, real_openings, band_edges
from drive.main import ROOT, CsvLog, MOVING_ACTIONS, Recorder, _text, draw_mask, draw_overlay
from drive.motors import Car

# ---------------------------------------------------------------- 트랙 (cm)
# 실습 트랙 도면(200x200cm, 테두리 10cm)을 따랐다. 블록 4개, 길은 3x3 격자.
# 도로 폭은 2026-11-11 줄자 실측값: 바깥 28cm, 가운데 가로 도로(위아래 블록 사이) 18cm, 가운데 세로 도로(좌우 블록 사이) 21cm.
# 블록 크기는 트랙이 200cm 가 되도록 계산한다 (가로 51.5cm, 세로 53cm).
PX_PER_CM = 2
BORDER = 10            # 바깥 녹색 테두리 폭
# (예전 값: 모두 20cm 로 가정. 도면 사진으로는 바깥이 넓고 안쪽이 좁아 보였지만 비스듬해 못 쟀다.)
ROAD_OUT = 28          # 바깥 도로 폭 (녹색 경계 사이, 노란 선 포함)
ROAD_IN_V = 21         # 가운데 세로 도로 폭 (좌우 블록 사이, x 방향 간격)
ROAD_IN_H = 18         # 가운데 가로 도로 폭 (위아래 블록 사이, y 방향 간격)
ROAD_IN = ROAD_IN_V    # 예전 이름 (시험 스크립트 호환)
ROAD = ROAD_OUT        # 예전 이름 (시험 스크립트 호환)
TRACK = 200            # 트랙 한 변
BLOCK_W = (TRACK - 2 * BORDER - 2 * ROAD_OUT - ROAD_IN_V) / 2   # 블록 가로 (x) 51.5
BLOCK_H = (TRACK - 2 * BORDER - 2 * ROAD_OUT - ROAD_IN_H) / 2   # 블록 세로 (y) 53
BLOCK = BLOCK_W        # 예전 이름
CURVE_RADIUS = 17      # 바깥 커브 안쪽 반지름 (구석 블록의 바깥 모서리). 바깥쪽은 이 값 + ROAD_OUT
CORNER_RADIUS = 7      # 나머지 블록 모서리 반지름
LINE_W = 1.5           # 녹색 경계를 따라 도로 쪽에 그린 노란 선 폭
MARGIN = 60            # 트랙 밖 바닥

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
CAM_HEIGHT = 8.8 * 24 / 20   # cm. 가까운 띠에서 도로 24cm 가 264px 로 보이는 높이.
# 실차 사진(20261110)에서 잰 띠별(가까운/중간/먼) 도로 폭: 가운데 세로 도로(21cm) 233/189/150px, 바깥 도로(28cm) 294/245/188px.
# 24cm 기준이면 21cm 도로가 231/180/125, 28cm 도로가 308/240/167px 로 실차와 가장 가깝다
# (21cm 기준으로 두면 바깥 도로가 가까운 띠에서 352px 로 화면보다 넓어져 바깥 도로 전체를 교차로로 봤다)
CAM_FORWARD = 9.0      # 차체 중심(도는 중심)에서 카메라까지 앞쪽 거리 cm. 카메라는 차 앞쪽 끝에 있다
CAM_MAX_DIST = 400.0   # 이보다 먼 바닥은 벽으로 칠한다
NOISE_SIGMA = 5.0      # 픽셀 잡음 표준편차

# ---------------------------------------------------------------- 차량
# 2026-11-09 실차 영상(KakaoTalk_Video_2026-11-09-22-31-52, 22-45-58)에서 잰 값.
#   직진 BASE_THROTTLE 0.3: 약 4.7 cm/s (교차로 앞 직선 24~33초, 횡단보도·도로 폭 20cm 기준)
#   교차로 회전(바깥 바퀴 0.32, 안쪽 정지): 90도에 약 2.5초 (약 35 도/s)
#   차체 약 20 x 15 cm, 바퀴 4개(좌우 같은 쪽은 같이 돈다). 도로(20cm) 폭을 거의 다 차지한다.
#   PWM 0.25 에서는 교차로 감속 중 바퀴가 멈췄고 손으로 밀면 다시 갔다 (멈춘 바퀴를 다시 돌리려면 더 큰 PWM 이 필요)
BODY_LEN = 20.0        # 차체 길이 cm
BODY_W = 15.0          # 차체 폭 cm
WHEEL_TRACK = 13.0     # 좌우 바퀴 중심 간격 cm
WHEEL_AXLE = 7.0       # 차체 중심에서 앞/뒤 바퀴까지 cm
YAW_SLIP = 1.15        # 4륜 미끄럼 조향: 실제로 도는 양 = 바퀴 속도 차 / (WHEEL_TRACK * 이 값)
WHEEL_BASE = WHEEL_TRACK * YAW_SLIP
PWM_KEEP = 0.24        # 굴러가는 바퀴가 멈추는 PWM
PWM_START = 0.27       # 멈춘 바퀴가 다시 도는 PWM (정지 마찰)
V_SLOPE = 120.0        # PWM_KEEP 위에서 PWM 1 당 바퀴 속도 cm/s. 0.3 -> 7cm/s 바퀴, 차는 미끄럼으로 약 4.7cm/s
DRIVE_EFF = 0.68       # 바퀴 속도 중 실제로 앞으로 가는 비율 (4륜 미끄럼, 실측 직진 속도에 맞춤)
MOTOR_TAU = 0.15       # 모터 응답 시간상수 s
REAL_LOOP_HZ = 17.0    # 실차 루프 실측 주기 (교차로 직진 표지판 2920 -> 3078 프레임이 영상 약 9초). 시뮬레이션 시간은 이걸로 흐른다
LATENCY_FRAMES = 2     # 카메라 촬영 -> 판단까지 프레임 지연 (Pi 카메라 버퍼 + 처리)
SUBSTEPS = 5

# 시작 위치: 이름 -> (x cm, y cm, 방향 도). 방향 0은 오른쪽(동), 90은 아래(남), -90은 위(북).
_MID_X = BORDER + ROAD_OUT + BLOCK_W + ROAD_IN_V / 2   # 가운데 세로 도로 중심선 x
_MID_Y = BORDER + ROAD_OUT + BLOCK_H + ROAD_IN_H / 2   # 가운데 가로 도로 중심선 y
_MID = _MID_X
_BOTTOM = TRACK - BORDER - ROAD_OUT / 2             # 아래쪽 바깥 도로 중심선
STARTS = {
    "outer": (50.0, _BOTTOM, 0.0),                  # 아래 바깥 도로, 동쪽으로
    "cross": (_MID, _BOTTOM - 20, -90.0),           # 가운데 세로 도로, 교차로를 향해 북쪽으로
    "corner": (140.0, _BOTTOM, 0.0),                # 오른쪽 아래 커브 앞
    "offset": (50.0, _BOTTOM + 5, -15.0),           # 테두리 쪽으로 5cm 치우치고 15도 틀어짐
    "left_outer": (BORDER + ROAD_OUT / 2, 152.0, -90.0),  # 왼쪽 바깥 도로, 북쪽으로 (오른쪽에 가운데 가로 도로가 열린다)
    # 보라 경로: 바깥 세로 도로를 북쪽으로 올라가 위쪽 커브를 돈 뒤, 위 바깥 도로(28cm)에서 가운데 세로 도로(21cm)로 회전
    "top_east": (BORDER + ROAD_OUT / 2, 80.0, -90.0),          # 왼쪽 위 커브 -> 동쪽 -> 우회전
    # 반대 방향은 커브 전에 출발하면 위쪽 좌회전 표지판이 세로 도로에서 비스듬히 보여(sim 에는 가림이 없다) 커브에서 돈다.
    # 그래서 커브를 지난 직선에서 출발한다
    "top_west": (TRACK - 60.0, BORDER + ROAD_OUT / 2, 180.0),  # 위 바깥 도로, 서쪽으로 -> 좌회전
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
    r_out = CURVE_RADIUS + ROAD_OUT + BORDER
    img[_rrect(shape, -2, -2, TRACK + 2, TRACK + 2, r_out + 2) > 0] = WHITE   # 바깥 흰 테이프
    img[_rrect(shape, 0, 0, TRACK, TRACK, r_out) > 0] = GREEN

    road = _rrect(shape, BORDER, BORDER, TRACK - BORDER, TRACK - BORDER, CURVE_RADIUS + ROAD_OUT)
    lo = BORDER + ROAD_OUT
    hx, hy = lo + BLOCK_W + ROAD_IN_V, lo + BLOCK_H + ROAD_IN_H
    big, small = CURVE_RADIUS, CORNER_RADIUS
    blocks = {  # 구석 블록마다 트랙 바깥을 향한 모서리 하나만 크게 둥글다
        (lo, lo): (big, small, small, small),
        (hx, lo): (small, big, small, small),
        (lo, hy): (small, small, big, small),
        (hx, hy): (small, small, small, big),
    }
    for (bx, by), radii in blocks.items():
        road[_rrect(shape, bx, by, bx + BLOCK_W, by + BLOCK_H, radii) > 0] = 0
    img[road > 0] = ROAD_COLOR
    k = 2 * int(round(LINE_W * PX_PER_CM)) + 1
    inner = cv2.erode(road, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    img[(road > 0) & (inner == 0)] = YELLOW

    # 횡단보도 (도면 위치): 가운데 교차로로 들어가는 네 도로 + 바깥 도로 네 곳. 가운데 것은 들어오는 차의 오른쪽 블록에 파란 원
    sw, gap, length, n = 2.5, 1.8, 8, 4
    span = n * sw + (n - 1) * gap

    def crossing(a0, vertical, road_lo, width=ROAD_OUT):
        first = road_lo + (width - span) / 2
        for i in range(n):
            b0 = first + i * (sw + gap)
            if vertical:
                p, q = (_px(b0), _px(a0)), (_px(b0 + sw), _px(a0 + length))
            else:
                p, q = (_px(a0), _px(b0)), (_px(a0 + length), _px(b0 + sw))
            cv2.rectangle(img, p, q, WHITE, -1)

    x0, x1 = lo + BLOCK_W, lo + BLOCK_W + ROAD_IN_V   # 가운데 교차로 x 범위 (세로 도로)
    y0, y1 = lo + BLOCK_H, lo + BLOCK_H + ROAD_IN_H   # 가운데 교차로 y 범위 (가로 도로)
    for a0, vertical, (bx, by) in (
            (y0 - 5 - length, True, (x0 - 8, y0 - 5 - length / 2)),
            (y1 + 5, True, (x1 + 8, y1 + 5 + length / 2)),
            (x0 - 5 - length, False, (x0 - 5 - length / 2, y1 + 8)),
            (x1 + 5, False, (x1 + 5 + length / 2, y0 - 8))):
        if vertical:
            crossing(a0, True, x0, ROAD_IN_V)
        else:
            crossing(a0, False, y0, ROAD_IN_H)
        cv2.circle(img, (_px(bx), _px(by)), int(4 * PX_PER_CM), BLUE, -1, cv2.LINE_AA)
    near, far = BORDER, TRACK - BORDER - ROAD_OUT                  # 바깥 도로 (위/왼쪽, 아래/오른쪽)
    crossing(45, False, near)                 # 위 바깥 도로, 왼쪽
    crossing(45, True, far)                   # 오른쪽 바깥 도로, 위쪽
    crossing(TRACK - 45 - length, False, far)  # 아래 바깥 도로, 오른쪽
    crossing(TRACK - 45 - length, True, near)  # 왼쪽 바깥 도로, 아래쪽
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
    """4륜 미끄럼 조향(좌우 차동). 좌우 PWM -> 바퀴 속도(정지 마찰, 1차 지연) -> 위치와 방향."""

    def __init__(self, x, y, psi_deg, right_gain=1.0):
        self.right_gain = right_gain
        self.place(x, y, psi_deg)

    def place(self, x, y, psi_deg):
        self.x, self.y, self.psi = x, y, math.radians(psi_deg)
        self.vl = self.vr = 0.0
        self.stuck = [True, True]          # 바퀴가 멈춰 있으면 PWM_START 를 넘어야 돈다

    def wheel_speed(self, pwm, i):
        mag = abs(pwm)
        if self.stuck[i]:
            if mag < PWM_START:
                return 0.0
            self.stuck[i] = False
        elif mag < PWM_KEEP:
            self.stuck[i] = True
            return 0.0
        return math.copysign(V_SLOPE * (mag - PWM_KEEP), pwm)

    def step(self, left, right, dt):
        h = dt / SUBSTEPS
        a = 1.0 - math.exp(-h / MOTOR_TAU)
        tl, tr = self.wheel_speed(left, 0), self.wheel_speed(right, 1) * self.right_gain
        moved = 0.0
        for _ in range(SUBSTEPS):
            self.vl += (tl - self.vl) * a
            self.vr += (tr - self.vr) * a
            v = (self.vl + self.vr) / 2.0 * DRIVE_EFF
            self.x += v * math.cos(self.psi) * h
            self.y += v * math.sin(self.psi) * h
            self.psi += (self.vl - self.vr) / WHEEL_BASE * h   # 왼쪽이 빠르면 오른쪽으로 돈다
            moved += abs(v) * h
        return moved

    @property
    def speed(self):
        return (self.vl + self.vr) / 2.0 * DRIVE_EFF

    def wheels_xy(self):
        """바퀴 4개가 바닥에 닿는 점 (앞왼, 앞오, 뒤왼, 뒤오)."""
        c, s = math.cos(self.psi), math.sin(self.psi)
        return [(self.x + f * c - r * s, self.y + f * s + r * c)
                for f in (WHEEL_AXLE, -WHEEL_AXLE) for r in (-WHEEL_TRACK / 2, WHEEL_TRACK / 2)]


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
        self.wheels_in = 0          # 바퀴 4개가 모두 도로 위였던 프레임 수
        self.steps = 0
        self.stops = 0
        self.states = Counter()
        self.trail = []
        self.res = None
        self.cmd = None
        self.wheels = (0.0, 0.0)
        self.map_scale = 480 / self.track.shape[0]
        self.map_small = cv2.resize(self.track, (480, 480), interpolation=cv2.INTER_AREA)
        self.signs = None           # 가상 표지판 (--signs). 없으면 표지판 없이 달린다
        self.sign_log = []          # (시각, SignEvent)
        if getattr(args, "signs", None):
            from drive.sim_signs import SCENARIOS, CameraModel, SimSigns
            cam = CameraModel(CAM_W, CAM_H, CAM_FOCAL, CAM_PITCH_DEG, CAM_HEIGHT, CAM_FORWARD)
            self.signs = SimSigns(cfg, cam, SCENARIOS[args.signs], latency_s=args.sign_latency,
                                  miss=args.sign_miss, seed=args.seed, hz=REAL_LOOP_HZ)

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

    def wheels_on_road(self):
        """바퀴 4개가 모두 도로(노란 선 포함) 위면 True. 하나라도 녹색 위면 선을 넘은 것."""
        h, w = self.road.shape
        for x, y in self.veh.wheels_xy():
            px, py = _px(x), _px(y)
            if not (0 <= px < w and 0 <= py < h and self.road[py, px] > 0):
                return False
        return True

    def step(self, log=None, recorder=None):
        cfg, veh = self.cfg, self.veh
        dt = 1.0 / REAL_LOOP_HZ                    # 코드는 프레임 단위로 돌고, 차는 실측 주기만큼 움직인다
        self.frames.append(self.cam.render(veh.x, veh.y, veh.psi))
        frame = self.frames[0]                     # LATENCY_FRAMES만큼 늦은 프레임
        self.res = self.det.process(frame)
        if self.signs is not None and not self.ctrl.sign_paused():   # main.py 와 같은 순서: 표지판 이벤트 먼저
            for ev in self.signs.step(self.steps, veh.x, veh.y, veh.psi, self.t):
                if ev.kind != "TURN_SEEN":
                    self.sign_log.append((self.t, ev))
                self.ctrl.on_sign(ev)
        prev = self.cmd.action if self.cmd else None
        self.cmd = self.ctrl.update(self.res.state, self.res.error, band_edges(self.res), near_error(self.res, self.cfg), real_openings(self.res, self.cfg), self.res.ahead_green,
                               [b.learned for b in self.res.bands])
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
        self.wheels_in += self.wheels_on_road()
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
        wpct = 100.0 * self.wheels_in / max(1, self.steps)
        out = (f"sim {self.t:5.1f}s | 주행 거리 {self.odo / 100:5.2f} m | 도로 위 {pct:5.1f}% | 바퀴 안 넘음 {wpct:5.1f}% | "
               f"멈춘 횟수 {self.stops} | 상태 {st}")
        if self.signs is not None:
            evs = " ".join(f"{t:.1f}s:{ev.cls}" for t, ev in self.sign_log)
            out += (f"\n표지판 이벤트: {evs or '없음'}\n도착(정지선): {'성공' if self.ctrl.finished else '실패'}"
                    f" | 검출 지연 {self.signs.latency_frames / REAL_LOOP_HZ:.2f}s")
        return out

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
        if self.signs is not None:   # 가상 표지판
            for s in self.signs.signs:
                p = self._map_pt(s.x, s.y)
                cls = s.cls_at(self.t)
                col = (0, 0, 255) if cls in ("red_light", "stop", "red_line") else (255, 120, 0)
                cv2.circle(img, p, 4, col, -1, cv2.LINE_AA)
                _text(img, cls, (p[0] + 5, p[1] - 4), scale=0.35)
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
        L, W = BODY_LEN / 2, BODY_W / 2
        body = [(-L, -W), (L, -W), (L, W), (-L, W)]     # 실측 차체 크기
        poly = np.array([self._map_pt(veh.x + f * c - r * s, veh.y + f * s + r * c) for f, r in body], np.int32)
        ok = self.wheels_on_road()
        cv2.fillPoly(img, [poly], (0, 140, 255) if ok else (0, 0, 230))
        for wx, wy in veh.wheels_xy():
            cv2.circle(img, self._map_pt(wx, wy), 2, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.line(img, self._map_pt(veh.x, veh.y), self._map_pt(veh.x + 14 * c, veh.y + 14 * s),
                 (0, 0, 0), 2, cv2.LINE_AA)
        pct = 100.0 * self.on_road / max(1, self.steps)
        _text(img, f"t {self.t:5.1f}s  v {veh.speed:5.1f} cm/s  {self.odo / 100:5.2f} m", (6, 18))
        wpct = 100.0 * self.wheels_in / max(1, self.steps)
        _text(img, f"on road {pct:5.1f}%  wheels in {wpct:5.1f}%  stops {self.stops}", (6, 36))
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
    p.add_argument("--signs", choices=["mission", "left", "red", "side_left", "side_right", "purple_right", "purple_left"],
                   help="가상 표지판 시나리오 (drive/sim_signs.py). mission, red 는 --start cross 로")
    p.add_argument("--sign-latency", type=float, default=0.0,
                   help="표지판 검출 지연 초 (Pi 의 YOLO 추론 시간). 검출 주기도 이만큼 길어진다")
    p.add_argument("--sign-miss", type=float, default=0.0, help="프레임마다 표지판을 놓칠 확률")
    p.add_argument("--seed", type=int, default=0, help="--sign-miss 난수 시드")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="drive/config.py 값을 이번 실행에서만 바꾼다 (예: --set SIGN_MIN_BOX_H=45). 여러 번 쓸 수 있다")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = config
    for kv in args.set:
        key, value = kv.split("=", 1)
        old = getattr(cfg, key)   # 없는 이름이면 AttributeError 로 바로 알린다
        setattr(cfg, key, type(old)(value))
        print(f"config {key}: {old} -> {getattr(cfg, key)}")
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
        period = 1.0 / REAL_LOOP_HZ / max(args.speed, 0.05)
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
