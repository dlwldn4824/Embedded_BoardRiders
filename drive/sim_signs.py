"""시뮬레이터용 가상 표지판과 검출 지연.

YOLO 를 돌리지 않고, 트랙 위에 놓은 가상 표지판이 카메라 화면에서 차지할 박스를 계산해서 검출 결과처럼 만든다.
그 결과를 실제 주행 코드의 SignInterpreter(신뢰도·박스 크기·연속 검출·쿨다운 필터)에 그대로 넣는다.

검출 지연은 drive/signs.py 의 SignWorker 와 같은 방식으로 흉내 낸다.
- 검출기는 한 번에 프레임 하나만 처리하고, 처리하는 동안 들어온 프레임은 버린다 (최신 것만 남김).
- 결과는 latency 초 뒤에 도착한다. 그래서 latency 가 길면 검출 주기도 그만큼 길어진다.

가정 (실제 트랙에서 다시 잰다):
- 기둥 표지판 판은 한 변 SIGN_SIZE_CM, 바닥에서 SIGN_ELEVATION_CM 위에 있다.
- 검출기는 박스 높이 MIN_DETECT_PX 이상이면 확률 (1 - miss) 로 잡고, 신뢰도는 DETECT_CONF 로 낸다.
- 가림(다른 물체 뒤에 숨음)은 계산하지 않는다.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from drive.signs import SignInterpreter

SIGN_SIZE_CM = 8.0       # 기둥 표지판·신호등 판 크기
SIGN_ELEVATION_CM = 8.0  # 판 아래쪽 높이
FLOOR_LINE_DEPTH_CM = 3.0
MIN_DETECT_PX = 12       # 이보다 작은 박스는 YOLO 가 못 잡는다고 본다
MAX_VIEW_DEG = 60        # 표지판 정면에서 이 각도보다 비껴 보면 못 읽는다고 본다
DETECT_CONF = 0.85


@dataclass
class VirtualSign:
    """트랙 좌표(cm)에 놓인 표지판. schedule 이 있으면 시간에 따라 클래스가 바뀐다 (신호등)."""

    cls: str
    x: float
    y: float
    floor: bool = False                 # 바닥 표시(정지선, 횡단보도). 도로를 가로지르는 선
    width: float = SIGN_SIZE_CM         # floor 면 선 길이 (도로 폭)
    heading_deg: float = 0.0            # floor 선이 뻗은 방향 (0 = 동서, 90 = 남북)
    depth: float = FLOOR_LINE_DEPTH_CM  # floor 면 선 두께 (진행 방향). 횡단보도는 줄무늬 길이
    height: float = SIGN_SIZE_CM        # 기둥 표지판 판 높이. 신호등은 몸통 전체 (YOLO 라벨 상자가 몸통 전체다)
    schedule: list = field(default_factory=list)   # [(이 시각(s) 전까지, cls), ...]
    for_heading: float | None = None    # 이 표지판을 읽을 차의 진행 방향 (도, sim 기준). 판은 그 반대쪽을 본다

    def __post_init__(self):
        if self.for_heading is None and not self.floor:
            raise ValueError(f"{self.cls}: 기둥 표지판은 for_heading 이 필요하다")

    def cls_at(self, t):
        for until, cls in self.schedule:
            if t < until:
                return cls
        return self.cls


class CameraModel:
    """sim.py 카메라와 같은 바늘구멍 모델 (높이, 아래로 기울기, 초점거리)."""

    def __init__(self, width, height, focal, pitch_deg, cam_height, cam_forward):
        self.w, self.h = width, height
        self.f = focal
        self.p = math.radians(pitch_deg)
        self.ch = cam_height
        self.fwd = cam_forward

    def project(self, veh_x, veh_y, psi, x, y, z):
        """세계 점 (x, y, 높이 z) → 화면 (u, v). 카메라 뒤면 None."""
        c, s = math.cos(psi), math.sin(psi)
        dx, dy = x - (veh_x + self.fwd * c), y - (veh_y + self.fwd * s)
        d = dx * c + dy * s            # 앞 거리
        r = -dx * s + dy * c           # 오른쪽 거리
        zc = d * math.cos(self.p) + (self.ch - z) * math.sin(self.p)
        yc = -d * math.sin(self.p) + (self.ch - z) * math.cos(self.p)
        if zc < 3.0:
            return None
        return self.w / 2 + self.f * r / zc, self.h / 2 + self.f * yc / zc

    def box(self, veh_x, veh_y, psi, sign: VirtualSign):
        """표지판의 화면 박스 (x1, y1, x2, y2). 화면 밖이거나 카메라 뒤면 None."""
        if sign.floor:
            a = math.radians(sign.heading_deg)
            ux, uy = math.cos(a) * sign.width / 2, math.sin(a) * sign.width / 2       # 선 방향
            vx, vy = -math.sin(a) * sign.depth / 2, math.cos(a) * sign.depth / 2
            pts = [(sign.x + i * ux + j * vx, sign.y + i * uy + j * vy, 0.0) for i in (-1, 1) for j in (-1, 1)]
        else:
            # 판의 앞면은 for_heading 반대쪽(다가오는 차)을 본다. 정면에서 MAX_VIEW_DEG 넘게 비껴 보면 못 읽는다.
            a = math.radians(sign.for_heading)
            nx, ny = -math.cos(a), -math.sin(a)                    # 판 앞면 방향
            cam_x = veh_x + self.fwd * math.cos(psi)
            cam_y = veh_y + self.fwd * math.sin(psi)
            tx, ty = cam_x - sign.x, cam_y - sign.y
            dist = math.hypot(tx, ty)
            if dist < 1e-6 or (tx * nx + ty * ny) / dist < math.cos(math.radians(MAX_VIEW_DEG)):
                return None
            hx, hy = -ny * sign.width / 2, nx * sign.width / 2      # 판 가로 방향
            pts = [(sign.x + i * hx, sign.y + i * hy, z) for i in (-1, 1)
                   for z in (SIGN_ELEVATION_CM, SIGN_ELEVATION_CM + sign.height)]
        uv = [self.project(veh_x, veh_y, psi, *p) for p in pts]
        if any(q is None for q in uv):
            return None
        us, vs = [q[0] for q in uv], [q[1] for q in uv]
        x1, x2 = max(0.0, min(us)), min(self.w - 1.0, max(us))
        y1, y2 = max(0.0, min(vs)), min(self.h - 1.0, max(vs))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        cx = (min(us) + max(us)) / 2
        if not 0 <= cx < self.w:       # 가운데가 화면 밖이면 못 본 것으로
            return None
        return x1, y1, x2, y2


class SimSigns:
    """가상 표지판 검출 + SignWorker 식 지연 + 실제 SignInterpreter."""

    def __init__(self, cfg, camera: CameraModel, signs, latency_s=0.0, miss=0.0, seed=0, hz=None):
        self.cfg = cfg
        self.cam = camera
        self.signs = signs
        self.latency_frames = max(0, int(round(latency_s * (hz or cfg.LOOP_HZ))))
        self.miss = miss
        self.rng = random.Random(seed)
        self.interp = SignInterpreter(cfg)
        self._pending = None            # (결과 도착 프레임, 검출 결과)
        self.n_detect_calls = 0

    def detect(self, veh_x, veh_y, psi, t, frame_id):
        out = []
        for s in self.signs:
            b = self.cam.box(veh_x, veh_y, psi, s)
            if b is None:
                continue
            x1, y1, x2, y2 = b
            if y2 - y1 < MIN_DETECT_PX and not s.floor:
                continue
            if s.floor and x2 - x1 < MIN_DETECT_PX * 3:
                continue
            if self.rng.random() < self.miss:
                continue
            out.append({"cls": s.cls_at(t), "conf": DETECT_CONF, "box_w": x2 - x1, "box_h": y2 - y1,
                        "box_cx": (x1 + x2) / 2, "box_y2": y2, "frame_id": frame_id})
        return out

    def step(self, frame_id, veh_x, veh_y, psi, t):
        """매 프레임 부른다. 이번 프레임에 도착한 이벤트를 돌려준다."""
        events = []
        if self._pending and frame_id >= self._pending[0]:
            events = self.interp(self._pending[1])
            self._pending = None
        if self._pending is None:       # 검출기가 비었으면 지금 프레임을 넣는다
            self.n_detect_calls += 1
            dets = self.detect(veh_x, veh_y, psi, t, frame_id)
            if self.latency_frames == 0:
                events += self.interp(dets)
            else:
                self._pending = (frame_id + self.latency_frames, dets)
        return events


# ---------------------------------------------------------------- 시나리오 (트랙 좌표 cm, sim.py 트랙 기준)
# 실습 트랙 도면(200x200cm): 도로 중심선 바깥 x/y = 20, 180 / 가운데 x/y = 100. 가운데 교차로는 90~110.
# 기둥 표지판은 차가 오는 방향의 오른쪽 길가(도로 중심에서 15cm)에 둔다.
# 가운데 교차로로 북쪽으로 들어가는 길의 횡단보도 (sim.py 트랙 그림과 같은 자리: 줄 4개 폭 15.4cm, 길이 8cm, y 115~123)
CROSSWALK_SOUTH = VirtualSign("crosswalk", 100, 119, floor=True, width=15.4, depth=8.0, heading_deg=0)


def light_north(red_until_s):
    """가운데 교차로를 북쪽으로 들어가는 차가 보는 신호등. 실습 트랙 사진처럼 교차로 건너편 왼쪽 모서리(왼쪽 위 블록)에
    둔다. 몸통 높이 15cm 로 잡으면 출발점(약 65cm)에서 이미 60px 를 넘어, 실차처럼 멀리서 빨간불이 확정된다."""
    return VirtualSign("green_light", 86, 86, schedule=[(red_until_s, "red_light")], for_heading=-90, height=15.0)


SCENARIOS = {
    # --start cross. 빨간불(6초 뒤 초록) + 좌회전 → T자에서 우회전 → 서행 → 정지 표지판 → 정지선 도착
    "mission": [
        # 진행 방향(for_heading): 0 = 동, 90 = 남, 180 = 서, -90 = 북
        light_north(6.0),           # 6초까지 빨간불. 횡단보도(CROSSWALK_SOUTH) 앞에서 서서 기다린다
        CROSSWALK_SOUTH,
        # 출발점이 교차로 앞이라 방향 표지판은 교차로 오른쪽 앞 모서리에 둔다
        VirtualSign("left", 116, 106, for_heading=-90),
        VirtualSign("right", 42, 86, for_heading=180),
        VirtualSign("speed_20", 35, 66, for_heading=-90),
        # 커브 바로 뒤에 두면 차가 비스듬할 땐 화면 밖, 정면일 땐 너무 가까워서 못 본다. 위쪽 직선 뒤쪽에 둔다
        VirtualSign("stop", 120, 35, for_heading=0),
        VirtualSign("red_line", 148, 20, floor=True, width=20, heading_deg=90),
    ],
    # --start cross. 좌회전 표지판 하나만 (회전 타이밍만 보기)
    "left": [VirtualSign("left", 116, 106, for_heading=-90)],
    # --start cross. 빨간불(12초 뒤 초록) + 횡단보도: 출발하자마자 빨간불을 봐도 횡단보도 앞까지 가서 서는지 본다
    "red": [light_north(12.0), CROSSWALK_SOUTH],
    # --start outer. 바깥 아래 도로에서 옆길(가운데 세로 도로)로 좌회전
    "side_left": [VirtualSign("left", 84, 195, for_heading=0)],   # 80 은 출발 직후 화면 끝에 잘린 채로만 보였다
    # --start left_outer. 왼쪽 바깥 도로를 북쪽으로 가다 옆길(가운데 가로 도로)로 우회전.
    # 표지판을 옆길 바로 앞에 둬서, 느린 실차처럼 표지판을 본 뒤 옆길까지 오래 걸리는 경우를 본다
    "side_right": [VirtualSign("right", 34, 118, for_heading=-90)],
}
