"""녹색 경계 사이 검은 도로의 중앙 찾기와 상태 판정. GPIO 의존 없음.

Donkeycar LineFollower (donkeycar/parts/line_follower.py, https://github.com/autorope/donkeycar)를
바탕으로 했다. 가로 띠를 잘라 색 마스크를 만들고 그 위치로 조향한다는 구조만 가져왔고,
노란 선의 최대값 대신 양쪽 녹색 경계 사이의 중앙을 찾는다.
원본은 RGB->HSV를 쓰지만 여기서는 카메라 프레임이 BGR이므로 BGR->LAB을 쓴다.
Copyright (c) 2017 Will Roscoe, MIT License. 전문은 drive/LICENSE-donkeycar 참고.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

LANE = "LANE"
ONE_SIDE = "ONE_SIDE"
INTERSECTION = "INTERSECTION"
OFF_ROAD = "OFF_ROAD"
LOST = "LOST"


def preprocess(frame, size, flip_code=-1):
    """flip -> PROC_SIZE로 리사이즈. 결과는 BGR."""
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if flip_code is not None:
        frame = cv2.flip(frame, flip_code)
    w, h = size
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return frame


def _kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def green_mask(bgr, a_thr, open_k=0, close_k=0):
    """LAB a* < a_thr 인 픽셀을 255로. 열기로 노이즈를, 닫기로 녹색 위 구멍을 정리한다."""
    a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 1]
    _, mask = cv2.threshold(a, a_thr - 1, 255, cv2.THRESH_BINARY_INV)
    if open_k > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(open_k))
    if close_k > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(close_k))
    return mask


def find_edges(col, start, min_run):
    """1차원 녹색 여부 배열에서 start부터 좌/우로 탐색해
    녹색이 min_run 이상 이어지는 첫 구간의 안쪽 끝 위치를 돌려준다. 없으면 None."""
    n = len(col)
    min_run = max(1, int(min_run))
    if min_run > n:
        return None, None
    g = np.asarray(col, dtype=np.int32)
    # full[i]: col[i : i+min_run]이 모두 녹색
    full = np.flatnonzero(np.convolve(g, np.ones(min_run, np.int32), mode="valid") >= min_run)
    left_win = full[full + min_run - 1 <= start]
    right_win = full[full >= start]
    left = int(left_win[-1] + min_run - 1) if left_win.size else None
    right = int(right_win[0]) if right_win.size else None
    return left, right


@dataclass
class Band:
    name: str
    y: int
    h: int
    weight: float
    start: int = 0
    left: Optional[int] = None
    right: Optional[int] = None
    width: Optional[int] = None       # 양쪽 경계가 있을 때 측정한 폭
    learned: Optional[float] = None   # 이 띠의 학습된 도로 폭
    center: Optional[float] = None    # 유효할 때만 값이 있다
    wide: bool = False                # 폭이 평소의 WIDTH_RATIO배 이상
    note: str = ""                    # 무효 사유 (overlay용)

    @property
    def edges(self):
        return ("L" if self.left is not None else "-") + ("R" if self.right is not None else "-")


@dataclass
class VisionResult:
    image: np.ndarray                 # 전처리된 BGR (PROC_SIZE)
    mask: np.ndarray                  # 녹색 마스크
    bands: List[Band] = field(default_factory=list)
    center: Optional[float] = None    # 최종 도로 중앙 x
    error: Optional[float] = None     # (center - 화면중앙) / (화면폭/2), -1~1
    candidate: Optional[str] = None   # 이 프레임만 보고 낸 상태 (None: 유효한 띠 없음)
    state: Optional[str] = None       # 떨림 방지를 거친 상태
    offroad_ratio: float = 0.0
    dark: bool = False

    @property
    def edges_found(self):
        return "/".join(b.edges for b in self.bands)


class LaneDetector:
    """매 프레임 process(frame)을 부른다. 띠별 직전 중앙과 도로 폭, 상태를 기억한다.
    cfg 값은 매 프레임 다시 읽으므로 tune.py에서 실행 중에 바꿀 수 있다."""

    def __init__(self, cfg, flip=True):
        self.cfg = cfg
        self.flip = flip
        self.reset()

    def reset(self):
        n = len(self.cfg.BANDS)
        self._starts = [self.cfg.PROC_SIZE[0] / 2.0] * n
        self.widths = (list(self.cfg.ROAD_WIDTH_INIT) + [None] * n)[:n]
        self.state = None
        self._pending = None
        self._pending_count = 0
        self._no_valid = 0

    # ------------------------------------------------------------ 스캔
    def _scan(self, i, b, col, w):
        cfg = self.cfg
        start = int(round(min(max(self._starts[i], 0), w - 1)))
        if col[start]:
            start = w // 2  # 직전 중앙이 녹색 위면 화면 중앙에서 다시 탐색
        b.start = start
        b.learned = self.widths[i]
        if col[start]:
            b.note = "start on green"
            return
        b.left, b.right = find_edges(col, start, cfg.MIN_RUN)
        learned = self.widths[i]

        if b.left is not None and b.right is not None:
            b.width = b.right - b.left
            if learned is None:
                if b.width >= cfg.MIN_ROAD_WIDTH_PX:
                    self.widths[i] = float(b.width)
                    b.center = (b.left + b.right) / 2.0
                else:
                    b.note = "narrow"
            elif b.width >= cfg.WIDTH_RATIO * learned:
                b.wide = True
                b.note = "wide"
            elif b.width < cfg.WIDTH_MIN_RATIO * learned:
                b.note = "narrow"
            else:
                a = cfg.WIDTH_EMA_ALPHA
                self.widths[i] = (1 - a) * learned + a * b.width
                b.center = (b.left + b.right) / 2.0
        elif learned is not None and b.left is not None:
            b.center = b.left + learned / 2.0
        elif learned is not None and b.right is not None:
            b.center = b.right - learned / 2.0
        b.learned = self.widths[i]

    # ------------------------------------------------------------ 상태
    def _confirm(self, cand):
        if self.state is None:
            self.state = cand
        elif cand == self.state:
            self._pending, self._pending_count = None, 0
        else:
            if cand == self._pending:
                self._pending_count += 1
            else:
                self._pending, self._pending_count = cand, 1
            if self._pending_count >= self.cfg.STATE_CONFIRM_FRAMES:
                self.state = cand
                self._pending, self._pending_count = None, 0

    def _candidate(self, bands, near, dark, offroad_ratio):
        if dark:
            return None
        if offroad_ratio >= self.cfg.OFFROAD_GREEN_RATIO:
            return OFF_ROAD
        valid = [b for b in bands if b.center is not None]
        if not valid:
            # 교차로 안: 모든 띠를 정상적으로 훑었는데 양쪽 다 녹색 경계가 없다 (도로가 사방으로 열림).
            # 교차로에 들어갈 때는 먼 띠부터 열리므로, 가까운 띠까지 열리는 순간 여기로 온다.
            if all(b.edges == "--" and not b.note for b in bands):
                return INTERSECTION
            return None
        if near.wide or (near.edges == "--" and not near.note):
            # 가까운 띠는 양쪽이 열렸는데 먼 띠는 도로가 보인다 -> 교차로
            return INTERSECTION
        primary = max(valid, key=lambda b: b.y)
        return LANE if primary.edges == "LR" else ONE_SIDE

    # ------------------------------------------------------------ 메인
    def process(self, frame):
        cfg = self.cfg
        if len(cfg.BANDS) != len(self._starts):
            self.reset()
        img = preprocess(frame, cfg.PROC_SIZE, cfg.FLIP_CODE if self.flip else None)
        h, w = img.shape[:2]
        mask = green_mask(img, cfg.A_THR, cfg.OPEN_KERNEL, cfg.CLOSE_KERNEL)
        limit = h - cfg.BOTTOM_EXCLUDE
        mask[limit:] = 0
        dark = float(img.mean()) < cfg.DARK_MEAN_THR

        bands = []
        for i, spec in enumerate(cfg.BANDS):
            y = int(min(max(spec["y"], 0), limit - 1))
            bh = int(max(1, min(spec["h"], limit - y)))
            b = Band(spec.get("name", str(i)), y, bh, float(spec.get("weight", 1.0)))
            if not dark:
                col = (mask[y:y + bh] > 0).mean(axis=0) >= cfg.COLUMN_GREEN_RATIO
                self._scan(i, b, col, w)
            else:
                b.note = "dark"
            bands.append(b)

        res = VisionResult(image=img, mask=mask, bands=bands, dark=dark)
        valid = [b for b in bands if b.center is not None]
        if valid:
            total = sum(b.weight for b in valid)
            res.center = sum(b.center * b.weight for b in valid) / total
            # LANE_CENTER_OFFSET: 차를 도로 한가운데에 똑바로 놓았을 때 화면에 나오는 오차. 카메라가 차 중심에서
            # 비껴 달렸거나 살짝 돌아가 있으면 0 이 아니고, 그만큼 차가 한쪽 선에 붙어 간다
            res.error = float(np.clip((res.center - w / 2.0) / (w / 2.0) - getattr(self.cfg, "LANE_CENTER_OFFSET", 0.0),
                                      -1.0, 1.0))
        for i, b in enumerate(bands):
            if b.center is not None:
                self._starts[i] = b.center
            else:
                self._starts[i] = res.center if res.center is not None else w / 2.0

        # 화면 중앙 아래쪽(가장 가까운 띠) 탐색 시작점이 녹색 위인지
        near = max(bands, key=lambda b: b.y)
        x0 = max(0, int(w / 2 - cfg.OFFROAD_HALF_W))
        x1 = min(w, int(w / 2 + cfg.OFFROAD_HALF_W) + 1)
        probe = mask[near.y:near.y + near.h, x0:x1]
        res.offroad_ratio = float((probe > 0).mean()) if probe.size else 0.0

        res.candidate = self._candidate(bands, near, dark, res.offroad_ratio)
        self._no_valid = self._no_valid + 1 if res.candidate is None else 0
        if res.candidate is None:
            if self._no_valid >= cfg.LOST_FRAMES:
                self.state = LOST  # LOST_FRAMES 자체가 확인 기간이므로 바로 바꾼다
                self._pending, self._pending_count = None, 0
        else:
            self._confirm(res.candidate)
        res.state = self.state
        return res
