from types import SimpleNamespace

from drive import config
from drive.vision import INTERSECTION, LANE, Band, LaneDetector


def _band(name, y, edges, learned):
    b = Band(name, y, 12, 1.0, learned=learned)
    if "L" in edges:
        b.left = 10
    if "R" in edges:
        b.right = 300
    if edges == "LR":
        b.center = 155.0
    return b


def _cand(near_learned, mid_learned, far_learned, edges, state=None):
    det = LaneDetector(config)
    det.state = state
    bands = [_band("near", 195, edges[0], near_learned), _band("mid", 168, edges[1], mid_learned),
             _band("far", 141, edges[2], far_learned)]
    return det._candidate(bands, bands[0], False, 0.0)


def test_near_band_cut_is_still_intersection():
    # 가까운 띠 경계만 잘린 --/LR/LR 도 교차로로 둔다 (실차에서 옆길 회전 신호가 이것뿐인 경우가 있다).
    # 늦게 도는 것은 controller 가 진입을 늘려 처리한다
    assert _cand(294, 245, 188, ["--", "LR", "LR"]) == INTERSECTION


def test_near_and_mid_open_is_intersection():
    assert _cand(233, 189, 150, ["--", "--", "LR"]) == INTERSECTION


def test_wide_road_side_road_in_mid_band_is_intersection():
    assert _cand(294, 245, 188, ["--", "L-", "LR"]) == INTERSECTION


def test_inside_intersection_near_open_stays_intersection():
    # 교차로 안: 가까운 띠는 교차로 위, 중간·먼 띠는 건너편 도로 -> 교차로 유지 (풀면 회전 진입이 끊긴다)
    assert _cand(233, 189, 150, ["--", "LR", "LR"], state=INTERSECTION) == INTERSECTION
