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


def _cand(near_learned, mid_learned, far_learned, edges):
    det = LaneDetector(config)
    bands = [_band("near", 195, edges[0], near_learned), _band("mid", 168, edges[1], mid_learned),
             _band("far", 141, edges[2], far_learned)]
    return det._candidate(bands, bands[0], False, 0.0)


def test_wide_road_near_band_cut_is_not_intersection():
    # 28cm 도로: 가까운 띠 경계만 화면 밖으로 나가고 중간·먼 띠는 양쪽 경계가 보인다 -> 옆길은 아직 안 왔다
    assert _cand(294, 245, 188, ["--", "LR", "LR"]) == LANE


def test_narrow_road_near_band_cut_is_not_intersection():
    # 옆길 폭이 띠 간격보다 넓어, 21cm 도로에서도 --/LR/LR 은 가까운 띠가 잘린 것이다
    assert _cand(233, 189, 150, ["--", "LR", "LR"]) == LANE


def test_near_and_mid_open_is_intersection():
    assert _cand(233, 189, 150, ["--", "--", "LR"]) == INTERSECTION


def test_wide_road_side_road_in_mid_band_is_intersection():
    assert _cand(294, 245, 188, ["--", "L-", "LR"]) == INTERSECTION
