"""표지판 이벤트에 따른 Controller 동작과 SignInterpreter 필터 확인.

    python -m pytest tests
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drive import config  # noqa: E402
from drive.controller import Controller  # noqa: E402
from drive.signs import SignEvent, SignInterpreter  # noqa: E402
from drive.vision import INTERSECTION, LANE, LOST  # noqa: E402


def make():
    c = Controller(config)
    c.set_running(True)
    return c


def ev(kind, value=None):
    return SignEvent(kind, value, "x", 0.9, 0)


def run(c, n, state=LANE, error=0.0):
    return [c.update(state, error).action for _ in range(n)]


def test_lane_follow_is_pid():
    assert run(make(), 3) == ["PID"] * 3


def test_stop_sign_waits_then_resumes():
    c = make()
    c.on_sign(ev("STOP"))
    acts = run(c, config.STOP_WAIT_FRAMES + 2)
    assert acts[: config.STOP_WAIT_FRAMES - 1] == ["STOP"] * (config.STOP_WAIT_FRAMES - 1)
    assert acts[-1] == "PID"


def test_stop_sign_seen_again_does_not_stop_forever():
    """표지판이 차 앞에 계속 보여 쿨다운마다 STOP 이 다시 와도, 한 번 서고 나면 출발한다."""
    c = make()
    acts = []
    for i in range(config.STOP_WAIT_FRAMES + config.STOP_IGNORE_FRAMES // 2):
        if i % config.SIGN_COOLDOWN_FRAMES == 0:
            c.on_sign(ev("STOP"))
        acts.append(c.update(LANE, 0.0).action)
    assert acts[:config.STOP_WAIT_FRAMES - 1] == ["STOP"] * (config.STOP_WAIT_FRAMES - 1)
    assert acts[-1] == "PID"


def test_stop_sign_works_again_after_ignore_window():
    c = make()
    c.on_sign(ev("STOP"))
    run(c, config.STOP_WAIT_FRAMES + config.STOP_IGNORE_FRAMES + 1)
    c.on_sign(ev("STOP"))   # 다음 정지 표지판
    assert run(c, 1) == ["STOP"]


def test_red_light_far_away_does_not_stop():
    # 실차: 신호등이 커서 멀리서도 빨간불이 확정돼 횡단보도 한참 앞에서 섰다. 빨간불만으로는 서지 않고 기억만 한다
    c = make()
    c.on_sign(ev("LIGHT", "RED"))
    assert set(run(c, 50)) == {"PID"}
    assert c._red_on


def test_red_light_stops_at_crosswalk_until_green():
    c = make()
    c.on_sign(ev("LIGHT", "RED"))
    run(c, 30)
    c.on_sign(ev("CROSSWALK"))          # 횡단보도가 차 바로 앞
    assert set(run(c, 50)) == {"RED"}
    c.on_sign(ev("LIGHT", "GREEN"))
    assert run(c, 1) == ["PID"]


def test_red_light_off_resumes_without_green():
    # 초록불을 못 잡아도 빨간불이 꺼지면(OFF) 출발한다
    c = make()
    c.on_sign(ev("LIGHT", "RED"))
    c.on_sign(ev("CROSSWALK"))
    assert run(c, 5) == ["RED"] * 5
    c.on_sign(ev("LIGHT", "OFF"))
    assert run(c, 1) == ["PID"]


def test_red_light_waits_while_still_red_then_times_out():
    # 빨간불이 계속 보이면(쿨다운마다 RED) 최대 대기를 넘겨도 기다린다. 더 못 보면 최대 대기 뒤 출발하고,
    # 그 빨간불은 놓는다 (교차로 입구에서 또 서지 않는다)
    c = make()
    c.on_sign(ev("LIGHT", "RED"))
    c.on_sign(ev("CROSSWALK"))
    acts = []
    for i in range(config.LIGHT_MAX_WAIT_FRAMES * 2):
        if i % config.SIGN_COOLDOWN_FRAMES == 0:
            c.on_sign(ev("LIGHT", "RED"))
        acts.append(c.update(LANE, 0.0).action)
    assert set(acts) == {"RED"}
    acts = run(c, config.LIGHT_MAX_WAIT_FRAMES + 2)
    assert acts[-1] == "PID" and not c._red_on
    assert c.update(INTERSECTION, None).action == "CROSS"


def test_crosswalk_without_red_does_not_stop():
    c = make()
    c.on_sign(ev("CROSSWALK"))
    assert run(c, 5) == ["PID"] * 5
    c2 = make()
    c2.on_sign(ev("LIGHT", "RED"))
    c2.on_sign(ev("LIGHT", "GREEN"))    # 다가가는 동안 초록불로 바뀜
    c2.on_sign(ev("CROSSWALK"))
    assert run(c2, 5) == ["PID"] * 5


def test_red_confirmed_just_after_crosswalk_stops():
    # 횡단보도 앞에 온 뒤에야 빨간불이 확정돼도 선다. 횡단보도를 본 지 오래면 서지 않는다 (이미 지나감)
    c = make()
    c.on_sign(ev("CROSSWALK"))
    run(c, config.RED_CROSSWALK_HOLD_FRAMES - 5)
    c.on_sign(ev("LIGHT", "RED"))
    assert run(c, 1) == ["RED"]
    c2 = make()
    c2.on_sign(ev("CROSSWALK"))
    run(c2, config.RED_CROSSWALK_HOLD_FRAMES + 5)
    c2.on_sign(ev("LIGHT", "RED"))
    assert run(c2, 1) == ["PID"]


def test_red_without_crosswalk_stops_at_intersection_entry():
    # 횡단보도를 못 잡았으면 교차로 입구(INTERSECTION)에서 선다. 빨간불이 꺼지면 교차로를 지난다
    c = make()
    c.on_sign(ev("LIGHT", "RED"))
    assert run(c, 10) == ["PID"] * 10
    assert run(c, 5, INTERSECTION, None) == ["RED"] * 5
    c.on_sign(ev("LIGHT", "OFF"))
    assert run(c, 1, INTERSECTION, None) == ["CROSS"]


def test_red_stop_during_turn_entry_keeps_detector_running():
    # 옆길 회전을 정하고 직진(ENTRY)하다 교차로 입구에서 빨간불로 서면, 회전 중이라도 YOLO 를 쉬지 않는다 (신호 바뀜을 봐야 한다).
    # 빨간불이 꺼지면 남은 직진부터 이어 간다
    c = make()
    c.on_sign(ev("LIGHT", "RED"))
    c._start_turn("LEFT", 30, "옆길")
    run(c, 5, INTERSECTION, None)
    assert c._wait and c._wait[0] == "RED" and c._turn[1] == "ENTRY"
    assert not c.sign_paused()
    c.on_sign(ev("LIGHT", "OFF"))
    m = c.update(INTERSECTION, None)
    assert m.action == "TURN" and m.steering == 0.0 and c.sign_paused()


def test_open_bands_before_intersection_state_keep_going_straight():
    # 교차로 입구: 상태 확정(3프레임) 전에 띠가 모두 열린 프레임(오차 없음)이 섞여 와도 서지 않고 곧게 간다.
    # (sim: 섰다 출발한 차가 이런 프레임마다 서서 모터 세기가 0부터 다시 올라가, 정지 바퀴를 끝내 못 돌렸다)
    from drive.vision import ONE_SIDE
    c = make()
    acts = []
    for i in range(12):
        if i % 3 == 2:
            acts.append(c.update(ONE_SIDE, None, ["--", "--", "--"]))
        else:
            acts.append(c.update(ONE_SIDE, -0.13, ["L-", "--", "--"]))
    assert all(m.action in ("PID", "CROSS") and m.throttle > 0 for m in acts)
    assert all(m.steering == 0.0 for m in acts if m.action == "CROSS")
    # 앞이 막힌 화면(먼 띠가 초록 위)이나 도로를 잃은 상태는 예전처럼 선다
    assert make().update(ONE_SIDE, None, ["--", "--", "GG"]).action == "STOP"
    assert make().update(LOST, None, ["--", "--", "--"]).action == "STOP"


def test_red_ignored_once_inside_intersection():
    # 교차로를 지나는 중에 빨간불로 바뀌거나 건너편 횡단보도가 보여도 서지 않고 빠져나간다
    c = make()
    run(c, 5, INTERSECTION, None)
    c.on_sign(ev("LIGHT", "RED"))
    c.on_sign(ev("CROSSWALK"))
    assert "RED" not in run(c, 10, INTERSECTION, None)
    assert not c._red_on


def test_green_without_red_does_nothing():
    c = make()
    c.on_sign(ev("LIGHT", "GREEN"))
    assert run(c, 1) == ["PID"]


def test_speed_sign_is_temporary():
    c = make()
    base = c.update(LANE, 0.0).throttle
    c.on_sign(ev("SPEED", 0.6))
    assert c.update(LANE, 0.0).throttle < base
    run(c, config.SPEED_FRAMES)
    assert c.update(LANE, 0.0).throttle == base


def test_speed_sign_seen_again_does_not_extend():
    """같은 서행 표지판을 여러 프레임 보더라도 감속은 처음 1초뿐이다 (연장도, 다시 켜지지도 않는다)."""
    c = make()
    base = c.update(LANE, 0.0).throttle
    c.on_sign(ev("SPEED", 0.2))
    run(c, config.SPEED_FRAMES // 2)
    c.on_sign(ev("SPEED", 0.2))            # 감속 중에 다시 봄 -> 늘어나지 않는다
    run(c, config.SPEED_FRAMES - config.SPEED_FRAMES // 2)
    assert c.update(LANE, 0.0).throttle == base
    c.on_sign(ev("SPEED", 0.2))            # 감속이 끝난 직후 다시 봄 -> 다시 켜지지 않는다
    assert c.update(LANE, 0.0).throttle == base
    run(c, config.SPEED_IGNORE_FRAMES)
    c.on_sign(ev("SPEED", 0.2))            # 한참 뒤 다른 서행 표지판 -> 다시 감속
    assert c.update(LANE, 0.0).throttle < base


def test_finish_ignored_right_after_start():
    c = make()
    c.on_sign(ev("FINISH"))
    assert "FINISH" not in run(c, 50)


def test_finish_stops_after_delay_and_restart_clears():
    c = make()
    run(c, config.FINISH_ARM_FRAMES)
    c.on_sign(ev("FINISH"))
    acts = run(c, config.FINISH_DELAY_FRAMES + 3)
    assert acts[: config.FINISH_DELAY_FRAMES] == ["PID"] * config.FINISH_DELAY_FRAMES
    assert acts[-1] == "FINISH"
    c.set_running(True)
    assert run(c, 1) == ["PID"]


def test_turn_sequence_at_intersection():
    c = make()
    c.on_sign(ev("TURN", "LEFT"))
    first = c.update(INTERSECTION, None)
    assert first.action == "TURN" and first.steering == 0.0   # ENTRY: 직진
    run(c, config.TURN_ENTRY_FRAMES - 1, INTERSECTION, None)
    spin = c.update(INTERSECTION, None)
    assert spin.action == "TURN" and spin.steering < 0       # SPIN: 왼쪽
    run(c, config.TURN_FRAMES - 1, INTERSECTION, None)
    # 최소 회전 뒤, 새 길이 정면(중간·먼 띠 양쪽 경계)에 TURN_DONE_CONFIRM 프레임 연속 보이면 PID 로 넘어간다
    acts = [c.update(LANE, 0.0, ["LR", "LR", "LR"]).action for _ in range(config.TURN_DONE_CONFIRM)]
    assert acts[:-1] == ["TURN"] * (config.TURN_DONE_CONFIRM - 1) and acts[-1] == "PID"
    assert c.pending_turn is None


def test_turn_exit_gives_up_without_lane():
    c = make()
    c.on_sign(ev("TURN", "RIGHT"))
    acts = run(c, config.TURN_ENTRY_FRAMES + config.TURN_FRAMES + config.TURN_EXIT_MAX_FRAMES + 2, LOST, None)
    assert acts[-1] == "STOP"


def test_straight_or_no_sign_crosses_then_stops():
    c = make()
    acts = run(c, config.INTERSECTION_MAX_FRAMES + 2, INTERSECTION, None)
    assert acts[0] == "CROSS" and acts[-1] == "STOP"


# ---------------------------------------------------------------- SignInterpreter
def det(cls, fid, conf=0.9, h=80, y2=300):
    return {"cls": cls, "conf": conf, "box_w": 60, "box_h": h, "box_cx": 320, "box_y2": y2, "frame_id": fid}


def test_interpreter_needs_consecutive_hits():
    it = SignInterpreter(config)
    assert it([det("stop", 1)]) == []
    out = it([det("stop", 4)])
    assert [e.kind for e in out] == ["STOP"]


def test_interpreter_filters_far_low_conf_and_unknown():
    it = SignInterpreter(config)
    for f in (1, 2, 3):
        assert it([det("stop", f, h=10), det("left", f, conf=0.2), det("crosswalk", f)]) == []


def test_interpreter_red_line_uses_position():
    it = SignInterpreter(config)
    assert it([det("red_line", 1, h=8, y2=200)]) == [] and it([det("red_line", 2, h=8, y2=200)]) == []
    it([det("red_line", 3, h=8, y2=450)])
    assert [e.kind for e in it([det("red_line", 4, h=8, y2=450)])] == ["FINISH"]


def test_interpreter_crosswalk_only_when_close_and_ahead():
    # 횡단보도는 차 바로 앞(상자 아래 끝 y2 >= min_y2, 화면 가운데를 덮음)일 때만 이벤트가 된다
    it = SignInterpreter(config)
    y2 = config.SIGN_CLASS_RULES["crosswalk"]["min_y2"]
    cw = lambda f, **k: {"cls": "crosswalk", "conf": 0.9, "frame_id": f, **k}   # noqa: E731
    evs = []
    for f in range(0, 8, 2):
        evs += it([cw(f, box_h=30, box_w=200, box_cx=320, box_y2=250),             # 멀다
                   cw(f, box_h=100, box_w=150, box_cx=100, box_y2=y2 + 20)])       # 가깝지만 옆길 (가운데를 안 덮음)
    assert evs == []
    for f in range(8, 12, 2):
        evs += it([cw(f, box_h=115, box_w=400, box_cx=405, box_y2=398)])           # 실차 사진: 횡단보도 바로 앞
    assert [e.kind for e in evs] == ["CROSSWALK"]


def test_interpreter_red_off_after_missing_runs():
    # 빨간불 이벤트를 낸 뒤 LIGHT_OFF_RUNS 번 연속 검출에서 빨간불이 안 보이면 OFF 를 한 번 낸다
    it = SignInterpreter(config)
    red = lambda f, conf=0.9: {"cls": "red_light", "conf": conf, "box_h": 100, "box_w": 40,   # noqa: E731
                               "box_cx": 200, "box_y2": 200, "frame_id": f}
    assert all(it([]) == [] for _ in range(config.LIGHT_OFF_RUNS + 1))     # 빨간불을 본 적 없으면 OFF 도 없다
    evs = it([red(1)]) + it([red(3)])
    assert [(e.kind, e.value) for e in evs] == [("LIGHT", "RED")]
    evs = []
    for _ in range(config.LIGHT_OFF_RUNS - 1):
        evs += it([])
    evs += it([red(9, conf=0.3)])             # 신뢰도가 낮아도 빨간불이 보이면 아직 켜진 것
    for _ in range(config.LIGHT_OFF_RUNS - 1):
        evs += it([])
    assert evs == []
    evs += it([])
    assert [(e.kind, e.value) for e in evs] == [("LIGHT", "OFF")]
    assert all(it([]) == [] for _ in range(5))                              # 한 번만
    evs = it([red(20)]) + it([red(22)])       # 다시 켜지면 쿨다운 없이 바로 알린다
    assert [(e.kind, e.value) for e in evs] == [("LIGHT", "RED")]


def test_interpreter_red_leaving_frame_edge_is_not_off():
    # 정지 위치에서 신호등이 화면 오른쪽 끝으로 빠지면 꺼진 건지 모른다 -> OFF 를 내지 않는다 (초록불이나 최대 대기로 출발)
    it = SignInterpreter(config)
    red = lambda f, cx: {"cls": "red_light", "conf": 0.9, "box_h": 120, "box_w": 60,   # noqa: E731
                         "box_cx": cx, "box_y2": 200, "frame_id": f}
    it([red(1, 400)])
    assert [(e.kind, e.value) for e in it([red(3, 400)])] == [("LIGHT", "RED")]
    it([red(5, 600)])                          # 오른쪽 끝 가까이(x2 = 630) 보이다가
    assert all(it([]) == [] for _ in range(config.LIGHT_OFF_RUNS * 3))      # 사라짐
    it([red(30, 400)])                         # 다시 화면 안에서 보였다가 사라지면 꺼진 것
    evs = []
    for _ in range(config.LIGHT_OFF_RUNS):
        evs += it([])
    assert [(e.kind, e.value) for e in evs] == [("LIGHT", "OFF")]


def test_interpreter_cooldown():
    it = SignInterpreter(config)
    it([det("horn", 1)])
    assert it([det("horn", 2)])
    it([det("horn", 5)])
    assert it([det("horn", 6)]) == []


def test_side_branch_turn_after_sign():
    # T자·옆길: 오른쪽 경계는 그대로이고 왼쪽만 열린다 (비전 상태는 ONE_SIDE)
    from drive.vision import ONE_SIDE
    c = make()
    c.on_sign(ev("TURN", "LEFT"))
    for _ in range(config.SIDE_CLOSED_FRAMES):
        assert c.update(LANE, 0.0, ["LR", "LR", "LR"]).action == "PID"
    acts = [c.update(ONE_SIDE, 0.0, ["-R", "-R", "-R"]).action for _ in range(config.SIDE_OPEN_FRAMES)]
    assert acts[-1] == "TURN" and c.pending_turn is None


def test_side_branch_ignored_without_closed_road_first():
    # 교차로를 막 빠져나올 때처럼 처음부터 한쪽이 열려 있으면 회전하지 않는다
    from drive.vision import ONE_SIDE
    c = make()
    c.on_sign(ev("TURN", "RIGHT"))
    acts = [c.update(ONE_SIDE, 0.0, ["L-", "L-", "LR"]).action for _ in range(10)]
    assert "TURN" not in acts and c.pending_turn == "RIGHT"


def test_spin_continues_until_new_road_is_ahead():
    # 최소 회전 시간이 지나도 새 길이 정면(중간·먼 띠 양쪽 경계, 오차 작음)에 안 보이면 계속 돈다
    from drive.vision import ONE_SIDE
    c = make()
    c.on_sign(ev("TURN", "LEFT"))
    c.update(INTERSECTION, None, ["--", "--", "--"])
    for _ in range(config.TURN_ENTRY_FRAMES + config.TURN_FRAMES + 5):
        cmd = c.update(INTERSECTION, 0.8, ["L-", "L-", "L-"])
    assert cmd.action == "TURN" and cmd.steering < 0
    acts = [c.update(ONE_SIDE, 0.05, ["L-", "LR", "LR"]).action for _ in range(config.TURN_DONE_CONFIRM)]
    assert acts[-1] == "PID"


def test_brief_road_glimpse_does_not_end_spin():
    # 도는 중 대각선 너머가 한 프레임 길처럼 보여도 회전을 끝내지 않는다
    c = make()
    c.on_sign(ev("TURN", "LEFT"))
    c.update(INTERSECTION, None, ["--", "--", "--"])
    run(c, config.TURN_ENTRY_FRAMES + config.TURN_FRAMES, INTERSECTION, None)
    assert c.update(LANE, 0.0, ["LR", "LR", "LR"]).action == "TURN"
    assert c.update(INTERSECTION, None, ["--", "--", "--"]).action == "TURN"
    assert c._turn is not None and c._turn[1] == "SPIN"


def test_side_branch_already_open_when_sign_seen():
    # 표지판을 옆길 바로 앞에 두면, 표지판을 알아볼 때 이미 옆길이 열려 있다. 그래도 돈다
    from drive.vision import ONE_SIDE
    c = make()
    for _ in range(config.SIDE_CLOSED_FRAMES):
        c.update(LANE, 0.0, ["LR", "LR", "LR"])
    for _ in range(5):
        c.update(ONE_SIDE, 0.0, ["-R", "-R", "-R"])   # 옆길이 먼저 열림
    c.on_sign(ev("TURN", "LEFT"))
    acts = [c.update(ONE_SIDE, 0.0, ["-R", "-R", "-R"]).action for _ in range(config.SIDE_OPEN_FRAMES)]
    assert acts[-1] == "TURN"


def test_turn_sign_ignored_while_crossing():
    # 직진으로 교차로를 지나는 중에 건너편 길의 우회전 표지판이 보여도 방향을 바꾸지 않는다
    c = make()
    c.on_sign(ev("TURN", "STRAIGHT"))
    run(c, 5, INTERSECTION, None)
    c.on_sign(ev("TURN", "RIGHT"))
    assert "TURN" not in run(c, 5, INTERSECTION, None)


def test_side_branch_when_hugging_far_edge():
    # 우회전 옆길에서 왼쪽 선에 붙어 가면 가까운 띠에는 오른쪽 경계가 늘 화면 밖이다. 중간 띠의 양쪽 경계로 기록을 센다
    from drive.vision import ONE_SIDE
    c = make()
    c.on_sign(ev("TURN", "RIGHT"))
    for _ in range(config.SIDE_CLOSED_FRAMES):
        assert c.update(ONE_SIDE, 0.0, ["L-", "LR", "LR"]).action == "PID"
    acts = [c.update(ONE_SIDE, 0.0, ["L-", "L-", "L-"]).action for _ in range(config.SIDE_OPEN_FRAMES)]
    assert acts[-1] == "TURN"


def test_side_window_restarts_when_sign_seen_again():
    # 느린 실차는 표지판을 처음 본 뒤 옆길까지 대기 시간보다 오래 걸린다. 다시 보면 대기 시간을 처음부터 센다
    from drive.vision import ONE_SIDE
    c = make()
    c.on_sign(ev("TURN", "LEFT"))
    for i in range(config.SIDE_TURN_WINDOW_FRAMES + 50):
        if i % 42 == 0:
            c.on_sign(ev("TURN", "LEFT"))
        c.update(LANE, 0.0, ["LR", "LR", "LR"])
    acts = [c.update(ONE_SIDE, 0.0, ["-R", "-R", "-R"]).action for _ in range(config.SIDE_OPEN_FRAMES)]
    assert acts[-1] == "TURN"


def test_hold_straight_just_before_intersection():
    # 교차로 바로 앞(중간·먼 띠 열림)에서는 가까운 띠의 한쪽 경계만 보고 조향하지 않는다
    from drive.vision import ONE_SIDE
    c = make()
    cmd = c.update(ONE_SIDE, 0.4, ["-R", "--", "--"])
    assert cmd.action == "PID" and cmd.steering == 0.0


def test_no_new_turn_until_new_lane_after_turn():
    # 좌회전을 끝낸 차가 아직 교차로 안에서 다음 길의 우회전 표지판을 봐도, 새 차선에 들어서기 전에는 돌지 않는다
    from drive.vision import ONE_SIDE
    c = make()
    c.on_sign(ev("TURN", "LEFT"))
    c.update(INTERSECTION, None, ["--", "--", "--"])
    for _ in range(config.TURN_ENTRY_FRAMES + config.TURN_FRAMES + 2):
        c.update(INTERSECTION, 0.0, ["--", "LR", "LR"])
    assert c._turn is None and c._post_turn
    c.on_sign(ev("TURN", "RIGHT"))
    assert c.pending_turn == "RIGHT"
    acts = [c.update(INTERSECTION, None, ["--", "--", "LR"]).action for _ in range(10)]
    acts += [c.update(ONE_SIDE, 0.0, ["L-", "L-", "L-"]).action for _ in range(10)]
    assert "TURN" not in acts
    for _ in range(config.POST_TURN_LANE_FRAMES if hasattr(config, "POST_TURN_LANE_FRAMES") else 8):
        c.update(LANE, 0.0, ["LR", "LR", "LR"])
    assert not c._post_turn and c.pending_turn == "RIGHT"


def test_brief_edge_loss_is_not_a_side_branch():
    # 왼쪽 선에 붙어 가다 잠깐 틀어져 중간 띠의 오른쪽 경계가 몇 프레임 화면 밖으로 나가도 우회전하지 않는다
    from drive.vision import ONE_SIDE
    c = make()
    c.on_sign(ev("TURN", "RIGHT"))
    acts = []
    for _ in range(3):
        acts += [c.update(ONE_SIDE, 0.0, ["L-", "LR", "LR"]).action for _ in range(10)]
        acts += [c.update(ONE_SIDE, 0.0, ["L-", "L-", "LR"]).action for _ in range(config.SIDE_OPEN_FRAMES - 1)]
    assert "TURN" not in acts


def test_later_sign_waits_until_current_direction_is_used():
    # 직진 표지판을 보고 교차로로 가는 중에 다음 교차로의 우회전 표지판이 보여도, 이번 교차로는 직진하고
    # 우회전은 그 다음에 쓴다
    c = make()
    c.on_sign(ev("TURN", "STRAIGHT"))
    run(c, 30)
    c.on_sign(ev("TURN", "RIGHT"))
    assert c.pending_turn == "STRAIGHT"
    acts = run(c, 20, INTERSECTION, None)
    assert "TURN" not in acts and "CROSS" in acts
    run(c, 3)
    assert c.pending_turn == "RIGHT"


def test_stale_sign_gives_way_to_later_sign():
    # 앞 방향 표지판을 오래 못 봤고 교차로도 없었으면 (잘못 본 것), 받아 둔 방향으로 바꾼다
    c = make()
    c.on_sign(ev("TURN", "LEFT"))
    c.on_sign(ev("TURN", "RIGHT"))
    run(c, config.PENDING_STALE_FRAMES + 1)
    assert c.pending_turn == "RIGHT"


def test_nearest_turn_sign_wins_in_one_frame():
    # 한 화면에 좌회전(가까움, 큰 박스)과 우회전(멀리, 작은 박스)이 보이면 가까운 것만 센다
    it = SignInterpreter(config)
    evs = []
    for fid in range(3):
        evs += it([{"cls": "left", "conf": 0.7, "box_h": 90, "box_y2": 300, "frame_id": fid},
                   {"cls": "right", "conf": 0.9, "box_h": 50, "box_y2": 200, "frame_id": fid}])
    assert [e.value for e in evs] == ["LEFT"]


def test_center_first_slows_and_steers_by_position():
    # 가운데 찾기 모드(켰을 때): 곧은 길에서 차가 많이 치우쳐 있으면 천천히 가며 치우침 쪽 반대로 조향한다
    from drive.vision import ONE_SIDE
    old = config.CENTER_FIRST
    config.CENTER_FIRST = True
    try:
        c = make()
        cmd = c.update(ONE_SIDE, 0.0, ["L-", "LR", "LR"], near=-0.4)
        assert cmd.throttle == config.CENTER_THROTTLE and cmd.steering < 0
        cmd = c.update(LANE, 0.0, ["LR", "LR", "LR"], near=0.0)
        assert cmd.throttle != config.CENTER_THROTTLE or cmd.steering == 0.0
    finally:
        config.CENTER_FIRST = old


def test_spin_continues_while_view_is_all_green():
    # 느리게 돌아 TURN_SPIN_MAX_FRAMES 가 지나도 화면이 블록의 초록만 보이면 직진으로 넘어가지 않고 계속 돈다
    from drive.vision import OFF_ROAD
    c = make()
    c.on_sign(ev("TURN", "LEFT"))
    c.update(INTERSECTION, None, ["--", "--", "--"])
    for _ in range(config.TURN_ENTRY_FRAMES + config.TURN_SPIN_MAX_FRAMES + 5):
        cmd = c.update(OFF_ROAD, None, ["--", "--", "--"])
    assert cmd.action == "TURN" and cmd.steering < 0 and c._turn[1] == "SPIN"


def test_hold_steering_briefly_when_road_disappears_in_curve():
    # 커브를 꺾던 중 잠깐 도로가 안 보이면 직전 조향을 유지하고, 계속 안 보이면 정지
    from drive.vision import OFF_ROAD
    c = make()
    last = c.update(LANE, 0.5, ["LR", "LR", "LR"])
    held = [c.update(OFF_ROAD, None, ["--", "--", "--"]) for _ in range(config.LOST_HOLD_FRAMES)]
    assert all(h.action == "PID" and h.steering == last.steering for h in held)
    assert c.update(OFF_ROAD, None, ["--", "--", "--"]).action == "STOP"


def test_block_view_mid_turn_does_not_end_spin():
    # 실차 사진: 왼쪽으로 45도쯤 돌면 블록을 정면으로 본다. 띠 경계는 LR, 오차 +0.20 이지만 상태는 OFF_ROAD.
    # 새 길로 보지 않고 계속 왼쪽으로 돈다
    from drive.vision import OFF_ROAD
    c = make()
    c.on_sign(ev("TURN", "LEFT"))
    c.update(INTERSECTION, None, ["--", "--", "--"])
    run(c, config.TURN_ENTRY_FRAMES + config.TURN_FRAMES, INTERSECTION, None)
    cmds = [c.update(OFF_ROAD, 0.197, ["LR", "LR", "LR"]) for _ in range(20)]
    assert all(m.action == "TURN" and m.steering < 0 for m in cmds)


def test_confusing_view_right_after_turn_keeps_turning():
    # 회전을 끝낸 직후 헷갈리는 화면이 나오면 PID(반대 조향)나 정지 대신 하던 방향으로 계속 돈다
    from drive.vision import OFF_ROAD
    c = make()
    c.on_sign(ev("TURN", "LEFT"))
    c.update(INTERSECTION, None, ["--", "--", "--"])
    run(c, config.TURN_ENTRY_FRAMES + config.TURN_FRAMES, INTERSECTION, None)
    for _ in range(config.TURN_DONE_CONFIRM):
        c.update(LANE, 0.0, ["LR", "LR", "LR"])
    assert c._turn is None and c._post_turn
    m = c.update(OFF_ROAD, 0.197, ["LR", "LR", "LR"])
    assert m.action == "TURN" and m.steering < 0
    assert c.update(LANE, 0.0, ["LR", "LR", "LR"]).action == "PID"


def test_side_branch_without_closed_history_far_from_intersection():
    # 실차 사진: 오른쪽 옆길이 계속 L-/L-/L- 로만 보이고 양쪽 경계(LR) 기록이 없다. 교차로에서 멀면 그래도 돈다
    from drive.vision import ONE_SIDE
    c = make()
    for _ in range(config.SIDE_AFTER_CROSS_FRAMES + 50):
        c.update(ONE_SIDE, 0.05, ["L-", "L-", "LR"])
    c.on_sign(ev("TURN", "RIGHT"))
    acts = [c.update(ONE_SIDE, 0.05, ["L-", "L-", "L-"]).action for _ in range(config.SIDE_OPEN_FRAMES)]
    assert acts[-1] == "TURN"


def test_open_side_right_after_intersection_is_not_a_branch():
    # 교차로를 막 나오면 한쪽이 열려 보인다. 그때는 옆길로 보지 않는다
    from drive.vision import ONE_SIDE
    c = make()
    run(c, 5, INTERSECTION, None)
    c.update(ONE_SIDE, 0.05, ["L-", "L-", "L-"])       # 교차로를 막 나옴
    c.on_sign(ev("TURN", "RIGHT"))
    assert c.pending_turn == "RIGHT"
    acts = [c.update(ONE_SIDE, 0.05, ["L-", "L-", "L-"]).action for _ in range(20)]
    assert "TURN" not in acts


def test_kick_when_spin_is_stuck():
    # 회전 중 화면이 그대로면(차가 블록 앞에서 멈춤) 잠깐 세게 민 뒤 원래 세기로 돌아온다
    from drive.vision import OFF_ROAD
    c = make()
    c.on_sign(ev("TURN", "LEFT"))
    c.update(INTERSECTION, None, ["--", "--", "--"])
    run(c, config.TURN_ENTRY_FRAMES, INTERSECTION, None)
    thr = [c.update(OFF_ROAD, 0.197, ["LR", "LR", "LR"]).throttle for _ in range(40)]
    first = thr.index(config.TURN_KICK_THROTTLE)
    kick = thr[first:first + config.TURN_KICK_FRAMES]
    assert kick == [config.TURN_KICK_THROTTLE] * config.TURN_KICK_FRAMES          # 짧게만
    assert thr[first + config.TURN_KICK_FRAMES] == config.TURN_THROTTLE           # 다시 원래 세기
    assert c.notice


def test_no_kick_while_view_changes():
    # 실제로 돌고 있으면(화면이 바뀜) 세게 밀지 않는다
    from drive.vision import OFF_ROAD
    c = make()
    c.on_sign(ev("TURN", "LEFT"))
    c.update(INTERSECTION, None, ["--", "--", "--"])
    run(c, config.TURN_ENTRY_FRAMES, INTERSECTION, None)
    thr = [c.update(OFF_ROAD, 0.1 + 0.03 * i, ["LR", "LR", "LR"]).throttle for i in range(20)]
    assert max(thr) == config.TURN_THROTTLE


def test_no_side_turn_when_missing_edge_is_just_off_screen():
    # 오른쪽 경계가 화면 밖이라 안 보이는 것("L?")은 열린 길이 아니다. 경적을 우회전으로 잘못 봐도 돌지 않는다
    from drive.vision import ONE_SIDE
    c = make()
    c._since_cross = 10 ** 6
    c.on_sign(ev("TURN", "RIGHT"))
    acts = [c.update(ONE_SIDE, 0.05, ["L-", "L-", "L-"], None, ["L?", "L?", "L?"]).action for _ in range(30)]
    assert "TURN" not in acts
    acts = [c.update(ONE_SIDE, 0.05, ["L-", "L-", "L-"], None, ["L-", "L-", "L-"]).action
            for _ in range(config.SIDE_OPEN_FRAMES)]
    assert acts[-1] == "TURN"


def test_real_openings_uses_expected_edge_position():
    from types import SimpleNamespace as NS
    from drive.controller import real_openings
    b = lambda e, l, r, learned: NS(edges=e, left=l, right=r, learned=learned)
    # 실차 오른쪽 옆길: 왼쪽 32 + 폭 264 = 296 < 320-10 -> 진짜 열림
    assert real_openings(NS(bands=[b("L-", 32, None, 264)]), config) == ["L-"]
    # 왼쪽에 붙음: 왼쪽 80 + 264 = 344 > 320 -> 오른쪽 경계는 화면 밖일 뿐
    assert real_openings(NS(bands=[b("L-", 80, None, 264)]), config) == ["L?"]
    # 커브: 오른쪽 236 - 264 < 0 -> 왼쪽 경계는 화면 밖
    assert real_openings(NS(bands=[b("-R", None, 236, 264)]), config) == ["?R"]


def test_turn_sign_at_horn_position_is_dropped():
    # 경적이 보였던 자리에서 곧 비슷한 크기의 우회전이 보이면 같은 경적 표지판으로 본다
    it = SignInterpreter(config)
    evs = []
    for fid in range(0, 12, 2):
        evs += it([{"cls": "horn", "conf": 0.7, "box_h": 50, "box_w": 50, "box_cx": 500, "box_y2": 200, "frame_id": fid}])
    for fid in range(12, 30, 2):
        evs += it([{"cls": "right", "conf": 0.6, "box_h": 55, "box_w": 52, "box_cx": 510, "box_y2": 210, "frame_id": fid}])
    assert not [e for e in evs if e.kind == "TURN"]


def test_turn_sign_elsewhere_after_horn_is_kept():
    it = SignInterpreter(config)
    it([{"cls": "horn", "conf": 0.7, "box_h": 50, "box_w": 50, "box_cx": 500, "box_y2": 200, "frame_id": 0}])
    evs = []
    for fid in range(2, 10, 2):
        evs += it([{"cls": "right", "conf": 0.8, "box_h": 60, "box_w": 55, "box_cx": 150, "box_y2": 220, "frame_id": fid}])
    assert [e.value for e in evs] == ["RIGHT"]


def test_turn_sign_cut_at_frame_edge_is_ignored():
    # 실차: 화면 왼쪽 끝에 반쯤 걸린 경적을 right 로 봄 (상자 x 0~29). 끝에 닿은 회전 표지판은 버린다
    it = SignInterpreter(config)
    evs = []
    for fid in range(0, 10, 2):
        evs += it([{"cls": "right", "conf": 0.7, "box_h": 102, "box_w": 29, "box_cx": 14.5, "box_y2": 169, "frame_id": fid}])
    assert evs == []
    for fid in range(10, 20, 2):   # 화면 안으로 다 들어오면 받는다
        evs += it([{"cls": "right", "conf": 0.7, "box_h": 90, "box_w": 70, "box_cx": 120, "box_y2": 200, "frame_id": fid}])
    assert [e.value for e in evs] == ["RIGHT"]


def test_invalid_both_edge_band_does_not_end_spin():
    # 실차: 30~40도 돌았을 때 블록 양옆 좁은 틈을 LR 로 읽음 (비전은 폭이 안 맞아 버림, center 없음).
    # band_edges 가 "lr" 로 바꾸므로 교차로 상태여도 회전을 끝내지 않는다
    from types import SimpleNamespace as NS
    from drive.controller import band_edges
    r = NS(bands=[NS(edges="LR", center=None), NS(edges="LR", center=None), NS(edges="LR", center=190.0)])
    assert band_edges(r) == ["lr", "lr", "LR"]
    c = make()
    c.on_sign(ev("TURN", "LEFT"))
    c.update(INTERSECTION, None, ["--", "--", "--"])
    run(c, config.TURN_ENTRY_FRAMES + config.TURN_FRAMES, INTERSECTION, None)
    cmds = [c.update(INTERSECTION, 0.2, band_edges(r)) for _ in range(10)]
    assert all(m.action == "TURN" and m.steering < 0 for m in cmds)


def test_position_term_pulls_back_to_centre():
    # 내다본 오차가 0 이어도 차 위치가 치우쳐 있으면(near) 그 반대쪽 도로 중앙으로 조향한다 (부호는 near 와 같다)
    c = make()
    m = c.update(LANE, 0.0, ["LR", "LR", "LR"], near=0.3)
    assert m.steering > 0


def test_slow_spin_ramps_up_and_does_not_exit_toward_old_road():
    # 실차 로그: 0.32 로 3초에 30도만 돌고, 60 프레임에 직진으로 넘어가 원래 길(오차 +1.0)로 오른쪽 꺾음.
    # 이제는 세기를 0.42 까지 올리며, 새 길이 정면 근처에 오기 전에는 직진으로 넘어가지 않는다
    c = make()
    c.on_sign(ev("TURN", "LEFT"))
    c.update(INTERSECTION, None, ["--", "--", "--"])
    run(c, config.TURN_ENTRY_FRAMES, INTERSECTION, None)
    # 화면은 조금씩 바뀐다 (느리게라도 돈다) -> 멈춤 감지(세게 밀기)와 구분
    cmds = [c.update(INTERSECTION, 0.4 + 0.01 * (i % 50), ["--", "L-", "L-"])
            for i in range(config.TURN_SPIN_MAX_FRAMES + 30)]
    assert all(m.action == "TURN" and m.steering < 0 for m in cmds)
    assert cmds[-1].throttle == config.TURN_THROTTLE_MAX
    assert cmds[10].throttle == config.TURN_THROTTLE
    c.update(INTERSECTION, 0.3, ["--", "LR", "LR"])   # 새 길이 정면 근처 -> 이제 회전을 마무리
    assert c._turn is None or c._turn[1] != "SPIN"


def test_oblique_turn_sign_is_ignored():
    # 비스듬히 본 직진 표지판을 right 로 읽음 (상자가 가늘다). 가로/세로 0.7 미만이면 버린다
    it = SignInterpreter(config)
    evs = []
    for fid in range(0, 10, 2):
        evs += it([{"cls": "right", "conf": 0.7, "box_h": 80, "box_w": 40, "box_cx": 400, "box_y2": 200, "frame_id": fid}])
    assert evs == []
    for fid in range(10, 20, 2):
        evs += it([{"cls": "right", "conf": 0.7, "box_h": 80, "box_w": 76, "box_cx": 400, "box_y2": 200, "frame_id": fid}])
    assert [e.value for e in evs] == ["RIGHT"]


def test_lane_offset_follows_pending_turn():
    # 평소 살짝 오른쪽, 좌회전이 정해지면 왼쪽, 우회전이면 오른쪽에 붙는다. 회전 직후에는 가운데
    c = make()
    assert c.lane_offset() == config.LANE_OFFSET
    c.on_sign(ev("TURN", "LEFT"))
    assert c.lane_offset() == -config.TURN_PREP_OFFSET
    m = c.update(LANE, 0.0, ["LR", "LR", "LR"])
    assert m.steering < 0          # 가운데에 있어도 왼쪽으로 붙으려고 조향
    c2 = make()
    c2.on_sign(ev("TURN", "RIGHT"))
    assert c2.lane_offset() == config.TURN_PREP_OFFSET


def test_turn_sign_seen_while_crossing_used_when_blocked_ahead():
    # 실차: T자 건너편 우회전 표지판이 교차로에 들어선 뒤에야 이벤트가 됨 -> 막힌 쪽으로 직진하다 LOST.
    # 받아 둔 방향으로, 앞이 막히면(먼 띠 초록 GG 또는 LOST) 바로 돈다
    from drive.vision import LOST
    c = make()
    run(c, config.CROSS_LATE_SIGN_FRAMES + 5, INTERSECTION, None)   # 표지판 없이 교차로 직진 중 (들어선 지 오래됨)
    c.on_sign(ev("TURN", "RIGHT"))
    assert c.pending_turn is None
    assert "TURN" not in run(c, 5, INTERSECTION, None)
    m = c.update(INTERSECTION, None, ["--", "--", "GG"])
    assert m.action == "TURN" and c._turn[0] == "RIGHT"
    c2 = make()
    run(c2, config.CROSS_LATE_SIGN_FRAMES + 5, INTERSECTION, None)
    c2.on_sign(ev("TURN", "LEFT"))
    assert c2.update(LOST, None, ["--", "--", "--"]).action == "TURN" and c2._turn[0] == "LEFT"


def test_turn_sign_while_crossing_open_ahead_keeps_straight():
    # 가운데 십자 교차로(앞이 열림)에서는 건너편 표지판을 봐도 계속 직진하고, 다음 교차로에 쓴다
    c = make()
    c.on_sign(ev("TURN", "STRAIGHT"))
    run(c, 3, INTERSECTION, None)
    c.on_sign(ev("TURN", "RIGHT"))
    assert "TURN" not in [c.update(INTERSECTION, None, ["--", "--", "--"]).action for _ in range(10)]
    run(c, 2)
    assert c.pending_turn == "RIGHT"


def test_straight_used_up_at_side_branch_then_next_sign_applies():
    # 직진 표지판을 보고 옆길(진짜 열린 길)을 지나치면 직진을 쓴 것. 보류했던 우회전이 다음에 쓰인다
    from drive.vision import ONE_SIDE
    c = make()
    c._since_cross = 10 ** 6
    c.on_sign(ev("TURN", "STRAIGHT"))
    c.on_sign(ev("TURN", "RIGHT"))
    assert c.pending_turn == "STRAIGHT"
    for _ in range(config.SIDE_OPEN_FRAMES):
        c.update(ONE_SIDE, 0.0, ["-R", "-R", "-R"], None, ["-R", "-R", "-R"])
    assert c.pending_turn == "RIGHT"
    acts = [c.update(ONE_SIDE, 0.0, ["-R", "-R", "-R"], None, ["-R", "-R", "-R"]).action for _ in range(15)]
    assert "TURN" not in acts                         # 같은 옆길에서 바로 돌지는 않는다


def test_same_sign_straight_and_right_picks_higher_confidence():
    # 같은 표지판에 straight(0.85)와 right(0.55, 상자가 약간 큼)가 겹쳐 나오면 straight
    it = SignInterpreter(config)
    evs = []
    for fid in range(0, 6, 2):
        evs += it([{"cls": "straight", "conf": 0.85, "box_h": 70, "box_w": 66, "box_cx": 400, "box_y2": 200, "frame_id": fid},
                   {"cls": "right", "conf": 0.55, "box_h": 74, "box_w": 70, "box_cx": 402, "box_y2": 202, "frame_id": fid}])
    assert [e.value for e in evs] == ["STRAIGHT"]


def test_vision_blocked_t_junction_is_intersection():
    # 가까운 띠는 양옆이 열리고 먼 띠는 초록(트랙 테두리) 위에서 시작 -> 교차로 (예전: LOST)
    from types import SimpleNamespace as NS
    from drive.vision import LaneDetector
    det = LaneDetector(config, flip=False)
    near = NS(edges="--", note="", center=None, wide=False, y=195)
    bands = [near, NS(edges="--", note="", center=None, wide=False, y=168),
             NS(edges="--", note="start on green", center=None, wide=False, y=141)]
    assert det._candidate(bands, near, False, 0.0) == INTERSECTION


def test_t_end_exit_turns_after_reaching_the_border():
    # 막힌 T자 끝(먼 띠 위 초록 띠 >= TEND_AHEAD_GREEN)에서 빠져나갈 때는 정해진 진입 프레임이 지나도 직진하고,
    # 먼 띠가 테두리(GG)에 닿은 뒤 TEND_EXTRA_FRAMES 더 가서 돈다. 십자 교차로는 예전 그대로
    c = make()
    c.on_sign(ev("TURN", "RIGHT"))
    c.update(INTERSECTION, None, ["--", "--", "--"], ahead=0.85)
    acts = [c.update(INTERSECTION, None, ["--", "--", "--"], ahead=0.85) for _ in range(config.TURN_ENTRY_FRAMES + 5)]
    assert all(m.steering == 0.0 for m in acts)                     # 아직 직진
    gg = [c.update(INTERSECTION, None, ["--", "--", "GG"], ahead=0.95) for _ in range(config.TEND_EXTRA_FRAMES + 1)]
    assert gg[-1].steering > 0                                       # 테두리에 닿은 뒤 우회전
    c2 = make()
    c2.on_sign(ev("TURN", "RIGHT"))
    c2.update(INTERSECTION, None, ["--", "--", "--"], ahead=0.5)
    acts = [c2.update(INTERSECTION, None, ["--", "--", "--"], ahead=0.5) for _ in range(config.TURN_ENTRY_FRAMES + 2)]
    assert acts[-1].steering > 0                                     # 십자 교차로: 정해진 진입 뒤 회전


def test_late_turn_sign_right_after_entering_intersection():
    # 정한 방향 없이 교차로에 막 들어섰는데 회전 표지판이 늦게 확정됨 -> 그 방향으로 돈다 (남은 진입만큼 직진 후)
    c = make()
    run(c, 8, INTERSECTION, None)
    c.on_sign(ev("TURN", "RIGHT"))
    acts = [c.update(INTERSECTION, None, ["--", "--", "--"]) for _ in range(config.TURN_ENTRY_FRAMES)]
    assert acts[0].action == "TURN" and acts[-1].steering > 0


def test_side_opening_seen_before_sign_counts():
    # 옆길이 표지판보다 먼저 열려 있었으면, 표지판이 와도 처음부터 다시 세지 않고 바로 돈다
    from drive.vision import ONE_SIDE
    c = make()
    c._since_cross = 10 ** 6
    for _ in range(config.SIDE_OPEN_FRAMES + 6):
        c.update(ONE_SIDE, 0.0, ["L-", "L-", "L-"], None, ["L?", "L-", "L-"])
    c.on_sign(ev("TURN", "RIGHT"))
    m = c.update(ONE_SIDE, 0.0, ["L-", "L-", "L-"], None, ["L?", "L?", "L?"])   # 모두 "모름"이어도 지우지 않는다
    assert m.action == "TURN"


def test_pending_turn_and_blocked_ahead_turns_instead_of_stopping():
    # 우회전이 정해졌고 오른쪽이 열려 보였는데 앞이 섬으로 막혀 OFF_ROAD -> 멈추지 않고 우회전
    from drive.vision import ONE_SIDE, OFF_ROAD
    c = make()
    c._since_cross = 10 ** 6
    c.on_sign(ev("TURN", "RIGHT"))
    c._side_runs["L-"] = 3          # 오른쪽 옆길이 막 열려 보였다 (회전 조건까지는 못 감)
    m = c.update(OFF_ROAD, None, ["--", "--", "--"])
    assert m.action == "TURN" and c._turn[0] == "RIGHT"
    c2 = make()                      # 그쪽이 열린 적이 없으면 예전처럼 정지
    c2.on_sign(ev("TURN", "RIGHT"))
    assert c2.update(OFF_ROAD, None, ["--", "--", "--"]).action == "STOP"


def test_close_sign_shortens_side_entry():
    # 옆길 회전을 정하고 직진하는 중에 표지판이 60px 를 넘으면 직진을 줄이고 곧 돈다
    c = make()
    c._start_turn("RIGHT", 60, "옆길")
    c._turn_ambiguous = True          # 경계로는 판단이 안 되는 옆길 (모두 "모름")
    run(c, 10, INTERSECTION, None)
    c.on_sign(SignEvent("TURN", "RIGHT", "right", 0.8, 0, (500, 60, 563, 123)))
    acts = [c.update(INTERSECTION, None, ["--", "--", "--"]) for _ in range(config.SIGN_NEAR_ENTRY_FRAMES + 2)]
    assert acts[-1].steering > 0


def test_ambiguous_wide_open_side_waits_for_close_sign():
    # 실차 run_041701 (빨강): 오른쪽이 오래 "모름" -> 옆길 회전을 정해도 바로 돌지 않고, 표지판이 60px 를 넘을 때 돈다
    from drive.vision import ONE_SIDE
    c = make()
    c._since_cross = 10 ** 6
    for _ in range(300):
        c.update(ONE_SIDE, 0.0, ["L-", "L-", "L-"], None, ["L?", "L?", "L?"])
    c.on_sign(SignEvent("TURN", "RIGHT", "right", 0.8, 0, (400, 70, 445, 115)))      # 45px
    c.update(ONE_SIDE, 0.0, ["L-", "L-", "LR"], None, ["L?", "L?", "LR"])
    assert c._turn is not None and c._turn[1] == "ENTRY" and c._turn[3] == config.SIDE_AMBIG_ENTRY_FRAMES
    run(c, 30, INTERSECTION, None)
    assert c._turn[1] == "ENTRY"                                                     # 아직 직진
    c.on_sign(SignEvent("TURN", "RIGHT", "right", 0.8, 0, (500, 60, 563, 123)))      # 63px
    acts = [c.update(INTERSECTION, None, ["--", "--", "--"]) for _ in range(config.SIGN_NEAR_ENTRY_FRAMES + 2)]
    assert acts[-1].steering > 0


def test_side_seek_arcs_then_line_traces():
    # SIDE_SEEK: 옆길 회전을 정하면 직진 뒤 제자리 회전 대신 살짝 꺾으며 전진하고, 회전할 쪽 초록과 새 길이 보이면 PID
    old = config.SIDE_SEEK
    config.SIDE_SEEK = True
    try:
        c = make()
        c._start_turn("RIGHT", 3, "옆길")
        c._seek = True
        run(c, 3, INTERSECTION, None)
        arc = c.update(INTERSECTION, None, ["--", "--", "--"])
        assert arc.action == "TURN" and 0 < arc.steering < 1.0 and arc.throttle == config.SEEK_THROTTLE
        run(c, config.SEEK_MIN_FRAMES, INTERSECTION, None)
        assert c.update(LANE, 0.0, ["LR", "LR", "LR"]).action == "PID"
    finally:
        config.SIDE_SEEK = old


def test_after_curve_shift_toward_inside():
    # 오른쪽 커브(오차가 계속 +)를 크게 돌고 나오면 한동안 커브 안쪽(오른쪽)으로 붙는다. 커브 도는 중에는 붙지 않는다
    c = make()
    for _ in range(config.CURVE_HIST_FRAMES + 5):
        c.update(LANE, 0.5, ["LR", "LR", "LR"])
    assert c._in_curve_now and c.lane_offset() == config.LANE_OFFSET
    for _ in range(config.CURVE_HIST_FRAMES + 5):
        c.update(LANE, 0.05, ["LR", "LR", "LR"])
    assert c._curve_exit > 0 and c.lane_offset() == config.CURVE_EXIT_OFFSET


def test_after_curve_follow_line_before_side_turn():
    # 커브 직후 안쪽으로 붙는 동안 커브 안쪽 모서리가 "열림"처럼 보여도 바로 돌지 않는다.
    # 회전할 쪽 경계를 양쪽 경계로 다시 본 뒤 열리는 옆길에서 돈다
    from drive.vision import ONE_SIDE
    c = make()
    c._since_cross = 10 ** 6
    c.on_sign(ev("TURN", "RIGHT"))
    for _ in range(config.CURVE_HIST_FRAMES + 5):
        c.update(ONE_SIDE, 0.5, ["L-", "L-", "L-"], None, ["L-", "L-", "L-"])     # 오른쪽 커브
    acts = [c.update(ONE_SIDE, 0.05, ["L-", "L-", "L-"], None, ["L-", "L-", "L-"]).action
            for _ in range(config.CURVE_HIST_FRAMES + 10)]                          # 커브 끝, 오른쪽이 열려 보임
    assert "TURN" not in acts
    for _ in range(config.SIDE_CLOSED_FRAMES + 2):
        c.update(LANE, 0.0, ["LR", "LR", "LR"], None, ["LR", "LR", "LR"])         # 블록 경계를 다시 봄
    acts = [c.update(ONE_SIDE, 0.0, ["L-", "L-", "L-"], None, ["L-", "L-", "L-"]).action
            for _ in range(config.SIDE_OPEN_FRAMES)]                                # 진짜 옆길
    assert acts[-1] == "TURN"


def test_cross_timeout_with_held_turn_turns_instead_of_stop():
    # 실차 run_060243: 교차로 직진 중 우회전 표지판이 늦게 확정돼 보류 -> 직진이 최대 시간을 넘으면 정지 대신 우회전
    c = make()
    run(c, config.CROSS_LATE_SIGN_FRAMES + 5, INTERSECTION, None)
    c.on_sign(ev("TURN", "RIGHT"))
    acts = [c.update(INTERSECTION, None, ["--", "-R", "-R"]) for _ in range(config.INTERSECTION_MAX_FRAMES)]
    assert "STOP" not in [a.action for a in acts] and c._turn is not None and c._turn[0] == "RIGHT"


def test_same_sign_right_after_turn_is_ignored():
    # 실차 run_060120: 우회전 직후 같은 우회전 표지판을 다시 보고 우회전이 또 예약됨 -> 무시
    c = make()
    c.on_sign(ev("TURN", "RIGHT"))
    c.update(INTERSECTION, None, ["--", "--", "--"])
    run(c, config.TURN_ENTRY_FRAMES + config.TURN_FRAMES, INTERSECTION, None)
    for _ in range(config.TURN_DONE_CONFIRM):
        c.update(LANE, 0.0, ["LR", "LR", "LR"])
    assert c._post_turn
    c.on_sign(ev("TURN", "RIGHT"))
    assert c.pending_turn is None
    c.on_sign(ev("TURN", "LEFT"))     # 다른 방향은 받는다
    assert c.pending_turn == "LEFT"


def test_keep_crossing_when_lost_but_road_ahead():
    """십자 교차로 직진 중 가까운 · 중간 띠가 비고 먼 띠만 건너편 길(LR)을 보면 비전은 LOST 를 낸다. 서지 않고 계속 간다
    (실차 run_20261111_191835: 초록불인데 교차로 한가운데서 정지)."""
    c = make()
    for _ in range(5):
        c.update(INTERSECTION, None, edges=["--", "--", "--"])
    acts = [c.update(LOST, None, edges=["--", "--", "LR"]).action for _ in range(10)]
    assert "STOP" not in acts and acts[-1] == "CROSS"
