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


def test_red_light_holds_until_green():
    c = make()
    c.on_sign(ev("LIGHT", "RED"))
    assert set(run(c, 50)) == {"RED"}
    c.on_sign(ev("LIGHT", "GREEN"))
    assert run(c, 1) == ["PID"]


def test_red_light_times_out():
    c = make()
    c.on_sign(ev("LIGHT", "RED"))
    acts = run(c, config.LIGHT_MAX_WAIT_FRAMES + 2)
    assert acts[-1] == "PID"


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
    # 최소 회전 뒤, 새 길이 정면(중간·먼 띠 양쪽 경계)에 보이면 PID 로 넘어간다
    assert c.update(LANE, 0.0, ["LR", "LR", "LR"]).action == "PID"
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
    assert c.update(ONE_SIDE, 0.05, ["L-", "LR", "LR"]).action == "PID"


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
