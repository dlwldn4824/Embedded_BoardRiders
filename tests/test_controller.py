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
    assert c.update(LANE, 0.0).action == "PID"                # 차선을 다시 잡으면 PID
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
