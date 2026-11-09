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
