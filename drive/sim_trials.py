"""시뮬레이터 무작위 반복 시험. 조건을 매번 흔들어 성공 횟수를 센다.

  python -m drive.sim_trials turns                 # 교차로 좌·우·직진 각 50회 (기본 조건)
  python -m drive.sim_trials turns --stress        # 조건을 2배로 흔든 한계 시험
  python -m drive.sim_trials mission -n 50         # 가상 표지판 미션 전체

turns: 가운데 교차로 앞에서 출발한다. 매번 출발 위치·각도, 오른쪽 바퀴 세기, 조명, 카메라 잡음을 바꾼다.
  성공 = 교차로(TURN/CROSS)를 지난 뒤 목표 방향 ±30° 안에서 PID 주행이 1초 이어지고,
         그동안 도로를 벗어나거나 LOST/OFF_ROAD 가 한 번도 없었다.
mission: sim_sweep.run_one 을 검출 지연 0~0.3초, 놓침 확률 0~20% 로 돌린다.
  성공 = 도착(FINISH)했고 LOST 정지가 없었다.
"""
from __future__ import annotations

import argparse
import math
import random

from drive import config
from drive import sim
from drive import sim_sweep

TARGET_DEG = {"LEFT": 180.0, "RIGHT": 0.0, "STRAIGHT": -90.0}
BASE = {"pos": 5.0, "dist": 8.0, "deg": 10.0, "gain": 0.07, "light": 0.2}
STRESS = {"pos": 10.0, "dist": 8.0, "deg": 20.0, "gain": 0.15, "light": 0.2}


def _angdiff(a, b):
    return abs((a - b + 180) % 360 - 180)


def turn_trial(turn, rng, spread, seconds=9.0):
    args = sim.parse_args(["--no-gui", "--start", "cross", "--turn", turn,
                           "--right-gain", f"{1 + rng.uniform(-spread['gain'], spread['gain']):.3f}",
                           "--light", f"{1 + rng.uniform(-spread['light'], spread['light']):.2f}"])
    s = sim.Sim(args, config)
    s.cam = sim.Camera(s.track, args.light, seed=rng.randrange(10 ** 6))
    x, y, psi = sim.STARTS["cross"]
    s.place(x + rng.uniform(-spread["pos"], spread["pos"]), y + rng.uniform(-spread["dist"], spread["dist"]),
            psi + rng.uniform(-spread["deg"], spread["deg"]))
    s.ctrl.pending_turn = turn
    s.ctrl.set_running(True)
    off, lost, passed, good = 0, False, False, 0
    while s.t < seconds:
        s.step()
        off += not s.is_on_road()
        lost |= s.res.state in ("LOST", "OFF_ROAD")
        a = s.cmd.action
        if a in ("TURN", "CROSS"):
            passed, good = True, 0
        elif passed and a == "PID" and _angdiff(math.degrees(s.veh.psi), TARGET_DEG[turn]) < 30:
            good += 1
        else:
            good = 0
        if good >= config.LOOP_HZ:
            break
    return good >= config.LOOP_HZ and off == 0 and not lost


def run_turns(n, seed, stress):
    rng = random.Random(seed)
    spread = STRESS if stress else BASE
    total = 0
    for turn in TARGET_DEG:
        ok = sum(turn_trial(turn, rng, spread) for _ in range(n))
        total += ok
        print(f"{turn:8s} {ok} / {n}", flush=True)
    print(f"합계     {total} / {3 * n}")


def run_mission(n, seed, box_h):
    rng = random.Random(seed)
    ok = 0
    for _ in range(n):
        lat, miss = rng.uniform(0, 0.3), rng.uniform(0, 0.2)
        r = sim_sweep.run_one(lat, miss, rng.randrange(10 ** 6), {"SIGN_MIN_BOX_H": box_h})
        good = r["finish"] is not None and r["lost"] is None
        ok += good
        if not good:
            missed = [k for k in sim_sweep.KINDS if k not in r["first"]]
            print(f"실패  지연 {lat:.2f}s  놓침 {miss:.2f}  못 잡은 표지판: {', '.join(missed) or '-'}", flush=True)
    print(f"미션 완주 {ok} / {n}  (SIGN_MIN_BOX_H={box_h})")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["turns", "mission"])
    p.add_argument("-n", type=int, default=None, help="반복 횟수 (turns 는 방향마다, 기본 50 / 한계 시험 30)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stress", action="store_true", help="turns: 조건을 2배로 흔든다")
    p.add_argument("--box-h", type=int, default=45, help="mission: SIGN_MIN_BOX_H (docs/sim-signs.md 5 참고)")
    args = p.parse_args(argv)
    if args.kind == "turns":
        run_turns(args.n or (30 if args.stress else 50), 11 if args.seed is None and args.stress else (args.seed or 7),
                  args.stress)
    else:
        run_mission(args.n or 50, 3 if args.seed is None else args.seed, args.box_h)


if __name__ == "__main__":
    main()
