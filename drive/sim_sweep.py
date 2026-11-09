"""가상 표지판 미션을 검출 지연·설정값별로 돌려 표로 정리한다.

  python -m drive.sim_sweep                                  # 기본: 지연 0~1초 x SIGN_MIN_BOX_H 40/45
  python -m drive.sim_sweep --latency 0.2 0.5 --miss 0.2 --seeds 0 1 2 --set SIGN_MIN_BOX_H=45

각 줄: 도착 여부·시각, 각 표지판 이벤트가 처음 난 시각, 빨간불에 서 있던 시간, LOST 정지 시각.
"""
from __future__ import annotations

import argparse
import itertools

from drive import config
from drive import sim

KINDS = ["red_light", "left", "green_light", "right", "speed_20", "stop", "red_line"]


def run_one(latency, miss, seed, overrides, seconds=240.0, scenario="mission"):
    saved = {k: getattr(config, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(config, k, v)
        args = sim.parse_args(["--no-gui", "--start", "cross", "--signs", scenario,
                               "--sign-latency", str(latency), "--sign-miss", str(miss), "--seed", str(seed)])
        s = sim.Sim(args, config)
        s.ctrl.set_running(True)
        red = lost = finish = None
        red_frames = 0
        while s.t < seconds:
            s.step()
            a = s.cmd.action
            red_frames += a == "RED"
            if a == "FINISH" and finish is None:
                finish = s.t
            if s.res.state == "LOST" and a == "STOP" and lost is None:
                lost = s.t
            if finish is not None or (lost is not None and s.t > lost + 1):
                break
        first = {}
        for t, ev in s.sign_log:
            first.setdefault(ev.cls, t)
        return {"finish": finish, "lost": lost, "red_wait": red_frames / config.LOOP_HZ, "first": first}
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--latency", type=float, nargs="+", default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])
    p.add_argument("--miss", type=float, nargs="+", default=[0.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--box-h", type=int, nargs="+", default=[40, 45], help="SIGN_MIN_BOX_H 후보")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)

    base = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        base[k] = type(getattr(config, k))(v)
    head = f"{'box_h':>5s} {'지연':>5s} {'miss':>5s} {'seed':>4s} | {'도착':>6s} {'LOST':>5s} {'빨간불':>6s} | " + \
           " ".join(f"{k[:8]:>8s}" for k in KINDS)
    print(head)
    print("-" * len(head))
    for h, lat, miss, seed in itertools.product(args.box_h, args.latency, args.miss, args.seeds):
        r = run_one(lat, miss, seed, {**base, "SIGN_MIN_BOX_H": h})
        f = lambda x: "-" if x is None else f"{x:.1f}"
        print(f"{h:5d} {lat:5.2f} {miss:5.2f} {seed:4d} | {f(r['finish']):>6s} {f(r['lost']):>5s} "
              f"{r['red_wait']:6.1f} | " + " ".join(f"{f(r['first'].get(k)):>8s}" for k in KINDS), flush=True)


if __name__ == "__main__":
    main()
