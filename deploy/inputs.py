"""조향 입력: 게임패드(권장), 키보드(SSH 터미널), 스크립트(테스트용).

poll() 은 매 프레임 호출되고 (steer, events) 를 돌려준다.
events: record, drive, recovery, bad, estop, quit, faster, slower
"""
import math
import threading
import time


class GamepadInput:
    """evdev 게임패드. 왼쪽 스틱 X축이 연속 조향값이 된다."""

    def __init__(self, cfg):
        import evdev

        self.ecodes = evdev.ecodes
        self.cfg = cfg["gamepad"]
        devs = [evdev.InputDevice(p) for p in evdev.list_devices()]
        axis = self.ecodes.ecodes[self.cfg["steer_axis"]]
        pads = [d for d in devs if axis in dict(d.capabilities().get(self.ecodes.EV_ABS, []))]
        if not pads:
            raise RuntimeError("게임패드를 찾지 못했습니다 (블루투스 연결 / 권한 확인, 또는 --input keyboard)")
        self.dev = pads[0]
        info = self.dev.absinfo(axis)
        self.axis, self.amin, self.amax = axis, info.min, info.max
        self.buttons = {self.ecodes.ecodes[v]: k for k, v in self.cfg.items() if k not in ("steer_axis", "deadzone")}
        self.steer = 0.0
        self.events = []
        self.connected = True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        e = self.ecodes
        try:
            for ev in self.dev.read_loop():
                with self.lock:
                    if ev.type == e.EV_ABS and ev.code == self.axis:
                        v = 2 * (ev.value - self.amin) / (self.amax - self.amin) - 1
                        self.steer = 0.0 if abs(v) < self.cfg["deadzone"] else max(-1.0, min(1.0, v))
                    elif ev.type == e.EV_ABS and ev.code == e.ABS_HAT0Y and ev.value:
                        self.events.append("faster" if ev.value < 0 else "slower")
                    elif ev.type == e.EV_KEY and ev.value == 1 and ev.code in self.buttons:
                        self.events.append(self.buttons[ev.code])
        except OSError:
            self.connected = False

    def poll(self):
        with self.lock:
            ev, self.events = self.events, []
            if not self.connected:
                ev.append("estop")
            return self.steer, ev

    def status(self, line):
        print("\r" + line.ljust(100), end="", flush=True)

    def close(self):
        print()


class KeyboardInput:
    """SSH 터미널용. a/d(←/→)를 누르는 동안 조향이 커지고, 떼면 0으로 돌아온다."""

    KEYS = {
        ord("r"): "record", ord("g"): "drive", ord("c"): "recovery", ord("x"): "bad",
        ord(" "): "estop", ord("q"): "quit", ord("+"): "faster", ord("="): "faster", ord("-"): "slower",
    }

    def __init__(self, cfg):
        import curses

        self.curses = curses
        self.step, self.decay = cfg["keyboard_step"], cfg["keyboard_decay"]
        self.scr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.scr.keypad(True)
        self.scr.nodelay(True)
        self.steer = 0.0
        self.scr.addstr(0, 0, "a/←: 좌  d/→: 우  g: 주행  r: 녹화  c: 복구태그  x: 방금실수  space: 정지  +/-: 속도  q: 종료")

    def poll(self):
        c = self.curses
        events, pressed = [], False
        while (k := self.scr.getch()) != -1:
            if k in (ord("a"), c.KEY_LEFT):
                self.steer, pressed = max(-1.0, self.steer - self.step), True
            elif k in (ord("d"), c.KEY_RIGHT):
                self.steer, pressed = min(1.0, self.steer + self.step), True
            elif k in self.KEYS:
                events.append(self.KEYS[k])
        if not pressed:
            self.steer *= self.decay
            if abs(self.steer) < 0.02:
                self.steer = 0.0
        return self.steer, events

    def status(self, line):
        self.scr.addstr(2, 0, line.ljust(100))
        self.scr.refresh()

    def close(self):
        c = self.curses
        self.scr.keypad(False)
        c.nocbreak()
        c.echo()
        c.endwin()


class ScriptedInput:
    """하드웨어 없이 파이프라인을 시험하는 입력. 시작하자마자 주행·녹화를 켜고 사인파로 조향한다."""

    def __init__(self, cfg):
        self.t0 = time.time()
        self.started = False

    def poll(self):
        ev = [] if self.started else ["drive", "record"]
        self.started = True
        t = time.time() - self.t0
        if 1.0 < t < 1.1:
            ev.append("recovery")
        return math.sin(t * 1.5), ev

    def status(self, line):
        pass

    def close(self):
        pass


def make_input(kind, cfg):
    return {"gamepad": GamepadInput, "keyboard": KeyboardInput, "script": ScriptedInput}[kind](cfg)
