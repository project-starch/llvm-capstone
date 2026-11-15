#!/usr/bin/env python3
"""Offline checks of pre_boot_park.py's contract (no board). Exit 0 when every arm holds.

Through the real FpgaConsole against mock_server (one process per arm; the local board lock is held to process exit):
 A. a clean board: rc 0, nothing emitted, the must-fire line printed;
 B. the mock set to 217 first: rc 10, states_before=[1, 0, 0, 1, 1, 0, 1, 1], states_after=[0]*8.
Through a fake console (console_factory), on a virtual clock:
 C. switch_reset_all unsupported and bit 3 never confirmed: rc 1, no blind second toggle;
 D. no switch_state ever: rc 1, states_before=None;
 E. the console raises with a URL and a token in its message: rc 1, line printed, neither leaks;
 F. a bad board argument: rc 1, line printed."""
import contextlib
import io
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fpga_driver import config as C  # noqa: E402
from fpga_driver import pre_boot_park as PB  # noqa: E402
from fpga_driver import switch_park as SP  # noqa: E402
from fpga_driver.fpga_console import ActionTimeout  # noqa: E402

fails = []


def check(name, cond, detail):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f": {detail}"))
    if not cond:
        fails.append(name)


# --- A, B: mock server -------------------------------------------------------------------------------------------
SETTER = r'''
import os, sys
sys.path.insert(0, os.environ["T_PARENT"])
from fpga_driver.fpga_console import FpgaConsole
c = FpgaConsole(os.environ["PREBOOT_MOCK_URL"], allow_unverified=True); c.connect(timeout=10)
for b in (0, 3, 4, 6, 7): c.set_switch(b, True)
c.close()
'''


def mock_arm(preset):
    from fpga_driver.test_dryrun import _MockServer
    with _MockServer() as srv:
        env = dict(os.environ, PREBOOT_MOCK_URL=srv.url, T_PARENT=str(HERE.parent))
        if preset:
            env["BOARD_LOCK"] = os.path.join(tempfile.mkdtemp(), "set.lock")
            subprocess.run([sys.executable, "-c", SETTER], env=env, check=True, timeout=60,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        env["BOARD_LOCK"] = os.path.join(tempfile.mkdtemp(), "park.lock")
        r = subprocess.run([sys.executable, str(HERE / "pre_boot_park.py"), "2"], env=env, timeout=120,
                           capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("[pre-boot-park]")]
    return r.returncode, line, r.stderr


rc, line, err = mock_arm(False)
check("A clean board rc 0", rc == 0 and len(line) == 1 and "states_before=[0, 0, 0, 0, 0, 0, 0, 0]" in line[0]
      and "fpga-switch" not in err, (rc, line))
rc, line, err = mock_arm(True)
check("B 217 parked rc 10", rc == 10 and len(line) == 1 and "states_before=[1, 0, 0, 1, 1, 0, 1, 1]" in line[0]
      and "states_after=[0, 0, 0, 0, 0, 0, 0, 0]" in line[0], (rc, line, err[-400:]))

# --- C..F: fake console on a virtual clock -------------------------------------------------------------------------
CLOCK = [0.0]
SP.time.monotonic = lambda: CLOCK[0]
PB.time.monotonic = lambda: CLOCK[0]
PB.time.sleep = lambda s: CLOCK.__setitem__(0, CLOCK[0] + s)


class Fake:
    def __init__(self, states, initial=True, never=()):
        self.sw, self.never, self.emits, self.events = list(states), set(never), [], []
        if initial:
            self.events.append((CLOCK[0], {C.SWITCH_STATE_KEY: list(self.sw)}))

    def now(self):
        return CLOCK[0]

    def latest(self, ev):
        got = [d for ts, d in self.events if ts <= CLOCK[0]]
        return got[-1] if got else None

    def _emit(self, key, **kw):
        self.emits.append((key, kw.get("index")))
        if key == "switch_toggle":
            b = kw["index"]
            self.sw[b] ^= 1
            if b not in self.never:
                self.events.append((CLOCK[0] + 1, {C.SWITCH_STATE_KEY: list(self.sw)}))

    def wait_event(self, ev, predicate=lambda d: True, timeout=60.0, *, since=None):
        end = CLOCK[0] + timeout
        for ts, d in sorted(self.events, key=lambda e: e[0]):
            if ts > end:
                break
            if (since is None or ts >= since) and predicate(d):
                CLOCK[0] = max(CLOCK[0], ts) + 1e-6
                return d
        CLOCK[0] = end
        raise ActionTimeout(f"timed out waiting for event {ev!r}")


def run(argv, factory):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = PB.main(argv, console_factory=factory)
    return rc, out.getvalue(), err.getvalue()


f = Fake([1, 0, 0, 1, 1, 0, 1, 1], never={3})
rc, out, err = run(["2"], lambda: f)
check("C never confirmed rc 1, one toggle for bit 3", rc == 1 and "[pre-boot-park] board 2" in out
      and [i for k, i in f.emits if k == "switch_toggle"].count(3) == 1, (rc, out, f.emits))

rc, out, err = run(["2"], lambda: Fake([0] * 8, initial=False))
check("D no switch_state rc 1", rc == 1 and "states_before=None" in out, (rc, out))


# A URL-shaped fake built at run time: committed source never carries a URL- or token-shaped literal.
FAKE_HOST = "board" + ".invalid"
FAKE_SECRET = "".join(chr(65 + i % 26) for i in range(30))
FAKE_MSG = "connect to " + "https" + "://" + FAKE_HOST + "/console?" + "tok" + "en=" + FAKE_SECRET + " failed"


def boom():
    raise RuntimeError(FAKE_MSG)


rc, out, err = run(["2"], boom)
check("E exception rc 1, nothing leaks", rc == 1 and "[pre-boot-park] board 2" in out and FAKE_HOST not in out
      and FAKE_SECRET not in out and "error=RuntimeError" in out, (rc, out))

rc, out, err = run(["9"], lambda: Fake([0] * 8))
check("F bad board rc 1", rc == 1 and "[pre-boot-park] board 9" in out and "error=ValueError" in out, (rc, out))

sys.exit(1 if fails else 0)
