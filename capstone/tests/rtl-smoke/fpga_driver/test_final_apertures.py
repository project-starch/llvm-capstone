#!/usr/bin/env python3
"""Offline check of run_baked_rungs_fpga's BAKED_FINAL_APERTURES (no board): a fake console runs the rung loop.

Four arms, each in a fresh process:
- healthy + FINAL=1: the transcript ends with the final apertures block;
- healthy + FINAL unset: no block (the default is unchanged);
- the last rung wedges + FINAL=1 + WEDGE=1: the wedge block is present and NO final block (the for-else does not run
  after a break);
- the first rung wedges: later rungs are COLLATERAL and no final block.
Exit 0 when all four hold."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

CHILD = r'''
import os, sys, types, pathlib
sys.path.insert(0, os.environ["T_PARENT"])
import fpga_driver.run_baked_rungs_fpga as d
WEDGE = os.environ.get("T_WEDGE", "")
class Fake:
    def __init__(self, *a, **k): self.uart_text = ""; self.sw = 0
    def connect(self): pass
    def lock(self): pass
    def upload_boot_image(self, *a): pass
    def uart_mark(self): return len(self.uart_text)
    def uart_since(self, m): return self.uart_text[m:]
    def run_command(self, cmd, pat, timeout=0, idle_timeout=0):
        r = cmd.split("### RUNG ")[1].split()[1]
        if r == WEDGE:
            self.uart_text += f"### RUNG x/x {r} ###\npartial\n"
            raise TimeoutError("fake wedge")
        self.uart_text += f"### RUNG x/x {r} ###\nRESULT {r} retval=0\n### RUNG x {r} END rc=0 ###\nDN_0\n"
    def set_switch(self, bit, on): self.sw = (self.sw | (1 << bit)) if on else (self.sw & ~(1 << bit))
    def latest(self, key): return {"states": [bool((self.sw * 7) >> i & 1) for i in range(8)]}
d.FpgaConsole = Fake
for n in ("install_resilient_emit", "install_release_on_signal", "require_preflight", "release_board"):
    setattr(d, n, lambda *a, **k: None)
d.cold_boot = lambda *a, **k: None
d.nvbit = lambda c: d.BITSTREAM
d.time.sleep = lambda s: None
d.main()
'''


def arm(name, env_extra):
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "oracles").mkdir()
        for r in ("ra", "rb"):
            (t / "oracles" / f"{r}.oracle").write_text("0\n")
        fw = t / "fw.bin"
        fw.write_bytes(b"x" * 64)
        env = dict(os.environ, T_PARENT=str(HERE.parent), FPGA_FW=str(fw),
                   FPGA_BITSTREAM="stub.bit", BAKED_RUNGS="ra rb", LADDER_FPGA_DIR=str(t / "oracles"),
                   BAKED_OUT=str(t / "board.txt"), BAKED_TIMEOUT="1", PREFLIGHT="0", **env_extra)
        env["FPGA_URL"] = "stub"  # the driver requires one; the fake console never connects. Never a real or URL-shaped value.
        for k in ("BAKED_FINAL_APERTURES", "BAKED_WEDGE_APERTURES", "T_WEDGE"):
            if k not in env_extra:
                env.pop(k, None)
        p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=120)
        out = (t / "board.txt").read_text() if (t / "board.txt").exists() else None
        return p, out


def main():
    bad = 0
    cases = [
        ("healthy, FINAL=1", {"BAKED_FINAL_APERTURES": "1"}, lambda o: "===== final apertures (no wedge) =====" in o
         and "sw=193" in o.split("final apertures")[1]),
        ("healthy, FINAL unset", {}, lambda o: "final apertures" not in o),
        ("last rung wedges, FINAL=1", {"BAKED_FINAL_APERTURES": "1", "BAKED_WEDGE_APERTURES": "1", "T_WEDGE": "rb"},
         lambda o: "===== rb wedge apertures =====" in o and "final apertures" not in o),
        ("first rung wedges, FINAL=1", {"BAKED_FINAL_APERTURES": "1", "T_WEDGE": "ra"},
         lambda o: "final apertures" not in o and "===== rb" not in o),
    ]
    for name, extra, ok in cases:
        p, out = arm(name, extra)
        good = out is not None and ok(out)
        bad += not good
        print(f"{'ok  ' if good else 'FAIL'} {name}" + ("" if good else f" rc={p.returncode} {p.stderr[-400:]!r}"))
    print("PASS" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
