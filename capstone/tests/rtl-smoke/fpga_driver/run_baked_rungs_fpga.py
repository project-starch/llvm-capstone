"""Run silicon-ladder rungs BAKED INTO THE FIRMWARE IMAGE, not shipped over UART.

Why this exists. `run_ladder_perf_fpga.py` transfers the controller and every `.dom` over the
console as gzip+base64 at 16 chars per emit, and each emit is an HTTPS round trip -- minutes per
domain, and it dominated every sweep. These rungs are ~10 KB each, so putting them in the
initramfs costs nothing on the JTAG upload that happens anyway, and the run becomes a shell
command per rung.

It also removes the R-3 confound that voided an earlier sweep: `run_ladder_perf_fpga.py` with
LADDER_ONE_BOOT=1 requires DISTINCT entry VAs, and all these rungs link at 0x10000. Here every
rung is invoked from a shell in ONE boot, and the controller reloads the domain each time, so
position-in-boot is recorded per rung rather than assumed away -- see `pos=` in the output.

Usage:
    BAKED_RUNGS="clp1 clp2 clp4 clp8" python3 -m fpga_driver.run_baked_rungs_fpga
Env: FPGA_URL, FPGA_FW (both required, no defaults -- an implicit FW silently boots whatever
was built last), BAKED_CTL (default /test-domains/lpc), BAKED_TIMEOUT (default 120).
"""
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fpga_driver import config as C                                    # noqa: E402
from fpga_driver.fpga_console import FpgaConsole                       # noqa: E402
from fpga_driver.safe_cleanup import (release_board, hard_exit,        # noqa: E402
                                      install_release_on_signal)
from fpga_driver.run_sqlite_baked_fpga import _hash_name  # noqa: E402
from fpga_driver.run_ladder_perf_fpga import (cold_boot, nvbit,        # noqa: E402
                                              install_resilient_emit)

URL = os.environ.get("FPGA_URL")
if not URL:
    raise SystemExit("FPGA_URL not set")
IMG = pathlib.Path(os.environ["FPGA_FW"])          # KeyError on purpose: never default the FW
# NAME THE STORED IMAGE BY CONTENT, never by a fixed filename.
#
# This used to default to "fw_payload_fpga_up_gpfree.bin" -- a single slot that every run
# overwrote. Two consequences, both bad for an A/B: the board's image store cannot tell two
# firmwares apart, and the reassuring "19563528 bytes written" line proves nothing about WHICH
# firmware reached DDR, because that byte count was identical across every build on 2026-08-05.
# A content hash makes the loaded image self-identifying and lets an unchanged firmware reuse
# the stored upload instead of re-sending ~19 MB. This is the stale-store trap cold_boot's own
# docstring warns about, and the other runners already avoid it.
IMG_NAME = os.environ.get("FPGA_FW_NAME") or _hash_name(IMG)
CTL = os.environ.get("BAKED_CTL") or "/test-domains/lpc"
RUNGS = (os.environ.get("BAKED_RUNGS") or "clp1 clp8").split()
TIMEOUT = float(os.environ.get("BAKED_TIMEOUT") or 120)
# A wedged domain emits TOTAL SILENCE, so waiting the full run budget for it is
# pure waste -- callers used to pass idle_timeout=TIMEOUT, which made the idle
# check a no-op and spent the whole 150s on a domain that died in the first
# millisecond. Any UART byte resets this clock, so a slow-but-live workload is
# unaffected; only silence trips it. Raise BAKED_IDLE_S if a workload can legitimately
# go quiet for longer than this between markers.
IDLE_S = float(os.environ.get("BAKED_IDLE_S", "25"))
ART = pathlib.Path(os.environ.get("LADDER_FPGA_DIR") or "/tmp/capstone/ladder-fpga")
OUT = os.environ.get("BAKED_OUT") or "/tmp/capstone/baked-rungs.txt"
# Resident-bitstream guard. Reflashed 2026-08-04 to caplifive_fixed_forward.bit, which
# carries the operand-forwarding fix (capstone-ariane 7aac52f93). Overridable so the next
# reflash needs no code change mid-session -- the guard exists to stop a run from silently
# measuring the WRONG silicon, and on 2026-08-04 it did exactly that.
BITSTREAM = os.environ.get("FPGA_BITSTREAM", "caplifive_fixed_forward.bit")


def log(m):
    print(f"[baked] {m}", file=sys.stderr, flush=True)


def main():
    oracles = {}
    for r in RUNGS:
        p = ART / f"{r}.oracle"
        if not p.is_file():
            raise SystemExit(f"missing oracle {p} -- run build-ladder-fpga.sh {r}")
        oracles[r] = int(p.read_text().split()[0])

    console = FpgaConsole(URL, logger=lambda m: print(f"[fpga] {m}", file=sys.stderr))
    console.connect()
    install_resilient_emit(console)
    results, transcript = [], []
    try:
        console.lock()
        install_release_on_signal(console)
        rb = nvbit(console)
        if rb and BITSTREAM not in rb:
            raise SystemExit(f"HARD STOP: resident bitstream is {rb!r}, expected {BITSTREAM!r}")
        console.upload_boot_image(IMG_NAME, str(IMG))
        cold_boot(console, C.GDB_PROMPT, IMG_NAME)
        log(f"booted; running {len(RUNGS)} baked rungs from the shell")

        for pos, r in enumerate(RUNGS, 1):
            mark = console.uart_mark()
            banner = f"### RUNG {pos}/{len(RUNGS)} {r} ###"
            try:
                console.run_command(
                    f"echo '{banner}'; {CTL} {r} /test-domains/{r}.dom; "
                    f"rc=$?; echo \"### RUNG {pos} {r} END rc=$rc ###\"; echo D''N_$rc",
                    r"DN_\d", timeout=TIMEOUT, idle_timeout=IDLE_S)
            except Exception as exc:
                log(f"{r}: no return within {TIMEOUT:.0f}s ({type(exc).__name__})")
            text = console.uart_since(mark)
            transcript.append(f"===== {r} (pos {pos}) =====\n{text}\n")
            m = re.search(r"RESULT\s+" + re.escape(r) + r"\s+retval=(\d+)", text)
            got = int(m.group(1)) if m else None
            results.append((r, pos, got, oracles[r]))
            log(f"  {r}: retval={got} oracle={oracles[r]} "
                f"{'OK' if got == oracles[r] else 'MISMATCH/NO-RESULT'}")
    finally:
        print("RUN_DONE", flush=True)
        try:
            release_board(console, label="baked rungs")
        except Exception as exc:
            print("release warn:", exc, flush=True)
        pathlib.Path(OUT).write_text("".join(transcript))
        print(f"\n==== BAKED RUNGS (one boot, {len(results)} rungs) ====", flush=True)
        print(f"{'rung':10} {'pos':>4} {'retval':>8} {'oracle':>8}  verdict", flush=True)
        for r, pos, got, orc in results:
            v = "OK" if got == orc else ("NO RESULT" if got is None else "WRONG")
            print(f"{r:10} {pos:>4} {str(got):>8} {orc:>8}  {v}", flush=True)
        print(f"transcript -> {OUT}", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()
