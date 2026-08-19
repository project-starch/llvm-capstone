"""Run the board-run preflight gate before spending a boot.

WHY THIS EXISTS. `capstone/tests/preflight-board-run.sh` encodes C1-C14, every one of them a
failure that already cost board time, and NOTHING CALLED IT. It was invoked by hand, by two
repro `run.sh` scripts, and named in the board-run skill -- so it fired only when the operator
remembered, which is exactly when it is least needed.

The cost of that is not hypothetical. C2 ("distinct images") exists because three "different"
images were byte-identical when the flags never reached the build. On 2026-08-19 that recurred:
five `wbuf` arms were built with EXTRA_CFLAGS, which `build-ladder-domain.sh` does not read (it
reads DOMAIN_EXTRA_CFLAGS), all five compiled successfully to the SAME binary, and five
identical results would have read as "no effect". It was caught by hashing out of habit, not by
the gate that was written for it.

ONE COPY, IMPORTED BY BOTH RUNNERS. Three separate bugs on 2026-08-19 came from two divergent
copies of the same logic in one file; a second copy of a gate is worse than that, because the
copies can disagree about whether to block.

The gate is a SCRIPT, not a subagent, deliberately -- see its own header. This module only
decides WHEN to run it, never what it checks.
"""
import os
import pathlib
import subprocess
import sys

# repo root: .../capstone/tests/rtl-smoke/fpga_driver/preflight.py -> up 4
REPO = pathlib.Path(__file__).resolve().parents[4]
GATE = REPO / "capstone" / "tests" / "preflight-board-run.sh"


def require_preflight():
    """Run the gate. Abort the run on BLOCK. Returns None; raises SystemExit on failure.

    Called BEFORE the board is touched, so a blocked run costs no lock, no power cycle and no
    JTAG upload.

    PREFLIGHT=0 skips it, and says so LOUDLY on stderr. An override that is silent is an
    override that becomes the default; this one has to be read in the transcript afterwards.
    """
    if os.environ.get("PREFLIGHT") == "0":
        print("!! PREFLIGHT SKIPPED (PREFLIGHT=0). Every C-gate is off: distinct images, "
              "construct-in-artifact, oracle present, control-has-a-record, DTS/bitstream "
              "match, firmware freshness, slot budget. Any verdict from this boot is "
              "UNGATED -- say so when reporting it.", file=sys.stderr, flush=True)
        return
    if not GATE.is_file():
        raise SystemExit(f"preflight gate missing: {GATE}\n"
                         f"Refusing to spend a boot without it. Set PREFLIGHT=0 to override "
                         f"deliberately.")
    # cwd=REPO: the gate resolves capstone/... paths relative to the repo root.
    # The environment is passed through unchanged -- it reads the same FPGA_FW / BAKED_RUNGS /
    # SQLITE_STAGE_DOMS the caller already set, which is what keeps the two in sync.
    r = subprocess.run(["bash", str(GATE)], cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(
            "\npreflight BLOCKED -- not spending a boot.\n"
            "Fix the checks above, or re-run with PREFLIGHT=0 if you have read them and are "
            "overriding deliberately. Do NOT weaken a check to make a run start.")
