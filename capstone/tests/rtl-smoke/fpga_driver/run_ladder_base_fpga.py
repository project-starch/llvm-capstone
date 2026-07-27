#!/usr/bin/env python3
"""Board runner for the silicon-ladder BASELINE sweep (plain RISC-V denominator).

Companion to run_ladder_perf_fpga.py, which runs each rung as a pure-capability
Capstone domain. This runs the SAME kernels as ordinary RISC-V userspace code and
reports the same counter delta, so the pair prices the capability ABI plus its
hardware enforcement (see build-ladder-base-fpga.sh for why both halves must come
from the same compiler at the same -O).

ONE BOOT, not one per rung. The perf runner power-cycles per rung (~2.5 min each)
because every domain must be the FIRST domain of a clean boot -- a second domain at
the same VA hangs on a stale icache. This runner creates no domain and never opens
/dev/capstone, so that constraint does not apply: a single cold boot, a single
binary transfer, then all seven rungs back to back.

COUNTER PROBES COME FIRST, and each is its own process invocation. The domain half
reads the M-mode `mcycle`; userspace must use the U-mode `cycle` mirror, which
traps if counteren gates it, and there is no libc here to catch SIGILL. So the run
probes cycle/time/instret separately -- a trap kills only that invocation, the boot
survives, and the sweep then uses the first counter that actually read. Never
assume the counter is available just because QEMU allowed it.

Board etiquette matches the perf runner: lock, verify the resident bitstream,
power off + unlock in `finally`. It never flashes a bitstream.
"""
import os, pathlib, re, sys, time

DRV = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(DRV.parent))
from fpga_driver import config as C
from fpga_driver.fpga_console import FpgaConsole, ActionTimeout
from fpga_driver.run_rtl_smoke import login_root, POWER_ON_SETTLE, POWER_CYCLE_OFF
from fpga_driver.fast_xfer import fast_put
from fpga_driver.run_ladder_perf_fpga import (
    URL, IMG, IMG_NAME, BITSTREAM, log, sha16, gzip_file, nvbit,
    wait_connected, install_resilient_emit, sh,
)
from socketio.exceptions import BadNamespaceError

ART = pathlib.Path(os.environ.get("LADDER_BASE_DIR") or
                   os.path.join(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone"),
                                "ladder-base"))
RUNGS = (os.environ.get("LADDER_RUNGS") or
         "null matmult_int coremark_matrix rv8_primes beebs_crc32 "
         "beebs_insertsort beebs_prime beebs_recursion").split()
# `null` is the empty-compute control; it has no kernel source and so no host
# oracle binary. Its expected retval is a zeroed volatile.
NO_ORACLE = {"null": "0"}
CAPTURE = "/tmp/capstone/board-run-ladder-base.uart.txt"
RESULTS_OUT = "/tmp/capstone/ladder-base-results.txt"
CTL_REMOTE = "/tmp/lbc"
COUNTERS = ("cycle", "time", "instret")


def cold_boot_plain(console, prompt):
    """Power-cycle + JTAG firmware reload -> fresh root shell.

    Deliberately does NOT insmod capstone.ko: the baseline touches no capability
    machinery, so requiring the module would add a failure mode that cannot affect
    the measurement. Otherwise identical to the perf runner's cold_boot, including
    the fact that a warm `monitor reset halt` does not work here (the fw_payload
    OpenSBI cannot re-run its one-time hart/DDR init).
    """
    console.power(False); time.sleep(POWER_CYCLE_OFF)
    console.power(True); time.sleep(POWER_ON_SETTLE)
    console.gdb_start()
    try:
        console.gdb_cmd("monitor reset halt", prompt, timeout=60.0)
        time.sleep(4.0)
        console.gdb_cmd(f"monitor load_image images/{IMG_NAME} 0x80000000 bin",
                        prompt, timeout=300.0)
        console.gdb_cmd("set $pc = 0x80000000", prompt)
        console.gdb_cmd("set $a0 = 0", prompt)
        boot_start = len(console.uart_text)
        console._emit("gdb_input", text="continue\n")
        if not login_root(console, search_from=boot_start):
            raise RuntimeError("cold boot: reached boot but no shell")
    finally:
        console.gdb_stop()
    time.sleep(4.0)
    sh(console, "echo 1 > /proc/sys/kernel/printk")


def probe_counters(console):
    """Return the counters that can actually be read from userspace, in order."""
    ok = []
    for c in COUNTERS:
        try:
            out = console.run_command(
                f"echo P''RB{c}; {CTL_REMOTE} probe {c}; echo P''RE{c}=$?",
                rf"PRE{c}=\d+", timeout=30)
        except ActionTimeout:
            log(f"  probe {c}: no marker (timeout) -- treating as unavailable")
            continue
        m = re.search(rf"BASE PROBE {c} ok v=(\d+) delta=(\d+)", out)
        if m:
            log(f"  probe {c}: OK (v={m.group(1)} delta={m.group(2)})")
            ok.append(c)
        else:
            # A gated CSR traps; busybox reports it and the exit status is nonzero.
            why = "illegal instruction" if "llegal" in out else "no OK line"
            log(f"  probe {c}: UNAVAILABLE ({why})")
    return ok


def main():
    ctl = ART / "ladder_base_ctl"
    if not ctl.is_file():
        raise SystemExit(f"missing {ctl} -- run build-ladder-base-fpga.sh first")
    oracles = {}
    for r in RUNGS:
        if r in NO_ORACLE:
            oracles[r] = NO_ORACLE[r]
            continue
        f = ART / f"{r}_host"
        if not f.is_file():
            raise SystemExit(f"missing oracle binary {f} -- run the build script")
        oracles[r] = os.popen(str(f)).read().strip()

    console = FpgaConsole(URL, logger=lambda m: print(f"[fpga] {m}", file=sys.stderr))
    console.connect()
    install_resilient_emit(console)
    locked = False
    results = {}
    counter_used = None
    try:
        log(f"users connected: {console.user_count()}")
        st = console._current_state(C.GDB_STATE_EVENT)
        if st in ("running", "starting", "error"):
            log(f"stopping stale gdb session ({st})")
            console.gdb_stop(); time.sleep(3)
        console.lock(); locked = True
        rb = nvbit(console)
        log(f"took the lock; resident NV bitstream = {rb!r}")
        if rb != BITSTREAM:
            # Flashing is a hard stop that needs explicit human authorization, and
            # the baseline does not need any particular bitstream badly enough to
            # take one. Stop rather than reprogram a shared board.
            raise RuntimeError(
                f"resident bitstream {rb!r} != expected {BITSTREAM!r}; refusing to "
                f"flash. Re-run after the board owner restores it.")

        prompt = C.GDB_PROMPT
        log(f"uploading {IMG_NAME} to store ({IMG.stat().st_size} bytes)")
        console.upload_boot_image(IMG_NAME, str(IMG))

        # Boot + transfer as one retryable unit. fast_put over the board's
        # UART/websocket wedges non-deterministically (documented in the 25-07
        # sweep note), and once the guest shell is wedged the only reliable
        # recovery is a fresh boot -- so retry the pair, not the transfer alone.
        # cold_boot_plain is idempotent, so a second attempt costs a reload and
        # nothing else.
        ctl_gz, ctl_sha = gzip_file(ctl), sha16(ctl)
        for boot_attempt in range(1, 4):
            log(f"cold boot (attempt {boot_attempt}; single boot for the whole sweep)")
            cold_boot_plain(console, prompt)
            try:
                fast_put(console, ctl_gz, "/tmp/lbc.gz", CTL_REMOTE, ctl_sha, True, log)
                break
            except (RuntimeError, ActionTimeout, BadNamespaceError) as e:
                log(f"  transfer failed ({e}); re-booting and retrying")
                if not getattr(console.sio, "connected", False):
                    wait_connected(console, 45)
        else:
            raise RuntimeError("controller transfer failed on every boot attempt")

        avail = probe_counters(console)
        # The sweep reads cycle AND instret around every pass, so both must work.
        # Failing here costs a re-run; failing mid-sweep kills the boot with a
        # SIGILL there is no handler for.
        missing = [c for c in ("cycle", "instret") if c not in avail]
        if missing:
            raise RuntimeError(
                f"counter(s) {', '.join(missing)} unreadable from userspace "
                f"(available: {', '.join(avail) or 'none'}); the sweep needs both")
        counter_used = "cycle+instret"
        log(f"counters OK (available: {', '.join(avail)})")

        # All seven in one invocation: no domain, so no per-rung boot needed.
        for attempt in range(1, 4):
            try:
                # No counter argument: `all` reads cycle AND instret around every
                # pass (run_pass), so there is nothing to select. Passing the
                # report label "cycle+instret" here made the controller reject it
                # as a counter name and cost a boot.
                out = console.run_command(
                    f"echo A''LLB; {CTL_REMOTE} all; echo A''LLE=$?",
                    r"ALLE=\d+", timeout=300)
            except ActionTimeout:
                log(f"  sweep: no END marker (attempt {attempt})"); time.sleep(2); continue
            except BadNamespaceError:
                log("  sweep: socket dropped; reconnecting"); wait_connected(console, 45); continue
            for r in RUNGS:
                passes = {}
                for m in re.finditer(
                        rf"BASE RESULT {r} pass=(\d+) retval=(\d+) "
                        rf"cycles=(\d+) instret=(\d+)", out):
                    passes[int(m.group(1))] = (m.group(2), int(m.group(3)),
                                               int(m.group(4)))
                if 1 in passes and 2 in passes:
                    results[r] = passes
                    (rv1, c1, i1), (rv2, c2, i2) = passes[1], passes[2]
                    idem = "" if rv2 == rv1 else "  [stateful: warm pass invalid]"
                    # Pick the LEAST-DISTURBED warm pass, not merely the second
                    # one (issue I-2). Timer interrupts land inside the bracket
                    # and inflate both counters; they arrive on a timer rather
                    # than in step with the kernel, so across many passes the
                    # minimum instret is the run that took fewest of them, and
                    # the number of passes tied at that minimum is the evidence
                    # that it is genuinely clean. The old check -- pass1 instret
                    # == pass2 instret -- only proved reproducibility, which
                    # beebs_cnt satisfied while still yielding a 0.684x ratio.
                    warm = {p: v for p, v in passes.items() if p >= 2}
                    if rv2 != rv1:          # stateful kernel: warm passes invalid
                        warm = {}
                    if warm:
                        best_i = min(v[2] for v in warm.values())
                        tied = [v for v in warm.values() if v[2] == best_i]
                        best_c = min(v[1] for v in tied)
                        spread = max(v[1] for v in warm.values()) - \
                            min(v[1] for v in warm.values())
                        clean = f"{len(tied)}/{len(warm)} passes at min instret"
                        log(f"  {r}: BEST cycles={best_c} instret={best_i} "
                            f"({clean}, cycle spread={spread}) retval={rv1}{idem}")
                        results[r]["best"] = (rv1, best_c, best_i,
                                              len(tied), len(warm), spread)
                    log(f"  {r}: cold={c1} warm={c2} (delta={c1 - c2}) "
                        f"instret={i1}/{i2} retval={rv1}{idem}")
            if len(results) == len(RUNGS):
                break
            log(f"  sweep: only {len(results)}/{len(RUNGS)} rungs parsed (attempt {attempt})")
    finally:
        if not getattr(console.sio, "connected", False):
            try: console.connect(); log("reconnected for cleanup")
            except Exception as e: log(f"cleanup reconnect err: {e}")
        try: console.power(False); log("powered off")
        except Exception as e: log(f"power off err: {e}")
        try:
            if locked: console.unlock(); log("unlocked")
        except Exception as e: log(f"unlock err: {e}")
        try:
            pathlib.Path(CAPTURE).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(CAPTURE).write_text(console.uart_text)
            log(f"UART capture -> {CAPTURE} ({len(console.uart_text)} chars)")
        except Exception as e: log(f"capture save err: {e}")
        console.close()

    lines = [f"counters = {counter_used}",
             "cold = first call (pays Linux demand-paging); warm = immediate repeat.",
             "The domain half has no paging, so WARM is the comparable column --",
             "but only where the kernel is idempotent (idem=YES).",
             "",
             f"{'rung':<18} {'oracle':<12} {'cold_cyc':>10} {'warm_cyc':>10} "
             f"{'delta':>9} {'cold_ins':>10} {'warm_ins':>10}  idem correct"]
    allok = True
    for r in RUNGS:
        p = results.get(r)
        if not p:
            lines.append(f"{r:<18} {oracles[r]:<12} {'--':>10} {'--':>10} "
                         f"{'--':>9} {'--':>10} {'--':>10}  --   NO")
            allok = False
            continue
        (rv1, c1, i1), (rv2, c2, i2) = p[1], p[2]
        ok = (rv1 == oracles[r])
        allok = allok and ok
        lines.append(f"{r:<18} {oracles[r]:<12} {c1:>10} {c2:>10} {c1 - c2:>9} "
                     f"{i1:>10} {i2:>10}  {'YES' if rv2 == rv1 else 'no ':<4} "
                     f"{'YES' if ok else 'NO'}")
    report = "\n".join(lines)
    print("\n==== silicon-ladder FPGA BASELINE (plain RISC-V) ====\n" + report)
    pathlib.Path(RESULTS_OUT).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(RESULTS_OUT).write_text(report + "\n")
    log(f"results -> {RESULTS_OUT}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
