#!/usr/bin/env python3
"""DEPRECATED 2026-08-03 -- this driver ships the program to the board over the UART
console. Do not use it for new work, and do not extend it.

Use the BAKED path instead: put the artifact in
`caplifive-system/sw/buildroot/overlay/test-domains/` AND `build/target/test-domains/`,
rebuild with `A=linux-rebuild` then `A=opensbi-rebuild`, and invoke it from the shell
with `run_baked_rungs_fpga.py` (ladder rungs) or `run_sqlite_stages_fpga.py` (SQLite /
staged probes). Rationale: UART delivery is 16 chars per socket.io emit and each emit is
an HTTPS round trip, so a ~10 KB domain costs MINUTES; baked, 10 domains run in ONE boot
in ~5 minutes because the JTAG upload happens anyway. See
`agent-handoff/ref/HOW-TO-LAUNCH-ON-FPGA.md` s"UART TRANSFER IS RETIRED".
"""
"""Run the silicon-ladder perf rungs on the Capstone CVA6 FPGA in ONE session.

Boots the working firmware prebuilt (fw_payload_fpga_up_gpfree.bin; embedded
FDT+kernel, /dev/capstone at boot -- do NOT rebuild the monitor), transfers the
generic controller + each rung's perf .dom (gzip+base64, per-chunk sha), runs
`ladder_perf_ctl <name> <rung.dom>` for each rung, and harvests the RESULT line
(retval + mcycle). Correctness gate: retval == the native cc -O0 oracle.

Board etiquette (non-negotiable): verifies the resident bitstream is
working-caplifive-captype-fixed.bit before measuring; ALWAYS powers off + unlocks
in finally; never leaves the board locked/on. The board URL/token is read from
~/.config/capstone/fpga-board-url at runtime (secret; never embedded/echoed).

Adapted from the proven /tmp/capstone/board_run_gpfree.py (the 554745961 silicon
run). Artifacts come from build-ladder-fpga.sh's OUT_DIR (default
$CAPSTONE_TMP_ROOT/ladder-fpga): ladder_perf_ctl, <rung>.dom, <rung>.oracle.
"""
import os
import sys, os, time, base64, hashlib, re, pathlib, gzip, subprocess

DRV = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(DRV.parent))          # .../rtl-smoke (for `import fpga_driver`)
from fpga_driver import config as C
from fpga_driver.fpga_console import FpgaConsole, ActionTimeout
from fpga_driver.run_rtl_smoke import login_root, POWER_ON_SETTLE, POWER_CYCLE_OFF
from fpga_driver.fast_xfer import fast_put
from socketio.exceptions import BadNamespaceError

# The console URL embeds the access token in its PATH, so it is a secret: it must
# never be committed, echoed into a captured log, or persisted. The project rule is
# to keep it in an env var for the duration of a run, so FPGA_URL wins; the dotfile
# is only a fallback for setups that already had one. Prefer FPGA_URL.
def _board_url():
    u = os.environ.get("FPGA_URL")
    if u:
        return u.strip()
    p = os.path.expanduser("~/.config/capstone/fpga-board-url")
    if os.path.exists(p):
        return open(p).read().strip()
    raise SystemExit("no board URL: export FPGA_URL='https://<host>/<token>/' "
                     "for this run (do not commit or persist it)")

URL = _board_url()
# FPGA_FW overrides the firmware, so a run can exercise a freshly built monitor
# instead of the checked-in prebuilt. Needed for anything that depends on a monitor
# change (the globals-offset unpacking, the loud capstone_error).
IMG = pathlib.Path(os.environ.get("FPGA_FW") or
                   os.path.expanduser("~/capstone-b-artifacts/fw_payload_fpga_up_gpfree.bin"))
IMG_NAME = os.environ.get("FPGA_FW_NAME") or "fw_payload_fpga_up_gpfree.bin"
# Resident-bitstream guard. Reflashed 2026-08-04 to caplifive_fixed_forward.bit, which
# carries the operand-forwarding fix (capstone-ariane 7aac52f93). Overridable so the next
# reflash needs no code change mid-session -- the guard exists to stop a run from silently
# measuring the WRONG silicon, and on 2026-08-04 it did exactly that.
BITSTREAM = os.environ.get("FPGA_BITSTREAM", "caplifive_fixed_forward.bit")

# Must match build-ladder-fpga.sh's OUT_DIR default ($CAPSTONE_TMP_ROOT/ladder-fpga),
# else the runner reads a different dir than the build writes and can pick up stale
# doms (the 2026-07-25 incident). preflight_artifacts() also passes OUT_DIR=ART.
ART = pathlib.Path(os.environ.get("LADDER_FPGA_DIR") or
                   os.path.join(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone"),
                                "ladder-fpga"))
ONE_BOOT = os.environ.get("LADDER_ONE_BOOT") == "1"
RUNGS = (os.environ.get("LADDER_RUNGS") or
         "matmult_int coremark_matrix rv8_primes beebs_crc32 "
         "beebs_insertsort beebs_prime beebs_recursion").split()
CAPTURE = "/tmp/capstone/board-run-ladder-perf.uart.txt"
RESULTS_OUT = "/tmp/capstone/ladder-perf-results.txt"
CTL_REMOTE = "/tmp/lpc"
# Seconds to wait for a rung's END marker. Threaded into the timeout AND the log
# message: they used to be 75 and "120s", so every wedge was mis-reported.
EXEC_TIMEOUT = int(os.environ.get("LADDER_EXEC_TIMEOUT") or 75)
# Abort on UART silence, same as the SQLite runner. A wedged rung in a multi-rung sweep
# otherwise burns the full EXEC_TIMEOUT per rung; with 6 rungs that is minutes of holding a
# shared board waiting for output that will never come. Progress resets the clock.
EXEC_IDLE = int(os.environ.get("LADDER_EXEC_IDLE") or 40)
# See the exec loop: a wedged board never recovers on a second attempt.
EXEC_ATTEMPTS = int(os.environ.get("LADDER_EXEC_ATTEMPTS") or 1)

def log(m): print(f"[run] {m}", file=sys.stderr, flush=True)
def sha16(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]

def gzip_file(src):
    dst = str(src) + ".gz"
    with open(src, "rb") as f, gzip.open(dst, "wb") as g:
        g.write(f.read())
    return dst

def nvbit(console, poll=8.0):
    end = time.time() + poll
    while True:
        with console._cond:
            fs = console._state.get("flash_state") or {}
        v = fs.get("nv_bitstream_name")
        if v is not None or time.time() >= end:
            return v
        time.sleep(0.5)

def wait_connected(console, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if getattr(console.sio, "connected", False):
            return True
        time.sleep(1.0)
    return False

def install_resilient_emit(console):
    """Wrap _emit so a websocket drop (BadNamespaceError) reconnects + retries.
    The board's socket drops mid-transfer; every run_command/send_file emit goes
    through here, so this makes the whole session survive a drop."""
    orig = console._emit
    def resilient(event, **kw):
        for _ in range(6):
            try:
                return orig(event, **kw)
            except BadNamespaceError:
                log("socket dropped mid-emit; waiting for reconnect")
                if not wait_connected(console, 45):
                    try: console.connect()
                    except Exception as e: log(f"explicit reconnect failed: {e}")
                time.sleep(1.0)
        return orig(event, **kw)
    console._emit = resilient

# The board's UART RX FIFO silently drops characters on a long bulk write. On 2026-08-05 a
# 204-char setup line arrived as `/proc/sys/kernel/ppstone ] || insmod ...` -- 20 characters
# vanished mid-word, the shell reported `can't create .../ppstone`, printk was never quieted,
# and the run died with no useful diagnosis. The console logged `ttyS0: 1 input overrun(s)`.
# Keep every typed line short, and FAIL LOUDLY rather than let a truncated command look like a
# board fault.
SH_MAX_CHARS = 120

def sh(console, cmd, timeout=20):
    line = f"{cmd}; echo D''N_$?"
    if len(line) > SH_MAX_CHARS:
        raise ValueError(
            f"command is {len(line)} chars, over the {SH_MAX_CHARS}-char UART safety limit; "
            f"the RX FIFO drops characters and the command arrives CORRUPTED. Split it.\n  {cmd}")
    return console.run_command(line, r"DN_\d+", timeout=timeout)

def quiet_setup(console, cmds, timeout=30):
    """Run SHORT setup commands with echo suppressed, then restore echo.

    Echo is toggled by its OWN short command rather than being chained onto the setup line.
    The chained version built a 204-char line, which the UART RX FIFO truncated mid-word
    (2026-08-05). With echo off first, the following commands produce no echo anyway, so this
    is both safer and quieter than the one-liner it replaces.

    `stty echo` is restored in a `finally`, so a failure in any setup command cannot leave the
    terminal silent. Best-effort throughout: an image with no `stty` just stays verbose.
    """
    def _quiet(c):
        try:
            sh(console, c, timeout=15)
        except Exception as exc:
            log(f"{c!r} unavailable ({type(exc).__name__}); console stays verbose")
    _quiet("stty -echo 2>/dev/null || true")
    out = ""
    try:
        for c in cmds:
            out += sh(console, c, timeout=timeout)
    finally:
        _quiet("stty echo 2>/dev/null || true")
    return out


def ensure_capstone_dev(console):
    """insmod capstone.ko (the UP image does not auto-load it) then verify the device
    exists. Gate on the trailing DEVNO/DEVOK token, NOT substring presence.

    printk quieting and the insmod are ONE command with ONE marker: they used to be two
    `sh()` calls, i.e. two echoed command lines and two DN_ markers, for no diagnostic
    gain -- if the device is missing the DEVNO branch already says so. Its echo is
    suppressed for THIS command only; per-test commands stay visible on the console."""
    out = quiet_setup(console, [
        "echo 1 > /proc/sys/kernel/printk",
        "[ -e /dev/capstone ] || insmod /capstone.ko 2>/dev/null",
        "[ -e /dev/capstone ] && echo DEV''OK || echo DEV''NO",
    ], timeout=30)
    if "DEVNO" in out or "DEVOK" not in out:
        raise RuntimeError(f"/dev/capstone missing (insmod failed?):\n{out}")
    drop_setup_log(console)


def drop_setup_log(console):
    """Wipe the harness's own setup exchanges from the OPERATOR's console.

    Three exchanges are pure harness bookkeeping and carry no information about the run:
    the login readiness probe (`echo RDYOK` / `RDYOK`), the printk quieting, and the
    insmod/device check (`DEVOK` / `DN_0`). They must still be TYPED and MATCHED -- they are
    the drivers' synchronisation contract -- so they cannot simply be deleted, and the login
    probe's echo cannot be suppressed at all (any `stty -echo` would itself echo first).

    They can, however, be dropped once they have served their purpose. `uart_clear` empties
    the board server's ring buffer, which is what the browser console renders; the driver
    matches against its OWN accumulated `_uart` string, which is untouched. So this changes
    only what a human sees, never what the driver sees.

    Deliberately NOT a blanket `stty -echo`: an earlier attempt did that and silenced every
    per-test command line as well, which is exactly the information the console is watched
    for. Set BOARD_KEEP_SETUP_LOG=1 to keep the setup chatter visible when debugging the
    harness itself."""
    if os.environ.get("BOARD_KEEP_SETUP_LOG") == "1":
        return
    try:
        console._emit("uart_clear")
    except Exception as exc:                       # never let cosmetics break a run
        log(f"uart_clear unavailable ({type(exc).__name__}); setup chatter stays visible")

def cold_boot(console, prompt, img_name=None):
    """Full power-cycle + JTAG firmware reload -> fresh boot. Each rung runs as the
    FIRST domain of a clean boot (clean icache). A warm `monitor reset halt` does
    NOT work here -- the fw_payload OpenSBI does not re-enter cleanly from a soft
    reset (its one-time hart/DDR init is not re-runnable), so a real power-cycle is
    required. The image must already be in the console store (upload_boot_image);
    load_image JTAG-copies store->DDR (~2 min). Returns with a confirmed root shell
    and /dev/capstone loaded (the initramfs is re-unpacked, so /tmp starts empty).

    PASS `img_name` EXPLICITLY when the caller uploaded under its own name. This function
    lives in THIS module, so a bare `IMG_NAME` resolves to THIS module's global -- the
    hardcoded ladder image below -- regardless of which runner calls it. That cost a whole
    session on 2026-07-30: run_sqlite_baked_fpga.py computes a CONTENT-HASHED name, uploaded
    its 17.4 MB firmware under it, and then cold_boot JTAG-loaded the ladder's stale stored
    image instead. The board booted a JULY 19 initramfs, so the freshly staged
    sqlite_host.user was absent and the shell answered `not found` (exit 127) -- which reads
    as a domain failure and sent the investigation into the monitor. It had worked until then
    only because callers exported FPGA_FW_NAME, which BOTH modules happen to read from the
    environment; a content-hash default silently broke that coupling."""
    name = img_name or IMG_NAME
    console.power(False); time.sleep(POWER_CYCLE_OFF)
    console.power(True); time.sleep(POWER_ON_SETTLE)
    console.gdb_start()
    try:
        console.gdb_cmd("monitor reset halt", prompt, timeout=60.0)
        time.sleep(4.0)
        console.gdb_cmd(f"monitor load_image images/{name} 0x80000000 bin",
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
    ensure_capstone_dev(console)

def send_file(console, local_gz, remote_gz, remote_bin, bin_sha, do_exec, chunk=200):
    b64 = base64.b64encode(open(local_gz, "rb").read()).decode()
    log(f"transfer {remote_bin}: {len(b64)} b64 chars in {(len(b64)+chunk-1)//chunk} chunks")
    sh(console, f": > {remote_gz}")
    for i in range(0, len(b64), chunk):
        piece = b64[i:i+chunk]
        want = hashlib.sha256(piece.encode()).hexdigest()[:16]
        for attempt in range(6):
            out = console.run_command(
                f"printf %s '{piece}' > /tmp/part; echo H$(sha256sum /tmp/part | cut -c1-16)H",
                r"H[0-9a-f]{16}H", timeout=20)
            m = re.search(r"H([0-9a-f]{16})H", out)
            if m and m.group(1) == want:
                sh(console, f"cat /tmp/part >> {remote_gz}")
                break
        else:
            raise RuntimeError(f"chunk @{i} failed after retries for {remote_bin}")
    out = console.run_command(
        f"base64 -d {remote_gz} | gunzip -c > {remote_bin}; echo S$(sha256sum {remote_bin} | cut -c1-16)S",
        r"S[0-9a-f]{16}S", timeout=30)
    m = re.search(r"S([0-9a-f]{16})S", out)
    if not m or m.group(1) != bin_sha:
        raise RuntimeError(f"{remote_bin} decompressed sha {m.group(1) if m else None} != {bin_sha}\n{out}")
    if do_exec:
        sh(console, f"chmod 0755 {remote_bin}")
    log(f"  {remote_bin} OK (sha {bin_sha})")

def preflight_artifacts():
    """Never feed the board a stale domain binary.

    A 2026-07-25 sweep ran pre-fix `.dom` files (built before the sub-cap memcpy
    fix) and reported 4 bogus "silicon miscompiles" that were actually an
    already-fixed compiler bug — because this runner used to reuse an existing
    `.dom` and never rebuild it. So by default we REBUILD every artifact against
    the current compiler via build-ladder-fpga.sh, and regardless of that we
    HARD-FAIL if any artifact is missing or older than the compiler binary.

    Env knobs: LADDER_REBUILD=0 skips the rebuild (then the freshness check must
    pass on the pre-built set); CAPSTONE_CLANG is used for the freshness compare.
    """
    build_sh = DRV.parent / "build-ladder-fpga.sh"
    if os.environ.get("LADDER_REBUILD", "1") != "0":
        if not build_sh.is_file():
            raise SystemExit(f"cannot rebuild: {build_sh} missing "
                             f"(set LADDER_REBUILD=0 to run a pre-built set)")
        log(f"rebuilding ladder artifacts with the current compiler -> {ART}")
        subprocess.run(["bash", str(build_sh), *RUNGS], check=True,
                       env=dict(os.environ, OUT_DIR=str(ART)))

    arts = [ART / "ladder_perf_ctl"] + [ART / f"{r}.dom" for r in RUNGS]
    missing = [str(a) for a in arts if not a.is_file()] + \
              [str(ART / f"{r}.oracle") for r in RUNGS
               if not (ART / f"{r}.oracle").is_file()]
    if missing:
        raise SystemExit("artifacts missing (run build-ladder-fpga.sh):\n  " +
                         "\n  ".join(missing))

    # -O-level cross-check (issue I-1). The capability and baseline halves must be
    # built at the SAME -O or the ratio measures optimisation, not capabilities.
    # On 2026-07-27 they silently were not: this runner REBUILDS by default, and
    # LADDER_OPT had been set only on a pre-build, so five rungs ran at -O0 against
    # -O1 baselines. That produced five bogus "silicon failures", a false
    # refutation of R-1 that was nearly sent to the board owner, and a nearly
    # published claim that a plain rebuild flips a passing rung. Only an unrelated
    # control rung caught it. Fail loudly instead.
    def _optlevels(path):
        try:
            return dict(l.split() for l in path.read_text().split("\n") if l.strip())
        except OSError:
            return {}

    mine = _optlevels(ART / "optlevels.txt")
    if mine:
        log("opt levels: " + ", ".join(f"{r}={mine.get(r, '?')}" for r in RUNGS))
    # Prefer the BARE-METAL baseline: it is the real denominator now (issue I-2).
    # The Linux dir is kept only as a fallback for older artifact sets -- checking
    # it first made this guard fire against a stale -O0 file and block a correctly
    # configured -O1 sweep.
    _root = pathlib.Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone"))
    base = {}
    for _d in ("ladder-base-bare", "ladder-base"):
        base = _optlevels(_root / _d / "optlevels.txt")
        if base:
            log(f"comparing -O levels against {_d}")
            break
    mismatch = [f"{r}: capability={mine[r]} baseline={base[r]}"
                for r in RUNGS if r in mine and r in base and mine[r] != base[r]]
    if mismatch:
        raise SystemExit(
            "-O LEVEL MISMATCH between the capability and baseline halves — the\n"
            "ratio would measure optimisation, not capabilities (issue I-1):\n  " +
            "\n  ".join(mismatch) +
            "\nSet LADDER_OPT on THIS command (the runner rebuilds; setting it only\n"
            "on a pre-build is discarded), or rebuild the baseline to match.")

    cc = os.environ.get("CAPSTONE_CLANG")
    if cc and os.path.exists(cc):
        cc_mtime = os.path.getmtime(os.path.realpath(cc))
        stale = [str(a) for a in arts if os.path.getmtime(a) < cc_mtime]
        if stale:
            raise SystemExit(
                "STALE domain artifacts OLDER than the compiler — would run "
                "pre-fix code (see the 2026-07-25 stale-dom incident):\n  " +
                "\n  ".join(stale) +
                f"\ncompiler: {os.path.realpath(cc)}\n"
                "Rebuild them (default) or delete them; do NOT run stale on the board.")
    else:
        log("WARNING: CAPSTONE_CLANG unset/absent — cannot verify dom freshness")


def main():
    preflight_artifacts()
    oracles = {r: open(ART / f"{r}.oracle").read().strip() for r in RUNGS}
    ctl = ART / "ladder_perf_ctl"

    console = FpgaConsole(URL, logger=lambda m: print(f"[fpga] {m}", file=sys.stderr))
    console.connect()
    install_resilient_emit(console)
    locked = False
    gdb_up = False
    results = {}
    debug_lines = {}      # rung -> raw per-probe values (gp_diag only)
    instret_lines = {}    # rung -> (retired instructions, minstret phase)
    try:
        users = console.user_count()
        log(f"users connected: {users}")
        st = console._current_state(C.GDB_STATE_EVENT)
        if st in ("running", "starting", "error"):
            log(f"stopping stale gdb session ({st})")
            console.gdb_stop(); time.sleep(3)
        console.lock(); locked = True
        rb = nvbit(console)
        log(f"took the lock; resident NV bitstream = {rb!r}")
        if rb != BITSTREAM:
            # HARD STOP. Flashing reprograms a SHARED physical board and is the one
            # persistent write this console exposes, so it needs an explicit human go
            # every time -- "authorized restore" was too weak a justification to do it
            # unattended mid-sweep. Opt in per run with FPGA_ALLOW_FLASH=1.
            if os.environ.get("FPGA_ALLOW_FLASH") != "1":
                raise SystemExit(
                    f"HARD STOP: resident bitstream is {rb!r}, expected {BITSTREAM!r}.\n"
                    "Re-flashing is NOT done automatically -- it reprograms a shared board.\n"
                    "Ask the user, then re-run with FPGA_ALLOW_FLASH=1 if they approve.")
            log(f"resident bitstream {rb!r} != {BITSTREAM!r}; re-flashing (explicitly approved)")
            console.power(True); time.sleep(POWER_ON_SETTLE)
            console.flash_bitstream(BITSTREAM)
            console.power(False); time.sleep(POWER_CYCLE_OFF)
            console.power(True); time.sleep(POWER_ON_SETTLE)
            rb = nvbit(console)
            if rb != BITSTREAM:
                raise RuntimeError(f"after flash+power-cycle bitstream is {rb!r}, not {BITSTREAM!r}")
            log(f"verified resident bitstream = {rb!r}")
            console.power(False); time.sleep(POWER_CYCLE_OFF)

        # Upload the fw_payload to the console image STORE once (HTTP; persists
        # across power-cycles). cold_boot() JTAG-reloads store->DDR per rung.
        prompt = C.GDB_PROMPT
        log(f"uploading {IMG_NAME} to store ({IMG.stat().st_size} bytes)")
        console.upload_boot_image(IMG_NAME, str(IMG))

        ctl_gz = gzip_file(ctl)
        ctl_sha = sha16(ctl)
        booted_once = [False]
        # Does the guest currently hold THIS controller binary? Cleared by any
        # cold boot (which wipes /tmp) so the controller is re-sent exactly once
        # per boot instead of once per rung. See the transfer block below.
        ctl_present = [False]
        # One rung per FULL power-cycle + reload: each domain runs as the first of
        # a clean boot, sidestepping the multi-domain same-VA icache hang. ~2.5 min
        # per rung (the JTAG reload dominates); the domains themselves are tiny.
        for r in RUNGS:
            # one-boot mode: a wedged rung must not poison the rest of the sweep
            dom = ART / f"{r}.dom"
            dom_remote = f"/tmp/{r}.dom"
            # Boot + transfer is one RETRYABLE unit. A single GDB timeout during
            # `monitor reset halt`, or a fast_put wedge, used to skip the rung
            # outright -- on 26-07 that silently dropped the beebs_prime_noins
            # CONTROL, which was the whole point of that session. cold_boot is
            # idempotent, so a retry costs a reload and nothing else; losing a
            # rung costs the run's conclusion.
            dom_gz = gzip_file(dom)
            dom_sha = sha16(dom)
            for boot_attempt in range(1, 4):
                try:
                    # LADDER_ONE_BOOT=1: boot ONCE for the whole sweep instead of
                    # power-cycling per rung. Only valid when the rungs are linked
                    # at DISTINCT entry VAs (DOMAIN_BASE_VA), because R-3 hangs a
                    # second domain reused at the SAME VA within one boot. The
                    # power-cycle is ~2.5 min and dominates every sweep, so this is
                    # the single biggest throughput lever available -- but it is
                    # opt-in, because if R-3 turns out not to be address-keyed the
                    # failure mode is a silent hang that looks like a rung result.
                    if ONE_BOOT and booted_once[0]:
                        log(f"[{r}] reusing the existing boot (LADDER_ONE_BOOT=1)")
                    else:
                        log(f"[{r}] power-cycle + reload firmware "
                            f"(attempt {boot_attempt})")
                        cold_boot(console, prompt)
                        booted_once[0] = True
                        ctl_present[0] = False   # a cold boot wipes the guest's /tmp
                    # THE CONTROLLER SURVIVES A REUSED BOOT, so re-sending it every rung
                    # was pure waste: it is the LARGER of the two transfers (2,808 base64
                    # chars vs ~1,950 for a domain) and it is byte-identical across the
                    # whole sweep. Measured over three sessions, re-transferring it cost
                    # 527-914 s -- 24-41% of total wall-clock -- because a transfer that
                    # loses a character restarts the WHOLE file and then drops to the
                    # 1-char tier. Sending it once per boot removes that exposure
                    # entirely for every rung after the first.
                    # Guarded on ctl_present rather than on ONE_BOOT so the invariant is
                    # "the guest has this exact controller", which a cold boot clears.
                    # fast_put's whole-file SHA-256 gate is untouched; if the controller
                    # is somehow absent or corrupt the rung fails loudly at exec rather
                    # than measuring a stale binary.
                    if not ctl_present[0]:
                        fast_put(console, ctl_gz, "/tmp/lpc.gz", CTL_REMOTE, ctl_sha, True, log)
                        ctl_present[0] = True
                    else:
                        log(f"  {r}: controller already on the guest, skipping transfer")
                    fast_put(console, dom_gz, f"/tmp/{r}.gz", dom_remote,
                             dom_sha, False, log)
                    break
                except (ActionTimeout, BadNamespaceError, RuntimeError) as e:
                    log(f"  {r}: boot/transfer failed ({e}); retrying")
                    if not getattr(console.sio, "connected", False):
                        wait_connected(console, 45)
            else:
                results[r] = (None, None, None)
                booted_once[0] = False   # force a fresh boot for the next rung
                ctl_present[0] = False   # ...which wipes /tmp, so re-send the ctl
                log(f"  {r}: boot/transfer failed on every attempt; skipping")
                if not getattr(console.sio, "connected", False):
                    wait_connected(console, 45)
                continue
            # As the first domain of a fresh boot, a rung either returns quickly
            # (matmult was <1 ms) or the cscall genuinely hangs -- retrying the same
            # first-domain condition won't help, so keep the budget tight.
            #
            # ONE attempt by default (was two). A rung that produces no END marker has
            # WEDGED the board in M-mode: attempt 2 re-runs the same command against a
            # dead machine, so it cannot recover and only burns EXEC_TIMEOUT seconds.
            # Measured: 8/8 second attempts across three sessions here, and 33/33 across
            # the wider log corpus, were followed by "FAILED to produce a RESULT" --
            # not one recovery ever. At 75 s each that was ~225 s wasted per 3-failure
            # session, and bisection sessions are mostly failures by design.
            # LADDER_EXEC_ATTEMPTS restores the old behaviour if a transient (non-wedge)
            # exec failure is ever observed.
            for attempt in range(1, EXEC_ATTEMPTS + 1):
                cmd = f"echo B''G{r}; {CTL_REMOTE} {r} {dom_remote}; echo E''ND{r}=$?"
                try:
                    out = console.run_command(cmd, rf"END{r}=\d", timeout=EXEC_TIMEOUT, idle_timeout=EXEC_IDLE)
                    m = re.search(rf"RESULT {r} retval=(\d+) cycles=(\d+) ran=(\d+)", out)
                    if m:
                        results[r] = (m.group(1), int(m.group(2)), m.group(3))
                        log(f"  {r}: retval={m.group(1)} cycles={m.group(2)} ran={m.group(3)}")
                        # instret/phase are optional (perf rungs only). phase=1
                        # means minstret faulted the domain; phase=2 means it read.
                        ins = re.search(rf"RESULT {r} .*instret=(\d+) phase=(\d+)", out)
                        if ins:
                            instret_lines[r] = (int(ins.group(1)), int(ins.group(2)))
                            log(f"  {r}: instret={ins.group(1)} (minstret phase="
                                f"{ins.group(2)})")
                        # The gp_diag rung also prints raw per-probe values; that
                        # line IS the point of running it, so surface it in the log
                        # rather than leaving it buried in the UART capture.
                        d = re.search(rf"DEBUG {r} ((?:dbg\d+=\d+ ?)+)", out)
                        if d:
                            log(f"  {r}: DEBUG {d.group(1).strip()}")
                            debug_lines[r] = d.group(1).strip()
                        break
                    log(f"  {r}: ran but no RESULT line; attempt {attempt}")
                except ActionTimeout:
                    log(f"  {r}: no END marker in {EXEC_TIMEOUT}s (attempt {attempt})")
                    # A rung that never returned may have WEDGED the boot. In
                    # one-boot mode the next rung would "reuse" a dead boot and
                    # fail too: on 2026-07-28 one hanging rung cost the four after
                    # it, all of which had worked minutes earlier. Force a fresh
                    # boot so one failure stays one failure.
                    booted_once[0] = False
                    ctl_present[0] = False
                except BadNamespaceError:
                    log(f"  {r}: socket dropped; reconnecting"); wait_connected(console, 45)
                time.sleep(2)
            else:
                results[r] = (None, None, None)
                log(f"  {r}: FAILED to produce a RESULT")
    finally:
        if not getattr(console.sio, "connected", False):
            try: console.connect(); log("reconnected for cleanup")
            except Exception as e: log(f"cleanup reconnect err: {e}")
        try:
            if gdb_up: console.gdb_stop()
        except Exception as e: log(f"gdb_stop err: {e}")
        try: console.power(False); log("powered off")
        except Exception as e: log(f"power off err: {e}")
        try:
            if locked: console.unlock(); log("unlocked")
        except Exception as e: log(f"unlock err: {e}")
        try:
            pathlib.Path(CAPTURE).write_text(console.uart_text)
            log(f"UART capture -> {CAPTURE} ({len(console.uart_text)} chars)")
        except Exception as e: log(f"capture save err: {e}")
        console.close()

    # report
    lines = ["rung                 retval        oracle        cycles(mcycle)  instret       correct"]
    allok = True
    for r in RUNGS:
        got, cyc, ran = results.get(r, (None, None, None))
        oracle = oracles[r]
        ok = (got == oracle and ran == str(0xD09E))
        allok = allok and ok
        ins = instret_lines.get(r, (None, None))[0]
        lines.append(f"{r:<20} {str(got):<13} {oracle:<13} {str(cyc):<15} "
                     f"{str(ins):<13} {'YES' if ok else 'NO'}")
    for r in RUNGS:
        if r in debug_lines:
            lines.append(f"{r} raw probes: {debug_lines[r]}")
    report = "\n".join(lines)
    print("\n==== silicon-ladder FPGA perf ====\n" + report)
    pathlib.Path(RESULTS_OUT).write_text(report + "\n")
    log(f"results -> {RESULTS_OUT}")
    return 0 if allok else 1

if __name__ == "__main__":
    sys.exit(main())
