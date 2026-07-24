#!/usr/bin/env python3
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
import sys, os, time, base64, hashlib, re, pathlib, gzip

DRV = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(DRV.parent))          # .../rtl-smoke (for `import fpga_driver`)
from fpga_driver import config as C
from fpga_driver.fpga_console import FpgaConsole, ActionTimeout
from fpga_driver.run_rtl_smoke import login_root, POWER_ON_SETTLE, POWER_CYCLE_OFF
from fpga_driver.fast_xfer import fast_put
from socketio.exceptions import BadNamespaceError

URL = open(os.path.expanduser("~/.config/capstone/fpga-board-url")).read().strip()
IMG = pathlib.Path(os.path.expanduser("~/capstone-b-artifacts/fw_payload_fpga_up_gpfree.bin"))
IMG_NAME = "fw_payload_fpga_up_gpfree.bin"
BITSTREAM = "working-caplifive-captype-fixed.bit"

ART = pathlib.Path(os.environ.get("LADDER_FPGA_DIR",
                                  os.path.expanduser("/tmp/capstone/ladder-fpga")))
RUNGS = (os.environ.get("LADDER_RUNGS") or
         "matmult_int coremark_matrix rv8_primes beebs_crc32 "
         "beebs_insertsort beebs_prime beebs_recursion").split()
CAPTURE = "/tmp/capstone/board-run-ladder-perf.uart.txt"
RESULTS_OUT = "/tmp/capstone/ladder-perf-results.txt"
CTL_REMOTE = "/tmp/lpc"

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

def sh(console, cmd, timeout=20):
    return console.run_command(f"{cmd}; echo D''N_$?", r"DN_\d+", timeout=timeout)

def ensure_capstone_dev(console):
    """insmod capstone.ko (the UP image does not auto-load it) then verify the
    device exists. Gate on the trailing DEVNO/DEVOK token, NOT substring presence
    (the echoed command line itself contains the word 'DEVOK')."""
    sh(console, "echo 1 > /proc/sys/kernel/printk")
    out = sh(console, "[ -e /dev/capstone ] || insmod /capstone.ko 2>/dev/null; "
                      "[ -e /dev/capstone ] && echo DEV''OK || echo DEV''NO", timeout=30)
    if "DEVNO" in out or "DEVOK" not in out:
        raise RuntimeError(f"/dev/capstone missing (insmod failed?):\n{out}")

def cold_boot(console, prompt):
    """Full power-cycle + JTAG firmware reload -> fresh boot. Each rung runs as the
    FIRST domain of a clean boot (clean icache). A warm `monitor reset halt` does
    NOT work here -- the fw_payload OpenSBI does not re-enter cleanly from a soft
    reset (its one-time hart/DDR init is not re-runnable), so a real power-cycle is
    required. The image must already be in the console store (upload_boot_image);
    load_image JTAG-copies store->DDR (~2 min). Returns with a confirmed root shell
    and /dev/capstone loaded (the initramfs is re-unpacked, so /tmp starts empty)."""
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

def main():
    oracles = {r: open(ART / f"{r}.oracle").read().strip() for r in RUNGS}
    ctl = ART / "ladder_perf_ctl"
    if not ctl.is_file():
        raise SystemExit(f"controller missing: {ctl} (run build-ladder-fpga.sh)")
    for r in RUNGS:
        if not (ART / f"{r}.dom").is_file():
            raise SystemExit(f"domain missing: {ART}/{r}.dom")

    console = FpgaConsole(URL, logger=lambda m: print(f"[fpga] {m}", file=sys.stderr))
    console.connect()
    install_resilient_emit(console)
    locked = False
    gdb_up = False
    results = {}
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
            log(f"resident bitstream {rb!r} != {BITSTREAM!r}; re-flashing (authorized restore)")
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
        # One rung per FULL power-cycle + reload: each domain runs as the first of
        # a clean boot, sidestepping the multi-domain same-VA icache hang. ~2.5 min
        # per rung (the JTAG reload dominates); the domains themselves are tiny.
        for r in RUNGS:
            dom = ART / f"{r}.dom"
            dom_remote = f"/tmp/{r}.dom"
            try:
                log(f"[{r}] power-cycle + reload firmware")
                cold_boot(console, prompt)
                fast_put(console, ctl_gz, "/tmp/lpc.gz", CTL_REMOTE, ctl_sha, True, log)
                fast_put(console, gzip_file(dom), f"/tmp/{r}.gz", dom_remote,
                         sha16(dom), False, log)
            except (ActionTimeout, BadNamespaceError, RuntimeError) as e:
                results[r] = (None, None, None)
                log(f"  {r}: boot/transfer failed ({e}); skipping")
                if not getattr(console.sio, "connected", False):
                    wait_connected(console, 45)
                continue
            # As the first domain of a fresh boot, a rung either returns quickly
            # (matmult was <1 ms) or the cscall genuinely hangs -- retrying the same
            # first-domain condition won't help, so keep the budget tight.
            for attempt in range(1, 3):
                cmd = f"echo B''G{r}; {CTL_REMOTE} {r} {dom_remote}; echo E''ND{r}=$?"
                try:
                    out = console.run_command(cmd, rf"END{r}=\d", timeout=75)
                    m = re.search(rf"RESULT {r} retval=(\d+) cycles=(\d+) ran=(\d+)", out)
                    if m:
                        results[r] = (m.group(1), int(m.group(2)), m.group(3))
                        log(f"  {r}: retval={m.group(1)} cycles={m.group(2)} ran={m.group(3)}")
                        break
                    log(f"  {r}: ran but no RESULT line; attempt {attempt}")
                except ActionTimeout:
                    log(f"  {r}: no END marker in 120s (attempt {attempt})")
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
    lines = ["rung                 retval        oracle        cycles(mcycle)  correct"]
    allok = True
    for r in RUNGS:
        got, cyc, ran = results.get(r, (None, None, None))
        oracle = oracles[r]
        ok = (got == oracle and ran == str(0xD09E))
        allok = allok and ok
        lines.append(f"{r:<20} {str(got):<13} {oracle:<13} {str(cyc):<15} "
                     f"{'YES' if ok else 'NO'}")
    report = "\n".join(lines)
    print("\n==== silicon-ladder FPGA perf ====\n" + report)
    pathlib.Path(RESULTS_OUT).write_text(report + "\n")
    log(f"results -> {RESULTS_OUT}")
    return 0 if allok else 1

if __name__ == "__main__":
    sys.exit(main())
