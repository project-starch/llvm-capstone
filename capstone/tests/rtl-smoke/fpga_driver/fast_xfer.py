"""Faster board file transfer for the CapliFive UART console.

The stock send_file (board_bisect_gpfree.py) is slow for two reasons:
  1. run_command types every char with a hardcoded 0.05s sleep (fpga_console.py:661)
     -- the board UART RX FIFO drops chars on bulk writes, so it throttles.
  2. It does THREE command round-trips per chunk: printf -> /tmp/part,
     echo sha, cat >> file. Plus a small chunk size (200).

fast_put fixes both, SAFELY (no board test needed to trust it):
  - Appends the base64 chunk DIRECTLY to the target (printf ... >> file), one
    round-trip per chunk, no /tmp/part, no per-chunk sha, no cat.
  - Verifies ONCE at the end with a whole-file decompress+sha.
  - Types at a reduced per-char delay, but the final sha is the guard: on ANY
    mismatch it automatically retries the whole file at the safe 0.05s/200-char
    settings. Worst case = the old speed + one retry; expected case ~3x faster.

base64 alphabet is [A-Za-z0-9+/=] -- no single quotes -- so a chunk is always
safe inside 'single quotes' with no escaping.
"""
import base64, hashlib, re, time

try:
    from .fpga_console import ActionTimeout
except Exception:  # pragma: no cover - tolerate flat/standalone import
    try:
        from fpga_driver.fpga_console import ActionTimeout
    except Exception:
        class ActionTimeout(Exception):
            pass


def _resync(console):
    """Land on a clean shell prompt, escaping any continuation/partial-line state.

    The board UART RX FIFO drops chars on bulk writes; a dropped closing quote
    leaves the shell at a `> ` continuation prompt that silently eats every
    following command (empty-file sha, DN_ marker timeouts). Ctrl-C aborts that
    continuation (Ctrl-U alone does NOT); Ctrl-U then clears any partial line.
    """
    console._emit("uart_send", text="\x03")   # Ctrl-C: abort any continuation/partial cmd
    time.sleep(0.2)
    console._emit("uart_send", text="\x15")    # Ctrl-U: clear partial line
    time.sleep(0.15)


def _type_line(console, line, delay):
    _resync(console)
    for ch in line:
        console._emit("uart_send", text=ch)
        time.sleep(delay)
    console._emit("uart_send", text="\r")


def _put_once(console, b64, remote_gz, remote_bin, bin_sha, delay, chunk, log):
    n = (len(b64) + chunk - 1) // chunk
    # A dropped char can wedge the shell at a `> ` continuation; that raises
    # ActionTimeout here. Catch it and return False so fast_put escalates to the
    # next (slower, resync-first) tier instead of aborting the whole transfer.
    try:
        # Escape any wedged continuation left by a prior attempt, then truncate
        # the target from a known-clean prompt.
        _resync(console)
        console.run_command(f": > {remote_gz}; echo D''N_$?", r"DN_\d", timeout=20)
        for k, i in enumerate(range(0, len(b64), chunk)):
            piece = b64[i:i + chunk]
            start = len(console.uart_text)
            _type_line(console, f"printf %s '{piece}' >> {remote_gz}; echo D''N_$?", delay)
            console.wait_uart(r"DN_\d", timeout=60, search_from=start)
        out = console.run_command(
            f"base64 -d {remote_gz} | gunzip -c > {remote_bin}; "
            f"echo S$(sha256sum {remote_bin} | cut -c1-16)S",
            r"S[0-9a-f]{16}S", timeout=60)
    except ActionTimeout:
        log(f"  {remote_bin}: {n} chunks @delay={delay} chunk={chunk} -> "
            f"TIMEOUT (shell wedged); will resync+retry")
        return False
    m = re.search(r"S([0-9a-f]{16})S", out)
    ok = bool(m) and m.group(1) == bin_sha
    log(f"  {remote_bin}: {n} chunks @delay={delay} chunk={chunk} -> "
        f"sha {m.group(1) if m else None} {'OK' if ok else 'MISMATCH'}")
    return ok


def fast_put(console, local_gz, remote_gz, remote_bin, bin_sha, do_exec, log,
             fast=(0.02, 400), safe=(0.05, 200), safest=(0.09, 100)):
    """Transfer local_gz -> remote_bin (decompressed) on the board, escalating
    fast -> safe -> safest on any whole-file sha mismatch. Each tier first
    Ctrl-C-resyncs the shell, so a continuation prompt left by a dropped char in
    the previous tier can't poison the retry. Returns True; raises on repeated
    failure."""
    b64 = base64.b64encode(open(local_gz, "rb").read()).decode()
    log(f"transfer {remote_bin}: {len(b64)} b64 chars")
    for delay, chunk in (fast, safe, safest):
        if _put_once(console, b64, remote_gz, remote_bin, bin_sha, delay, chunk, log):
            if do_exec:
                console.run_command(f"chmod 0755 {remote_bin}; echo D''N_$?", r"DN_\d", 20)
            return True
        log(f"  {remote_bin}: retrying at slower settings")
    raise RuntimeError(f"{remote_bin} failed even at safest settings")
