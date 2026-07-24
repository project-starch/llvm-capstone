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


def _type_line(console, line, delay):
    console._emit("uart_send", text="\x15")   # Ctrl-U: clear any partial line
    time.sleep(0.15)
    for ch in line:
        console._emit("uart_send", text=ch)
        time.sleep(delay)
    console._emit("uart_send", text="\r")


def _put_once(console, b64, remote_gz, remote_bin, bin_sha, delay, chunk, log):
    # truncate target
    console.run_command(f": > {remote_gz}; echo D''N_$?", r"DN_\d", timeout=20)
    n = (len(b64) + chunk - 1) // chunk
    for k, i in enumerate(range(0, len(b64), chunk)):
        piece = b64[i:i + chunk]
        start = len(console.uart_text)
        _type_line(console, f"printf %s '{piece}' >> {remote_gz}; echo D''N_$?", delay)
        console.wait_uart(r"DN_\d", timeout=60, search_from=start)
    out = console.run_command(
        f"base64 -d {remote_gz} | gunzip -c > {remote_bin}; "
        f"echo S$(sha256sum {remote_bin} | cut -c1-16)S",
        r"S[0-9a-f]{16}S", timeout=60)
    m = re.search(r"S([0-9a-f]{16})S", out)
    ok = bool(m) and m.group(1) == bin_sha
    log(f"  {remote_bin}: {n} chunks @delay={delay} chunk={chunk} -> "
        f"sha {m.group(1) if m else None} {'OK' if ok else 'MISMATCH'}")
    return ok


def fast_put(console, local_gz, remote_gz, remote_bin, bin_sha, do_exec, log,
             fast=(0.02, 400), safe=(0.05, 200)):
    """Transfer local_gz -> remote_bin (decompressed) on the board, fast then
    safe-retry. Returns True; raises on repeated failure."""
    b64 = base64.b64encode(open(local_gz, "rb").read()).decode()
    log(f"transfer {remote_bin}: {len(b64)} b64 chars")
    for delay, chunk in (fast, safe):
        if _put_once(console, b64, remote_gz, remote_bin, bin_sha, delay, chunk, log):
            if do_exec:
                console.run_command(f"chmod 0755 {remote_bin}; echo D''N_$?", r"DN_\d", 20)
            return True
        log(f"  {remote_bin}: retrying at safe settings")
    raise RuntimeError(f"{remote_bin} failed even at safe settings")
