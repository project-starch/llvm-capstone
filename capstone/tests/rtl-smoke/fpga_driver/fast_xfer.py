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


def _type_line(console, line, delay, burst=1):
    """Type `line` then Enter, `burst` characters per socket.io emit.

    WHY BURSTING. The char-by-char throttle exists because the board's ns16550a RX
    FIFO overruns on a bulk write and silently drops characters (fpga_console.
    run_command documents a real instance: a path arriving as `row_cost` instead of
    `borrow_cost`). That is a real constraint -- but it is a constraint on how many
    bytes may be in flight before the UART drains, NOT on how many bytes may ride in
    one emit. Sending one character per emit paid a full HTTPS/socket.io round-trip
    per character, and the round-trip -- not the FIFO -- was the actual wall clock:
    a 6,032-char domain transfer is 6,032 round-trips.

    A 16-byte burst is bounded by the hardware, not guessed: 16 bytes is the
    ns16550a FIFO depth, so a burst fills it at most exactly once, and at 115200
    baud it drains in ~1.4 ms -- an order of magnitude under the ~20 ms inter-burst
    delay. Round-trips drop ~16x.

    This is still guarded end-to-end: fast_put verifies a whole-file sha after every
    attempt and escalates to slower, smaller-burst tiers on any mismatch, with the
    final tier at burst=1 (the old behaviour). So the worst case of bursting too
    hard is one wasted attempt, never a corrupt domain -- the same bargain the
    original fast_put already made with its delay/chunk tiers."""
    _resync(console)
    if burst <= 1:
        for ch in line:
            console._emit("uart_send", text=ch)
            time.sleep(delay)
    else:
        for i in range(0, len(line), burst):
            console._emit("uart_send", text=line[i:i + burst])
            time.sleep(delay)
    console._emit("uart_send", text="\r")


def _put_once(console, b64, remote_gz, remote_bin, bin_sha, delay, chunk, log,
              burst=1):
    n = (len(b64) + chunk - 1) // chunk
    # A dropped char can wedge the shell at a `> ` continuation; that raises
    # ActionTimeout here. Catch it and return False so fast_put escalates to the
    # next (slower, resync-first) tier instead of aborting the whole transfer.
    try:
        # Escape any wedged continuation left by a prior attempt, then truncate
        # the target from a known-clean prompt.
        _resync(console)
        console.run_command(f": > {remote_gz}; echo D''N_$?", r"DN_\d", timeout=20,
                            idle_timeout=8)
        for k, i in enumerate(range(0, len(b64), chunk)):
            piece = b64[i:i + chunk]
            start = len(console.uart_text)
            _type_line(console, f"printf %s '{piece}' >> {remote_gz}; echo D''N_$?",
                       delay, burst)
            # WEDGE DETECTED BY SILENCE, not by a 60s wall clock. This wait starts AFTER
            # the line is typed, and uart_send is fire-and-forget over websocket
            # (config.EMIT has no expects_ack), so there is no delivery backlog to absorb
            # -- all that remains is the board echoing the line and running one printf,
            # i.e. well under a second. The 60s only ever elapsed on the failure path: a
            # dropped quote parks the shell at a `> ` continuation that eats every
            # following command, so DN_ never arrives and each wedged tier burned a full
            # minute before escalating.
            #
            # idle_timeout is the better guard than a smaller total, because the board
            # ECHOES the line as it arrives: progress keeps resetting the clock, so a
            # slow-but-live delivery is untouched, while a wedge (total silence) trips in
            # ~8s. Total kept generous for the same reason.
            console.wait_uart(r"DN_\d", timeout=60, search_from=start, idle_timeout=8)
        # 60s KEPT, and deliberately NO idle_timeout. This is the one wait here that does
        # real board-side work: base64-decode, gunzip and sha256sum of the decompressed
        # image, which for a ladder domain is up to ~1.4 MB on a ~50 MHz soft core with no
        # crypto acceleration. I have no measurement of that throughput, so shrinking this
        # would be guessing. idle_timeout is actively WRONG here: the board emits nothing
        # at all while computing, so silence is the expected state and an idle guard would
        # abort a perfectly healthy hash.
        out = console.run_command(
            f"base64 -d {remote_gz} | gunzip -c > {remote_bin}; "
            f"echo S$(sha256sum {remote_bin} | cut -c1-16)S",
            r"S[0-9a-f]{16}S", timeout=60)
    except ActionTimeout:
        log(f"  {remote_bin}: {n} chunks @delay={delay} chunk={chunk} burst={burst} -> "
            f"TIMEOUT (shell wedged); will resync+retry")
        return False
    m = re.search(r"S([0-9a-f]{16})S", out)
    ok = bool(m) and m.group(1) == bin_sha
    log(f"  {remote_bin}: {n} chunks @delay={delay} chunk={chunk} burst={burst} -> "
        f"sha {m.group(1) if m else None} {'OK' if ok else 'MISMATCH'}")
    return ok


def fast_put(console, local_gz, remote_gz, remote_bin, bin_sha, do_exec, log,
             burst=(0.02, 400, 16), fast=(0.02, 400, 1), safe=(0.05, 200, 1),
             safest=(0.09, 100, 1)):
    """Transfer local_gz -> remote_bin (decompressed) on the board, escalating
    burst -> fast -> safe -> safest on any whole-file sha mismatch. The first tier
    sends 16 characters per socket.io emit (see _type_line); every later tier falls
    back to one character per emit, i.e. the original behaviour, so a board that
    cannot take bursts costs one extra attempt and nothing else. Each tier first
    Ctrl-C-resyncs the shell, so a continuation prompt left by a dropped char in
    the previous tier can't poison the retry. Returns True; raises on repeated
    failure."""
    b64 = base64.b64encode(open(local_gz, "rb").read()).decode()
    log(f"transfer {remote_bin}: {len(b64)} b64 chars")
    for delay, chunk, bu in (burst, fast, safe, safest):
        if _put_once(console, b64, remote_gz, remote_bin, bin_sha, delay, chunk, log,
                     bu):
            if do_exec:
                console.run_command(f"chmod 0755 {remote_bin}; echo D''N_$?", r"DN_\d", 20,
                                    idle_timeout=8)
            return True
        log(f"  {remote_bin}: retrying at slower settings")
    raise RuntimeError(f"{remote_bin} failed even at safest settings")
