"""Bounded cleanup for board sessions: a session must ALWAYS release the board.

WHY THIS EXISTS. Every teardown step in these drivers waits on a board event --
`set_switch` waits for switch_state, `power(False)` waits for power_state, `unlock()` waits
for lock_state. Any one of them blocks indefinitely if the confirmation never arrives, and
because teardown runs inside `finally` while holding the flock, a hung cleanup strands the
board AND every later session queued behind the lock.

That is not hypothetical. Measured 2026-07-31: probe_revnode.py produced all four of its LED
reads and then sat for 16 minutes in `set_switches(console, 0)`, holding the lock with the
board still powered, which blocked the next probe entirely. Earlier the same day
probe_sqlite_wedge.py held the lock for 24 minutes after its results were already on disk.
Both looked from the outside like "the board is busy" when the board was simply abandoned.

THE RULE: cleanup is best-effort and time-boxed. Releasing the board matters more than
tidying it, so a step that will not complete is abandoned rather than waited on. Ordering in
the caller should therefore be least-important-first (cosmetic state), then power, then
unlock -- so that if the budget is consumed, what gets skipped is the least harmful.

SIGALRM is used rather than a worker thread because these are single-threaded blocking
network waits: the alarm interrupts the syscall, which a thread-based timeout cannot do
without leaving the thread stuck forever. Only usable on the main thread of a POSIX process,
which is exactly how these drivers run.
"""
import signal
import sys


class _Timeout(Exception):
    pass


def bounded(label, seconds, fn, *args, **kwargs):
    """Run one cleanup step with a hard deadline. Never raises.

    Returns True if the step completed, False if it timed out or errored. The outcome is
    printed so an abandoned step is visible in the log rather than silently skipped -- a
    board left powered on is worth knowing about.
    """
    def _fire(_sig, _frm):
        raise _Timeout()

    prev = None
    try:
        prev = signal.signal(signal.SIGALRM, _fire)
        signal.alarm(int(max(1, seconds)))
    except ValueError:
        # Not on the main thread; run unbounded rather than not at all.
        try:
            fn(*args, **kwargs)
            return True
        except Exception as exc:
            print(f"[cleanup] {label}: FAILED ({type(exc).__name__})", flush=True)
            return False
    try:
        fn(*args, **kwargs)
        return True
    except _Timeout:
        print(f"[cleanup] {label}: TIMED OUT after {seconds}s -- abandoned so the board is "
              f"released; it may be left powered on", flush=True)
        return False
    except Exception as exc:
        print(f"[cleanup] {label}: FAILED ({type(exc).__name__})", flush=True)
        return False
    finally:
        try:
            signal.alarm(0)
            if prev is not None:
                signal.signal(signal.SIGALRM, prev)
        except Exception:
            pass


def release_board(console, *, switches_off=None, label="session"):
    """Standard teardown: cosmetic state, then power, then unlock -- each time-boxed.

    Order is deliberate. If the budget runs out, the step that gets skipped should be the
    one that matters least; unlock goes last but gets its own budget so it is never the
    casualty of a slow power-off.
    """
    if switches_off is not None:
        bounded("switches->0", 20, switches_off)
    bounded("power off", 30, console.power, False)
    bounded("unlock", 20, console.unlock)
    print("BOARD_RELEASED", flush=True)
