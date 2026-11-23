#!/usr/bin/env python3
"""board-idle-watch.py [--once] [--bookings FILE] [--period S] [--idle S] [--max S]

Watches boards 1-3 for idle time. A board is busy while the lab hub holds a booking for it (start <= now < end).
For each board, the script also lists the chain processes that name it: nat-run3*.sh <b>, thrun.sh <b>,
wait-pid-thrun.sh <pid> <b>, and wait-marker.sh lines that carry a conf for the board.

It exits 0 and prints "IDLE board <b> ..." once some board has had no booking for --idle seconds (default 900), so
the session that launched it in the background is woken. A refused chain start leaves its board free without a
word: on 2026-11-24 h16s stopped on a stale output directory, and nothing reported it. A chain between two of its
stages also holds no booking, which is why --idle is set above the gap between stages.

The script exits 2 after three failed hub reads in a row: an unreadable hub is an error, never "busy". It exits 1
when --max seconds pass (default 86,400) without an idle board. --bookings reads a JSON list in place of the hub;
it is used for the positive and negative controls."""
import json, os, sys, time, urllib.request

BOARDS = ("1", "2", "3")


def arg(name, default):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def hub_bookings():
    f = arg("--bookings", None)
    if f:
        return json.load(open(f))
    tok = open(os.path.expanduser("~/.claude-kisp/secrets/fpga-lab-token")).read().strip()
    hub = open("/tmp/capstone/lab-hub/hub-url").read().strip().rstrip("/")
    req = urllib.request.Request(hub + "/api/bookings", headers={"X-Lab-Token": tok})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def chains():
    """board -> list of 'pid script args' for chain processes naming that board."""
    out = {b: [] for b in BOARDS}
    me = os.getpid()
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) == me:
            continue
        try:
            argv = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        argv = [x.decode(errors="replace") for x in argv if x]
        if len(argv) < 3 or os.path.basename(argv[0]) != "bash":
            continue
        s, rest = os.path.basename(argv[1]), argv[2:]
        b = None
        if s.startswith("nat-run3") or s == "thrun.sh":
            b = rest[0]
        elif s == "wait-pid-thrun.sh" and len(rest) > 1:
            b = rest[1]
        elif s == "wait-marker.sh":
            b = next((x for x in BOARDS if any(f"-b{x}." in r for r in rest)), None)
        if b in out:
            out[b].append(f"{p} {s} {' '.join(rest)}")
    return out


def main():
    period, idle, mx = int(arg("--period", 300)), int(arg("--idle", 900)), int(arg("--max", 86400))
    once = "--once" in sys.argv
    t0, fails, free_since = time.time(), 0, {}
    while True:
        now = time.time()
        try:
            bk = hub_bookings()
            fails = 0
        except Exception as e:
            fails += 1
            print(f"{time.strftime('%F %T')} HUB READ FAILED ({fails}): {str(e)[:120]}", flush=True)
            if fails >= 3:
                return 2
            time.sleep(period)
            continue
        busy = {b: [x for x in bk if str(x.get("board")) == b and x["start"] <= now < x["end"]] for b in BOARDS}
        ch = chains()
        for b in BOARDS:
            if busy[b]:
                free_since.pop(b, None)
            else:
                free_since.setdefault(b, now)
        line = " | ".join(f"b{b}: " + (f"booked {busy[b][0]['id']}" if busy[b] else
                                       f"FREE {int(now - free_since[b])}s") + f", {len(ch[b])} chain(s)"
                          for b in BOARDS)
        print(f"{time.strftime('%F %T')} {line}", flush=True)
        hit = [b for b in BOARDS if b in free_since and (once or now - free_since[b] >= idle)]
        if hit:
            for b in hit:
                print(f"IDLE board {b}: no booking for {int(now - free_since[b])} s; chains naming it: "
                      + ("; ".join(ch[b]) if ch[b] else "NONE"), flush=True)
            return 0
        if once:
            return 1
        if now - t0 >= mx:
            print("MAX reached, no idle board", flush=True)
            return 1
        time.sleep(period)


sys.exit(main())
