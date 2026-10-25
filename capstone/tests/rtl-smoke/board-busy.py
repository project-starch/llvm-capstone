#!/usr/bin/env python3
"""board-busy.py <board>: is board 1 or 2 in use from this host? Exit 0 FREE, 1 BUSY (the reason printed), 2 on bad input.
Two independent signs, either one is enough (2026-10-19, phase 1 of board #2):
- the board's host lock is HELD: fpga_console takes an exclusive flock on /tmp/capstone/.board.lock (board 1) or
  /tmp/capstone/.board2.lock (board 2) for every console it opens. It is read from /proc/locks, never by taking the
  lock, so the check cannot make a starting run fail its own lock;
- a board-user process names the board: a fpga_driver module, a bare runner (run_sup_bare*.py), infra-retry.py,
  flash-bit.py or diag_noboot_fpga.py whose environment has FPGA_BOARD=<board>. An unset FPGA_BOARD is board 1, which
  fpga_console enforces by refusing any other board's URL. This catches a run in its pre-connect phase (a bake, a
  wait), which holds no lock yet. The caller's own ancestors are excluded, so a chain never sees itself.
BOARD_LOCK_DIR replaces /tmp/capstone for the lock files, for the guard's own fire/quiet test."""
import os, re, sys
LOCKS = {"1": ".board.lock", "2": ".board2.lock"}
USER = re.compile(r"python3? (-u )?(-m fpga_driver\S*|\S*run_sup_bare\S*\.py|\S*infra-retry\.py|\S*flash-bit\.py|\S*diag_noboot_fpga\S*)")
def ancestors():
    a, p = set(), os.getpid()
    while p > 1:
        a.add(p)
        try: p = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
        except Exception: break
    return a
def main():
    if len(sys.argv) != 2 or sys.argv[1] not in LOCKS:
        print(f"usage: board-busy.py 1|2 (got {sys.argv[1:]})"); return 2
    b = sys.argv[1]; path = os.path.join(os.environ.get("BOARD_LOCK_DIR", "/tmp/capstone"), LOCKS[b])
    if os.path.exists(path):
        st = os.stat(path); key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
        for l in open("/proc/locks"):
            f = l.split()
            if len(f) > 5 and f[1] == "FLOCK" and f[5] == key:
                print(f"BUSY board {b}: its lock {path} is held by pid {f[4]}"); return 1
    mine = ancestors()
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) in mine: continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            if not USER.search(cmd): continue
            env = dict(e.split("=", 1) for e in open(f"/proc/{d}/environ", "rb").read().decode(errors="replace").split("\0") if "=" in e)
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
        if (env.get("FPGA_BOARD", "1").strip() or "1") == b:
            print(f"BUSY board {b}: pid {d} ({cmd.strip()[:80]}) has FPGA_BOARD={env.get('FPGA_BOARD', 'unset -> 1')}"); return 1
    print(f"FREE board {b}"); return 0
if __name__ == "__main__":
    sys.exit(main())
