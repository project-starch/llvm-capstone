#!/usr/bin/env python3
"""board-env-check.py <board 1|2> <kit dir> [stage dir ...]: what a stubbed dry run and a QEMU gate cannot see, checked
against the board the run goes to. Run it before arming a board run; exit 0 = every check holds, 1 = a check fails,
2 = the check itself could not run (a missing file it needs). Each line names its check.

C1 console URL: the board's own URL file exists (~/.claude-kisp/secrets/fpga-console-url, board 2: ...-board2), and the
   kit's board-run.sh will read it: board.env's BOARD_URL_FILE, or the BOARD_URL_FILE this process exports, names that
   file. A board 1 kit carries no BOARD_URL_FILE and its board-run.sh then defaults to board 1's (5 Nov: a board 2 boot
   stopped at the driver for this).
C2 node table width: on bitstream 3 (e7af48457) and C.5a (1fc2cd5aa) the kit's capstone_vm.ko must be the 18-bit build,
   which alone exports the index_bits parameter. The 16-bit default holds 65,536 records; gcc train01 ran out of it
   (5 Nov, QEMU) while every test input fit.
C3 kernel and module interfaces: every /proc or /sys path a stage reads (its run.sh, and the strings of its stage files)
   must exist on the board: a path that needs a kernel option is checked against the board kernel's .config (QEMU's
   kernel is configured differently: /proc/self/pagemap is there and not on the board, 5 Nov), and every
   /sys/module/capstone_vm/parameters/<p> against the kit module's parameters. A run.sh line
   `# board-env-check: optional <path>` downgrades that path to a note, for a reader with a fallback."""
import os, re, subprocess, sys

HOME = os.path.expanduser("~")
# the board kernel's .config (BOARD_KERNEL_CONFIG overrides): the main checkout's buildroot, since worktrees have no submodules
BOARD_CONFIG = os.environ.get("BOARD_KERNEL_CONFIG", os.path.expanduser(
    "~/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot/build-fpga/build/linux-6.4.14/.config"))
WIDE = {"caplifive_vp2c_e7af48457", "caplifive_vp2c_1fc2cd5aa"}   # bitstreams with the 18-bit lifetime table
NEEDS = [(r"^/proc/(self|\d+|\[pid\])/pagemap$|^/proc/kpage", "CONFIG_PROC_PAGE_MONITOR"),
         (r"^/proc/(self|\d+)/(task/\d+/)?schedstat$|^/proc/schedstat$", "CONFIG_SCHED_INFO"),
         (r"^/proc/(self|\d+)/io$", "CONFIG_TASK_IO_ACCOUNTING"),
         (r"^/proc/pressure", "CONFIG_PSI"),
         (r"^/proc/config\.gz$", "CONFIG_IKCONFIG_PROC"),
         (r"^/sys/kernel/debug", "CONFIG_DEBUG_FS"),
         (r"^/sys/kernel/tracing", "CONFIG_TRACING")]
fail = 0; broken = 0
def say(ok, msg):
    global fail
    if not ok: fail = 1
    print(("PASS " if ok else "FAIL ") + msg)

def env_file(path):
    d = {}
    for l in open(path, errors="replace"):
        m = re.match(r"^(\w+)=(.*)$", l.strip())
        if m: d[m[1]] = m[2].strip('"')
    return d

def module_params(ko):
    s = subprocess.run(["strings", ko], capture_output=True, text=True).stdout
    # module_param() leaves parmtype=<name>:; module_param_cb() leaves only the name as its own string (__param_str_)
    named = set(re.findall(r"^parm(?:type)?=(\w+):", s, re.M))
    return named | {l for l in s.split("\n") if re.fullmatch(r"[a-z_][a-z0-9_]{2,40}", l)}

def main():
    global broken
    if len(sys.argv) < 3: print(__doc__); return 2
    brd, kit, stages = sys.argv[1], sys.argv[2], sys.argv[3:]
    envp = os.path.join(kit, "board.env"); ko = os.path.join(kit, "adapter", "module", "capstone_vm.ko")
    for p in (envp, ko, BOARD_CONFIG):
        if not os.path.exists(p): print(f"BROKEN: {p} is missing"); return 2
    env = env_file(envp)
    # C1
    want = os.path.join(HOME, ".claude-kisp/secrets/fpga-console-url" + ("-board2" if brd == "2" else ""))
    say(os.path.isfile(want) and os.path.getsize(want) > 0, f"C1 board {brd}'s console URL file {want} exists")
    got = env.get("BOARD_URL_FILE") or os.environ.get("BOARD_URL_FILE") or os.path.join(HOME, ".claude-kisp/secrets/fpga-console-url")
    say(os.path.realpath(got) == os.path.realpath(want), f"C1 the kit's board-run.sh reads {got}"
        + ("" if env.get("BOARD_URL_FILE") else " (no BOARD_URL_FILE in board.env: from the environment or the default)"))
    # C2
    params = module_params(ko); bs = env.get("BOARD_BITSTREAM", "?")
    if bs in WIDE:
        say("index_bits" in params, f"C2 {bs} has the 18-bit table: the kit module exports index_bits"
            + ("" if "index_bits" in params else " -- it does not: the 16-bit default build"))
    else:
        print(f"NOTE C2 {bs}: not a known 18-bit bitstream, width not checked")
    # C3
    cfg = open(BOARD_CONFIG).read()
    for st in stages:
        sdir = os.path.join(st, "stage"); run = os.path.join(sdir, "run.sh")
        if not os.path.exists(run): print(f"BROKEN: {run} is missing"); broken = 1; continue
        text = open(run, errors="replace").read()
        optional = set(re.findall(r"#\s*board-env-check:\s*optional\s+(\S+)", text))
        paths = set(re.findall(r"(/(?:proc|sys)/[A-Za-z0-9_./\[\]-]+)", text))
        for root, _, files in os.walk(sdir):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.getsize(fp) > 64 << 20: continue
                s = subprocess.run(["strings", "-n", "6", fp], capture_output=True, text=True).stdout
                paths |= set(m.rstrip(".") for m in re.findall(r"(/(?:proc|sys)/[A-Za-z0-9_./-]+)", s))
        name = os.path.basename(st.rstrip("/"))
        for p in sorted(paths):
            m = re.match(r"^/sys/module/capstone_vm/parameters/(\w+)$", p)
            if m and not p.endswith("_"):
                ok = m[1] in params
                (say(ok, f"C3 {name}: {p} is a parameter of the kit module") if p not in optional or ok
                 else print(f"NOTE C3 {name}: optional {p} absent from the kit module"))
                continue
            for rx, opt in NEEDS:
                if re.search(rx, p):
                    ok = re.search(rf"^{opt}=y", cfg, re.M) is not None
                    if ok or p not in optional:
                        say(ok, f"C3 {name}: {p} needs {opt}, which the board kernel {'has' if ok else 'does NOT have'}")
                    else:
                        print(f"NOTE C3 {name}: optional {p} needs {opt}, absent on the board; the stage has a fallback")
    return 2 if broken else fail

if __name__ == "__main__":
    sys.exit(main())
