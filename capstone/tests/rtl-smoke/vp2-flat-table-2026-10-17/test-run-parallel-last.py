#!/usr/bin/env python3
"""Pin run.py's --parallel-last: it MOVES the parallel pair, and the default is untouched.

run.py is a shared runner whose argv several suites pin, so the risk of adding an option is not
that the option is wrong but that the DEFAULT moved. This asserts both directions against the real
module rather than against a copy of the text.

Run: python3 test-run-parallel-last.py   (exit 0 = pinned)
"""
import pathlib, sys, types

HERE = pathlib.Path(__file__).resolve().parent
# HERE is <repo>/capstone/tests/rtl-smoke/<folder>, so parents[2] is <repo>/capstone.
RUNPY = HERE.parents[2] / "runtime" / "virtual" / "run.py"


def load():
    """Compile the LIVE source text, never import it.

    spec_from_file_location goes through the bytecode cache, which is keyed on (size, mtime at
    one-second granularity). A block move inside a file preserves its size exactly, and a scripted
    edit-and-restore lands inside the same second -- so a stale __pycache__ entry compares equal
    and is served. That is not hypothetical: it made this test report "the move changed nothing"
    against a file whose text was provably correct (2026-10-18). Compiling the text read from disk
    has no cache to go stale.
    """
    ns = {"__name__": "run_py_under_test", "__file__": str(RUNPY)}
    exec(compile(RUNPY.read_text(), str(RUNPY), "exec"), ns)
    return types.SimpleNamespace(**ns)


def main():
    if not RUNPY.exists():
        print(f"NO DATA: {RUNPY} not found", file=sys.stderr)
        return 2
    m = load()
    S, PAR = m.SCRIPT, m.PARALLEL_BLOCK
    fails = []

    # the block must be in the default script exactly once, or the move cannot be verbatim
    if S.count(PAR) != 1:
        fails.append(f"PARALLEL_BLOCK appears {S.count(PAR)} times in SCRIPT, expected 1")

    moved = S.replace(PAR, "", 1).replace("rmmod capstone_vm\n", PAR + "rmmod capstone_vm\n", 1)

    # 1. nothing is lost or invented: the same SET of lines, only the order differs
    if sorted(l for l in S.splitlines() if l.strip()) != \
       sorted(l for l in moved.splitlines() if l.strip()):
        fails.append("the moved script is not a permutation of the default")
    if S == moved:
        fails.append("the move changed nothing")

    # 2. the pair lands immediately BEFORE rmmod, while the module is still loaded. After rmmod it
    #    would launch with no module on a machine where rmmod succeeds and test nothing.
    lines = moved.splitlines()
    ri = lines.index("rmmod capstone_vm")
    if lines[ri - 1] != "cat /tmp/one.out /tmp/two.out":
        fails.append(f"line before rmmod is {lines[ri-1]!r}, not the end of the parallel block")
    if "VIRTUAL_RUNTIME_DONE" in "\n".join(lines[:ri]):
        fails.append("the pair was placed after VIRTUAL_RUNTIME_DONE")

    # 3. the default is NOT the moved order (the thing other suites depend on)
    dl = S.splitlines()
    if dl.index(PAR.splitlines()[0]) >= dl.index("rmmod capstone_vm"):
        fails.append("the DEFAULT script already runs the pair last; the default moved")

    # 4. negative control: the checks above must be able to fail. Prove it on a script whose pair
    #    is placed after rmmod, which is the mistake this test exists to catch.
    bad = S.replace(PAR, "", 1).replace("rmmod capstone_vm\n", "rmmod capstone_vm\n" + PAR, 1)
    bl = bad.splitlines()
    if bl[bl.index("rmmod capstone_vm") - 1] == "cat /tmp/one.out /tmp/two.out":
        fails.append("NEGATIVE CONTROL DID NOT FIRE: the placement check cannot detect "
                     "a pair placed after rmmod, so its pass means nothing")

    for f in fails:
        print("FAIL " + f)
    if not fails:
        print(f"ok   --parallel-last moves the pair from line "
              f"{dl.index(PAR.splitlines()[0])+1} to {lines.index(PAR.splitlines()[0])+1} "
              f"of {len(lines)}, immediately before rmmod; default unchanged; "
              f"negative control fires")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
