#!/usr/bin/env python3
"""uaf-repeat.py --board-kit <kit> --tools <tools dir> --controls <controls.dom> --helper <juliet-timeout> --n <N> --out <fresh dir>

The Juliet runner's use-after-free control, repeated N times in ONE cold board boot (2026-10-22). On tree B it read
cause 13 (a load page fault) at control_read_probe+0x38 in 5 of 74 Juliet runs (temporal 2 of 24, spatial 3 of 48),
where tree A reads cause 25 (the revocation check) in 74 of 74; one-sided Fisher p ~ 0.03. Each run is the Juliet
gate's own command, byte for byte apart from its name: `./juliet-timeout <name> 10000 -- ./capstone-vexec
./images/controls.dom uaf-malloc`, with the same controls.dom, helper and kit launcher. Odd runs are untraced (as in
Juliet); even runs set CAPSTONE_VM_TRACE=1, so a cause-13 run has the launcher's service/boundary trace beside it.
A bounds-control run comes first, as in every Juliet batch. Every uaf step prints its fault line(s) (FL ...); the
whole output (TR ...) is printed for every cause-13 run and every tenth run, the tenth runs being the cause-25 traces
to compare against.
--no-trace runs every uaf run untraced (as in Juliet). --dmesg also prints, for those same runs, the module's last 12
kernel lines (`dmesg | grep capstone_vm`, DM ...): a cause-13 fault the module returns unresolved passes through
resolve(), whose refusal line ("resolve refused at <site>", capstone_vm.c CV_RESOLVE_FAIL) is pr_warn unless
diag_emerg, so it is in the ring buffer but never on the console. Reading the ring changes nothing about the run.
Judges nothing; writes <out>/steps.json and prints the tally: per arm (plain, traced) the count of runs with
cause 25, cause 13, other, and no fault line. Exits 2 unless every step is in the transcript."""
import argparse
import collections
import json
import re
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    kits = p.add_mutually_exclusive_group(required=True)
    kits.add_argument("--board-kit", type=Path)
    p.add_argument("--tools", type=Path, required=True, help="bug-corpora/tools of a clean worktree")
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--helper", type=Path, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--no-trace", action="store_true", help="every uaf run untraced")
    p.add_argument("--mapped", type=Path, help="controls-mapped.dom: after each uaf run, one uaf-mapped run (the freed page kept mapped)")
    p.add_argument("--keep-retired", action="store_true", help="after each uaf run, one with CAPSTONE_VM_TEST_KEEP_RETIRED=1 (a launcher whose hook keeps a retired region mapped)")
    kits.add_argument("--qemu-kit", type=Path, help="run the same steps in one QEMU boot of this virtual kit instead of a board")
    p.add_argument("--refuse-pte", action="store_true", help="set the module's refuse_pte=1 (0644) before the runs and read it back")
    p.add_argument("--dmesg", action="store_true", help="print the module's last 12 dmesg lines with each printed run")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.tools))
    import boardvm  # noqa: E402
    if a.out.exists():
        sys.exit(f"{a.out} exists; a run goes into a fresh directory")
    a.out.mkdir(parents=True)
    if a.qemu_kit:
        import virtualvm  # noqa: E402
        b = virtualvm.Batch(a.out / "stage")
    else:
        b = boardvm.BoardBatch(a.out / "stage")
    if a.mapped:
        b.put("images/controls-mapped.dom", a.mapped)
        import hashlib
        print(f"MAPPED-IMAGE sha256 {hashlib.sha256(a.mapped.read_bytes()).hexdigest()[:16]}")
    b.put("images/controls.dom", a.controls)
    b.put("juliet-timeout", a.helper)
    (a.out / "stage" / "juliet-timeout").chmod(0o755)
    b.bounds = {}
    # As in every Juliet batch, the bounds control runs first, so run 1 sits where ctl-uaf-0 sat (the second program
    # after the module loads), the position of the spatial set's three cause-13 readings.
    if a.refuse_pte:
        # Module d0d5f758 (runtime lane, runtime/arena-refusal-pte 3feb24ac2c61): refuse_pte adds the faulting PTE and the
        # covering arena to the `arena` refusal line. Set through sysfs after the batch's insmod, and read back: a run
        # whose read-back is not Y did not run that configuration.
        b.step("refuse-pte", "P=/sys/module/capstone_vm/parameters/refuse_pte; echo 1 > $P; echo \"RPTE $(cat $P 2>/dev/null || echo ABSENT)\"")
    b.step("bounds-0", "./juliet-timeout bounds-0 10000 -- ./capstone-vexec ./images/controls.dom bounds-malloc")
    for k in range(1, a.n + 1):
        env = "CAPSTONE_VM_TRACE=1 " if k % 2 == 0 and not a.no_trace else ""
        dm = "; dmesg | grep -a capstone_vm | tail -n 12 | sed 's/^/DM /'" if a.dmesg else ""
        name = f"uaf-{k}-{'traced' if env else 'plain'}"
        b.step(name, f"( {env}./juliet-timeout {name} 10000 -- ./capstone-vexec ./images/controls.dom uaf-malloc ) "
                     f"> /tmp/u.txt 2>&1; rc=$?; grep -a 'domain fault' /tmp/u.txt | sed 's/^/FL /'; "
                     f"if grep -q 'cause=13 ' /tmp/u.txt || [ {k % 10} = 0 ]; then tail -n 300 /tmp/u.txt | sed 's/^/TR /'{dm}; fi; "
                     f"exit $rc")
        if a.mapped:
            # every tenth mapped run is traced, to record which operation the free issued (an object revoke, or a region
            # UNMAP/retire as the original control's group free does); tallied apart, since the trace changes timing
            mname = f"map-{k}" + ("-traced" if k % 10 == 0 else "")
            menv = "CAPSTONE_VM_TRACE=1 " if k % 10 == 0 else ""
            b.step(mname, f"( {menv}./juliet-timeout {mname} 10000 -- ./capstone-vexec ./images/controls-mapped.dom uaf-mapped ) "
                          f"> /tmp/u.txt 2>&1; rc=$?; grep -a -E 'domain fault|STALE_READ_SURVIVED|CONTROL uaf-mapped' /tmp/u.txt | sed 's/^/FL /'; "
                          f"if grep -q -E 'cause=13 |STALE_READ_SURVIVED' /tmp/u.txt || [ {k % 10} = 0 ]; then tail -n 300 /tmp/u.txt | sed 's/^/TR /'{dm}; fi; "
                          f"exit $rc")
        if a.keep_retired:
            # The runtime lane's launcher 518a66b9 (runtime/vexec-keep-retired 0738f494c199): with the variable set, the
            # free's unmap still retires the region (revokes, unpins) but does not munmap it, so the read cannot page-fault.
            kname = f"keep-{k}" + ("-traced" if k % 10 == 0 else "")
            kenv = "CAPSTONE_VM_TEST_KEEP_RETIRED=1 " + ("CAPSTONE_VM_TRACE=1 " if k % 10 == 0 else "")
            b.step(kname, f"( {kenv}./juliet-timeout {kname} 10000 -- ./capstone-vexec ./images/controls.dom uaf-malloc ) "
                          f"> /tmp/u.txt 2>&1; rc=$?; grep -a -E 'domain fault|CONTROL uaf-malloc|CAPSTONE_VM_TEST_KEEP_RETIRED' /tmp/u.txt | sed 's/^/FL /'; "
                          f"if grep -q -E 'cause=13 |RETURNED' /tmp/u.txt || [ {k % 10} = 0 ]; then tail -n 300 /tmp/u.txt | sed 's/^/TR /'{dm}; fi; "
                          f"exit $rc")
    if a.dmesg:
        # The whole boot's refusals, read once at the end (the runtime lane's form): each unresolved fault is expected
        # to print twice (the module's in-step resolve, then the launcher's CV_RESOLVE).
        # The ring is 128 KiB and a long boot wraps it (2026-10-22, tree A: REFUSED 0 beside 7 per-run lines), so the
        # step also prints the ring's OLDEST line: a count is valid only if that line is from the boot's first seconds.
        b.step("dmesg-all", "echo OLDEST $(dmesg | head -n 1); "
                            "echo REFUSED $(dmesg | grep -a -c 'capstone_vm: resolve refused'); "
                            "dmesg | grep -a 'capstone_vm: resolve refused' | tail -n 40 | sed 's/^/DA /'")
    steps, transcript, completed = b.execute(a.qemu_kit or a.board_kit, a.out / "work", 3600 + 30 * a.n * (1 + (1 if a.mapped else 0) + (1 if a.keep_retired else 0)))
    rows = {n: {"result": r, "text": t} for n, (t, r) in steps.items()}
    (a.out / "steps.json").write_text(json.dumps({"completed": completed, "transcript": str(transcript),
                                                   "n": a.n, "steps": rows}, indent=1))
    tally = collections.Counter()
    for n, (t, r) in steps.items():
        if n == "refuse-pte":
            print(f"REFUSE-PTE {t.strip()[:80]}" + ("" if re.search(r"^RPTE Y$", t, re.M) else " VOID: refuse_pte did not read back Y"))
            continue
        if n == "bounds-0":
            print(f"BOUNDS-0 {r} {t.strip()[:200]}")
            continue
        if n == "dmesg-all":
            for l in t.strip().splitlines():
                print(f"DMESG-ALL {l[:300]}")
            m = re.search(r"^OLDEST \[\s*([0-9.]+)\]", t, re.M)
            if not m or float(m.group(1)) > 60:
                print(f"DMESG-ALL VOID: the ring's oldest line is {'unread' if not m else m.group(1) + ' s'} into the boot, "
                      f"so it has wrapped and REFUSED undercounts; the per-run reads are the reading")
            continue
        if n.startswith("keep-"):
            causes = re.findall(r"^FL .*domain fault cause=(\d+) ", t, re.M)
            ret = re.search(r"^FL CONTROL uaf-malloc RETURNED", t, re.M)
            mark = re.search(r"^FL CONTROL uaf-malloc mark", t, re.M)
            hook = re.findall(r"^FL CAPSTONE_VM_TEST_KEEP_RETIRED base=0x([0-9a-f]+) bytes=(\d+)", t, re.M)
            ad = re.search(r"^FL .*domain fault cause=\d+ pc=0x[0-9a-f]+ address=0x([0-9a-f]+)", t, re.M)
            inkept = bool(ad and any(int(b_, 16) <= int(ad.group(1), 16) < int(b_, 16) + int(n_) for b_, n_ in hook))
            kind = "SILENT" if ret else ("none" if not causes else causes[0] if causes[0] in ("13", "25") else "other")
            tally[("keep-traced" if n.endswith("-traced") else "keep", kind)] += 1
            tally[("keep-hook", "fired" if hook else "NOT-FIRED")] += 1
            tally[("keep-reached-read", "mark" if mark else "no-mark")] += 1
            if causes:
                tally[("keep-fault-in-kept-region", inkept)] += 1
            if kind != "25":
                print(f"KEEP {n} rc={r} kind={kind} causes={causes} hook={hook}")
            continue
        if n.startswith("map-"):
            causes = re.findall(r"^FL .*domain fault cause=(\d+) ", t, re.M)
            surv = re.findall(r"^FL STALE_READ_SURVIVED value=(\d+)", t, re.M)
            mark = re.findall(r"^FL CONTROL uaf-mapped mark same_page_keep=(\d) same_page_keep2=(\d)", t, re.M)
            kind = "SURVIVED" if surv else ("none" if not causes else causes[0] if causes[0] in ("13", "25") else "other")
            tally[("mapped-traced" if n.endswith("-traced") else "mapped", kind)] += 1
            tally[("mapped-reached-read", "mark" if mark else "no-mark")] += 1
            if mark:
                tally[("mapped-same-page", "keep=%s keep2=%s" % mark[0])] += 1
            if kind != "25":
                print(f"MAP {n} rc={r} kind={kind} surv={surv} causes={causes}")
            continue
        causes = re.findall(r"^FL .*domain fault cause=(\d+) ", t, re.M)
        arm = n.rsplit("-", 1)[1]
        kind = "none" if not causes else causes[0] if causes[0] in ("13", "25") else "other"
        if len(causes) > 1:
            kind += "+more"
        tally[(arm, kind)] += 1
        if kind != "25":
            print(f"RUN {n} rc={r} causes={causes}")
        if a.dmesg and ("cause=13 " in t or int(n.split("-")[1]) % 10 == 0):
            dm = re.findall(r"^DM .*$", t, re.M)
            ref = [l for l in dm if "resolve refused" in l]
            # must-fire: every printed run's ring holds at least the module's own lines (the pr_emerg "release retires"
            # of the previous run, or the load banner), so 0 DM lines means the read did not run, not "no refusal"
            print(f"DMESG {n} cause={causes[0] if causes else None} dm_lines={len(dm)} refused={len(ref)}")
            for l in ref:
                print(f"  {l[:300]}")
    for (arm, kind), c in sorted(tally.items()):
        print(f"TALLY {arm} cause {kind}: {c}")
    want = a.n * (1 + (1 if a.mapped else 0) + (1 if a.keep_retired else 0)) + 1 + (1 if a.dmesg else 0) + (1 if a.refuse_pte else 0)
    print(f"completed={completed} steps={len(steps)} of {want} transcript={transcript}")
    if len(steps) != want:
        print(f"NO DATA: {len(steps)} of {want} steps in the transcript")
        sys.exit(2)


if __name__ == "__main__":
    main()
