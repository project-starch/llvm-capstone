#!/usr/bin/env python3
"""spec-classify.py <transcript> [--twin <twin serial.log>]: the SPEC readings of a boot-2/3 vbatch segment (or a twin
serial.log). Per command: bench, arm, rc, wall seconds (uptime delta) and the output files' sha256. With --twin, each
command's output hashes are compared with the twin's for the same bench and arm (the same binary on QEMU).

Two transcript formats are read (kitfix 2026-11-23):
- module route: `@@SPEC-END <i> <bench> <arm> rc=...`, per-command `@@SPEC-HASH-BEGIN <i>`, `@@SPEC-DONE <n>`;
- native route (run-stage.py's run.sh): `@@SPEC-STAGE <bench> <size> ...`, `@@SPEC-END <i> <kind> <arm> <rep> rc=...`,
  one `@@SPEC-HASH-BEGIN <arm>` block per arm, `@@SPEC-DONE <bench> <size>`. Before this fix every native transcript
  read NO RESULT, because the regexes were the module route's only.
On the native route the hashes are per arm, so --twin compares each arm's output set once.

Exit 0: every command that began ended, @@SPEC-DONE was printed, rc 0 everywhere, and with --twin every output hash
matches. 1: a command failed or a hash differs (the host then runs specdiff on the in-band outputs). 2: NO RESULT (no
@@SPEC-DONE: the command that began and never ended is named) or MALFORMED. Never prints a pass it did not read."""
import argparse
import re
import sys


def segment(path):
    t = open(path, errors="replace").read().replace("\r", "")
    s = re.search(r"^### RUNG \d+/\d+ vbatch ###$", t, re.M)
    if s:
        t = t[s.end():]
        e = re.search(r"^### RUNG \d+ vbatch END rc=-?\d+ ###$", t, re.M)
        t = t[:e.start()] if e else t
    return t


def hashes_in(block):
    return dict((f, h) for h, f in re.findall(r"^([0-9a-f]{64})  (\S+)$", block, re.M)) if block else {}


def read(path):
    t = segment(path)
    stage = re.search(r"^@@SPEC-STAGE (\S+) (\S+)", t, re.M)
    cmds = {}
    if stage:
        bench = stage.group(1)
        arm_hashes = {}
        for m in re.finditer(r"^@@SPEC-HASH-BEGIN (\S+)$(.*?)^@@SPEC-HASH-END \1$", t, re.M | re.S):
            arm_hashes[m.group(1)] = hashes_in(m.group(2))
        for m in re.finditer(r"^@@SPEC-END (\d+) (\S+) (\S+) (\d+) rc=(-?\d+) up0=([\d.]+) up1=([\d.]+)", t, re.M):
            i = int(m.group(1))
            cmds[(m.group(3), i)] = dict(bench=bench, kind=m.group(2), arm=m.group(3), rc=int(m.group(5)),
                                         secs=float(m.group(7)) - float(m.group(6)))
        begun = [((a, int(i)), f"{i} {k} {a}") for i, k, a in re.findall(r"^@@SPEC-BEGIN (\d+) (\S+) (\S+) \d+$", t, re.M)]
        # `@@SPEC-DONE <bench> <size>` (SPEC stages) or `@@SPEC-DONE <n>` (the pgbench-derived stages' run.sh)
        done = [d.split() for d in re.findall(r"^@@SPEC-DONE (\S+(?: \S+)?)$", t, re.M)]
        done_ok = bool(done) and (done[-1] == [stage.group(1), stage.group(2)] or
                                  (len(done[-1]) == 1 and done[-1][0].isdigit() and int(done[-1][0]) == len(cmds)))
        return dict(native=True, cmds=cmds, arm_hashes=arm_hashes, begun=begun, done=done, done_ok=done_ok,
                    skipped=re.findall(r"^@@SPEC-SKIPPED (.*)$", t, re.M))
    for m in re.finditer(r"^@@SPEC-END (\d+) (\S+) (\S+) rc=(-?\d+) up0=([\d.]+) up1=([\d.]+)", t, re.M):
        i = int(m.group(1))
        hb = re.search(rf"^@@SPEC-HASH-BEGIN {i}$(.*?)^@@SPEC-HASH-END {i}$", t, re.M | re.S)
        cmds[i] = dict(bench=m.group(2), kind="run", arm=m.group(3), rc=int(m.group(4)),
                       secs=float(m.group(6)) - float(m.group(5)), hashes=hashes_in(hb.group(1) if hb else ""))
    begun = [(int(i), f"{i} {b} {a}") for i, b, a in re.findall(r"^@@SPEC-BEGIN (\d+) (\S+) (\S+)$", t, re.M)]
    done = re.findall(r"^@@SPEC-DONE (\d+)$", t, re.M)
    return dict(native=False, cmds=cmds, begun=begun, done=done,
                done_ok=bool(done) and int(done[-1]) == len(cmds),
                skipped=re.findall(r"^@@SPEC-SKIPPED (.*)$", t, re.M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--twin")
    a = ap.parse_args()
    r = read(a.transcript)
    if r["skipped"]:
        print(f"NO RESULT: SPEC skipped ({r['skipped'][0]})")
        return 2
    tw = read(a.twin) if a.twin else None
    bad = 0
    for key in sorted(r["cmds"], key=str):
        c = r["cmds"][key]
        bad += c["rc"] != 0
        print(f"{key[1] if r['native'] else key} {c['bench']:10s} {c['kind']:4s} {c['arm']:6s} rc={c['rc']} {c['secs']:.2f}s")
    if a.twin:
        if r["native"]:
            for arm, hs in sorted(r["arm_hashes"].items()):
                th = tw["arm_hashes"].get(arm) if tw and tw["native"] else None
                if th is None:
                    print(f"{arm}: twin absent"); bad += 1; continue
                diff = sorted(f for f in set(hs) | set(th) if hs.get(f) != th.get(f))
                print(f"{arm}: {len(hs)} outputs, twin " + ("identical" if not diff else f"DIFFER {diff}"))
                bad += bool(diff)
        else:
            twin = {(c["bench"], c["arm"]): c for c in tw["cmds"].values()} if tw and not tw["native"] else {}
            for i, c in sorted(r["cmds"].items()):
                t = twin.get((c["bench"], c["arm"]))
                if t is None:
                    print(f"{i}: twin absent"); bad += 1; continue
                diff = sorted(f for f in set(c["hashes"]) | set(t["hashes"]) if c["hashes"].get(f) != t["hashes"].get(f))
                print(f"{i}: twin " + ("outputs identical" if not diff else f"DIFFER {diff}"))
                bad += bool(diff)
    open_ = [name for key, name in r["begun"] if key not in r["cmds"]]
    if not r["done"]:
        print(f"NO RESULT: no @@SPEC-DONE; began and never ended: {open_ or 'none'}")
        return 2
    if open_ or not r["done_ok"] or not r["cmds"]:
        print(f"MALFORMED: @@SPEC-DONE {' '.join(map(str, r['done'][-1])) if r['native'] else r['done'][-1]} "
              f"with {len(r['cmds'])} commands ended, never ended: {open_ or 'none'}")
        return 2
    print("PASS" if not bad else f"FAIL: {bad} command(s) failed or differ from the twin")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
