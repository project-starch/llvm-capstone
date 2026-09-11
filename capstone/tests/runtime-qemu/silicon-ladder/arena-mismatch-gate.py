#!/usr/bin/env python3
"""Refuse a speedtest1 run whose arena is not the size it was built for.

THE FAILURE THIS EXISTS TO CATCH. The arena size reaches the HOST only
(run-speedtest1-measure.sh:111 puts SPEEDTEST1_ARENA_SIZE in HOST_EXTRA_DEFS and nowhere
else), the host creates the region, and the domain uses whatever it is handed. Nothing in
that chain compares the three numbers, so a run that quietly got a smaller arena than it
was built for looks exactly like a healthy one -- until it dies hours into a multi-hour
arm, or worse, completes at a size nobody intended and gets recorded as a measurement.
On an 8-hour board pair that is the whole session.

THREE NUMBERS, AND THEY MUST ALL AGREE:

  1. the size compiled into the host binary          (static, read back from the artifact)
  2. `SQ: arena_bytes=<N>` in the transcript         (what the host actually created)
  3. `HEAP <N>` in the transcript                    (what the domain actually used)

Checking only 2 against 3 would pass a run that consistently used the wrong size, which is
precisely the fallback case; checking only 1 would pass a run that never reached the arena
at all. So the gate wants all three, and wants them equal to --expect.

"CANNOT CHECK" IS AN ERROR, NOT A PASS. A missing marker, an unreadable binary, an empty
disassembly and an absent objdump all exit non-zero and say where the gate looked. A tool
that renders "no data" as a clean result is how a wasted boot gets signed off.

  arena-mismatch-gate.py --expect 134217728 --host sqlite_host.user --log run.log
  arena-mismatch-gate.py --selftest --host sqlite_host.user       # negative-test it

WHAT IT CANNOT SEE, stated so nobody reads a pass as more than it is:
  - The static half recognises the RISC-V `lui` form, which covers any size whose low 12
    bits are zero -- every power-of-two arena we use. A size that needs lui+addi is
    reported as UNCHECKABLE rather than passed.
  - It reads the transcript, so it cannot distinguish a region that was created at the
    right size from one that was created and then not used for the heap. `HEAP` is the
    domain's own report and is the closest available witness to the latter.
  - It CANNOT tell a static-heap run from a region-arena run that fell back to the static
    heap: neither emits `SQ: arena_bytes=`. It blocks both and says so. Run it only on
    SPEEDTEST1_REGION_ARENA=1 runs, where a missing marker really is the failure.
  - It says nothing about whether the arena was big ENOUGH -- that is the heap sweep's job.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path


class Cannot(Exception):
    """The gate could not complete a check. Never reported as a pass.

    `kind` separates the two failures that must not be collapsed:
      "tool" -- the GATE is broken: no disassembler, unreadable artifact, a form it does
                not recognise. Says nothing about the run. Exit 2.
      "run"  -- the RUN is wrong or silent: a missing marker, an empty transcript,
                contradictory values. A real finding about the subject. Exit 1.
    Both are non-zero, because an unproven check is not a pass.
    """

    def __init__(self, message, kind="tool"):
        super().__init__(message)
        self.kind = kind


def disassemble(binary: Path, objdump: str) -> str:
    if not binary.is_file():
        raise Cannot(f"host binary not found: {binary}")
    try:
        out = subprocess.run([objdump, "-d", str(binary)],
                             capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        raise Cannot(f"no disassembler at {objdump} (pass --objdump)")
    except subprocess.TimeoutExpired:
        raise Cannot(f"{objdump} timed out on {binary}")
    if out.returncode != 0:
        raise Cannot(f"{objdump} failed on {binary}: {out.stderr.strip()[:200]}")
    if len(out.stdout.splitlines()) < 10:
        raise Cannot(f"{objdump} produced an empty disassembly for {binary} -- "
                     "an empty search space finds nothing and reads like a pass")
    return out.stdout


def static_sites(dis: str, expect: int):
    """Count the sites materialising `expect`. Raises Cannot if the form is unsupported."""
    if expect & 0xFFF:
        raise Cannot(f"--expect {expect} has non-zero low 12 bits; this gate only "
                     "recognises the lui form. Treat as UNCHECKABLE, not as a pass.")
    upper = expect >> 12
    pat = re.compile(rf"\blui\s+\w+,\s*(?:0x{upper:x}|{upper})\b")
    return [ln.strip() for ln in dis.splitlines() if pat.search(ln)]


MARKERS = {
    "arena_bytes": re.compile(rb"SQ: arena_bytes=(\d+)"),
    "HEAP":        re.compile(rb"\bHEAP (\d+)"),
}


def log_values(log: Path):
    if not log.is_file():
        raise Cannot(f"transcript not found: {log}", kind="tool")
    blob = log.read_bytes()
    if not blob:
        raise Cannot(f"transcript is empty: {log} -- a run that produced no output "
                     "is a finding about the run, not about this gate", kind="run")
    found = {}
    for name, pat in MARKERS.items():
        hits = {int(m.group(1)) for m in pat.finditer(blob)}
        if not hits:
            # THE ONE AMBIGUITY THIS GATE CANNOT RESOLVE, said out loud rather than
            # implied. `SQ: arena_bytes=` is emitted only on the region-arena path, so a
            # transcript without it is either a static-heap run (this is the wrong tool)
            # or a region-arena run that fell back to the static heap (the exact failure
            # the gate exists to catch). Both are BLOCKED, because passing either one
            # would mean passing the second.
            if name == "arena_bytes" and MARKERS["HEAP"].search(blob):
                raise Cannot(
                    f"no `arena_bytes` marker in {log}, but `HEAP` is present. This gate "
                    "cannot tell a static-heap run (wrong tool -- use it only on "
                    "SPEEDTEST1_REGION_ARENA=1 runs) from a region-arena run that fell "
                    "back to the static heap (the failure it exists to catch). BLOCKED "
                    "either way, because passing the first would pass the second.",
                    kind="run")
            raise Cannot(f"no `{name}` marker in {log} -- the run did not report its arena, "
                         "so there is nothing to check and this is not a pass", kind="run")
        if len(hits) > 1:
            raise Cannot(f"`{name}` reported more than one value in {log}: "
                         f"{sorted(hits)} -- ambiguous, refusing to pick one", kind="run")
        found[name] = hits.pop()
    return found


def run(expect, host, log, objdump, out=sys.stdout):
    problems, checked, tool_faults = [], [], 0
    if host is not None:
        try:
            sites = static_sites(disassemble(Path(host), objdump), expect)
            if not sites:
                problems.append(f"host binary {host} never materialises {expect:,} "
                                f"(lui 0x{expect >> 12:x}) -- it was built for a different arena")
            else:
                checked.append(f"host binary materialises {expect:,} at {len(sites)} site(s)")
        except Cannot as exc:
            problems.append(f"STATIC CHECK COULD NOT RUN: {exc}")
            tool_faults += exc.kind == "tool"
    if log is not None:
        try:
            vals = log_values(Path(log))
            for name, got in sorted(vals.items()):
                if got != expect:
                    problems.append(f"{name} reports {got:,}, expected {expect:,}")
                else:
                    checked.append(f"{name} reports {expect:,}")
            if len(set(vals.values())) > 1:
                problems.append("the transcript's own numbers disagree with each other: "
                                + ", ".join(f"{k}={v:,}" for k, v in sorted(vals.items())))
        except Cannot as exc:
            problems.append(f"TRANSCRIPT CHECK COULD NOT RUN: {exc}")
            tool_faults += exc.kind == "tool"
    if host is None and log is None:
        print("arena-mismatch-gate: nothing to check -- pass --host and/or --log", file=out)
        return 2
    for line in checked:
        print(f"  ok   {line}", file=out)
    for line in problems:
        print(f"  FAIL {line}", file=out)
    if not problems:
        print("arena-mismatch-gate: clean", file=out)
        return 0
    # TWO KINDS OF FAILURE, AND THEY ARE NOT THE SAME NEWS. A size mismatch means the run is
    # wrong; "could not run" means the GATE is wrong -- a missing disassembler, an unreadable
    # artifact. Both are non-zero, because an unproven check is never a pass, but collapsing
    # them into one verdict sends someone hunting an arena bug when the real fault is a tool
    # path. That is not hypothetical: in a git worktree CAPSTONE_LLVM_BIN points at a build
    # directory that does not exist, and every run would have read as BLOCKED.
    if tool_faults == len(problems):
        print("arena-mismatch-gate: CANNOT CHECK -- the gate could not do its job, which is "
              "not a pass and not an arena fault; fix the tool path and re-run", file=out)
        return 2
    print("arena-mismatch-gate: BLOCKED", file=out)
    return 1


def selftest(host, objdump):
    """Negative-test the gate the day it is written, against the REAL artifacts.

    A gate that has never blocked anything is not a passing gate, it is an unproven one.
    """
    import io
    import tempfile

    good = b"SQ: F2/share3\nSQ: arena_bytes=134217728\nHEAP 134217728\nSPEEDTEST1-CYCLES 1\n"
    cases = [
        ("matching transcript passes",                   good, 134217728, 0),
        ("a domain that fell back to the static heap",
         b"SQ: arena_bytes=134217728\nHEAP 2097152\n",         134217728, 1),
        ("a host that created a smaller region",
         b"SQ: arena_bytes=2097152\nHEAP 2097152\n",           134217728, 1),
        ("a transcript missing the arena marker",
         b"HEAP 134217728\nSPEEDTEST1-CYCLES 1\n",             134217728, 1),
        ("a real static-heap transcript -- indistinguishable from a fallback, so BLOCKED",
         b"SQ: G/enter\nHEAP 2097152\nSPEEDTEST1-CYCLES 1\n",  2097152,   1),
        ("an empty transcript is an ERROR, not a zero",  b"",  134217728, 1),
        ("the right transcript against the wrong --expect",
         good, 6291456, 1),
    ]
    missing_tool = [("a missing disassembler is CANNOT CHECK (2), not BLOCKED (1)", 2)]
    ok = True
    for name, blob, expect, want in cases:
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as fh:
            fh.write(blob)
            path = fh.name
        got = run(expect, None, path, objdump, out=io.StringIO())
        Path(path).unlink()
        mark = "ok  " if got == want else "FAIL"
        ok &= got == want
        print(f"  {mark} {name}: exit {got} (wanted {want})")
    for name, want in missing_tool:
        got = run(134217728, host or __file__, None, "/nonexistent/llvm-objdump",
                  out=io.StringIO())
        mark = "ok  " if got == want else "FAIL"
        ok &= got == want
        print(f"  {mark} {name}: exit {got} (wanted {want})")
    if host:
        got = run(134217728, host, None, objdump, out=io.StringIO())
        print(f"  {'ok  ' if got == 0 else 'FAIL'} real host binary at its real size: "
              f"exit {got} (wanted 0)")
        ok &= got == 0
        got = run(6291456, host, None, objdump, out=io.StringIO())
        print(f"  {'ok  ' if got == 1 else 'FAIL'} real host binary at a size it was NOT "
              f"built for: exit {got} (wanted 1)")
        ok &= got == 1
    else:
        print("  WARN no --host given: the static half was not exercised, so this "
              "selftest proves only the transcript half")
    print("selftest: " + ("every case behaved as intended" if ok else "THE GATE IS BROKEN"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expect", type=int, help="intended arena size in bytes")
    ap.add_argument("--host", help="the sqlite_host.user artifact")
    ap.add_argument("--log", help="the run transcript")
    ap.add_argument("--objdump", default="llvm-objdump")
    ap.add_argument("--selftest", action="store_true",
                    help="negative-test the gate; pass --host to exercise the static half")
    args = ap.parse_args()
    if args.selftest:
        return selftest(args.host, args.objdump)
    if args.expect is None:
        ap.error("--expect is required (or use --selftest)")
    return run(args.expect, args.host, args.log, args.objdump)


if __name__ == "__main__":
    sys.exit(main())
