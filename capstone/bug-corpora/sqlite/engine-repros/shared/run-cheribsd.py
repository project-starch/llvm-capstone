#!/usr/bin/env python3
"""Run the SQLite engine corpus on CheriBSD purecap, and record what each case showed.

    run-cheribsd.py OUT --mode nested|non-nested [--only 00,19,...] [--no-build]

THE CASE LIST AND THE CASE SOURCES COME FROM THIS CORPUS. `ports/sqlite/cheribsd/` holds
the cross-compiler driver and the guest scripts; this runner invokes the first and does the
rest itself, so what is measured is decided here. That matters: for most of this review the
CheriBSD arm compiled hand-maintained copies of the cases under `ports/.../cases/`, three of
which had drifted into different programs (cases 03, 21 and 30 -- 30's copy ran a PRAGMA
before its own trigger, so the defect was unreachable and the arm scored a clean completion).
build-corpus-cheri.sh now takes the case program from the corpus; this runner checks that it
actually did, per binary, before trusting a verdict.

THE TWO MODES are one build flag and one environment variable apart, and they are the two
halves of the nesting axis for this program:

  nested      memsys5 over `static unsigned char sqlite_heap[arena_bytes]`. Every SQLite
              object is a sub-allocation inside one array, so the bounds a pointer carries
              are the ARENA's and revocation never sees a sub-allocation end.
  non-nested  no SQLITE_CONFIG_HEAP, so SQLite's allocator is CheriBSD's malloc: each object
              bounded to itself and its lifetime known to the mechanism. Built SYSALLOC=1
              (the `_sys` binaries, with the purecap alignment fix mem1.c needs) and run with
              NOMEM5=1, which also forces a revoke on every free at SQLite's allocator
              boundary -- without that the guest's quarantine has not swept when the stale
              load happens and every case reports PASS for a reason that is not the mechanism.

GATE. `poscontrol` -- a plain system-allocator use-after-free -- must FAULT in the same
session or the run is void and no verdict is written. A silent case proves nothing on an arm
that cannot be shown to report.

VERDICTS. exit 162 = SIGPROT(34), a capability violation, and `si_code` says which kind:
si_code=1 PROT_CHERI_BOUNDS is a valid capability used out of bounds -- the mechanism
reporting the violation itself; si_code=2 PROT_CHERI_TAG is dereferencing an untagged value,
which on this corpus has always meant the defective access already succeeded and this is its
consequence. The two are recorded separately and are not the same evidence.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve()
CORPUS = HERE.parents[1]
REPO = HERE.parents[5]
PORT = REPO / "capstone/ports/sqlite/cheribsd"
TAGS = REPO / "capstone/ports/sqlite/repro322/tags.tsv"
SSH = ["-i", str(pathlib.Path.home() / ".ssh/id_ed25519"), "-o", "BatchMode=yes",
       "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
       "-o", "ConnectTimeout=20"]
PORT_NO = "10086"


def guest(cmd, timeout=400):
    r = subprocess.run(["ssh", "-n", *SSH, "-p", PORT_NO, "root@localhost", cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout, r.returncode


def case_tags():
    """{case dir name: tag}, from tags.tsv -- the corpus's own accounting of what is a case."""
    out = {}
    for line in TAGS.read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 3 and f[2] == "case":
            out[f[1]] = f[0]
    return out


def tag_groups():
    """{tag: build group}, parsed out of the driver's manifest."""
    body = re.search(r"manifest\(\)\s*\{(.*?)\n\}", (PORT / "build-corpus-cheri.sh").read_text(), re.S)
    groups, cur = {}, None
    for line in body.group(1).splitlines():
        m = re.match(r"\s*([a-zA-Z0-9]+)\)\s*cat", line)
        if m:
            cur = m.group(1)
            continue
        # Two shapes in the manifest: a heredoc listing "<file>.c <tag>" per line, and a
        # one-liner `group) echo "<file>.c <tag>" ;;`. Parsing only the first missed the
        # whole coreT group, so case 05 was never built and then skipped as unbuilt.
        m = re.match(r'\s*([a-zA-Z0-9]+)\)\s*echo\s+"([^"]*)"', line)
        if m:
            one = re.match(r"\s*\S+\.c\s+(\S+)", m.group(2))
            if one:
                groups[one.group(1)] = m.group(1)
            continue
        m = re.match(r"\s*\S+\.c\s+(\S+)", line)
        if m and cur:
            groups[m.group(1)] = cur
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=pathlib.Path)
    p.add_argument("--mode", required=True, choices=["nested", "non-nested"])
    p.add_argument("--only", help="comma-separated case number prefixes")
    p.add_argument("--no-build", action="store_true", help="use what is already in WORK/out")
    p.add_argument("--work", type=pathlib.Path,
                   default=pathlib.Path.home() / "arms/sqlite/cheribsd")
    a = p.parse_args()

    sysalloc = a.mode == "non-nested"
    suffix = "_sys" if sysalloc else ""
    envp = "NOMEM5=1 " if sysalloc else ""

    tags = case_tags()
    groups = tag_groups()
    want = sorted(d for d in CORPUS.glob("[0-9][0-9]_*")
                  if d.is_dir() and d.name in tags
                  and (not a.only or d.name[:2] in a.only.split(",")))
    if not want:
        sys.exit("no cases selected")
    print(f"mode={a.mode}  cases={len(want)}  (list and sources from {CORPUS})", flush=True)

    # ---- build, through the port's driver ----
    if not a.no_build:
        need = sorted({groups[tags[d.name]] for d in want if tags[d.name] in groups})
        for g in need:
            env = {"SYSALLOC": "1"} if sysalloc else {}
            r = subprocess.run(["bash", "./build-corpus-cheri.sh", g], cwd=PORT,
                               capture_output=True, text=True,
                               env={**os.environ, **env})
            bad = [l for l in r.stdout.splitlines() if "FAIL" in l]
            print(f"  build {g:<8} rc={r.returncode} {'  '.join(bad) if bad else 'ok'}", flush=True)
            if r.returncode != 0:
                # A non-zero rc is a failed build, whatever the per-case lines say. Reading
                # only the FAIL lines reported "ok" for eleven groups that had all aborted
                # on an unset variable, and the run then scored nothing at all.
                sys.exit(f"build of group {g} failed (rc={r.returncode}):\n"
                         + "\n".join(r.stdout.splitlines()[-6:] + r.stderr.splitlines()[-6:]))

    # ---- per-binary checks, before anything is believed ----
    # The driver writes out/sources.tsv: tag -> the file it actually compiled. That is the
    # only way to tell a corpus-built binary from one built out of ports/.../cases/, because
    # only the nine image cases carry anything distinctive in the binary itself.
    sources = {}
    for smap in sorted((a.work / "out").glob("sources-*.tsv")):
        for line in smap.read_text().splitlines():
            f = line.split("\t")
            if len(f) == 2:
                sources[f[0]] = f[1]
    rows, skipped = [], []
    for d in want:
        tag = tags[d.name]
        binary = a.work / "out" / f"{tag}{suffix}"
        if not binary.is_file():
            skipped.append((d.name, f"no {binary.name}"))
            continue
        src = sources.get(f"{tag}{suffix}")
        if src is None:
            skipped.append((d.name, "not in the build's sources.tsv -- rebuild"))
            continue
        if not src.startswith(str(CORPUS)):
            skipped.append((d.name, f"built from {src}, not the corpus case.c"))
            continue
        rows.append((d, tag, binary))
    for name, why in skipped:
        print(f"  SKIP {name}: {why}", flush=True)

    # ---- ship ----
    if rows:
        subprocess.run(["scp", *SSH, "-P", PORT_NO, *[str(b) for _, _, b in rows],
                        "root@localhost:/root/corpus/"], capture_output=True, check=False)

    # ---- gate ----
    rev, _ = guest("sysctl -n security.cheri.runtime_revocation_default")
    pc_out, _ = guest("cd /root/corpus && ./poscontrol >/dev/null 2>&1; echo $?")
    pc_code = pc_out.strip().splitlines()[-1] if pc_out.strip() else "?"
    print(f"  gate poscontrol exit={pc_code}  revocation={rev.strip()}", flush=True)
    if pc_code != "162":
        sys.exit("GATE FAILED: the positive control did not fault; the run is void")

    # ---- run ----
    results = {}
    for d, tag, binary in rows:
        out, _ = guest(f"cd /root/corpus && chmod +x {tag}{suffix} && "
                       f"env {envp}timeout 300 ./{tag}{suffix} 2>&1; echo __EXIT=$?")
        code = re.search(r"__EXIT=(\d+)", out)
        code = int(code.group(1)) if code else -1
        si = re.search(r"si_code=(\d+) \(([^)]*)\)", out)
        alloc = re.search(r"allocator=(\S+)", out)
        revoke = re.search(r"revoke=(\S+)", out)
        verdict = ("FAULT" if code == 162 else "TIMEOUT" if code == 124 else
                   "PASS" if code == 0 and "RETURNED" in out else "ERR")
        results[d.name] = {
            "tag": tag, "mode": a.mode, "exit": code, "verdict": verdict,
            "si_code": si.group(1) if si else None,
            "si_meaning": si.group(2) if si else None,
            "allocator": alloc.group(1) if alloc else None,
            "revoke": revoke.group(1) if revoke else None,
            "last": [l for l in out.splitlines() if l.strip() and "__EXIT" not in l][-1:],
        }
        print(f"  {d.name[:48]:<48} {verdict:<8} exit={code:<4} "
              f"{('si_code=' + si.group(1) + ' ' + si.group(2)) if si else ''}")

    tally = {}
    for r in results.values():
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({
        "arm": "cheribsd-revocation", "mode": a.mode,
        "revocation_default": rev.strip(), "poscontrol_exit": pc_code,
        "skipped": dict(skipped), "tally": tally, "cases": results}, indent=2) + "\n")
    print(f"--- cheribsd-revocation ({a.mode}): {tally}   -> {a.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
