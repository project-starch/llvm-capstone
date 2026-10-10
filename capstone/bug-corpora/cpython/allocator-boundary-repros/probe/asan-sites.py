#!/usr/bin/env python3
"""Where host ASan reports each case's access: the functions a Capstone fault may land in (fault_sites).

    asan-sites.py run   ASAN_PYTHON OUT.tsv [--only NN,NN]   run every trigger on a host ASan build
    asan-sites.py apply SITES.tsv                             write fault_sites into each case.json

`run` executes each case's trigger.py, from its own directory, with PYTHONMALLOC=malloc so every
allocation is libc's and ASan sees the access whichever allocator owns the object on Capstone
(README.md, "The side is measured"), and ASAN_OPTIONS=detect_leaks=0 so a leak summary cannot bury
the report. From the FIRST report it takes the innermost stack frames:

  * every function at the first frame's pc. ASan prints an inlined chain as several frames with
    one pc; a Capstone fault resolves through the image's symbol table to the outermost function of
    that chain, so the whole chain is admissible and nothing else at that depth is.
  * when that first pc is an interceptor or libc routine (memcpy, strlen, ...), its libc name and
    the functions at the next pc too: on Capstone the same access faults inside musl's routine.

This run is the independent earlier run SCHEMA.md asks fault_sites to come from: it uses no Capstone
arm and is committed before any arm runs. A case whose trigger produces no ASan report gets no
sites, and its Capstone faults then cannot be attributed (NO-READING: unattributed), which is the
honest outcome for an access nothing located.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1]
LIBC = {"memcpy", "memmove", "memset", "memcmp", "strlen", "strnlen", "strcmp", "strncmp", "strcpy",
        "strncpy", "strchr", "strrchr", "memchr", "wcslen", "wmemcpy", "wmemcmp", "bcmp", "free",
        "realloc", "malloc", "calloc"}
REPORT = re.compile(r"ERROR: AddressSanitizer: (\S+)")
ACCESS = re.compile(r"^(READ|WRITE) of size (\d+)")
FRAME = re.compile(r"^\s*#(\d+) (0x[0-9a-f]+) in (\S+)(?: (\S+))?")


def libc_name(func, where):
    name = re.sub(r"^(__interceptor_|__asan_|___interceptor_|__interceptor_trampoline_)", "", func)
    if name in LIBC or "libsanitizer" in (where or "") or "sanitizer_common" in (where or ""):
        return name
    return None


def parse(text):
    """(kind, access, [frame lines used], [sites]) of the first report, or None."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if REPORT.search(l)), None)
    if start is None:
        return None
    kind = REPORT.search(lines[start]).group(1)
    access, frames = "", []
    for line in lines[start + 1:]:
        m = ACCESS.match(line)
        if m and not access:
            access = f"{m.group(1)} {m.group(2)}"
        f = FRAME.match(line)
        if f:
            frames.append((int(f.group(1)), f.group(2), f.group(3), f.group(4) or ""))
        elif frames and not line.strip():
            break                                    # the first stack ends at a blank line
    if not frames:
        return kind, access, [], []
    groups = []
    for _, pc, func, where in frames:
        if groups and groups[-1][0] == pc:
            groups[-1][1].append((func, where))
        else:
            groups.append((pc, [(func, where)]))
    used = list(groups[0][1])
    sites = [f for f, _ in groups[0][1]]
    if len(groups[0][1]) == 1 and libc_name(*groups[0][1][0]):
        sites = [libc_name(*groups[0][1][0])]
        if len(groups) > 1:
            used += groups[1][1]
            sites += [f for f, _ in groups[1][1]]
    shown = [f"{f} {re.sub(r'^.*?/Python-3.13.7/', '', w)}" for f, w in used]
    return kind, access, shown, list(dict.fromkeys(sites))


def run(python, out, only):
    env = dict(os.environ, PYTHONMALLOC="malloc", PYTHONDONTWRITEBYTECODE="1",
               ASAN_OPTIONS="detect_leaks=0:abort_on_error=0:symbolize=1")
    rows = []
    for d in sorted(CORPUS.glob("[0-9][0-9]_*")):
        if only and d.name[:2] not in only:
            continue
        try:
            p = subprocess.run([python, "trigger.py"], cwd=d, env=env, capture_output=True,
                               text=True, errors="replace", timeout=180)
            text, rc = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired:
            text, rc = "", "timeout"
        got = parse(text)
        if got is None:
            rows.append((d.name, "no-report", "", "", "", f"rc={rc}"))
        else:
            kind, access, shown, sites = got
            rows.append((d.name, kind, access, ",".join(sites), " | ".join(shown), f"rc={rc}"))
        print("\t".join(rows[-1][:4]), flush=True)
    with open(out, "w") as f:
        f.write("case\tkind\taccess\tsites\tframes\tstatus\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def apply(tsv):
    rows = [l.rstrip("\n").split("\t") for l in Path(tsv).read_text().splitlines()[1:]]
    stamp = Path(tsv).parent.name
    for case, kind, access, sites, frames, status in rows:
        path = CORPUS / case / "case.json"
        j = json.loads(path.read_text())
        if sites:
            j["fault_sites"] = sites.split(",")
            j["fault_sites_why"] = (
                f"Host ASan (CPython 3.13.7 --with-address-sanitizer, PYTHONMALLOC=malloc, "
                f"results/{stamp}/sites.tsv) reports {kind} ({access}) at {frames}. That run uses no "
                f"Capstone arm and was recorded before any virtual arm ran.")
        else:
            j.pop("fault_sites", None)
            j.pop("fault_sites_why", None)
        path.write_text(json.dumps(j, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "run":
        only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else set()
        run(sys.argv[2], sys.argv[3], only)
    elif len(sys.argv) == 3 and sys.argv[1] == "apply":
        apply(sys.argv[2])
    else:
        sys.exit(__doc__)
