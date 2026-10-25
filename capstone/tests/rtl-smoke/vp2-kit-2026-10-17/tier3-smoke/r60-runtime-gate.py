#!/usr/bin/env python3
"""r60-runtime-gate.py MANIFEST | --rev REV: may this image boot on a bitstream carrying R-60's RTL fix (CINCOFFSET/SCC clear the
tag of an out-of-window result)? A runtime without 232d13b2's integers-first slot offsets faults at its first allocation
there (cause 24 in __capstone_malloc_group_map, the compiler lane's QEMU probe). The fix is carried by descent (232d13b2) or
by port (5620badda510, under 8f132305), so ancestry is NOT the test (it refuses 8f132305): the manifest's runtime_revision
must have capstone/runtime/virtual/heap.c containing BOTH hunks' text (the runtime lane's rule, 2026-10-20), and
runtime_dirty must be false. MANIFEST may also be a bare revision (--rev REV). The repository is $CAPSTONE_REPO, default ~/dev/llvm-capstone. Exit 0 PASS, 1 REFUSE, 2 NO DATA."""
import json, os, subprocess, sys
HUNKS = ["(char *)group + (start - base)", "(r->key - __builtin_capstone_cap_get_base(r->parent))"]
def main():
    a = sys.argv[1:]
    repo = os.environ.get("CAPSTONE_REPO", os.path.expanduser("~/dev/llvm-capstone"))
    if a and a[0] == "--rev":
        rev, dirty = a[1], False
    else:
        try:
            m = json.load(open(a[0])); rev, dirty = m["runtime_revision"], m.get("runtime_dirty")
        except Exception as e:
            print(f"NO DATA: {e}"); return 2
    if dirty is not False:
        print(f"REFUSE: runtime_dirty is {dirty!r}, so the revision does not describe the bytes"); return 1
    p = subprocess.run(["git", "-C", repo, "show", f"{rev}:capstone/runtime/virtual/heap.c"], capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout:
        print(f"NO DATA: git show {rev}:capstone/runtime/virtual/heap.c failed: {p.stderr.strip()[:200]}"); return 2
    have = [h in p.stdout for h in HUNKS]
    if all(have):
        print(f"PASS: {rev[:12]} heap.c carries both hunks"); return 0
    print(f"REFUSE: {rev[:12]} heap.c hunks present {have} (bind_slots, backend_pointer)"); return 1
sys.exit(main())
