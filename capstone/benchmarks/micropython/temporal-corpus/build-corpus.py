#!/usr/bin/env python3
"""Build the MicroPython temporal-allocator corpus CSV.

Title, state, date and URL are COPIED from verified API responses (GitHub GraphQL
and the NVD REST API), never typed by hand. Only the classification columns are
authored here. `traps_unmodified` follows from a measurement (see
evidence/heap-bounds-model.s); `traps_if_gc_cap_aware` is a prediction about a
runtime that does not exist yet.
"""
import csv, json, sys, os

S = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(S, "temporal-allocator-corpus.csv")

gql = json.load(open(os.path.join(S, "github-issues.json")))["data"]["repository"]
issues = {v["number"]: v for v in gql.values() if v}
nvd = {v["cve"]["id"]: v["cve"] for v in json.load(open(os.path.join(S, "nvd-cves.json")))["vulnerabilities"]}

# id, ref, class, cwe, component, scope, trigger, capstone_hypothesis, notes
ROWS = [
    ("MPY-T01", "CVE-2023-7152", "uaf", "CWE-416", "extmod/modselect.c:poll_set_add_fd", "gc-managed",
     "poll object registers an fd, object freed, poll set still references it",
     "trapped", "Patch 8b24aa36ba97. Same defect as issue #12887."),
    ("MPY-T02", "CVE-2024-8947", "uaf", "CWE-416", "py/objarray.c", "gc-managed",
     "array/memoryview outlives the buffer it points into", "trapped",
     "Fixed in 1.23.0. Same defect family as issue #13283."),
    ("MPY-T03", "CVE-2026-1998", "memory-corruption", "CWE-119,CWE-787", "py/runtime.c:mp_import_all", "gc-managed",
     "import * over a module whose globals map is mutated during iteration",
     "unclear", "NVD says CWE-119/787, not 416. Patch 570744d06c5b."),
    ("MPY-T04", "#12887", "uaf", "CWE-416", "extmod/modselect.c:151", "gc-managed",
     "select object used after the registered stream was collected", "trapped",
     "Reporter-provided ASan trace. Underlies CVE-2023-7152."),
    ("MPY-T05", "#13283", "uaf", "CWE-416", "py/objarray.c:509", "gc-managed",
     "array resized while a view onto its old buffer is still live", "trapped",
     "ASan heap-use-after-free. Underlies CVE-2024-8947."),
    ("MPY-T06", "#12543", "uaf", "CWE-416", "extmod/modbtree.c", "gc-managed",
     "btree object used again after close() released its backing store", "trapped",
     "ASan heap-use-after-free, explicit reuse-after-close."),
    ("MPY-T07", "#4128", "uaf", "CWE-416", "py/lexer.c,py/compile.c", "gc-managed",
     "lex->source_name read after the lexer allocation was freed", "trapped",
     "Classic dangling pointer into a freed struct field."),
    ("MPY-T08", "#12670", "uaf", "CWE-416", "extmod/vfs_posix_file.c", "gc-managed",
     "close() on stdin/stdout then further use of the stream object", "trapped",
     "Reuse-after-close on a singleton stream."),
    ("MPY-T09", "#18168", "dangling-view", "CWE-416", "py/objarray.c", "gc-managed",
     "bytearray resized while active memoryviews still point at the old buffer",
     "trapped", "Open. Resize reallocates, views are not invalidated."),
    ("MPY-T10", "#18171", "dangling-view", "CWE-416", "py/objarray.c", "gc-managed",
     "array('I') resized leaves a stale memoryview; readinto() then writes through it",
     "trapped", "Open. Stale view becomes a heap-corruption write primitive."),
    ("MPY-T11", "#17941", "premature-free", "CWE-416", "py/objarray.c,py/gc.c", "gc-managed",
     "sort over an array while the collector runs, element buffer moved or freed",
     "trapped", "Open. GC interaction, not reproducible without collection pressure."),
    ("MPY-T12", "#18619", "reentrancy", "CWE-476", "py/objdict.c", "gc-managed",
     "re-entrant __bool__ clears the RHS map during dict equality, map pointer goes NULL",
     "unclear", "Open. Container mutated under an in-flight operation."),
    ("MPY-T13", "#19075", "dangling-buffer", "CWE-416", "extmod/modio.c", "gc-managed",
     "write() where slice assignment enlarges buf, reallocating under the caller",
     "trapped", "Open. Callback reallocates a buffer the caller still holds."),
    ("MPY-T14", "#19060", "dangling-buffer", "CWE-416", "extmod/vfs_blockdev.c", "gc-managed",
     "readblocks implementation enlarges the buffer it was handed", "trapped",
     "Same shape as MPY-T13 and MPY-T15, different call site."),
    ("MPY-T15", "#17848", "dangling-buffer", "CWE-416", "extmod/vfs_blockdev.c", "gc-managed",
     "readblocks enlarges buf via slice assignment, caller keeps the old pointer",
     "trapped", "Fixed. The canonical reallocate-under-caller case."),
    ("MPY-T16", "#5487", "lifetime-order", "CWE-416", "py/gc.c:gc_sweep_all", "gc-core",
     "port deinit hook runs after the sweep that already freed what it touches",
     "trapped", "Open. Shutdown ordering defect in the collector itself."),
    ("MPY-T17", "#3627", "lifetime-order", "CWE-667", "py/gc.c,py/objtype.c", "gc-core",
     "exception raised inside a __del__ finaliser deadlocks with threading on",
     "not-trapped", "Open. Liveness, not memory safety. Kept for taxonomy contrast."),
    ("MPY-T18", "#19413", "premature-free", "CWE-416", "py/gc.c + LVGL binding", "gc-managed",
     "gc.collect() frees a structure the C binding still holds, draw_task_head corrupt",
     "trapped", "Closed. C-side pointer invisible to the mark phase."),
    ("MPY-T19", "#5226", "premature-free", "CWE-416", "extmod/modbluetooth.c", "gc-managed",
     "gc.collect() collects a BLE object still referenced only from C", "trapped",
     "Closed. Same root shape as MPY-T18."),
    ("MPY-T20", "#6988", "premature-free", "CWE-416", "ports/esp32 + py/gc.c", "port-heap",
     "gc.collect() panics, collector state inconsistent with the IDF heap", "unclear",
     "Closed. Two allocators sharing one address space."),
    ("MPY-T21", "#8550", "premature-free", "CWE-416", "ports/rp2 + py/gc.c", "port-heap",
     "thread stack memory collected because it is not a scanned root", "trapped",
     "Closed. Root-set gap: a live pointer the mark phase cannot see."),
    ("MPY-T22", "#17442", "race-uaf", "CWE-362", "py/gc.c + vfs", "gc-core",
     "concurrent file access, allocator state mutated from two threads", "not-trapped",
     "Open. A data race; capabilities bound reach, not concurrency."),
    ("MPY-T23", "#12638", "premature-free", "CWE-416", "ports/esp32 lwip glue", "port-heap",
     "sporadic fault, buffer freed by the network stack still referenced", "unclear",
     "Closed as unreproducible. Kept as a known-suspect entry."),
    ("MPY-T24", "#11781", "uaf", "CWE-416", "ports/embed, py/compile.c", "gc-managed",
     "mp_compile() on an embed port faults on a freed lexer or parse tree", "trapped",
     "Closed. Related to MPY-T07."),
    ("MPY-T25", "#10402", "reentrancy", "CWE-416", "py/objstringio.c", "gc-managed",
     "printing to a StringIO subclass re-enters and invalidates the target buffer",
     "trapped", "Open. Buffer reallocated under an in-flight print."),
    ("MPY-T26", "#18645", "dangling-pointer", "CWE-476", "py/nativeglue.c:mp_native_relocate", "gc-managed",
     "relocation walks a pointer table that is NULL or already freed", "unclear",
     "Closed. NULL deref rather than a stale-but-mapped read."),
    ("MPY-T27", "#5272", "lifetime-order", "CWE-772", "ports/esp32 usocket", "port-heap",
     "failed connect() leaves the socket considered open, never cleaned up", "not-trapped",
     "Open. Leak plus a later reuse hazard, not a spatial violation."),
    ("MPY-T28", "#322", "alloc-invariant", "CWE-476", "py/gc.c:gc_realloc", "gc-core",
     "gc_realloc returns NULL after a few calls, caller keeps using the old block",
     "unclear", "Closed, 2014. Oldest specimen; allocator invariant, not yet a UAF."),
    ("MPY-T29", "#4705", "alloc-invariant", "CWE-617", "py/gc.c:gc_free,gc_realloc", "gc-core",
     "assertions in gc_free and gc_realloc fail on a pointer they consider invalid",
     "trapped", "Closed. Direct evidence the allocator cannot validate its own input."),
    ("MPY-T30", "#11698", "alloc-invariant", "CWE-401", "py/gc.c:gc_collect", "gc-core",
     "gc.collect() does not reclaim what the caller expects, blocks stay marked live",
     "not-trapped", "Open, needs-info. Over-retention, the dual of premature free."),
]

HEADER = ["id", "source", "ref", "url", "title", "state", "first_seen",
          "fix_commit", "fix_date", "present_at_pin", "repro_base",
          "class", "cwe", "component", "scope", "trigger",
          "traps_unmodified", "traps_if_gc_cap_aware", "repro_status", "stock_behaviour", "domain_behaviour", "notes"]

# Which MicroPython source actually has to be built to see each defect. Computed
# by gen-fix-status against a full clone (git merge-base --is-ancestor), never
# inferred from dates. Rows absent from this file stay "unknown" on purpose:
# a closed issue with no identifiable fix commit is not evidence of anything.
FIX = json.load(open(os.path.join(S, "fix-status.json")))
PIN = FIX["pin"][:12]

# Measured, not predicted. Produced by repros/run-on-stock.sh against a stock
# host build of the pinned commit. `stock_behaviour` is the column that matters
# for this project: a defect that corrupts silently on ordinary hardware is a
# better Capstone specimen than one that already crashes, because the crash is
# the platform doing our job for us.
MEASURED = {
    #        repro_status  stock_behaviour      domain_behaviour
    "#18168": ("confirmed", "silent-corruption", "untrapped-identical"),
    "#18171": ("confirmed", "silent-corruption", "untrapped-identical"),
    "#17941": ("confirmed", "crash-sigsegv",     "fault-cause24"),
    "#18619": ("confirmed", "crash-sigsegv",     "fault-cause24"),
    "#10402": ("confirmed", "crash-sigsegv",     "fault-cause24"),
    # Measured and does NOT reproduce. Upstream closed 322 COMPLETED with the
    # reporter confirming the retest passed, and at the pin growth fails only
    # once the required CONTIGUOUS block exceeds free memory, which is correct
    # behaviour for a non-compacting allocator rather than the reported defect.
    "#322":   ("built",     "not-reproducible",  "not-run"),
    # Measured at the fix commit's PARENT (ce491ab0d1) with gcc-12, not at the pin.
    # Both cases named in the fix run to completion with no crash and no visible
    # difference, which matches the fix author's own note that the bug "exists but
    # has no impact" in default configurations. Silent, and not merely unreported:
    # the dangling pointer is internal to array_extend and never reaches a
    # Python-visible value.
    "CVE-2024-8947": ("confirmed", "silent-no-effect", "not-run"),
    "#13283":        ("confirmed", "silent-no-effect", "not-run"),
    # #19075: the trigger IS created in the domain -- a diagnostic confirms
    # json.dump hands write() a bytearray there too, so `buf += buf` mutates in
    # place and reallocates under the caller. The domain then does not fault
    # where stock segfaults. Recorded as untrapped-no-crash rather than
    # untrapped-identical because, unlike #18168 and #18171, corruption was
    # NOT verified here: survival is all that was measured.
    "#19075": ("confirmed", "crash-sigsegv",     "untrapped-no-crash"),
}

out = []
missing = []
for rid, ref, cls, cwe, comp, scope, trig, hyp, notes in ROWS:
    if ref.startswith("CVE-"):
        c = nvd.get(ref)
        if not c:
            missing.append(ref); continue
        title = [x["value"] for x in c["descriptions"] if x["lang"] == "en"][0]
        title = title.split(". The manipulation")[0].split(". It has been")[0].strip()
        title = " ".join(title.split())[:150]
        state = "patched"
        date = c["published"][:10]
        url = f"https://nvd.nist.gov/vuln/detail/{ref}"
        src = "CVE"
    else:
        n = int(ref.lstrip("#"))
        v = issues.get(n)
        if not v:
            missing.append(ref); continue
        title = " ".join(v["title"].split())[:150]
        state = v["state"].lower()
        date = v["createdAt"][:10]
        url = v["url"]
        src = "issue"
    f = FIX["rows"].get(ref, {})
    fix_commit = f.get("fix_commit", "")
    fix_date = f.get("fix_date", "")
    present = f.get("present_at_pin", "unknown")
    base = f.get("repro_base", "unknown")
    if base == "pin":
        base = PIN
    repro, stock, dom = MEASURED.get(ref, ("none", "not-run", "not-run"))
    # What an UNMODIFIED MicroPython gets from Capstone, which is the honest
    # baseline and is almost always nothing. mpy_domain.c carves the heap as one
    # 384 KiB static array and gc_init hands that single object to the collector,
    # so every block it sub-allocates inherits a capability spanning the whole
    # heap (measured: evidence/heap-bounds-model.s). gc_free is bookkeeping in a
    # software bitmap and never reaches the hardware, so nothing is revoked and a
    # stale pointer stays indistinguishable from a live one.
    unmod = "unclear" if scope == "port-heap" else "no"
    out.append([rid, src, ref, url, title, state, date,
                fix_commit, fix_date, present, base,
                cls, cwe, comp, scope, trig, unmod, hyp, repro, stock, dom, notes])

if missing:
    print("FEHLT, nicht verifiziert:", missing, file=sys.stderr)
    sys.exit(1)

with open(OUT, "w", newline="\n") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(HEADER)
    w.writerows(out)

print(f"{len(out)} Zeilen geschrieben nach {OUT}")
from collections import Counter
for col, name in ((11, "class"), (14, "scope"), (16, "traps_unmodified"), (17, "traps_if_gc_cap_aware"), (9, "present_at_pin"), (19, "stock_behaviour"), (20, "domain_behaviour")):
    print(f"  {name:22s}", dict(Counter(r[col] for r in out)))
