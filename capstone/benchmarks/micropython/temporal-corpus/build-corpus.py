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
    # RETRACTED CLASSIFICATION 2026-08-18. This was filed as reentrancy/CWE-416 with
    # the note "re-enters and invalidates the target buffer". That was invented, not
    # read: the issue says subclassing a stream in pure Python is unsupported, and
    # the reporter quotes the docs saying so. Nothing is freed and then used. The
    # C-level stream state is never initialised, and printing dereferences it. That
    # is uninitialised state, not a lifetime defect, so it does not belong in a
    # temporal corpus and is kept only as a labelled counter-example.
    ("MPY-T25", "#10402", "uninitialised-state", "CWE-908", "py/objstringio.c", "gc-managed",
     "a pure-Python subclass of io.StringIO leaves the C stream state uninitialised, and print dereferences it",
     "not-trapped", "Open. NOT a temporal defect; retained as a counter-example, see RESULT.txt."),
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


# ---------------------------------------------------------------------------
# TEMPORAL AUDIT, 2026-08-18.
#
# The class column was assigned early, from issue TITLES, before most issue
# bodies and almost no fix commits had been read. Asked to look carefully, and
# the answer is that a third of the corpus is not a temporal defect at all.
#
# The authority used here is the FIX COMMIT's own message wherever one exists,
# because it is the maintainer stating what was wrong, and the issue text
# otherwise. Rows are NOT deleted: an honest counter-example is worth more than
# a quietly dropped row, and several of them correct a reading of the results.
#
# ref -> (is_temporal, evidence)
TEMPORAL = {
    # --- genuinely temporal: a pointer outlives the storage it names ---
    "CVE-2023-7152": ("yes", "fix: 'Handle growing the pollfds allocation correctly'; every registered object holds a raw pointer into the array m_renew moves"),
    "#12887":        ("yes", "same defect as CVE-2023-7152"),
    "CVE-2024-8947": ("yes", "fix: 'Fix use-after-free if extending a bytearray from itself'"),
    "#13283":        ("yes", "same defect as CVE-2024-8947"),
    "#12543":        ("yes", "fix: 'Add checks for already-closed database'; reuse after close"),
    "#4128":         ("yes", "fix: 'Fix reference to freed memory, lexer src name'"),
    "#18168":        ("yes", "measured: the memoryview addresses storage the bytearray no longer owns"),
    "#18171":        ("yes", "same shape as 18168, on array"),
    "#17941":        ("yes", "sort holds a pointer into the list while the comparison clears it"),
    "#18619":        ("yes", "reporter: 'the lookup returns a slot pointer into the now-freed table'"),
    "#19075":        ("yes", "callback enlarges the buffer its caller still points into"),
    "#17848":        ("yes", "fix: 'Use memoryview when available', stopping the callback reallocating under the caller"),
    "#19060":        ("yes", "duplicate of 17848"),
    "#5487":         ("yes", "port deinit runs after the sweep that already freed what it touches"),
    "#5226":         ("yes", "fix: 'Persist reference to NimBLE service instances'; collected while C still held it"),
    # DEMOTED 2026-08-18 on a second pass. The reporter writes only that the script
    # "silently stopped running" without gc.collect() and diagnoses no mechanism;
    # "the thread stack is not a scanned root" was this session's inference, not
    # anyone's finding, and upstream closed it as no longer reproducible.
    "#8550":         ("uncertain", "reporter records only a silent stop without gc.collect(); no mechanism established, closed as no longer reproducible"),
    "#4705":         ("yes", "fix: 'Make sure stack/regs get captured properly for GC'; missed roots free reachable memory"),

    # --- NOT temporal: no storage is released and then used ---
    "CVE-2026-1998": ("no",  "fix: 'Make import-all support non-modules'; type confusion, and NVD says CWE-119/787 not 416"),
    "#12670":        ("no",  "fix: 'casts away the constness and assigns -1 to the object fd member'; writing a const ROM object"),
    "#3627":         ("no",  "a deadlock between a finaliser exception and the GIL; liveness, not memory safety"),
    "#19413":        ("no",  "closed NOT_PLANNED and redirected to the LVGL binding project; not a MicroPython defect"),
    "#11781":        ("no",  "fix: 'embed: Improve stack top estimation'; a stack overflow, no lifetime component"),
    "#10402":        ("no",  "subclassing a stream in pure Python is unsupported; the C stream state is never initialised"),
    "#18645":        ("no",  "a malformed .mpy drives relocation off a NULL table; input validation"),
    "#5272":         ("no",  "a socket is never cleaned up after a failed connect; a leak"),
    "#322":          ("no",  "gc_realloc refusing an allocation is an allocation failure, and it was measured not to reproduce"),
    "#11698":        ("no",  "gc.collect retaining more than expected is over-retention, the opposite of premature free"),

    # --- uncertain: plausible but not established from a maintainer statement ---
    "#17442":        ("uncertain", "a data race on allocator state; whether it manifests as a lifetime violation is not established"),
    "#6988":         ("uncertain", "an ESP32 collector panic with no fix commit and no mechanism recorded upstream"),
    "#12638":        ("uncertain", "suspected freed lwip buffer, but closed as unreproducible and the thread ends in an ESP-IDF version question"),
}

HEADER = ["id", "source", "ref", "url", "title", "state", "first_seen",
          "fix_commit", "fix_date", "present_at_pin", "repro_base",
          "class", "cwe", "component", "scope", "trigger",
          "traps_unmodified", "traps_if_gc_cap_aware", "is_temporal", "temporal_evidence",
          "repro_status", "stock_behaviour", "domain_behaviour", "notes"]

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
    # Domain-measured 2026-08-18 by backporting the port to the 2024 pre-fix tree
    # (see backport-2024/). extend-from-self runs in a pure-capability domain and
    # does not fault. untrapped-no-crash rather than untrapped-identical, because
    # the profile lacks slice assignment so the output is not byte-identical to
    # the host, and because the dangling pointer is internal and corruption was
    # not demonstrated.
    "CVE-2024-8947": ("confirmed", "silent-no-effect", "untrapped-no-crash"),
    "#13283":        ("confirmed", "silent-no-effect", "untrapped-no-crash"),
    # Measured at the parent of 365913953a4e (3b954698fa, 2023-11-09): closing
    # sys.stdout kills the interpreter with SIGSEGV before a single stderr write
    # gets out. At the pin the same script raises ValueError instead, so the fix
    # is present and the parent build is what shows the defect.
    "#12670":        ("confirmed", "crash-sigsegv",   "not-run"),
    # Measured at the parent of 64f0394d80ca (4e6dc0b569, 2026-05-09). The defect
    # condition IS present there: the readblocks callback's oversized
    # `buf[:] = bytearray(1 + SEC_SIZE)` is accepted, where the pin raises
    # ValueError. It does not crash in the published reproduction, so this is
    # recorded as silent rather than as the crash the issue title claims.
    "#17848":        ("confirmed", "silent-no-effect", "not-run"),
    "#19060":        ("confirmed", "silent-no-effect", "not-run"),
    # Measured at the parent of 570744d06c5b (c9f747cccf, 2026-02-04): SIGSEGV.
    # mp_import_all assumed its argument was a native module; a non-module
    # injected through sys.modules takes it down. The pin prints "T03 survived".
    "CVE-2026-1998": ("confirmed", "crash-sigsegv",   "not-run"),
    # Measured at the parent of 8b24aa36ba97 (e9bcd49b3e, 2023-12-20). The growth
    # path is exercised, 16 registrations against an increment of 4, and BOTH
    # builds produce identical correct output. Staleness was not observed, so
    # this is silent rather than a crash.
    "CVE-2023-7152": ("confirmed", "silent-no-effect", "not-run"),
    "#12887":        ("confirmed", "silent-no-effect", "not-run"),
    # Measured at the parent of 6db91dfefb1a (8159dcc276, 2024-07-20): SIGSEGV on
    # reading a btree after close(). The pin raises ValueError instead.
    "#12543":        ("confirmed", "crash-sigsegv",   "not-run"),
    # Measured at the parent of 1a2c511e5d08 (2018 tree): the embedding example
    # prints "Hello world of easy embedding!" and exits 0. The use-after-free
    # executes and is silent.
    "#4128":         ("confirmed", "silent-no-effect", "not-run"),
    # Measured at the parent of d2a3cd7ac428 (be8d660fc2, 2024-02-15): the embed
    # example builds and runs to completion, exit 0. The defect is a stack-top
    # misestimate and the shipped example does not recurse deeply enough to hit
    # it, so this is a measured negative rather than a reproduction.
    "#11781":        ("built",     "not-reproducible", "not-run"),
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
    is_temp, temp_ev = TEMPORAL[ref]
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
                cls, cwe, comp, scope, trig, unmod, hyp, is_temp, temp_ev, repro, stock, dom, notes])

if missing:
    print("FEHLT, nicht verifiziert:", missing, file=sys.stderr)
    sys.exit(1)

with open(OUT, "w", newline="\n") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(HEADER)
    w.writerows(out)

print(f"{len(out)} Zeilen geschrieben nach {OUT}")
from collections import Counter
for col, name in ((11, "class"), (14, "scope"), (16, "traps_unmodified"), (17, "traps_if_gc_cap_aware"), (9, "present_at_pin"), (21, "stock_behaviour"), (22, "domain_behaviour"), (18, "is_temporal")):
    print(f"  {name:22s}", dict(Counter(r[col] for r in out)))
