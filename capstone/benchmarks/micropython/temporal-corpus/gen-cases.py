#!/usr/bin/env python3
"""Generate one self-contained reproduction directory per corpus case.

The CSV stays the single source of truth for status; this script renders it into
cases/<ID>_<slug>/README.md so a reader can open one folder and see the whole
case without cross-referencing a table. Repro scripts and RESULT.txt files are
hand-written and are never overwritten here.

    python3 gen-cases.py           # write/refresh every case README
    python3 gen-cases.py --check   # fail if a README is stale or a repro missing
"""
import csv, os, sys

S = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(S, "temporal-allocator-corpus.csv")
CASES = os.path.join(S, "cases")

# id -> (slug, runnable_here, why)
# runnable_here is the honest gate: can this defect be triggered in OUR domain,
# which has MICROPY_VFS=0, no threads, no sockets and no port hardware.
PLAN = {
    "MPY-T01": ("modselect-poll-uaf", "parent-measured", "fixed at the pin; needs the fix commit's parent"),
    "MPY-T02": ("objarray-bytes-self-copy-uaf", "parent-measured", "fixed at the pin; NVD states the trigger, a bytes object resized and copied into itself"),
    "MPY-T03": ("import-all-memory-corruption", "parent-measured", "fixed at the pin"),
    "MPY-T04": ("modselect-line151-uaf", "parent-measured", "fixed at the pin; same defect as MPY-T01"),
    "MPY-T05": ("objarray-line509-uaf", "parent-measured", "fixed at the pin; same defect as MPY-T02"),
    "MPY-T06": ("btree-reuse-after-close", "parent-measured", "fixed at the pin; also needs the btree module, which this domain does not build"),
        "MPY-T07": ("lexer-source-name-uaf", "parent-measured", "measured at the 2018 parent: the use-after-free executes and is silent"),
    "MPY-T08": ("stdio-close-then-use", "parent-measured", "fixed at the pin; also needs stdio streams, and MICROPY_PY_SYS_STDFILES is 0 here"),
    "MPY-T09": ("bytearray-resize-stale-view", "yes", "measured"),
    "MPY-T10": ("array-resize-stale-view", "yes", "measured, with the file vehicle replaced by a direct write"),
    "MPY-T11": ("sort-mutates-under-gc", "yes", "measured"),
    "MPY-T12": ("dict-eq-reentrant-clear", "yes", "measured"),
    "MPY-T13": ("write-callback-grows-buffer", "yes", "measured"),
    "MPY-T14": ("readblocks-grows-buffer-dup", "parent-measured", "fixed at the pin; duplicate of MPY-T15"),
    "MPY-T15": ("readblocks-grows-buffer", "parent-measured", "fixed at the pin; needs a block device, which this domain has no VFS for"),
    "MPY-T16": ("deinit-after-gc-sweep-all", "no", "needs a port shutdown hook; this domain's teardown is not the one with the defect"),
    "MPY-T17": ("finaliser-exception-deadlock", "no", "needs MICROPY_PY_THREAD, which is off here"),
    "MPY-T18": ("gc-collect-frees-live-binding", "no", "closed NOT_PLANNED upstream and redirected to the LVGL binding project; not a MicroPython defect"),
    "MPY-T19": ("ble-object-collected", "no", "the fix touches extmod/modbluetooth_nimble.c; reproducing needs a NimBLE stack, which no host build provides"),
    "MPY-T20": ("esp32-gc-collect-panic", "unknown", "needs the ESP32 IDF heap alongside the collector"),
    "MPY-T21": ("rp2-thread-stack-not-scanned", "unknown", "closed upstream as no-longer-reproducible rather than fixed, and it needs the rp2 port and threads"),
    "MPY-T22": ("concurrent-vfs-gc-race", "no", "needs threads and a filesystem, both absent here"),
    "MPY-T23": ("esp32-lwip-freed-buffer", "unknown", "the thread resolves into an ESP-IDF version question, not a MicroPython fix; needs the ESP32 network stack"),
        "MPY-T24": ("embed-compile-freed-lexer", "parent-measured", "embed example built at the parent and runs clean; the stack-top defect needs recursion the example does not do"),
    "MPY-T25": ("stringio-subclass-print", "yes", "measured"),
    "MPY-T26": ("native-relocate-null", "unknown", "needs the native emitter and a relocatable .mpy"),
    "MPY-T27": ("socket-connect-fail-leak", "no", "needs the ESP32 network stack"),
    "MPY-T28": ("gc-realloc-returns-null", "yes", "measured at the pin and does NOT reproduce; upstream closed it COMPLETED with the reporter confirming"),
    "MPY-T29": ("gc-free-assert-invalid-block", "no", "the fix is ports/unix/gccollect.c, making the GC capture stack and registers properly; the trigger depends on what happens to be in registers and is not deterministically reproducible"),
    "MPY-T30": ("gc-collect-over-retains", "no", "reported against the ESP32 port and closed needs-info upstream"),
}

STATUS_LINE = {
    "yes": "MEASURED in the domain.",
    "no": "NOT REPRODUCIBLE HERE. The trigger cannot be expressed in this domain.",
    "parent-build": "BLOCKED on a parent build. Already fixed in the pinned source.",
    "parent-measured": "MEASURED at the fix commit's parent. Fixed in the pinned source.",
    "unknown": "BLOCKED. Upstream status unresolved, see below.",
    "c-level": "BLOCKED on a C-level harness. No Python trigger exists.",
}


def render(r, slug, runnable, why):
    ref, rid = r["ref"], r["id"]
    out = []
    out.append(f"# {rid}: {r['title']}\n")
    out.append(f"Source: {ref}, {r['url']}  \n")
    out.append(f"Upstream state: {r['state']}, first seen {r['first_seen']}\n")
    out.append(f"\n**{STATUS_LINE[runnable]}**\n")
    out.append(f"\n## The defect\n\n{r['trigger']}.\n")
    out.append(f"\nClass `{r['class']}`, {r['cwe']}, in `{r['component']}`. ")
    out.append(f"Scope `{r['scope']}`")
    if r["scope"] in ("gc-core", "gc-managed"):
        out.append(", so it lives on memory MicroPython's own collector manages,\n"
                   "inside the single region `gc_init` was handed.\n")
    else:
        out.append(", so a second allocator is involved.\n")
    out.append("\n## What Capstone does about it\n\n")
    out.append(f"`traps_unmodified` = **{r['traps_unmodified']}**. ")
    if r["traps_unmodified"] == "no":
        out.append("An unmodified runtime gets no temporal protection here: the heap is one\n"
                   "object, every sub-allocation inherits its bounds, and `gc_free` never\n"
                   "reaches the hardware, so there is nothing to revoke. See\n"
                   "`../../evidence/heap-bounds-model.s` and\n"
                   "`../../evidence/nested-uaf-qemu-2026-08-17.txt`.\n"
                   "\nThe same blindness is not special to capabilities. AddressSanitizer misses\n"
                   "this defect family too, in a toolchain where it catches an ordinary `malloc`\n"
                   "use-after-free, because the runtime's frees never reach it either:\n"
                   "`../../evidence/asan-blindness-2026-08-17.txt`.\n")
    else:
        out.append("A second allocator is involved and its behaviour has not been examined.\n")
    out.append(f"\n`traps_if_gc_cap_aware` = {r['traps_if_gc_cap_aware']}, which is a prediction "
               "about a capability-aware\ncollector that does not exist yet. Not evidence.\n")
    out.append("\n## Measured\n\n")
    if r["stock_behaviour"] != "not-run" or r["domain_behaviour"] != "not-run":
        out.append(f"- stock MicroPython at the pin: **{r['stock_behaviour']}**\n")
        out.append(f"- Capstone domain under QEMU: **{r['domain_behaviour']}**\n")
        out.append("\nSee `RESULT.txt` in this directory.\n")
    else:
        out.append(f"Not run. {why.capitalize()}.\n")
    out.append("\n## Reproducing\n\n")
    if runnable == "yes":
        out.append("`repro.py` is the script. On stock:\n\n")
        out.append("```bash\n"
                   "MPY=/tmp/capstone/mpy-stock-pin/ports/unix/build-standard/micropython\n"
                   "$MPY repro.py\n"
                   "```\n\n")
        out.append("In the domain, copy it into the test directory the image is built from and\n"
                   "follow `../../README.md`; the driver is `tools/run-resumable-suite.py`\n"
                   "and it must be run with `--capture-output`, because a test that dies on a\n"
                   "missing builtin still returns a retval and reads exactly like an untrapped one.\n")
    elif runnable in ("parent-build", "parent-measured"):
        out.append(f"The defect is fixed in the pinned source, so building the pin measures nothing.\n"
                   f"Build `{r['repro_base']}` instead, the fix commit's parent:\n\n")
        out.append("```bash\n"
                   f"MPY_COMMIT={r['repro_base']} bash ../../../fetch-micropython.sh\n"
                   "```\n\n")
        out.append("**This works.** Use gcc-12, a compiler contemporary with the commit; the\n"
                   "default gcc 15 rejects the tree. Full recipe, and the two non-obvious\n"
                   "flags it needs, in `../../evidence/parent-build-attempt-2026-08-17.txt`.\n"
                   "Do NOT add AddressSanitizer: it breaks the MicroPython unix port outright,\n"
                   "and the question it would have answered is already settled in\n"
                   "`../../evidence/asan-blindness-2026-08-17.txt`.\n")
    else:
        out.append(f"Not reproducible with the current setup: {why}.\n")
    return "".join(out)


def render_status(rows):
    """One table, generated, so it cannot drift from the case directories."""
    buckets = {"yes": [], "parent-measured": [], "parent-build": [], "no": [],
               "unknown": [], "c-level": []}
    for r in rows:
        buckets[PLAN[r["id"]][1]].append(r)
    out = ["# Case status\n",
           "\nGenerated by `gen-cases.py`. One directory per case under `cases/`, each\n"
           "self-contained. Do not edit this file by hand.\n"]
    heads = [
        ("yes", "Measured",
         "These ran. Both columns are measurements, not predictions. MPY-T28 was run on "
         "stock only, and came back negative, which is why its domain column is empty: "
         "there was nothing left to take into the domain."),
        ("parent-measured", "Measured at the fix commit's parent",
         "Already fixed in the pinned source, so these were measured on the last commit "
         "that still carries the defect. Built with gcc-12 per "
         "`evidence/parent-build-attempt-2026-08-17.txt`."),
        ("parent-build", "Blocked: already fixed, parent not yet built",
         "Measuring these needs the fix commit's parent. Attempted and currently "
         "blocked by toolchain age, not by our patches: see "
         "`evidence/parent-build-attempt-2026-08-17.txt`."),
        ("no", "Not expressible in this domain",
         "The trigger needs threads, sockets, a filesystem or port hardware that this "
         "domain does not have. No amount of work on our side reaches these without "
         "changing what the domain is."),
        ("unknown", "Upstream status unresolved",
         "Closed upstream with no fix commit naming the issue, and no runnable trigger "
         "published. Recorded as unknown rather than guessed."),
        ("c-level", "Needs a C-level harness",
         "The published reproduction calls the allocator directly from C. No Python "
         "trigger exists, so the corpus harness cannot reach them."),
    ]
    for key, title, note in heads:
        rs = buckets[key]
        if not rs:
            continue          # an empty bucket is noise, not information
        out.append(f"\n## {title} ({len(rs)})\n\n{note}\n\n")
        out.append("| case | source | stock | domain |\n|---|---|---|---|\n")
        for r in rs:
            out.append(f"| `{r['id']}` | {r['ref']} | {r['stock_behaviour']} "
                       f"| {r['domain_behaviour']} |\n")
    out.append("\n## What the measurements say\n\n")
    n_un = sum(1 for r in rows if r["traps_unmodified"] == "no")
    from collections import Counter
    stock = Counter(r["stock_behaviour"] for r in rows if r["stock_behaviour"] != "not-run")
    dom = Counter(r["domain_behaviour"] for r in rows if r["domain_behaviour"] != "not-run")
    measured = sum(1 for r in rows
                   if r["stock_behaviour"] != "not-run" or r["domain_behaviour"] != "not-run")
    out.append(f"{measured} of {len(rows)} rows have been run. On stock or at a fix's parent: "
               + ", ".join(f"{v} {k}" for k, v in sorted(stock.items())) + ".\n\n")
    out.append("In the Capstone domain: "
               + ", ".join(f"{v} {k}" for k, v in sorted(dom.items())) + ".\n\n")
    out.append(f"`traps_unmodified` is **no** for {n_un} of {len(rows)} rows, and that is measured\n"
               "rather than assumed. Of the rows run in the domain, not one was trapped for a\n"
               "temporal reason: the untrapped ones completed exactly as unprotected stock does,\n"
               "and the faulting ones failed on `cause 24`, an untagged word used as a pointer,\n"
               "which is what an MMU already catches.\n\n")
    out.append("The single most useful number is the count of rows that execute a real defect\n"
               "and produce NOTHING observable: `silent-no-effect` plus `silent-corruption`\n"
               f"is {stock.get('silent-no-effect', 0) + stock.get('silent-corruption', 0)}. "
               "Those are the cases where a nested allocator hides a\n"
               "defect from the language, from the crash, and from AddressSanitizer alike.\n")
    return "".join(out)


def main():
    check = "--check" in sys.argv
    rows = list(csv.DictReader(open(CSV)))
    problems, made = [], 0
    status_path = os.path.join(S, "STATUS.md")
    status_text = render_status(rows)
    if check:
        if not os.path.exists(status_path) or open(status_path).read() != status_text:
            problems.append("STATUS.md stale, rerun gen-cases.py")
    else:
        open(status_path, "w").write(status_text)
    for r in rows:
        rid = r["id"]
        if rid not in PLAN:
            problems.append(f"{rid}: no entry in PLAN")
            continue
        slug, runnable, why = PLAN[rid]
        d = os.path.join(CASES, f"{rid}_{slug}")
        readme = os.path.join(d, "README.md")
        text = render(r, slug, runnable, why)
        if check:
            if not os.path.isdir(d):
                problems.append(f"{rid}: directory missing")
            elif open(readme).read() != text:
                problems.append(f"{rid}: README stale, rerun gen-cases.py")
            elif runnable == "yes" and not os.path.exists(os.path.join(d, "repro.py")):
                problems.append(f"{rid}: marked runnable but has no repro.py")
            elif runnable == "yes" and not os.path.exists(os.path.join(d, "RESULT.txt")):
                problems.append(f"{rid}: marked runnable but has no RESULT.txt")
        else:
            os.makedirs(d, exist_ok=True)
            open(readme, "w").write(text)
            made += 1
    if problems:
        print(f"{'FAIL' if check else 'WARN'}: {len(problems)} problems")
        for p in problems:
            print("   ", p)
        return 1
    print(f"{'OK' if check else 'wrote'}: {made or len(rows)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
