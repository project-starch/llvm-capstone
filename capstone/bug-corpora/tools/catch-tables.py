#!/usr/bin/env python3
"""The two tables (temporal, spatial) per program and allocator layer, every column, for the cases of
FFmpeg, tshark and memcached; then the inside-one-allocation table. Computed from case.json only. A
cell that is not a reading is printed as NOT RUN / UNCLASSIFIED and counted as neither, and the run
exits 1 if any exists -- a table with a hole in it is not printed as if it were complete.

    catch-tables.py [bug-corpora dir] [--markdown] [--board] [--board-json FILE]

--board prints the THREE-COLUMN tables instead: per bug, (1) CHERI -- stock CheriBSD, revocation on,
where a freed chunk HELD in quarantine (the stale pointer followed, the chunk never reissued) counts
as caught and is shown apart from a fault; (2) Sublet in malloc -- Sublet only as the system
allocator, the program's nested allocator stock; (3) Sublet in nested -- the nested allocator's
Sublet port (for a plain case, the program's whole configuration with that port live). Then one row
per bug. --board-json also writes the rows, for the slides.

--all adds the other six programs -- CPython, httpd, mruby, Perl, PostgreSQL, SQLite -- to the board,
and only to the board: their corpora carry the three columns' arms but not the four-column set.
Every one of their cells is an explicit `verdict` read out of the bundle that measured it (the
arm's `verdict_from` names the row); their oracles predate the MEASURED convention and are not
read. A column no bundle has measured for a group stays a hole, so --all exits 1 until the holes
are measured, by design.

The source of the per-program tables in docs/ref/spatial-vs-temporal-three-programs.md section 0.

Verdict sources, in order: an arm's explicit `verdict`; then the opening words of its oracle, after
a `MEASURED ...:` prefix -- caught: CAUGHT / REPORTED / ASan REPORTS / SIGPROT / fault; missed:
NOT CAUGHT / NO FAULT / NO ASan report / SILENT / complete / the sequence completes. An arm whose
status is predicted / not written / declined, or whose oracle begins PREDICTED, is NOT RUN. A
`verdict` of NO-READING is a run that ended without the mechanism answering either way (the
program's own check stopped it, a non-capability trap, a timeout): neither caught nor missed.

STRICT (2026-10-10): an oracle is a PREDICTION unless the arm says it was measured -- an explicit
`verdict`, `status: measured`, or a `MEASURED` prefix. Until then a bare oracle such as "fault at the
labelled read probe" was read as a catch on its opening word, so a cell with no run behind it could
not be told from one with a record; the audit found 26 such cells in memcached alone, and every one
then had to be traced by hand. A bare oracle is now NOT RUN, and so is a cell the old ASan-text
fallback used to accept.

A case whose case.json carries `duplicate_of` (the same upstream defect as another case of the same
corpus, e.g. a release-branch backport of the same fix) is left out of every table and listed."""
import collections
import json
import pathlib
import re
import sys

_VALUED = {"--board-json"}  # flags that take a value: their value is not the corpora directory
POS = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] not in _VALUED]
BC = pathlib.Path(POS[0]) if POS else pathlib.Path(__file__).resolve().parents[1]
TARGETS = ("ffmpeg", "wireshark", "memcached")
# --all: the board only, for these as well.
EXTRA = ("cpython", "httpd", "mruby", "perl", "postgres", "sqlite")
ALL = "--all" in sys.argv
if ALL and not ({"--board", "--board-json", "--board-virtual"} & set(sys.argv)):
    sys.exit("catch-tables: --all is for the board; the four-column tables cover FFmpeg, tshark and memcached")
PROGRAMS = TARGETS + EXTRA if ALL else TARGETS
NOTRUN = {"predicted", "not written", "declined"}
LAYER = {"pool-repros": "AVBufferPool", "plain-temporal-repros": "direct malloc", "plain-heap-repros": "direct malloc",
         "subobject-repros": "inside one struct", "plane-repros": "frame-pool plane", "carved-repros": "carved buffer",
         "wmem-repros": "wmem", "allocator-repros": "slabs.c / cache.c"}


VERDICT_WORDS = {"CAUGHT": "caught", "MISSED": "missed", "NOT CAUGHT": "missed",
                 # CheriBSD: freed chunk held in quarantine. A miss here; column 1 counts it as held.
                 "NOT-REISSUED": "missed",
                 "NO-READING": "no reading",
                 # native-fix-differential: buggy reproduces and fixed does not. Not a mechanism.
                 "TWO-SIDED": "two-sided"}


def verdict(arm):
    if arm is None:
        return "absent"
    if arm.get("status") in NOTRUN:
        return "not run"
    if arm.get("status") == "not-applicable":
        return "n/a"
    v = arm.get("verdict")
    if v:
        # Every verdict word the corpora use, each with its reading. A word not listed here is NOT
        # a miss: it lands in the NOT A READING list, which stops the board and names the cell.
        # (Until 2026-10-10 any word but CAUGHT read as "missed", so a legacy `detected` -- on
        # postgres' spatial arms -- would have scored as a miss had a table ever read it.)
        r = VERDICT_WORDS.get(str(v).strip().upper())
        return r if r else f"UNKNOWN VERDICT {v!r}"
    o = str(arm.get("oracle", "")).strip()
    if o.upper().startswith("PREDICTED"):
        return "not run"
    if arm.get("status") != "measured" and not re.match(r"\s*\**\s*MEASURED", o):
        return "not run"  # a bare oracle is a prediction, not a reading
    m = re.match(r"MEASURED[^:]*:\s*(.*)", o, re.S)
    head = (m.group(1) if m else o).lstrip("*").lstrip().lower()
    if head.startswith(("not caught", "no fault", "no asan", "silent", "complete", "the sequence completes")):
        return "missed"
    if head.startswith(("caught", "reported", "asan reports", "sigprot", "fault")):
        return "caught"
    if o.lower().startswith("no asan report"):
        return "missed"
    return "UNCLASSIFIED"


# The three columns per bug. Column 2 is Sublet ONLY as the system allocator; on a plain corpus that
# is the `sublet` arm itself, on a nested one the arm that leaves the nested allocator stock.
COL2 = {"pool-repros": "sublet-malloc", "wmem-repros": "sublet-malloc", "allocator-repros": "sublet-malloc"}
# Column 3 is the Sublet port of the INNERMOST allocator that made the object the access belongs to; a
# plain case runs in the program's full configuration. Where code carves the object out of a block that
# allocator handed out -- a codec's carve, av_frame_get_buffer's planes, memcached's ITEM_key/ITEM_suffix
# inside a slab item -- that carve is the innermost allocator, and its port is the case's `sublet-carve`
# arm (memcached 06/07: the slab port plus the key/suffix carve). One rule, so the two corpora agree.
COL3 = {"pool-repros": "sublet-port", "wmem-repros": "sublet-chunks", "allocator-repros": "sublet",
        "carved-repros": "sublet-carve", "plane-repros": "sublet-carve"}


def col3_arm(corpus, arms):
    return "sublet-carve" if "sublet-carve" in arms else COL3.get(corpus, "sublet-full")


# --all: per group of the other six programs, (allocator layer, nested, column-2 arm, column-3 arm).
# nested None: a whole-program corpus whose cases cross different boundaries and are not split here.
# Column 2 for a nested group is `sublet-malloc`, the same arm the three programs use; none of these
# groups has measured it yet, so those cells are holes. Perl's `sysalloc-sublet` is that arm under
# its own name (the Sublet heap, Perl's SV arenas stock); mruby has no such run. Column 3 is the
# group's Sublet port: the APR ports' `sublet`, Perl's SV-head port, mruby's GC port. pymalloc,
# PostgreSQL and SQLite are measured on the virtual profile instead: column 2 is `virtual-malloc`,
# column 3 the nested allocator's port on a block virtual mallocng lends. postgres/c-repros are
# frontend programs with no nested allocator linked at all, so both columns are the protected
# system allocator itself: virtual mallocng, `virtual-malloc`
# (the physical Sublet heap is not used, project decision 2026-10-10). Their recorded `sublet` arm
# was level0 bounds under a Sublet label and is dropped.
# Corpora of these programs that are deliberately NOT on the board yet, each with the reason. A new
# corpus is not silently skipped: it stops --all until it is either given a group above or named here.
NOT_ON_BOARD = {
    ("cpython", "allocator-boundary-repros"):
        "landed on dev in #213 after this board was built; its rows are oracle prose with no verdict "
        "field, and its nested/plain split is decided per case in its README, so placing it needs its "
        "own pass rather than a guess",
    ("sqlite", "capi-repros"):
        "a host-ASan row corpus (`sqlite-row`): its only arm is host-asan, its cases are `rowN_` "
        "directories with no Capstone, Sublet or CheriBSD arm, and its top-level `verdict` is a "
        "provenance verdict (literal / modeled / out-of-scope), not a mechanism's reading",
    ("mruby", "gc-slot-repros"):
        "planned, no cases yet (its hand-found rows live on another branch)",
}
EXTRA_GROUPS = {
    ("cpython", "pymalloc-repros"): ("pymalloc", True, "virtual-malloc", "virtual-nested-pools"),
    ("httpd", "apr-pool-repros"): ("APR pools", True, "sublet-malloc", "sublet"),
    ("httpd", "bucket-repros"): ("APR buckets", True, "sublet-malloc", "sublet"),
    ("mruby", "release-differential"): ("whole program", None, "sysalloc-sublet", "sublet-gc"),
    ("perl", "release-differential"): ("whole program", None, "sysalloc-sublet", "sublet-svheads"),
    ("postgres", "c-repros"): ("direct malloc", False, "virtual-malloc", "virtual-malloc"),
    ("postgres", "mmgr-repros"): ("memory contexts", True, "virtual-malloc", "virtual-pg-pools"),
    ("postgres", "sql-repros"): ("palloc, whole server", True, "virtual-malloc", "virtual-pg-pools"),
    ("sqlite", "engine-repros"): ("memsys5", True, "virtual-malloc", "virtual-nested-pools"),
}


def extra_kind(d):
    """These corpora name the class outright, or say in lifetime_ender that nothing ended a lifetime."""
    c = str(d.get("class", "")).lower()
    if c:
        return "spatial" if c.startswith("spatial") else "temporal" if c == "temporal" else "other"
    le = str(d.get("lifetime_ender", "")).strip().lower()
    return "spatial" if le.startswith(("n/a", "not a lifetime defect", "none")) else "temporal"


def cheri_board(arm):
    """Column 1: CheriBSD's reading with the quarantine rule -- 'held' is caught, kept apart."""
    if arm and str(arm.get("verdict", "")).upper() == "NOT-REISSUED":
        return "held"
    v = verdict(arm)
    return "fault" if v == "caught" else v


def protected(arms, plain, port):
    """The stronger arm where a port measured one, else the plain one."""
    a = arms.get(port)
    return a if (a and verdict(a) in ("caught", "missed")) else arms.get(plain)


rows, odd, dups = [], [], []
for p in TARGETS:
    for cj in sorted((BC / p).glob("*/[0-9][0-9]_*/case.json")):
        d = json.loads(cj.read_text())
        if d.get("duplicate_of"):
            dups.append(f"{p}/{cj.parts[-3]}/{cj.parent.name} = {d['duplicate_of']}")
            continue
        corpus = cj.parts[-3]
        le = str(d.get("lifetime_ender", "")).strip().upper()
        kind = "spatial" if (le.startswith("NONE") or "SPATIAL" in le) else "temporal"
        a = d["arms"]
        layer = LAYER.get(corpus, corpus)
        if p == "wireshark" and layer == "direct malloc":
            layer = "direct g_malloc"
        r = dict(prog=p, corpus=corpus, layer=layer, case=cj.parent.name, kind=kind,
                 nested=bool(d.get("nested")),
                 asan=verdict(a.get("native-detect")),
                 cheri=verdict(a.get("cheribsd-revocation")),
                 cap=verdict(a.get("spatial")),
                 sub=verdict(protected(a, "sublet", "sublet-chunks") if corpus != "pool-repros"
                             else protected(a, "sublet", "sublet-port")),
                 fcap=verdict(a.get("capstone-subobject")), fcheri=verdict(a.get("cheribsd-subobject")),
                 kcap=verdict(a.get("capstone-carve-bounds")), kcheri=verdict(a.get("cheribsd-carve-bounds")),
                 scarve=verdict(a.get("sublet-carve")),
                 c1=cheri_board(a.get("cheribsd-revocation")),
                 # The virtual-Capstone columns (2026-10-10), read only by --board-virtual: column 2 on
                 # virtual mallocng, column 3 the nested allocator's port on it where one was measured,
                 # else the same virtual-malloc run (a plain case has no nested allocator).
                 v2=verdict(a.get("virtual-malloc")),
                 v3=verdict(a.get("virtual-nested-pools") or a.get("virtual-malloc")),
                 c2=verdict(a.get(COL2.get(corpus, "sublet"))), c2arm=COL2.get(corpus, "sublet"),
                 c3=verdict(a.get(col3_arm(corpus, a))), c3arm=col3_arm(corpus, a),
                 title=str(d.get("title", ""))[:140])
        for k in ("c2", "c3"):
            if r[k] not in ("caught", "missed"):
                odd.append(f"{p}/{corpus}/{r['case']} {k}({r[k + 'arm']})={r[k]}")
        if r["c1"] not in ("fault", "held", "missed"):
            odd.append(f"{p}/{corpus}/{r['case']} c1={r['c1']}")
        for k in ("asan", "cheri", "cap", "sub"):
            if r[k] not in ("caught", "missed"):
                odd.append(f"{p}/{corpus}/{r['case']} {k}={r[k]}")
        rows.append(r)

if {r["prog"] for r in rows} != set(TARGETS):
    # No data is an ERROR, not an empty table: a wrong directory reads exactly like a clean one.
    sys.exit(f"catch-tables: {len(rows)} cases under {BC}; every one of {TARGETS} must have some")

extra = []
if ALL:
    # A corpus is found by its corpus.json, not by the case glob: a corpus whose case directories do
    # not match the glob (sqlite/capi-repros' rowN_) would otherwise vanish from --all silently.
    for decl in sorted(BC.glob("*/*/corpus.json")):
        key = (decl.parts[-3], decl.parts[-2])
        if key[0] in EXTRA and key not in EXTRA_GROUPS and key not in NOT_ON_BOARD:
            sys.exit(f"catch-tables: {key[0]}/{key[1]} is a corpus with no entry in EXTRA_GROUPS or NOT_ON_BOARD")
for p in EXTRA if ALL else ():
    for cj in sorted((BC / p).glob("*/[0-9][0-9]_*/case.json")):
        corpus = cj.parts[-3]
        if (p, corpus) in NOT_ON_BOARD:
            continue
        if (p, corpus) not in EXTRA_GROUPS:
            # A group this table does not know would vanish from it silently; refuse instead.
            sys.exit(f"catch-tables: {p}/{corpus} has cases but no entry in EXTRA_GROUPS")
        layer, nested, c2arm, c3arm = EXTRA_GROUPS[(p, corpus)]
        d = json.loads(cj.read_text())
        if d.get("duplicate_of"):
            dups.append(f"{p}/{corpus}/{cj.parent.name} = {d['duplicate_of']}")
            continue
        a = d["arms"]
        r = dict(prog=p, corpus=corpus, layer=layer, case=cj.parent.name, kind=extra_kind(d), nested=nested,
                 c1=cheri_board(a.get("cheribsd-revocation")),
                 c2=verdict(a.get(c2arm)), c2arm=c2arm, c3=verdict(a.get(c3arm)), c3arm=c3arm,
                 title=str(d.get("title", ""))[:140])
        for k in ("c2", "c3"):
            if r[k] not in ("caught", "missed"):
                odd.append(f"{p}/{corpus}/{r['case']} {k}({r[k + 'arm']})={r[k]}")
        if r["c1"] not in ("fault", "held", "missed"):
            odd.append(f"{p}/{corpus}/{r['case']} c1={r['c1']}")
        extra.append(r)
if ALL and {r["prog"] for r in extra} != set(EXTRA):
    sys.exit(f"catch-tables: --all found cases for {sorted({r['prog'] for r in extra})}, not all of {EXTRA}")
BOARD_ROWS = rows + extra

COLS = (("asan", "ASan"), ("cheri", "CheriBSD"), ("cap", "Cap bounds"), ("sub", "Cap+Sublet"))


def cell(g, k):
    c = collections.Counter(r[k] for r in g)
    run = c["caught"] + c["missed"]
    extra = len(g) - run
    return f"{c['caught']}/{run}" + (f" (+{extra} ?)" if extra else "")


print(f"cases: {len(rows)}" + (f" (and {len(dups)} duplicate{'s' if len(dups) > 1 else ''} left out: "
                                 + "; ".join(dups) + ")" if dups else ""))
for kind in ("temporal", "spatial"):
    print(f"\n{kind.upper()}")
    print(f"  {'program':<10}{'layer':<20}{'axis':<8}{'n':>4} | " + " | ".join(f"{h:>12}" for _, h in COLS))
    groups = collections.OrderedDict()
    for r in rows:
        if r["kind"] == kind:
            groups.setdefault((r["prog"], r["layer"], "nested" if r["nested"] else "plain"), []).append(r)
    for (p, layer, ax), g in groups.items():
        print(f"  {p:<10}{layer:<20}{ax:<8}{len(g):>4} | " + " | ".join(f"{cell(g, k):>12}" for k, _ in COLS))
    for ax in ("nested", "plain"):
        g = [r for r in rows if r["kind"] == kind and (r["nested"] == (ax == "nested"))]
        print(f"  {'':<10}{'subtotal':<20}{ax:<8}{len(g):>4} | " + " | ".join(f"{cell(g, k):>12}" for k, _ in COLS))
    g = [r for r in rows if r["kind"] == kind]
    print(f"  {'':<10}{'TOTAL':<20}{'':<8}{len(g):>4} | " + " | ".join(f"{cell(g, k):>12}" for k, _ in COLS))

print("\nINSIDE ONE ALLOCATION (spatial cases Capstone bounds misses)")
inside = [r for r in rows if r["kind"] == "spatial" and r["cap"] == "missed"]
groups = collections.OrderedDict()
for r in inside:
    groups.setdefault((r["prog"], r["layer"]), []).append(r)
IC = (("sub", "Cap+Sublet"), ("fcap", "Cap field"), ("fcheri", "CHERI field"), ("kcap", "Cap carve"), ("kcheri", "CHERI carve"),
      ("scarve", "Sublet carve"))
for (p, layer), g in groups.items():
    print(f"  {p:<10}{layer:<20}{len(g):>4} | " + " | ".join(f"{h}: {cell(g, k):>10}" for k, h in IC))
    for r in g:
        print(f"      {r['case'][:58]:<58} " + " ".join(f"{k}={r[k]}" for k, _ in IC))
if odd:
    print("\nNOT A READING:", *odd, sep="\n  ")


# ---- emitters: the same cells as markdown (docs) and as slide rows (html) ----------------------
def table_rows(kind):
    groups = collections.OrderedDict()
    for r in rows:
        if r["kind"] == kind:
            groups.setdefault((r["prog"], r["layer"], "nested" if r["nested"] else "plain"), []).append(r)
    order = {"ffmpeg": 0, "wireshark": 1, "memcached": 2}
    out = []
    for (p, layer, ax), g in sorted(groups.items(), key=lambda kv: (order[kv[0][0]], kv[0][2] != "nested", kv[0][1])):
        out.append((p, layer, ax, len(g), [cell(g, k) for k, _ in COLS]))
    tot = [r for r in rows if r["kind"] == kind]
    nn = sum(1 for r in tot if r["nested"])
    out.append(("Total", "", f"{nn} n · {len(tot) - nn} p", len(tot), [cell(tot, k) for k, _ in COLS]))
    return out


def board_cell(g, k):
    if k == "c1":
        held = sum(r["c1"] == "held" for r in g)
        caught = sum(r["c1"] in ("fault", "held") for r in g)
        run = sum(r["c1"] in ("fault", "held", "missed") for r in g)
        extra = len(g) - run  # a cell with no reading is shown, as columns 2 and 3 show theirs
        return f"{caught}/{run}" + (f" ({held} held)" if held else "") + (f" (+{extra} ?)" if extra else "")
    return cell(g, k)


BOARD = (("c1", "CHERI (quarantine = caught)"), ("c2", "Sublet in malloc"), ("c3", "Sublet in nested"))
NAMES = {"ffmpeg": "FFmpeg", "wireshark": "tshark", "memcached": "memcached", "cpython": "CPython",
         "httpd": "httpd", "mruby": "mruby", "perl": "Perl", "postgres": "PostgreSQL", "sqlite": "SQLite"}
ORDER = {p: i for i, p in enumerate(TARGETS + EXTRA)}
WORD = {"fault": "caught", "held": "caught (held)", "missed": "missed", "caught": "caught"}
KINDS = ("temporal", "spatial") + (("other",) if any(r["kind"] == "other" for r in BOARD_ROWS) else ())


def axis(r):
    return "nested" if r["nested"] else "unsplit" if r["nested"] is None else "plain"


def board_rows(kind):
    groups = collections.OrderedDict()
    for r in BOARD_ROWS:
        if r["kind"] == kind:
            groups.setdefault((r["prog"], r["layer"], axis(r)), []).append(r)
    out = [(p, layer, ax, len(g), [board_cell(g, k) for k, _ in BOARD])
           for (p, layer, ax), g in sorted(groups.items(), key=lambda kv: (ORDER[kv[0][0]], kv[0][2] != "nested", kv[0][1]))]
    tot = [r for r in BOARD_ROWS if r["kind"] == kind]
    nn = sum(1 for r in tot if r["nested"])
    nu = sum(1 for r in tot if r["nested"] is None)
    out.append(("Total", "", f"{nn} n · {len(tot) - nn - nu} p" + (f" · {nu} u" if nu else ""), len(tot),
                [board_cell(tot, k) for k, _ in BOARD]))
    return out


if "--board" in sys.argv:
    for kind in KINDS:
        tr = board_rows(kind)
        print(f"\n**{kind.capitalize()} ({tr[-1][3]})**\n")
        print("| program | allocator layer | axis | n | " + " | ".join(h for _, h in BOARD) + " |")
        print("|---|---|---|---:|---:|---:|---:|")
        for p, layer, ax, n, cells in tr:
            name = "**Total**" if p == "Total" else NAMES[p]
            print(f"| {name} | {layer} | {ax} | {n} | " + " | ".join(c.replace('/', ' / ') for c in cells) + " |")
    print("\n**Every bug**\n")
    print("| program | corpus | case | axis | nested | " + " | ".join(h for _, h in BOARD) + " |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted(BOARD_ROWS, key=lambda r: (ORDER[r["prog"]], r["kind"], r["corpus"], r["case"])):
        nested = {True: "yes", False: "no", None: "unsplit"}[r["nested"]]
        print(f"| {NAMES[r['prog']]} | {r['corpus']} | {r['case']} | {r['kind']} | {nested} | "
              f"{WORD.get(r['c1'], r['c1'])} | {WORD.get(r['c2'], r['c2'])} | {WORD.get(r['c3'], r['c3'])} |")

if "--board-virtual" in sys.argv:
    # The same rows on VIRTUAL Capstone: CHERI | virtual malloc | virtual nested. Holes are expected while
    # the nested corpora's virtual arms are unmeasured; they are printed as (+N ?) and counted, and this
    # table never changes --board's exit status. With --all, the six programs' rows read their own
    # virtual arms where their EXTRA_GROUPS column is a virtual one (pymalloc, PostgreSQL, SQLite).
    VB = (("c1", "CHERI (quarantine = caught)"), ("v2", "virtual malloc"), ("v3", "virtual nested"))
    vrows = [r for r in rows]
    for r in extra:
        if r["c2arm"].startswith("virtual-"):
            vrows.append(dict(r, v2=r["c2"], v3=r["c3"]))
    for kind in KINDS:
        groups = collections.OrderedDict()
        for r in vrows:
            if r["kind"] == kind:
                groups.setdefault((r["prog"], r["layer"], axis(r)), []).append(r)
        if not groups:
            continue  # a kind with no rows on this board (mruby's "other" has no virtual arm)
        print(f"\n**{kind.capitalize()}, virtual Capstone**\n")
        print("| program | allocator layer | axis | n | " + " | ".join(h for _, h in VB) + " |")
        print("|---|---|---|---:|---:|---:|---:|")
        allr = []
        for (pg, layer, ax), g in sorted(groups.items(), key=lambda kv: (ORDER[kv[0][0]], kv[0][2] != "nested", kv[0][1])):
            allr += g
            print(f"| {NAMES[pg]} | {layer} | {ax} | {len(g)} | " + " | ".join(
                (board_cell(g, k) if k == "c1" else cell(g, k)).replace('/', ' / ') for k, _ in VB) + " |")
        print(f"| **Total** |  |  | {len(allr)} | " + " | ".join(
            (board_cell(allr, k) if k == "c1" else cell(allr, k)).replace('/', ' / ') for k, _ in VB) + " |")
    holes = sum(r[k] not in ("caught", "missed") for r in vrows for k in ("v2", "v3"))
    print(f"\nvirtual cells that are not a reading yet: {holes}")

if "--board-json" in sys.argv:
    out = sys.argv[sys.argv.index("--board-json") + 1]
    pathlib.Path(out).write_text(json.dumps(dict(
        rows=[{k: r.get(k) for k in ("prog", "corpus", "layer", "case", "kind", "nested", "c1", "c2", "c3", "c2arm",
                                     "c3arm", "title", "cap", "sub", "fcap", "fcheri", "kcap", "kcheri", "scarve")}
              for r in BOARD_ROWS],
        **{kind: board_rows(kind) for kind in KINDS}), indent=1, ensure_ascii=False) + "\n")

if "--markdown" in sys.argv:
    NAME = {"ffmpeg": "FFmpeg", "wireshark": "tshark", "memcached": "memcached", "Total": "**Total**"}
    for kind in ("temporal", "spatial"):
        tr = table_rows(kind)
        print(f"\n**{kind.capitalize()} ({tr[-1][3]})**\n")
        print("| program | allocator layer | axis | n | ASan | CheriBSD | Capstone bounds | Capstone + Sublet |")
        print("|---|---|---|---:|---:|---:|---:|---:|")
        for p, layer, ax, n, cells in tr:
            print(f"| {NAME[p]} | {layer} | {ax} | {n} | " + " | ".join(c.replace('/', ' / ') for c in cells) + " |")


if odd:
    sys.exit(1)
