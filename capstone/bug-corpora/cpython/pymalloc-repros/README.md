# CPython pymalloc defect corpus

Consumer-side temporal-safety defects in CPython whose memory comes from
pymalloc, so that no `free()` reaches `malloc` and a malloc-level tool has no
event to see. Each defect is one program around the real `Objects/obmalloc.c`
(ports/cpython/pymalloc), and a fault counts only at its labelled probe.

The arms: `virtual-malloc` (pymalloc stock on virtual Capstone), `virtual-nested-pools`
(the same with the port's patch 0003: every block a child lifetime of its
arena, `CDERIVE`/`CREVOKE`) and `cheribsd-revocation` (stock CheriBSD, libc
revocation as shipped).

**Every one of the 20 reachable defects has a driver.** That is the whole
reachable set at the pin, not a sample — the inventory, the triage and the three
corrections that took it from 23 to 20 are in
`docs/ref/cpython-pymalloc-defects.md`. Each arm's verdict names its bundle
(`results/2026-10-10-patch0003/` for the two virtual arms).

## The twenty

| # | upstream fix | consumer | allocator shape |
|---|---|---|---|
| 0 | `gh-143543` | `itertools.groupby` | free / reuse / stale read |
| 1 | `gh-146613` | `itertools._grouper` | free / reuse / stale read |
| 2 | `gh-142829` | `Context.__eq__`, `Python/hamt.c` | interior pointer into a freed block |
| 3 | `gh-142831` | the JSON encoder's items list | stale entry reached through a live array |
| 4 | `gh-145244` | the JSON encoder's dict key | bulk free, stale read on the error path |
| 5 | `gh-148660` | `OrderedDict.copy()` | **pointer load out of a freed block, then followed** |
| 6 | `gh-151295` | `bytes.join()` via `__buffer__` | payload buffer, pinned sub-512 |
| 7 | `gh-148395` | `{LZMA,BZ2,_Zlib}Decompressor` | cursor surviving across two API calls |
| 8 | `gh-112127` | `atexit.unregister()` | free / reuse / stale read |
| 9 | `gh-139210` | `ElementTree.iterparse()` | payload buffer, error path |
| 10 | `gh-142560` | `bytearray` search methods | **block ended by a REALLOC that moved it** |
| 11 | `gh-142783` | the `zoneinfo` weak cache | free and use on adjacent lines |
| 12 | `gh-143004` | `collections.Counter.update()` | free / reuse / stale read |
| 13 | `gh-144833` | the SSL module on `SSL_new()` failure | reads a field of the object it just freed |
| 14 | `gh-146011` | `_decimal`'s signal dict | dangling pointer parked in a surviving object |
| 15 | `gh-149449` | `unicodedata`'s capsule | **bare `PyMem_Malloc` block, cached elsewhere** |
| 16 | `gh-151403` | `subprocess` fork_exec via `__fspath__` | free / reuse / stale read |
| 17 | `gh-151416` | `os.spawnv` via `__fspath__` | free / reuse / stale read |
| 18 | `gh-151695` | the curses screen encoding | dangling pointer parked in a global |
| 19 | `gh-153539` | `TextIOWrapper.tell()` | free / reuse / stale read |

Nineteen are proven live at the pin by their own backports into 3.13 after our
tag. Case 4 is not: it was never back-ported, and the apply test fails on it only
because upstream renamed the function. It is live by inspection —
`Modules/_json.c:1621` of `v3.13.7` hands the borrowed key straight on with no
`Py_INCREF` at all. Its `PROVENANCE.md` quotes the pinned source.

## Twenty reports, ten shapes

Eight of the twenty — 0, 1, 3, 8, 12, 16, 17, 19 — reduce to one sequence: free
a small object, allocate the same size again, read through the pointer that was
kept. Eight separately reported defects, seven modules, fixed one at a time over
more than a year. They are kept apart rather than merged because the sameness is
the point: one revocation mechanism covers a class upstream keeps rediscovering.

The ten shapes, and what each is there to show:

| shape | cases | why it is not the same test |
|---|---|---|
| free / reuse / stale read | 0, 1, 3, 8, 12, 16, 17, 19 | the base case |
| interior pointer into a freed block | 2, 13 | the stale pointer is not the block's base, so base-address validation cannot see it |
| stale entry in a live array | 4 | the container survives; only one thing it points at died |
| pointer loaded out of a freed block | 5 | the unprotected outcome is a **plausible wrong answer**, not a crash |
| payload buffer | 6, 9 | a `char*` into a payload, not a `PyObject*` anything could follow |
| cursor across two API calls | 7 | dormant between calls; the object is consistent, the pointer is stale |
| realloc moved the block | 10 | the block is ended by a realloc, not a free |
| free and use adjacent | 11 | no callback, no re-entrancy: two consecutive lines |
| parked with no bound on reuse | 14, 18 | the dangling pointer waits in another object, or in a global |
| bare `PyMem_Malloc` block | 15 | not a `PyObject` at all; exercises the `PYMEM_DOMAIN_MEM` path |

Case 5 is the most interesting: `_odict_FOREACH` reads `node->next` *out of* the
node it just processed, so the freed block's link reads back valid after reuse
and the walk continues into the wrong node.

## What is different from the PostgreSQL corpus

**Three allocator layers, not two.** A freed object may stop at a per-type free
list (ten of them) and never reach pymalloc at all; the parser has its own arena
besides. The port covers pymalloc only, so every case records which layer its
object came from — `case.json` carries it.

**A size threshold that decides whether the case is blind at all.** pymalloc
serves requests up to 512 bytes; above that `PyObject_Malloc` falls through to
`malloc` and ASan *does* report the use-after-free. This binds cases 6, 10 and
19, whose upstream defects carry buffers that can exceed it. The threshold also
**removed a case**: `gh-143544`'s freed object is an exception class, a heap type
measuring 1704 bytes, so it is a malloc allocation and not a blindspot at all.

**Upstream states the problem itself**, in `Doc/using/configure.rst`: to use
AddressSanitizer you should combine it with `--without-pymalloc`, "to disable the
specialized small-object allocator whose allocations are not tracked by ASan".
That is the corpus's thesis in the project's own build documentation.

## Layout

    corpus.json          this corpus's declaration; ../../SCHEMA.md is the contract
    NN_<gh-NNNNN>_<slug>/   NN is the case number a run selects
        case.c           the sequence; one program per defect
        case.json        machine-readable claims: layer, size, shape, arms, oracles
        PROVENANCE.md    the upstream hunk, quoted; what is real and what is reduced
    shared/corpus.h      probes; what every case includes
    shared/build-cases.sh  builds one program per case, for either target
    controls/            the use-after-free control the virtual arms run first
    runners/virtual/     the virtual Capstone runner and its manual
    runners/cheribsd/    the stock CheriBSD revocation run
    observe/supervise.c  the CheriBSD supervisor that observes a case's fault
    tools/               the survey scripts behind the inventory's numbers
    results/<stamp>/     matrix.tsv and input hashes; never raw serial captures

One rule decides where a file goes. A case directory is **self-contained**: the
claim, its provenance and its sequence. `shared/` is what every case includes
and the script that builds them. Each target owns a directory under `runners/`
with its runner and its manual; `tools/` is what produces the inventory's numbers.

**One program per defect.** A `case.c` includes `shared/corpus.h` and writes
its sequence inside `PYC_CASE(N)`; the macro supplies `pym_replay`, so the file
is a complete translation unit and the port's one-source seam builds it on its
own. `shared/build-cases.sh` invokes that seam once per case and puts the
programs side by side as `bin/defect-NN`. All twenty build in under half a
minute for either target.

[`../../SCHEMA.md`](../../SCHEMA.md) states what a case is and what every field
means, and [`../../tools/check-corpus.py`](../../tools/check-corpus.py) enforces
it over every corpus -- required fields, dense case numbers, a
`PROVENANCE.md` beside every claim, every arm's oracle, and the shape table
actually partitioning the cases. It found the headline ratio wrong on the day
it was written: the table has ten rows and the prose said nine. Run it after
touching anything here:

    python3 ../../tools/check-corpus.py

## Running it

One source, one set of twenty cases, two targets:

| target | arms | manual |
|---|---|---|
| virtual Capstone (the board's columns 2 and 3) | `virtual-malloc` / `virtual-nested-pools` | [`runners/virtual/README.md`](runners/virtual/README.md) |
| stock CheriBSD purecap (the board's CHERI column) | `cheribsd-revocation` | [`runners/cheribsd/README.md`](runners/cheribsd/README.md) |

All build from the same `case.c` files.

## History

Until 2026-10-10 the corpus also ran a `spatial`/`sublet` pair in a physical Capstone domain
(`results/20260919-qemu-20`) and a PoisonCap pair on CheriBSD (`poisoncap-spatial`,
`poisoncap-protected`, `results/20260919-poisoncap` in the port). Both used the port's lifetime
adapter, which `CDERIVE`/`CREVOKE` made unnecessary; their targets and runners are removed and
their results stay as recorded.

## What is NOT here

- **No native arm.** The PostgreSQL corpus pairs its arms with an x86
  `before.c`; this one does not. A native arm must claim nothing from ASan's
  silence — that claim is tautological, because the memory never reached
  `malloc` — so it needs a Valgrind run to say anything, and Valgrind is not
  installed on this host.
- **Nothing above pymalloc.** The two free-list cases and the one `PyArena` case
  need ports that do not exist. `gh-126703` is in the free-list layer despite
  sitting in the pymalloc bucket, and is excluded here for that reason.
