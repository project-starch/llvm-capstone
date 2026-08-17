# MicroPython temporal-safety corpus

Tracking table for temporal memory-safety defects on MicroPython's own allocator,
the mark-and-sweep collector in `py/gc.c` plus the `m_malloc` layer above it.
Thirty cases in `temporal-allocator-corpus.csv`.

MicroPython is interesting here precisely because it does not use the system
allocator. It carves one heap at startup and manages it itself, so every case
below is a lifetime bug on memory the collector believes it owns.

**And that is why Capstone catches none of them.** This is the point of the
corpus, not a disappointment in it. `mpy_domain.c` declares the heap as a single
384 KiB static array and hands that one object to `gc_init`, so every block the
collector sub-allocates inherits a capability spanning the whole heap.
`evidence/heap-bounds-model.s` is the compiled proof: the bounds are set once, to
the entire object (`lui a3, 96`, so 96 << 12 = 393216 bytes), and `cincoffset`
then just moves a cursor inside them. A write through a stale pointer is a bare
`sb` that the hardware has no grounds to reject. `gc_free` compounds it by being
pure bookkeeping in a software bitmap: it never reaches the hardware, so nothing
is ever revoked and a dangling pointer stays indistinguishable from a live one.

Capstone does have the mechanism that would catch these, and the cross-language
work in `agent-handoff` shows revocation catching heap use-after-free where the
C library's `malloc`/`free` drives it. A nested allocator is exactly the case
where that driver is missing. The corpus therefore has two separate columns: what
an unmodified runtime gets today, which is nothing, and what a
capability-aware collector could get, which is the open research question.

## One directory per case

`cases/<ID>_<slug>/` holds each case on its own: a README rendered from the CSV,
the reproduction script where one can run here, and RESULT.txt where it has been
measured. `STATUS.md` is the generated index. Both come from `gen-cases.py`, and
`gen-cases.py --check` fails if a README has drifted from the CSV or if a case
marked runnable is missing its script or its result.

Six cases are measured. The other twenty-four are blocked, each with the specific
reason in its own directory rather than a shared shrug: eleven need a parent build
that a current toolchain cannot produce, five need threads, sockets or port
hardware this domain does not have, six are unresolved upstream, and two have only
a C-level reproduction.

## What counts, and what does not

In scope: use-after-free, double free, dangling views and buffers, premature
collection, lifetime and shutdown ordering, re-entrancy that invalidates a
pointer already in flight, and allocator invariants that fail on their own input.

Deliberately excluded: purely spatial defects. Three of the six MicroPython CVEs
in NVD are heap buffer overflows (`CVE-2023-7158` in `slice_indices`,
`CVE-2024-8946` in `mp_vfs_umount`, `CVE-2024-8948` in `mpz_as_bytes`), as are
issues 12587 and 13006. They are real, they are just a different question, and
bounds checking already answers it.

Worth stating plainly: **NVD lists only six MicroPython CVEs in total**, so a
corpus of thirty cannot be CVEs alone. Three CVEs are in the table and the other
twenty-seven are issues from the upstream tracker, each one a dated, linked,
reproducible report. Two of the three CVEs have their originating issue in the
table as well (`CVE-2023-7152` with issue 12887, `CVE-2024-8947` with issue
13283), which is useful: those pairs give both a repro and an official patch.

## Provenance

Every factual column was copied from an API response, not typed:

- issue rows: GitHub GraphQL, one query over all thirty-four candidates
- CVE rows: NVD REST API, `keywordSearch=micropython`
- the builder is `build-corpus.py`; rerun it to refresh state and titles

`verify-corpus.py` re-reads the CSV and compares every title and state against
the stored API responses. It was negative-tested by corrupting two rows, and it
reported both.

## Columns

| column | meaning |
|---|---|
| `id` | stable local identifier, `MPY-T01` to `MPY-T30` |
| `source`, `ref`, `url` | where the case comes from, and the link |
| `title`, `state`, `first_seen` | copied verbatim from the source API |
| `fix_commit`, `fix_date` | the upstream commit that fixed it, where one is established |
| `present_at_pin` | is the defect in the source we actually build: `yes`, `no`, `unknown` |
| `repro_base` | which MicroPython commit to build to see it |
| `class` | defect shape: `uaf`, `dangling-view`, `dangling-buffer`, `premature-free`, `lifetime-order`, `reentrancy`, `race-uaf`, `alloc-invariant` |
| `cwe` | CWE where the source assigns one, otherwise the closest fit |
| `component` | file or function, from the report |
| `scope` | `gc-core` (inside the collector), `gc-managed` (on collector memory), `port-heap` (port allocator interaction) |
| `trigger` | one line on what has to happen for it to fire |
| `traps_unmodified` | what Capstone gives an unmodified runtime today: `no` inside the collector's heap, measured |
| `traps_if_gc_cap_aware` | what a capability-aware `gc_alloc`/`gc_free` could give: **a prediction** |
| `repro_status` | `confirmed` for three rows, `none` for the rest |
| `stock_behaviour` | measured on a stock host build: `crash-sigsegv`, `silent-corruption`, `not-run` |
| `notes` | provenance and cross-references |

## Which source to build, per case

`fetch-micropython.sh` pins MicroPython at `2e3304a128b3`, dated **2026-08-16**.
That pin is only days old, so almost everything upstream has ever fixed is
already fixed in the tree we compile. Knowing a bug exists is therefore not the
same as being able to run it, and the corpus says which of the three situations
each row is in:

- **`present_at_pin=yes`, 11 rows.** Open upstream, so building the pinned tree
  is enough. These are the cases to start with.
- **`present_at_pin=no`, 11 rows.** Already fixed in our tree. To see the defect
  you must build `repro_base`, the fix commit's parent, which is the last commit
  that still has the bug. `fetch-micropython.sh` takes `MPY_COMMIT`, so this
  costs one environment variable, but note the Capstone portability patches in
  `../patches/` are written against the pin and will not all apply to a
  2019 or 2023 tree.
- **`present_at_pin=unknown`, 8 rows.** Closed upstream with no fix commit that
  names the issue. This is a negative search result and is recorded as such: no
  commit message references the issue, which is not the same as the bug being
  unfixed. Each needs a look at the source before it can be scheduled.

Ancestry is decided by `git merge-base --is-ancestor <fix> <pin>` on a full
clone, never by comparing dates, and `gen-fix-status.py` refuses to emit
anything if its own positive and negative controls do not both behave.

## What is actually expressible in our domain

"Builds on the pin" is necessary, not sufficient. The Capstone port is configured
with `MICROPY_VFS=0`, no threads, no sockets, `MICROPY_PY_SYS_STDFILES=0` and a
minimum ROM level, so several of the eleven pin-buildable rows describe triggers
the domain cannot even express: `MPY-T17` and `MPY-T22` need threads, `MPY-T27`
and `MPY-T30` need an ESP32 port with networking, `MPY-T16` needs a port shutdown
path that is not ours, and `MPY-T10` needs a filesystem for its `readinto()`
vehicle. What survives that filter is `MPY-T09`, `MPY-T11`, `MPY-T12` and
`MPY-T13`, of which the first three need no modules beyond `gc`.

## Measured on stock, at the pin

`repros/` holds the reproductions and `repros/run-on-stock.sh` runs them against a
stock host build of the pinned commit, which takes seconds and needs neither the
domain nor the board. Three rows have been run:

| row | issue | stock behaviour at `2e3304a` |
|---|---|---|
| `MPY-T11` | 17941 | SIGSEGV |
| `MPY-T12` | 18619 | SIGSEGV |
| `MPY-T09` | 18168 | **no crash** |

`MPY-T09` not crashing is the most useful result in the table so far, and it is
the reason `stock_behaviour` exists as a column. The reporter's script runs to
completion, but the defect is fully present: after resizing a bytearray that has
an active `memoryview`, the view is left addressing an orphaned buffer, reads
recycled heap content, and remains writable, while the bytearray it came from is
untouched by writes through it. `repros/stale-view-proof.py` measures exactly
that and `run-on-stock.sh` treats it as a positive control, failing loudly if it
ever reports the resize happening in place.

So `MPY-T09` is a use-after-free **write** that ordinary hardware performs in
silence. That makes it a better specimen than either of the two that segfault,
because a crash is the platform already doing the job we are asking Capstone to
do, whereas here there is nothing to see until the capability check exists.

The general lesson for the other rows: **upstream still having the issue open is
a proxy, and this is a case where the proxy was wrong about reproducibility.**
Run it on the host before spending a domain build on it.

## The two hypothesis columns

`traps_unmodified` is not a guess. It reads `no` for all 26 `gc-core` and
`gc-managed` rows because the heap is one capability and `gc_free` never talks to
the hardware, which is measured in `evidence/`. It reads `unclear` for the four
`port-heap` rows only because those involve a second allocator whose behaviour
has not been examined. `verify-corpus.py` enforces the rule: a row inside the
collector's own heap that claims to trap unmodified is rejected as a mistake.

`traps_if_gc_cap_aware` is the prediction, and it is worth exactly as much as any
unrun prediction. It asks what happens if `gc_alloc` narrows a capability per
block and `gc_free` drives revocation. It reads `trapped` for 20 rows, `unclear`
for 6 and `not-trapped` for 4. Nothing behind it has been built or run.

The four `not-trapped` entries are the honest ones and the most useful: a
finaliser deadlock, a data race, a leak and over-retention are not reach
violations, so no amount of capability discipline addresses them, and a corpus
where every row confirmed the thesis would not be worth having.

The distinction that makes a row valuable is therefore not "does it crash" but
"is it invisible today and visible under an instrumented collector". By that
measure `MPY-T09` is the strongest row in the table.

## Next step

Nothing in this folder has been reproduced yet. The obvious first targets are
`MPY-T09` and `MPY-T10`, issues 18168 and 18171: both are open, both build on
the pin with no version archaeology, both are pure Python triggers needing no
port hardware, and 18171 turns a stale memoryview into a heap-corrupting write,
which is the sharpest primitive in the table.

`MPY-T02` is the best of the already-fixed rows to attempt second, because NVD
states its trigger outright, a bytes object resized and copied into itself, and
it comes with both a public repro (issue 13283) and an official patch.
