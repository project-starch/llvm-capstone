# What is still open in the three-arm study

Hand-written companion to [PROTECTION.md](PROTECTION.md), which is the record: one
row per bug, three columns, every cell carrying the bundle it was read from. That
file says what was measured. This one says what was **not**, why, and what each
missing answer would cost to get -- because a blank cell and an unanswerable cell
weigh differently and the table alone does not separate them.

Counts below come from `protection.json` and were recomputed on 2026-10-09. The
gates were green at that point: `tools/check-corpus.py` CLEAN over 13 corpora and
192 declared cases, its `--self-test` rejecting all 19 corruptions, and
`tools/protection-matrix.py --check` finding the generated files fresh.

## The two kinds of open cell

**85 `not-run` cells.** No measurement exists. Every one carries its own reason in
`protection.json`; they are grouped by that reason below. 50 distinct cases are
involved, because a case blocked for one arm is usually blocked for the others too.

**26 `cheribsd` misses with no disposition.** The mechanism ran and said nothing,
and no run has yet established *why* -- whether the object reached the system
allocator at all. These are the cells the `disposition` mechanism was built for and
the only arm it is required of; `capstone-sysalloc`'s 62 misses are not in this
ledger, because at the malloc boundary a synchronous revoke leaves nothing to
explain: the object either arrived or was never freed, and the nested cases are
answered by the boundary split rather than per cell.

## Open items, largest lever first

### 1. The SQL and SQLite silences -- 37 cases, up to 71 cells

`sqlite/engine-repros` (28 cases) and `postgres/sql-repros` (9) print only their
query's result. **A completion there is not a measured silence**: without a marker
printed after the defective access, a clean exit cannot be told apart from a case
that never reached its defect, so all three arms declare `not-run` rather than
claim a silence they cannot prove.

What it needs: a `defect_marker` in each of the 37 cases, then one run per arm.
The marker work is by hand and is the bulk of it; the runs are cheap afterwards.
This is the single largest lever in the study -- it is 71 of the 85 blank cells.

### 2. PostgreSQL on `capstone-sublet` -- 9 cells

Independent of the marker work above: the Sublet port exists (five patches under
`ports/postgres/memory-contexts/patches`) and the image is built at
`/tmp/capstone/virtual-corpora-ml/pg-sublet/link/postgres.dom`. Measured cost of
the run: about 41 minutes. Blocked only in the sense that nobody has run it.

### 3. Wireshark's wmem silences -- 22 `cheribsd` cells, undisposed

The disposition would come from the quarantine probe, and the probe needs the case
to run on CheriBSD. **There is no tshark CheriBSD build**, so these 22 cannot be
explained on the present vehicle. They are the largest block of undisposed misses
and the one with no cheap path.

### 4. Wireshark cases 18-21 on `capstone-sublet` -- 4 cells

The chunk port covers cases 0-17; 18-21 were added afterwards and no nested-arm run
includes them. Separately, `WM_PAYLOAD_BYTES` is 256 MiB and mallocng refuses it,
against a measured peak need of about 2.4 MiB -- the constant, not the allocator,
is the problem.

### 5. The eight withdrawn cells -- httpd 4, memcached 4

`httpd/bucket-repros` 0, 1, 4, 6 and `memcached/allocator-repros` 2-5 were
**withdrawn by their own control on 2026-10-08**: each case was run beside its
upstream-fixed sequence in the same image and the fault fires on both, so the
column was reading a fault that is not the defect. They are `not-run`, not catches.

Both withdrawals trace to the open port defect written up in
`../ports/memcached/allocators/README.md`, section OPEN DEFECT: every case that
releases a slab chunk faults at `slabs.c:533:10`, the first read of `ptr` entering
`do_slabs_free`, cause 24 untagged, on the fixed sequence too. At `0x226d8` a
pointer is spilled through a `shrink`-narrowed stack slot (`stc` then `ldc`) and
reloads untagged. The first hypothesis is **refuted**: a silent image of the same
campaign has 187 `shrink`->`stc` pairs against this one's 317, so pair counting is
not the discriminator. Next step is the register state at the trap, not more
disassembly reading.

### 6. FFmpeg's buffer pools -- 4 `cheribsd` + 4 `capstone-sysalloc` cells

`ffmpeg/pool-repros` has no `build-cases.sh` and no CheriBSD runner, so the
dispositions cannot be read. The path has to be built the way `cpython/pymalloc`'s
was, which is the template to copy.

### 7. mruby case 11 -- 2 cells

Unstable in a different way on each vehicle that failed to measure it: an abort
under `cheribsd`, and cause 12 (`INST_PAGE_FAULT`) at `pc=0x3f967e4190` under
`capstone-sysalloc`, which is not a Capstone capability cause. `capstone-sublet`
does have a verdict for it -- a fault with cause c2 -- so only the two
malloc-boundary cells are open. **Diagnose it; do not re-run it.** Repetition has
produced a different failure each time.

## A process item: nothing asks whether a measurement is stale

`tools/run-virtual-corpus.py` records `platform_sha256` for the campaign and an
`image_sha256` per row. **Nothing reads them back.** Three full campaigns ran in one
night for at most 56 open cells because the platform moved twice underneath and no
check noticed. Before the next campaign: a staleness query that refuses to re-run
rows whose inputs have not changed, and flags rows whose inputs have.

## Not open, on purpose

- **PoisonCap.** Excluded by `arms.json`: it is our adapter for a competitor's
  platform, so a reader is entitled to discount it. The `poisoncap-*` arms stay in
  the cases as a record and no cell here reads them. No number from this study
  fills a PoisonCap cell anywhere.
- **The 17 ignored cases.** Subobject crossings inside one allocation that no arm
  in the study claims, plus the one unpaired memcached cell. They stay as material
  and leave every denominator; `--focus ignored` lists them.
- **CheriBSD's 44 `never-freed` and 5 `not-temporal` misses.** These are answered,
  not open: no sweep policy reaches an object that never arrives at the system
  allocator.
