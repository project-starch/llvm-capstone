# Review of the predictions, recorded before any result was read

`PREDICTIONS.md` was committed at 3cd43d008369 and both runs started; this independent review
(Codex, gpt-6.1-sol, read-only, against PoisonCap's `mem5.c` and SQLite 3.22.0's sources) arrived
while they were running. `PREDICTIONS.md` stays as committed. What the review shows is wrong or
unjustified in it is listed here, and the results are read against both.

## Corrections to the predictions

- **The quarantine does flush in a 1 MiB heap.** PREDICTIONS.md says a flush needs 4,096 frees or
  16 MiB in use "which a 1 MiB case heap never reaches". The queue holds 4,096 entries regardless of
  size; 4,096 blocks of the 64-byte minimum are 256 KiB. When the queue is full, the next free flushes
  it through `memsys5FreeUnsafe` without revocation (`mem5.c:587`), and the next allocation's zeroing
  through the heap capability clears the poison (their QEMU clears it on a write from a capability whose
  bounds are wider than the poisoned block). A stale pointer can then reach a reused block unreported.
  A case with more than 4,096 frees between the free and the stale access can therefore read as a miss
  by their design.
- **Case 02: "site, poison" is not justified.** The backup's cached `Btree` is allocated on its own
  (`sqlite3MallocZero`, btree.c:2304); moving `db->aDb` copies the pointer and frees nothing it points
  to, and this case's first ATTACH copies `aDbStatic` rather than reallocating. The declared site can
  dereference a NULL `pPage1` (btree.c:8983). Expected: a fault at the site that is not the poison report.
- **Case 24: the lookaside-on prediction is wrong.** Parsing CREATE turns lookaside off before
  `sqlite3StartTable` allocates the Table (parse.y:105, :158; build.c:934), so the Table comes from
  memsys5 in both rows. Expected in both rows: site, poison.
- **Case 03: right outcome, wrong object.** The stale object is the `sqlite3` connection
  (`sqlite3MallocZero`, main.c:2858), freed when the last statement is finalized; `sqlite3DbFreeNN` then
  reads `db->pnBytesFreed` (malloc.c:345). Not the lookaside buffer.
- **Case 12: right outcome, wrong object.** The host oracle's use-after-free is in the separately
  allocated `aSeg` array freed by `fts5StructureExtendLevel` (fts5_index.c:962). It is poisoned only if
  the reallocation moves the array.
- **Case 07** sweeps OOM injections over repeated connections without resetting the allocator, so
  quarantine can exhaust the heap before an injection reaches the defect.

Corrected totals the review implies (not a new prediction, a reading aid): both rows identical; case 02
at most "fault at site" without the signature; flushes and heap exhaustion can lower either count.

## Scoring changes, made before any result was read

- `verdicts.py`: a returned process that printed a failing `... rc=N` line first is
  RETURNED-WITH-ERROR, its own outcome (the harness's REPRO322_MAIN ignores `run_case()`'s result, so an
  early stop would otherwise read as a clean return). The runner records those lines from now on; the
  first two runs predate this and record only the last three lines of output.
- `verdicts.py`: a fault's function is the sized symbol containing the PC (`llvm-nm --print-size`); a PC
  in no sized symbol, or an object that cannot be read, is unresolved and never at a site.
- `verdicts.py`: signed requires SIGPROT as well as si_code 3 and the console's poison line.
- `verdicts.py`: with a spatial control, a control that did not read cannot support "attributable".

## Not changed, and why

- The console's poison lines are taken from the window of each process (processes run one at a time in
  a boot, and QEMU prints before it raises the fault). They are not correlated with the faulting
  address; a reader can compare the printed poison bounds with the fault by hand in `results.jsonl`.
- Function-level matching is weakest for the cases whose site is `memcpy` or `memcmp` (16, 17, 21, 23):
  any fault there counts. Their console poison bounds are listed with the result for that reason.
- The two mechanisms differ in allocation policy as well as in lifetime enforcement (request-sized
  against block-sized bounds; in-heap against external free-list metadata; quarantine). This is a
  comparison of the two implementations of memsys5 protection as published, not of the checks alone.
- The plan's overflow control is not run: the comparison is temporal only.
