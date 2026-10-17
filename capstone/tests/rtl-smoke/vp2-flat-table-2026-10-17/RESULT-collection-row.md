# Result — the collection row, both arms, and three corrections it forced

Sibling documents in this folder: `PREREGISTRATION-ceiling-row.md` (the 70,000-live-node ceiling
row, still unrun) and `PREREGISTRATION-boot2-module-179686bc.md` (boot 2 of the bug-corpus kit).
This document is the **collection** row and reads nothing about the ceiling. Section 3 corrects a
step in the ceiling row's pre-registration, which must not itself be edited.

## No boot was spent on this

The row was already in hand. `run.py`'s `sparse-churn` case is exactly the specified workload —
200,000 allocations, both arms, same module revision — and its own gate at `run.py:233-234` already
requires `collections > 0`. That check passed on both arms in the `179686bcf198` re-gate, so the
numbers below are extracted from runs that were made for another purpose rather than from a boot
commissioned for this row.

Provenance: records `<kit>/out/result-{flat,paged}-179686bc.json`, logs
`<kit>/runpy-{flat,paged}-179686bc.log`, both PASS 41/41; modules `c4440adde0d02347` (flat) and
`69a7016b6d10cd0a` (paged); source `179686bcf198`. The churn stats line is index 5 of the eleven
`CAPSTONE_VM_STATS` lines, which is independently confirmed to be the churn arm two ways: it is the
only line of the eleven with `collections > 0`, and `run.py` reads that same index for its own
`sparse_recycling` check, which passed.

## The row

Same workload on both arms, and the workload really is the same — not merely the same invocation:

| quantity | flat | paged |
|---|---:|---:|
| allocations (test's own marker) | 200,000 | 200,000 |
| nodes allocated | 600,041 | 600,041 |
| arenas / pages / faults / rounds | 10 / 489 / 24 / 8 | 10 / 489 / 24 / 8 |
| **collections** | **9** | **28** |
| **reclaimed** | **584,902** | **586,825** |
| node_capacity | 65,536 | 21,504 |
| node_bytes | 1,048,576 | 356,352 |
| node_growths | 0 | 5 |
| test's own pass | `VIRTUAL_SPARSE_CHURN_OK allocations=200000 protected_stale_tags` | identical |

Both arms collect, and reclaim within **0.33%** of each other (584,902 vs 586,825) — so the flat
table recycles identities as effectively as the paged one. The difference is in the *shape* of the
collection, not its effectiveness: 9 sweeps of ~64,989 against 28 of ~20,958.

## The shape difference is predicted by the source, not explained after the fact

`ensure_nodes` gates its opportunistic collection on

    collect_min = max(sweep_pages, node_batch_pages * 256)   /* = 4,096 at the default */
    collect_min = max(collect_min, node_capacity(c) / 4)

so the floor scales with capacity. The flat table is created at its full 65,536 and never grows, so
its floor is 16,384 from the first allocation. The paged table grows into 21,504, so its floor
averages far lower and ends at 5,376.

| | flat | paged |
|---|---:|---:|
| binding floor `capacity / 4` | 16,384 | 5,376 |
| ratio of floors | | **3.05x** |
| ratio of measured collection counts (28 / 9) | | **3.11x** |

Predicted 3.05x, measured 3.11x — within 2%, from the floor alone. That is the reading: **the flat
mode does not collect less, it collects less often and more per sweep, and the factor is its fixed
capacity.**

## Correction 1 — the stated rationale for `node_flat_records` is refuted by this row

`179686bcf198`'s source comment says lower values are needed because "a collection row needs both
arms to collect, and at the default that means driving them to ~65,278 identities, which would
exercise the ceiling in a row meant to be independent of it."

**That is wrong, and this row is the refutation.** At the default capacity both arms collect — 9 and
28 times — and neither goes near the 65,278 live-node ceiling, because churn *retires as it goes*:
total lifetimes (600,041) and simultaneously-live nodes are different quantities, and only the
latter is what the ceiling bounds. The collection row needed no cap at all.

The parameter is still correct and still useful — it is the only way to reach a *live-node* pressure
reading without 65,278 allocations, and `997ef5781f9b` made its committed contract true. But the
justification written into the commit that introduced it does not hold, and the row that was planned
at 4,096 records on both arms was unnecessary.

## Correction 2 — `node_max_records` would have voided the planned row

Recorded because the planned-but-unnecessary row would have used it to cap the paged arm.
`node_max_records` is an **id ceiling, not a capacity cap**:

    :475   if (node_max_records && required > node_max_records) return -ENOMEM;
    :1017  if (node_max_records && r.available > node_max_records - CAP_REV_TABLE_RESERVE) rc = -ENOMEM;

Both return `-ENOMEM` *before* either collection site, so capping the paged arm that way produces
exhaustion and never a collection — and "the paged arm does not collect" would have read as a
finding about collection when it was an artefact of the knob. The comparable cap is
`node_max_pages`: paged capacity grows 256 records per leaf page, so 4,096 records is
`node_max_pages=16`, and `node_bytes` is then `(leaves + 3) * 4096` — the three directory levels
cost three pages, confirmed against three measured points (4 leaves -> 7 pages, 20 -> 23, 84 -> 87)
and against `run-nodes.py`'s own `node_bytes == 7 * 4096` at capacity 1024.

## Correction 3 — a mechanism step in the ceiling row's pre-registration is wrong

`PREREGISTRATION-ceiling-row.md` step 2 says "A **software collection** runs and reclaims
**nothing**." No collection runs at all. There are two collect sites and `node-contract` reaches
neither:

    :488   if (owner && node_retired(c) >= max(required - available, collect_min)) {   /* floor */
    :489       error = collect(c, owner); ... collected = true;
    :495   error = node_grow(c, required);          /* flat: never grows, :211 */
    :498   if (!collected && owner && node_retired(c)) {        /* no floor */
    :499       error = collect(c, owner);

`node-contract` keeps every structural node live by construction, so `node_retired(c)` is 0:
`0 >= 16384` fails at :488 and `node_retired(c)` is falsy at :498. The flat ceiling is in any case
an immediate return at :476, before either site.

The predicted **reading** is unaffected — no `NODE_GROWTH_OK`, no `NODE_LIVE:0:70000`, cause 30 —
so the ceiling row still stands as pre-registered. Only the intermediate mechanism claim is
withdrawn. Note also that the pre-registration's step 1 pressure point (`available <= 256`) is the
*processor's* signal and not a module event: `ensure_nodes` returns 0 while `available >= required`,
so the first time the module does anything is at exhaustion, not at the reserve point.

One subtlety for anyone reading a future non-collection: `:498` is guarded by `!collected`, so if
the floored sweep at :488 ran and reclaimed nothing, the last-resort sweep is skipped. That cannot
arise below the floor, where `node_retired` cannot reach `capacity / 4`.
