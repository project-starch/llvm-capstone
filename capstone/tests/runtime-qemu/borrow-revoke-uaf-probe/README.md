# M0 borrow → revoke → use-after-revoke probe

First milestone of the capability-mediated SQLite marshalling direction
(`../../../agent-handoff/design/sqlite-marshalling-feasibility.md`). It tests the
proposal's load-bearing claim directly: *after the lender revokes a borrowed
region, the borrower's later dereference must fault.*

## Topology
- `borrow_revoke_uaf_probe_guest.c` — the **lender / controller** (ordinary guest
  Linux, uses `libcapstone`). Owns the region, lends it as a **revocable borrow**
  (`shared_region_annotated(.., PERM_OUT, REV_BORROWED)`), and calls
  `revoke_region()` between the two domain calls.
- `borrow_revoke_uaf_probe.smode.c` — the **borrower** (runs inside the domain).
  Round 1: caches the delegated pointer, writes the stage-1 sentinel. Round 2
  (after the lender has revoked): dereferences the **cached** pointer — the
  use-after-revoke.

Build/run: `../build-borrow-revoke-uaf-probe.sh`, `../run-borrow-revoke-uaf-probe.sh`.

## Result (2026-06-29) — a GAP, not yet safe-fail

The borrow and the revoke both work, but the use-after-revoke is **not trapped**:

```
round 1 retval = 0x101
word after round 1 = 0x1111111111111111      <- borrow live, stage 1 written
revoking borrowed region
region revoked                                <- revoke_region() succeeds
entering round 2 (use-after-revoke)
round 2 returned 0x202
word after round 2 = 0x2222222222222222      <- stage-2 store LANDED
NO-TRAP-GAP use-after-revoke store landed
```

So the borrower's cached pointer to a revoked region is still usable; the store
reaches shared physical memory (the lender sees `0x2222…`). This contradicts the
proposal's "a subsequent dereference then faults" guarantee **in this runtime
configuration**.

### Two earlier observations along the way
- Sharing with `REV_SHARED` (0x2) instead of `REV_BORROWED` (0x1) makes
  `revoke_region()` assert in `helper_csrevoke` (`type == CAP_TYPE_REV` fails):
  only `REV_BORROWED` establishes the revocable relationship the lender can later
  act on. (Distinct from the known `helper_csmrev` `CAP_TYPE_LIN` assertion on
  re-share-without-revoke.)
- With `REV_BORROWED`, `revoke_region()` itself is clean.

### Root-cause hypothesis (needs confirmation)
The borrower obtains the region via an SBI region query
(`SBI_EXT_CAPSTONE_REGION_QUERY`), which appears to hand it a mapping that is
**not a tracked child of the lender's revocable capability** (likely an ambient /
NONLIN mapping into the domain's address space). The QEMU revocation sweep
(`cap_rev_tree_revoke`) only invalidates capabilities tracked as descendants in
the revocation tree, so it misses the borrower's mapping. For revocation to have
teeth, the borrower must reach the region **only** through the delegated,
tracked, linear capability — not through an independent mapping. This is the
concrete instance of the "linear authority must originate from / flow through the
delegated cap" risk in the feasibility doc.

This probe is kept green on the *observed* (gap) behaviour so it is a stable
regression artifact. When revocation is made to bite, flip the final marker in
`../run-borrow-revoke-uaf-probe.sh` from the NO-TRAP-GAP line to the QEMU
`Cap mem access` fault diagnostic.
