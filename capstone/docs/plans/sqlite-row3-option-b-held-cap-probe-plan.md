# Plan — row3 Option B, faithful path: single-domain **held-cap** revoke probe

*A-lane. Supersedes the "Tier-1 monitor-mediated" idea (see why below). Depends on
findings in `history/09-07-2026_18-10-43_option-b-intra-domain-revoke-firmware-spike.md`
(spike + RESOLVED + delivery-ABI follow-on) and B's task-005 mechanism proof.*

## Goal

Prove the **literal** single-domain BORROW-REVOKE (row3 Option B "gold standard")
at the runtime level: **one** domain receives a **real monitor-granted linear
capability**, `MREV`s it, uses an alias derived from it, `REVOKE`s at a lifecycle
point, and the cached alias **faults** — with the revocable arena reachable *only*
through the tracked cap. This is stronger than the two-domain Option A (collapses
to one domain, revoke is intra-domain over a held cap) and than B's task-005
(which used a `csdebuggencap` hand-minted cap; this uses the real delivery path).

## Why not "Tier-1 monitor-mediated single-domain"

`revoke` sweeps the **junior** lineage, not the root: the lender's own mapping
survives revoke (`borrow-revoke-uaf-probe` reads its word fine post-revoke). So a
lone entity cannot revoke-and-fault its **own** root cap. A monitor-mediated
"single domain" therefore either needs a second borrower entity (= the existing
2-entity probe ≈ Option A, already built) or collapses into the held-cap path
anyway. The faithful single-domain proof *is* the held-cap path.

## Key enabling finding (delivery already exists)

- `sbi_capstone.c` `shared_region_annotated(...)` →
  `__domcallsaves(d, CAPSTONE_DPI_REGION_SHARE, r)` hands the domain the **linear**
  region cap `r` (`REV_BORROWED` = monitor retains an `__mrev`; `REV_TRANSFERRED` =
  no monitor revoke, domain owns it).
- `my_first_domain/start.S` receives it as a **capability** (`stc a1,sp,80` →
  `ldc a0,sp,80`), surfaced to `domain_main` as a cap argument.
- So no new monitor/ABI primitive is required — only **domain-side glue** to bind
  that delivered cap and drive `mrev`/`revoke` through it.

## Approach (incremental, de-risked)

**Step 0 — locate the domain-side receive point.** The probe domains use the
`sbi.dom` scaffold (not `my_first_domain`). Find where the `REGION_SHARE`-delivered
cap lands on the domain side (the saved-cap slot / `domain_main` cap arg equivalent
in the `.smode` payload path) so domain C can bind it as a `void * __capability`
(vs the current `REGION_QUERY`→address pattern). If the `.smode` entry does not
already surface the cap, add the minimal receive glue in the probe's own entry (do
**not** edit shared `start.S`/monitor).

**Step 1 — `-O0` mechanism proof (dodges C1).** New probe under
`capstone/tests/runtime-qemu/` (e.g. `intra-domain-mrev-revoke-probe/`):
- one domain gets a region shared to itself as `REV_TRANSFERRED` (linear, domain
  owns) or `REV_BORROWED`;
- bind the delivered cap `arena`; `R = __builtin_capstone_cap_mrev(arena)`;
  `alias = __builtin_capstone_cap_delin(arena)` (C3: passing a LINEAR cap by value
  is consumed — `delin` first; see B task-005);
- write through `alias` (live); `__builtin_capstone_cap_revoke(R)`; re-deref
  `alias` → **must fault** (assert exact cause, per B's method note: cause-25 is
  self-proving; every cause-24 needs a no-revoke control);
- a control cap / unrelated buffer must **survive**. Compile the payload at `-O0`
  to avoid C1 (fastcc + cap-arg ICE).
- Land: source + `build-*.sh` + `run-*.sh` + README, mirroring the existing
  runtime-qemu probe layout. Serialize the QEMU run (single `rootfs.ext2` lock).

**Step 2 — `-O1/-O2` build (after B task-006 lands C1 fix).** Rebuild the same
probe at `-O1/-O2` to confirm optimized codegen preserves the fault (and that C2's
MREV-purity fix keeps the `MREV` from being DCE/CSE'd). This is also the gate for
the SQLite integration.

**Step 3 (separate, later) — SQLite integration.** Single SQLite domain, its
column-name/heap pointer `MREV`'d, revoke at `sqlite3_finalize`, cached alias
faults. Needs the linear-backed-heap question (memsys5 arena linear-backed) — its
own plan; not this doc.

## Q1 tie-in (evidence bar, lead)

This probe is the concrete artifact behind the Q1 fork: it demonstrates the
faithful single-domain held-cap revoke. If the lead's bar accepts it as the row3
"after," Option A becomes unnecessary as the headline. Frame for the lead: (a) real
monitor-delivered linear cap, (b) intra-domain instruction-level revoke, (c)
cached alias faults with asserted cause, (d) provenance rule enforced (arena
reachable only through the tracked cap).

## Constraints
No `Co-Authored-By:`; never commit debug/report files; `git add` exact paths;
commit only when asked; keep `capstone-qemu`/`caplifive-buildroot` submodules
clean; serialize QEMU matrix runs (one `rootfs.ext2` write-lock); work on
`capstone-bootstrap`. Do **not** edit shared `start.S`, the monitor, or
`capstone-c`; confine changes to the new probe dir + its build/run scripts.

## Verification
- Step-1 probe GREEN: post-revoke `alias` deref faults with the **asserted** cause;
  control survives; no-revoke control does not fault.
- The arena is reachable only via the tracked cap (no `gp`/ambient second path) —
  add an ambient-miss control like B's `mrev_ambient_miss` to prove the negative.
- Step-2: same GREEN at `-O1/-O2` on the C1/C2-fixed compiler.
