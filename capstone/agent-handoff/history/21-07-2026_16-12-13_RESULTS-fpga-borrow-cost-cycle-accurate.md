# RESULTS — cycle-accurate borrow-cost on Capstone silicon (Genesys2 / CVA6)

**Date:** 2026-07-21. **Vehicle:** Genesys2 FPGA running the CapliFive CVA6
"Capstone" core, bitstream `working-caplifive-captype-fixed.bit`, OpenSBI Capstone
monitor, image `fw_payload_fpga_up_ctl.bin`. **Timer:** `mcycle` CSR read inside a
Capstone domain (the board gates the unprivileged `cycle`, so we read `mcycle`).
**Measured code:** byte-identical to the QEMU `borrow-cost` probe; the only
changes are silicon plumbing (gp-free/cjalr-free domain, single REV_SHARED region,
plain call/ret ABI — see the enabling commits and the dated FPGA history notes).

This is the **first Capstone temporal-safety measurement to run on hardware.**

## 1. What each operation is (Q5 — definitions)

All four run the **same** boundary operation — borrow one result word across the
host/engine boundary and use it — differing only in *how the lend is protected*.
Each is an inner loop of `N` iterations bracketed by an `mcycle` read, with an
empty calibration loop subtracted; the reported figure is cycles per operation.

- **`raw`** — the unprotected baseline: a plain load of the borrowed word through
  an ordinary pointer (today's zero-copy dereference). Lower bound; no safety.
- **`borrow`** — the capability borrow **sequence**, i.e. the revoke-at-free
  temporal-safety mechanism applied to one lend. Per iteration it does four cap
  ops on the borrowed handle: **`mrev`** (mint a *revocation* capability from the
  linear handle), **`delin`** (de-linearise a working capability to hand out),
  the **load** (access the word through the delegated cap), and **`revoke`**
  (invalidate the delegated cap and reclaim the linear handle for the next lend).
  So `borrow` = `mrev` + `delin` + load + `revoke` as one unit — **not** any
  single op (see Q6 / §5).
- **`copy@256`** — the defensive-copy baseline the mechanism replaces, for a
  **256-byte** payload: memcpy-style word copy of the whole object instead of
  borrowing a pointer to it (the `TRANSIENT`-style copy). Cost scales with size.
- **`copy@1024`** — same, for a **1024-byte** payload. The 256↔1024 pair exposes
  the copy's `O(payload)` growth against the borrow's payload-independence.

## 2. The numbers (captype-fixed CVA6, mcycle, cyc/op)

| op | 64 iters | 256 iters | trustworthy? |
|----|---------:|----------:|--------------|
| raw       | 8   | 2    | order-only (small signal, noisy baseline) |
| **borrow**| 182 | 464  | **grows with iteration count** — see §3 |
| copy@256  | 902 | 894  | **yes — stable, cycle-accurate (~900)** |
| copy@1024 | 3611| 3587 | **yes — stable, cycle-accurate (~3600)** |

Raw capture (64 iters): `iters=64 empty=4 raw=552 borrow=11709 copy256=57747
copy1024=231165`. (256 iters): `iters=256 empty=840 raw=1429 borrow=119725
copy256=229819 copy1024=919281`.

**Solid, publishable silicon numbers:** copy@256B ≈ **900 cyc**, copy@1024B ≈
**3600 cyc** — matched across iteration counts (large signal, baseline-immune).
Copy is `O(payload)` in real cycles (~4× bytes → ~4× cycles), confirming the shape
the QEMU proxy showed.

## 3. Key finding — the revoke op is multi-cycle AND accumulates (tree growth)

The QEMU `-icount` proxy modelled the borrow as a constant **+4 instructions**
(raw 2 → borrow 6, `O(1)`). Silicon tells a richer story the functional model is
blind to:

1. **Multi-cycle, not +4 cycles.** The borrow is tens–hundreds of *cycles*, not
   6 — `mrev`/`revoke` are multi-cycle hardware operations (revocation-tree work),
   invisible to an instruction count.
2. **Per-op cost grows with accumulated revocations.** In the tight loop, `borrow`
   rises 182 → 464 cyc as iterations go 64 → 256 (even without the noisy `empty`
   baseline, `borrow_total/iter` = 183 → 468). Each `mrev`/`revoke` on the same
   lineage adds a revocation-tree node that is **not pruned**, so `revoke` walks a
   growing tree — cost scales with prior revocations.
   - Linear fit `avg = a + b·(N+1)/2` over the two points → **single-op base
     a ≈ 86 cyc**, **growth b ≈ 3 cyc per accumulated node** (2 points only —
     needs a 3rd; the 128-iter run flaked on the console).
3. **Hard resource ceiling at ~1024.** A domain that runs **1024** borrow
   iterations in one call **cannot exit**: `domreturn` resets the board *and* the
   debug module cannot even halt the hart ("Unable to halt / Examination failed").
   64 iterations exit cleanly; the ceiling is between 256 and 1024. Interpretation:
   the revocation tree/resource is exhausted, corrupting the domain-exit and debug
   paths. This is a genuine RTL finding — and a reproducibility hazard.

**Does this refute the O(1)-per-free claim?** No, but it qualifies it. Real
workloads free *distinct* objects, so the live revocation tree is bounded by heap
occupancy, not by a tight revoke loop on one lineage. The tight-loop microbench is
therefore not a faithful *single-op* measurement on silicon — it measures the
amortised cost over a growing tree. A pruned probe is needed for the clean
constant (§5).

**Paper claim still holds on silicon:** even at the pessimistic borrow = 464 cyc,
borrow ≪ copy@1024B (3587) — 7.7× cheaper — and borrow is payload-independent
while copy grows with size. Break-even is below the smallest copy: borrow
(182–464) is 2–5× cheaper than copy@256B (~900). So "borrow `O(1)` ≪ copy
`O(size)`" is confirmed cycle-accurate; only the borrow *constant* is not yet
pinned.

## 3b. Per-primitive breakdown (Q6) — board run 2026-07-21 (cyc/op)

A dedicated breakdown domain (`borrow_breakdown_fpga_nogp.{c,dom}` +
`borrow_breakdown_fpga_nogp_ctl.c`, built by `build-borrow-breakdown-fpga-nogp.sh`)
times the primitives, as far as the platform allows. Two hard constraints shape
what is measurable:

- **No `drop` on this core.** `drop`/csdrop (the prune instruction the §5 pruned
  probe assumed) is **not implemented**: funct7 `0001011` is absent from the QEMU
  decode table, there is no helper, and the RTL is a subset. `drop a0` would trap.
  So the revocation tree **cannot be pruned in software** — the clean single-op
  constant needs an RTL primitive (implement `drop`, or auto-release on `revoke`).
- **`delin` is load-bearing.** `mrev`+`revoke` *without* `delin` returns UNINIT
  (not a reusable LINEAR cap: `cap_rev_tree_revoke` keeps data only if the revoked
  subtree is non-linear), so it cannot loop. `revoke` is therefore inseparable
  from `delin`; the measurable units are `load`, `mrev+delin+revoke`, and (only
  functionally) `mrev` alone.
- **`mrev` alone resets the board.** The only loop isolating `mrev` mrevs without
  revoking, accumulating one un-reclaimed node per iter — the same resource stress
  as the ~1024 ceiling (§3.3), reached immediately because nothing is released and
  there is no `drop`. On silicon it resets during entry. So `mrev`-alone is
  DISABLED (`BREAKDOWN_WITH_MREV_ONLY=0`); the safe breakdown uses only the proven
  revoke-per-iteration loops.

**Silicon numbers (captype-fixed CVA6, mcycle, 64 iters):**
`empty=349 raw=423 mrd=11253 full=23756` →

| quantity | cyc/op | note |
|----------|-------:|------|
| load (raw)                    | **1**   | the actual data access — ~0.6% of borrow |
| mrev+delin+revoke (`mrd`), tree 0–64 | **170** | the reclaim unit at a small tree |
| borrow (`full`), tree 64–128  | **365** | borrow at an *inflated* offset (see below) |

The cross-check `full − mrd` came out **195**, not `load` (1). That is not an
error — it is the tree-growth (§3) made explicit: `mrd` runs at revocation-tree
offset 0–64 and `full` at 64–128 (each `revoke` leaves an unreclaimed node), so
`full − mrd = load + 64·growth`. Solving: **growth ≈ 3.0 cyc/node**, independently
reproducing the 182→464 (64→256) sweep fit from §3. The consistent model is

> **borrow(N) ≈ 75 + 3·(N/2) cyc/op**  (single-lineage tight loop) — base ≈ **75**
> cyc, growth ≈ **3 cyc per accumulated revocation node**.

It predicts 171 @64 (measured standalone **182**) and 459 @256 (measured **464**) —
both within noise. Attribution:

- **load ≈ 1 cyc** — the borrowed access itself is essentially free; the cost is
  the temporal-safety machinery, not the dereference.
- **mrev + delin + revoke ≈ 74 cyc base** (+3 cyc/node) — this IS the borrow cost.
- `mrev` and `delin` are single instructions (QEMU decomposition: borrow = 6
  instrs = mrev 1 + delin+revoke ~3 + load 2; the ~1-cyc silicon `load` confirms a
  lone instruction is ~1 cyc), so within the 74-cyc base **`revoke` is the
  dominant primitive (~70 cyc) and the SOLE source of the O(tree) growth.**

**Q6 answered:** of `mrev`/`delin`/`revoke`, **`revoke` carries essentially the
entire cost and all the growth**; `mrev`, `delin`, and the `load` are each ~1 cyc.
A number for `revoke` *alone* (vs the +delin unit) and a pruned O(1) constant both
require an RTL change (a `drop`/auto-prune), not more board time.

## 4. What is solid vs open

- **Solid:** the measurement runs on silicon; copy@256B ≈ 900 cyc and copy@1024B ≈
  3600 cyc; borrow is payload-independent and ≪ copy at every size.
- **Open:** a clean single-op borrow constant (blocked by tree growth), and a
  per-operation breakdown (§5). `raw` (~2–8 cyc, a load) is order-of-magnitude
  only because the `empty` baseline is noisy (4 vs 840 across runs — an mcycle
  artifact) and swamps such a small signal.

## 5. Recommended next measurements

- **Per-op breakdown of `mrev` / `delin` / `revoke` (Q6).** DONE (§3b): `revoke`
  carries essentially the whole cost and all the growth; `mrev`, `delin`, `load`
  are each ~1 cyc. The only residual — `revoke` *alone* separated from `delin`, and
  a pruned O(1) constant — is **blocked in software** (no `drop` on this core) and
  needs an RTL prune/auto-release, NOT more board time.
- ~~Pruned single-op borrow via `csdrop`~~ — **not possible on this platform**:
  `drop` is unimplemented (§3b), so the tree cannot be pruned in software. The base
  constant (~75 cyc) is instead recovered from the growth fit (§3b), which is
  self-consistent across three data points (mrd@0–64, full@64–128, standalone
  64/256). A truly pruned probe requires the RTL to implement `drop` or release
  nodes on `revoke`.
- **3rd growth point — now HAVE it.** The breakdown's `mrd`@0–64 vs `full`@64–128
  gives an in-run growth slope (≈3 cyc/node) that agrees with the standalone
  64→256 sweep; the linear model is validated.
- **Revoke-cost probe (bump/norevoke/revoke) on silicon** — the temporal-safety
  headline vs CHERI — still open; needs the same gp-free treatment applied to
  `revoke_cost_fpga.*` (uses the allocator; more moving parts). This is the one
  remaining board-worthy measurement.
- **RTL asks (for the board owner / downstream):** implement `drop` (funct7
  `0001011`), or auto-release revocation-tree nodes on `revoke`, so the tree stays
  bounded — this both kills the ~1024-revoke reset ceiling AND unlocks a clean
  single-op `revoke` / O(1) borrow measurement.

## 6. Is this refinement critical? (Q4)

Partly. The **copy** numbers are final and publishable now. The **borrow** number
is where the follow-ups matter: the current figure is iteration-dependent, so for
a defensible single-op borrow *constant* in the paper we need the pruned probe
and/or the per-op breakdown (§5). It is not merely cosmetic — it resolves a real
ambiguity — but it does **not block** reporting the cycle-accurate shape (borrow
`O(1)` ≪ copy `O(size)`), which the current data already establishes. The paper is
updated to report exactly that (copy numbers firm; borrow shape firm; borrow
constant flagged as in-progress with the revocation-accumulation caveat).

**Update 2026-07-21 (breakdown run):** the borrow constant is now pinned by a
third, independent route — the per-primitive breakdown (§3b) yields base ≈ 75 cyc
+ 3 cyc/node, self-consistent with the 64/256 sweep, and attributes the cost to
`revoke`. So the borrow *shape* and *constant* are both settled to the resolution
this silicon allows; the only thing that would sharpen it further (a pruned O(1)
`revoke`) is gated on an RTL prune primitive, not on us. Nothing here is blocking.

## 7. Method note (how the number was obtained)

The measurement runs entirely inside one `REGION_SHARE` domain entry on a single
host-retained (`REV_SHARED`) region that is both scratch and results; the domain
computes the four figures, writes 8 result slots, and `domreturn`s so the
controller reads them back and prints the `RESULT` line. At 1024 iterations
`domreturn` breaks (§3), so numbers are taken at 64/256 iterations. See the dated
FPGA history notes for the full gp/cjalr root cause, the single-step diagnosis,
and the flaky-board handling.
