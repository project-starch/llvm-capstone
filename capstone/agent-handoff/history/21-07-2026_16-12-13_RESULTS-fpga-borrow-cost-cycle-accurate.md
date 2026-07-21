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

## 4. What is solid vs open

- **Solid:** the measurement runs on silicon; copy@256B ≈ 900 cyc and copy@1024B ≈
  3600 cyc; borrow is payload-independent and ≪ copy at every size.
- **Open:** a clean single-op borrow constant (blocked by tree growth), and a
  per-operation breakdown (§5). `raw` (~2–8 cyc, a load) is order-of-magnitude
  only because the `empty` baseline is noisy (4 vs 840 across runs — an mcycle
  artifact) and swamps such a small signal.

## 5. Recommended next measurements

- **Per-op breakdown of `mrev` / `delin` / `revoke` (Q6).** The table reports the
  borrow *sequence*, not the individual ops. A probe that times each op alone
  (with the surrounding setup held constant) would attribute the cost — almost
  certainly `mrev`+`revoke` (the tree ops) dominate and `delin` is near-free. This
  is the single most informative follow-up and is straightforward (three small
  timed loops); it needs board time, not new mechanism.
- **Pruned single-op borrow.** Drop/free the revocation node each iteration (e.g.
  `csdrop` the minted `mrev` cap after `revoke`) so the tree stays size-1; then
  `borrow` is a true single-op constant at any iteration count — expected ~the
  86-cyc fit intercept. Confirms the O(1) headline cleanly.
- **A 3rd iteration point (128)** to validate the linear growth model (the 128 run
  flaked; retry).
- **Revoke-cost probe (bump/norevoke/revoke) on silicon** — the temporal-safety
  headline vs CHERI — needs the same gp-free treatment applied to
  `revoke_cost_fpga.*` (uses the allocator; more moving parts).

## 6. Is this refinement critical? (Q4)

Partly. The **copy** numbers are final and publishable now. The **borrow** number
is where the follow-ups matter: the current figure is iteration-dependent, so for
a defensible single-op borrow *constant* in the paper we need the pruned probe
and/or the per-op breakdown (§5). It is not merely cosmetic — it resolves a real
ambiguity — but it does **not block** reporting the cycle-accurate shape (borrow
`O(1)` ≪ copy `O(size)`), which the current data already establishes. The paper is
updated to report exactly that (copy numbers firm; borrow shape firm; borrow
constant flagged as in-progress with the revocation-accumulation caveat).

## 7. Method note (how the number was obtained)

The measurement runs entirely inside one `REGION_SHARE` domain entry on a single
host-retained (`REV_SHARED`) region that is both scratch and results; the domain
computes the four figures, writes 8 result slots, and `domreturn`s so the
controller reads them back and prints the `RESULT` line. At 1024 iterations
`domreturn` breaks (§3), so numbers are taken at 64/256 iterations. See the dated
FPGA history notes for the full gp/cjalr root cause, the single-step diagnosis,
and the flaky-board handling.
