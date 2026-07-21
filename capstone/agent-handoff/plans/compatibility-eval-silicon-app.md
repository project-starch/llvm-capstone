# Plan (proposal): Compatibility evaluation + running a real program on silicon

**Status:** PROPOSAL — for review before implementation.
**Date:** 2026-07-21.
**Owner goal:** close the third evaluation axis. Of the three eval goals —
Performance, Security, Compatibility — the first two are near-complete; Compatibility
has no handle yet. This plan is the Compatibility story plus the silicon-app work that
backs it.

## 0. The compatibility question we must answer

The open questions (from the project chat) are:

1. How is the defense *turned on* at an interface?
2. Does **every** pointer get a "pointer contract" (PureCap), and does that break
   normal functionality?
3. Or is it turned on **selectively** for a host–engine interface — and if so, how is
   that subset determined?

Our answer has **two dimensions**, and the evaluation must show both:

- **Spatial safety is pervasive (PureCap).** Every pointer is a 128-bit capability with
  bounds, always on, set at materialization (`SHRINK`). The "pointer contract" is the
  bounds. *Claim to defend:* this does **not** break correct programs.
- **Temporal safety is selective (boundary-only).** The revoke-at-free machinery
  (`mrev`/`revoke`) is applied only to the pointers that cross the host↔engine
  boundary — the objects the host *lends* to the engine. The subset is determined by
  the interface: the lend points are the host→engine calls that pass pointers. These
  are exactly where **Cross-Domain Pointer Bugs** occur.

## 1. Two experiments (do not conflate them)

A recurring mistake to avoid: simple compute benchmarks have **no boundary**. So they
serve one experiment, not the other.

### Experiment A — Pervasive-spatial compatibility (no boundary)
*Answers Q2: does making every pointer a capability break functionality, and at what
cost?*

- Vehicles: **CoreMark, BEEBS, RV8, SQLite** — self-contained; no host/engine split.
- Result 1 (functionality): they compile and run **correctly** under PureCap (already
  green in QEMU: CoreMark/RV8 7/7, BEEBS 82/82). Correct execution *is* the
  compatibility evidence.
- Result 2 (overhead): ambient PureCap cost (wider pointers, HW bounds, `SHRINK` at
  materialization). Cycle-accurate on silicon.
- **This is the "we ran a real program on silicon" existence proof** — and it is the
  right vehicle for it, because it needs no boundary.

### Experiment B — Selective-boundary temporal protection (the real use case)
*Answers Q1/Q3: how is protection turned on at an interface, and how is the subset
determined?*

- Vehicles with a **real host↔engine boundary**: the existing **Host–SQLite** study
  and the **Lua** stretch case study (`capstone/paper/lua/lua-pointerbugs.tex`; ~23k
  LoC, already isolated, produces cross-domain-pointer bugs in the host).
- The boundary = the host↔engine API. The host lends pointers to the engine; a
  `borrow` protects each lend; `revoke` reclaims at the contract point.
- Result 1 (security): boundary protection catches the Cross-Domain Pointer Bugs
  (temporal + spatial across the domain line).
- Result 2 (overhead): *this* is the PI's "protect a small number of boundary pointers,
  revoke is rare" measurement — aggregate overhead expected negligible because lends
  are sparse relative to total work.

## 2. The enabling engineering: the `gp` bring-up (gate for any real app in a domain on silicon)

Today only hand-crafted, global-free, `cjalr`-free domains run on silicon. A real app
(globals + a real call graph) does not, because our backend reaches globals and forms
return capabilities through a `gp = PCC(cursor 0)` convention that **only our QEMU
fork fabricates**. This is **our toolchain gap, not a hardware flaw** (the RTL
correctly faults on `delin` of a null capability). Full root cause:
`history/20-07-2026_19-45-09_fpga-gp-free-domain-plain-call-ret.md`.

Two sub-problems:

- **Calls/returns:** lower intra-domain `cjalr` to plain `jal`/`jalr` (bounds-checked
  within PCC). This is the proper backend version of the one-line `.s` rewrite already
  proven to work. Bounded.
- **Global addressing without `gp`:** the crux. Two candidate fixes:
  - **Monitor-side (potentially small):** the OpenSBI Capstone monitor installs a
    *real, representable* capability in `gp` at domain entry (image-covering, cursor 0
    — a base-0 cap is the most representable case). Existing domains would run
    unmodified.
  - **Backend-side (larger):** teach the backend a `gp`-free global-addressing sequence
    deriving from an on-entry data-capability base.

### Step 1 (de-risking spike, 1–2 days)
Answer the single deciding question: **can the M-mode monitor mint a valid,
representable `gp` capability covering the domain image at entry?** Try the monitor
path on a trivial globals-using domain in QEMU (representability) then on the board.
- If **yes** → monitor path; ~1 week to a real integer app running in a domain on
  silicon.
- If **no** → backend `gp`-free codegen; several weeks. (`plans/backend-compiler-fixes.md`
  is the home for that work.)

### Step 2 (given the spike outcome)
Land the chosen fix + the plain-call/ret lowering; build **one integer-only benchmark**
(BEEBS kernel) into a domain; run on silicon.

## 3. Platform constraints to design around

- **FPU:** the board's glibc hard-float path traps → pick **integer-only** apps
  (CoreMark is integer-only; BEEBS has integer kernels). Removes the FPU wall.
- **Rev-node ceiling:** the rev-node pool is a fixed **1024-entry bump allocator with
  no slot reclamation** (`drop` invalidates but doesn't free; `head` monotonic). So a
  domain call is capped at ~1024 `mrev`s. Irrelevant to Experiment A (spatial `SHRINK`
  allocates **no** rev-node), and naturally satisfied by Experiment B (few boundary
  lends). Only matters if a single domain call revokes >1024 times.

## 4. Proposed sequence

1. **Now (no board):** QEMU experiment for Experiment B's overhead question — instrument
   a boundary workload (Host–SQLite first, it exists) to count how often `revoke`
   actually fires vs total instructions, and report aggregate overhead. Directly tests
   "revoke is rare." Matches the QEMU-to-QEMU method already in the paper.
2. **1–2 days:** the `gp` spike (§2 Step 1).
3. **~1 week after spike:** first real integer program (BEEBS kernel, then CoreMark) in
   a domain **on silicon** — Experiment A existence proof + ambient PureCap cycle cost.
4. **Lua boundary case study (Experiment B) — QEMU only, not silicon.** Lua (~23k LoC,
   heavy globals, `longjmp`, default floating-point) plus the documented board
   instabilities (UART-only transfer, console drops, silent in-domain resets, no HW
   breakpoints) make a silicon run a bad risk/reward — and unnecessary: an app case
   study is a functional + instruction-count story, which the functional model
   delivers. Silicon's unique value (cycle-accurate primitives) is already captured.
   Reserve silicon for the primitives (done) + one small integer benchmark (step 3).

## 5. What this delivers for the paper

- Compatibility Q answered on both axes: PureCap runs real programs correctly (A);
  boundary protection is selectively applied at the host↔engine interface, subset =
  the lent pointers (B).
- "We have silicon **and** run a real program on it under capability protection."
- The Cross-Domain Pointer Bugs framing gets a second case study (Lua) beyond
  Host–SQLite.

## Constraints (repo hygiene)
Propose-before-implement (this doc is the proposal). No `Co-Authored-By`; no submodule
source commits (`capstone-ariane`/`-qemu`/buildroot/system); `capstone/paper` is ours.
No real-person names in committed content. Board etiquette per
`ref/fpga-borrow-cost-reproduction.md`.
