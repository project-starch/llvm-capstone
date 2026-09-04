# Sharing-bug taxonomy, and where the novelty is

*Framing note for the paper. Not an investigation — see `history/` for those.
Written 2026-08-13.*

## TL;DR

* **The bugs are ordinary. The boundary is what makes them undetectable and the mechanism
  affordable.**
* Classify by **which dimension of the loan is violated** (duration / exclusivity /
  permissions / extent / validity / identity), then cross-cut by **who ends the lifetime** —
  that second axis is the sharper one, because quarantine-and-sweep keys on `free()`, so every
  lifetime that ends by other means is a *structural* miss, not a timing miss.
* **Security novelty** = reuse-not-free (3), hierarchical (4), the **structural half of class 1**
  (GC-ended and internal-allocator-ended lifetimes), and TOCTOU (6) if we build it.
* **Performance novelty** = the **timing half of class 1** — plain free-ended UAF, where CHERI
  matches at `eager` and the argument is cost.
* **Design-only, do not evaluate**: confused deputy (8), sub-object (9), uninitialised (10),
  provenance (11).
* Splitting class 1 is not cosmetic: without it the security claim rests on rows with
  **1, 6, 0, 0** rows of evidence while ~40 reproduced rows sit in the performance column.

## The novelty split

| Column | Classes | Why it belongs there |
|---|---|---|
| **Security** | 3, 4, **1-structural**, (6) | No CHERI configuration catches these at any cost — there is no knob |
| **Performance** | **1-timing** | CHERI *can* match with `eager`; the claim is that it costs ~14–16.8 M instr/free against our +5 |
| **Design only** | 8, 9, 10, 11 | Real properties, but no corpus and no realistic path to one |

**Promote row 3.** The motivating figure (`parts/borrowing.tex:48-66`) is
`sqlite3_column_text()` cached across a `step`, and the prose calls it a UAF. But
`evaluation.tex:89-91` describes row 3r as *"reuses a statement-owned buffer in place without
freeing it; the capability stays tagged and in-bounds, so the stale read succeeds under every
configuration."* **Same shape.** If the `column_text` path reuses rather than frees — confirm
against SQLite internals — then the paper's opening example *is* class 3, and the strongest
sentence we own applies to it. Right now the paper labels it UAF and gives that away.

## Taxonomy

| # | Class — what it is | Dimension | Where it happens | Evidence | Column |
|---|---|---|---|---|---|
| 1 | **UAF / use-after-close** — A frees or `close()`s; B still dereferences | Duration | Every FFI seam: CPython, PHP, cgo, Rust FFI, JNI, plugin hosts | **~40 rows**, 3 corpora | split, see above |
| 2 | **Use-after-reallocation** — the slot now holds someone else's object; the exploitable step | Duration (weaker) | Same as 1, plus heap grooming | baseline only | — (this is what CHERI provides) |
| 3 | **Reuse-not-free** — owner recycles the buffer *in place*; no `free()` ever happens, pointer stays tagged and in-bounds, only the data's identity changes | Duration, no allocator event | SQLite column text, DB drivers, VM register stacks, protocol parsers, ring buffers | **1 row (3r)** + likely the motivating example | **Security** |
| 4 | **Hierarchical lifetime violation** — parent destroyed, derived children survive | Duration, transitively | Handle APIs: conn→stmt, txn→value, ctx→object, window→widget | **6 rows** | **Security** |
| 5 | **Double-free** — both sides believe they own the handle | Exclusivity | FFI ownership confusion: GC finalizer *plus* explicit close | 3 rows | supporting |
| 6 | **TOCTOU / double-fetch** — both sides hold write access concurrently; A validates then uses, B mutates between | Exclusivity | **kernel↔user syscall args**, hypervisor↔guest, enclave↔host, shm IPC, wasm host calls | **0 rows** | **Security, if built** |
| 7 | **Callback re-entrancy** — B calls back into A while A's loan is outstanding | Exclusivity across control transfer | Hooks, authorizers, progress handlers, finalizers calling native code | 4 rows | supporting |
| 8 | **Confused deputy** — loan grants more authority than the task needs | Permissions | Syscall interfaces, plugin APIs, IPC | **0 rows** | design only |
| 9 | **Over-wide loan / sub-object** — bounds cover the allocation, not the shared field | Extent | Struct passing across FFI | 2 rows, both *intra*-domain | design only |
| 10 | **Uninitialised read** — object handed over before construction finished | Validity | Two-phase init, error paths | 1 row | design only |
| 11 | **Provenance forgery** — rebuild a pointer from an integer to regain revoked authority | Identity | ABI seams, serialization, JIT | analysis only | design only |

**Frequency reality check.** 1 and 6 are the two genuinely high-frequency, heavily-CVE'd
classes in the wild; 6 is the one we have no evidence for. 3 is *underreported precisely
because no allocator event occurs* — there is nothing to log — not because it is rare. 4 is
common in any handle-based API. 8 is structural and almost never filed under that name, which
is why it cannot carry an evaluation.

## Who ends the lifetime — the axis that explains the blindness

lender frees · lender **reuses without freeing** (3) · a **collector** reclaims — the borrower
is invisible to it (7 of 13 Lua rows) · a **parent destroy** cascades (4) · an **internal
allocator** recycles — invisible to malloc-granularity quarantine.

Only the first is visible to quarantine-and-sweep. Everything else is a structural miss.

## Blindness map — the novelty argument, read by column

| Class | ASan/MTE | GC | Rust | CHERI spatial | CHERI async *(deployed)* | CHERI eager | **Capstone** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 UAF, free-ended | partial¹ | ✗ across FFI | ✗ at FFI | ✗ | ✗ *at contract point* | ✓ late | **✓ sync** |
| 1 UAF, GC/allocator-ended | ✗ | ✗ | ✗ | ✗ | **✗** | **✗** | **✓** |
| 2 Use-after-realloc | partial | ✗ | ✗ | ✗ | **✓** | ✓ | ✓ |
| 3 **Reuse-not-free** | ✗ | ✗ | ✗ | ✗ | **✗** | **✗** | **✓** |
| 4 **Hierarchical** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (H)** |
| 5 Double-free | allocator | ✗ | ✓ in-lang | ✗ | allocator | allocator | **✓ (L)** |
| 6 **TOCTOU / double-fetch** | ✗ | ✗ | ✓ in-lang | ✗ | ✗ | ✗ | **✓ (L)** |
| 7 Callback re-entrancy | ✗ | ✗ | partial | ✗ | ✗ | ✗ | **✓ (L+R+S)** |
| 8 Confused deputy | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (tighten)** |
| 9 Over-wide / sub-object | partial | ✗ | ✓ | partial² | partial² | partial² | partial² |
| 10 Uninitialised | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | **✓ (U)** |
| 11 Provenance forgery | ✗ | ✗ | ✓ | ✓ tag | ✓ tag | ✓ tag | ✓ tag |

¹ intra-domain only  ² allocation-granular, not sub-object

Rust covers a lot — inside one language, with escape hatches exactly at the FFI seam. CHERI
covers extent and identity, and its deployed configuration covers nothing temporal at the
contract point. Capstone is the only column spanning duration, exclusivity, permissions and
validity *across* a boundary.

## What we CAN and CANNOT claim

**CAN — performance, and it is the primary axis.** *"To match our temporal security CHERI must
run `eager`, which costs ~14 M instructions per `free` (microbench) or ~16.8 M (BST), against
our +5"* is measured, QEMU-to-QEMU, on a shared workload. Four repairs:

1. **Drop "O(1)", keep the number.** The RTL never prunes; measured `borrow(N) ≈ 75 + 3N/2`.
   Even at 10,000 live nodes that is ~15 k cycles against 14 M instructions — still ~1000×.
   The *quantitative* claim survives easily; only the *complexity* claim dies.
2. **Frame `eager` as "the configuration that matches our security"**, never as "CHERI's cost".
3. **Re-measure on the current bitstream** — every silicon number predates the 2026-08-04 reflash.
4. **Re-run the BST arm with a real allocator.** Whole-workload Capstone is ~105× (−O2) against
   CHERI async's 1.91×, entirely from the Phase-0 O(n) allocator. `tab:perftree` prints only the
   `+5` mechanism row and omits that denominator, while `tab:perfcompare` prints its 9.3×.

**CANNOT:** *"CHERI does not solve temporal safety"* (Cornucopia Reloaded ships UAR safety;
PoisonCap — preprint, ISA-extension proposal, unimplemented — targets at-access UAF at ~0.1%);
*"O(1) revocation"* on silicon; any end-to-end application-level silicon number;
silicon-enforced spatial safety for plain integer loads/stores (the LSU check is inert in our
domains); linear exclusivity while R-21/R-22 stand.

## Implications for the draft

* **§3** — organise by loan dimension, cross-cut by lifetime-ender.
* **§7** — lead with the security column above; class 1 split, not assigned wholesale to cost.
* **§5** — revocation is **not** a derivation tree. Spec says aliasing + creation order; both
  implementations say depth-tagged DFS-linearised list.
* **Threat model** — rewrite for boundary type (b) below.
* **Build one TOCTOU case.** It is the only item that changes what the paper *is*.

---

## Appendix — terminology: "cross-domain" is a context, not a bug class

A cross-domain UAF is a UAF. Treating "CDP" as a distinct class was tried and produced no CVEs,
no privilege asymmetry, and an exploit not worth leaning on; an audit found only 4 of 15 xlang
rows qualified. That is what happens when you search for a class that is not one.

Cross-domain is the **condition under which existing mechanisms fail**, and it does three jobs:

| Job | Without it | With it |
|---|---|---|
| Novelty | We are a faster Cornucopia, on ASan/MTE/MarkUs/PoisonCap's turf | We address where all of them are structurally blind — their metadata is domain-local |
| Cost | 22.8× per borrow is indefensible; SPEC2006 ends it | Crossings are rare — 1 per 21,423 instructions in SQLite — so the same mechanism is ~1% |
| Mechanism shape | "Lender-controlled duration" is meaningless in one program | Two parties, so the loan contract is the natural abstraction |

**Never write "cross-domain bugs" as a category.** Write "memory-safety bugs at a sharing
boundary". The paper title already has the grammar right: cross-domain modifies *sharing*, not
*bugs*.

**Two boundary types, and we have only one.** (a) an **enforced isolation** boundary — SFI,
wasm, process, hardware domain; what the drafted threat model describes. (b) a
**memory-management** boundary — two components in one address space with different lifetime
regimes (a GC on one side, `malloc` on the other; an engine and its host binding).
**Every bug in all three corpora is (b).** Rewrite the threat model for (b): it needs no
deployment assumption, the surface is enormous, and it is where the GC-root-set argument
actually applies.
