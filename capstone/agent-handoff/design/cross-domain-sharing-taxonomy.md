# Cross-domain sharing: taxonomy, and what we can claim

*Framing note for the paper. Not an investigation — see `history/` for those. Written
2026-08-13.*

## The one-line version

**The bugs are ordinary. The boundary is what makes them undetectable and the mechanism
affordable.**

## Cross-domain is a CONTEXT, not a bug class

A cross-domain UAF is a UAF. Treating "CDP" as a distinct class was tried and produced no
CVEs, no privilege asymmetry, and an exploit not worth leaning on; an audit of the xlang
corpus found only 4 of 15 rows qualified. That is not bad luck — it is what happens when you
search for a class that is not one.

Cross-domain is the **condition under which existing mechanisms fail**, and it does three
jobs nothing else in the paper can do:

| Job | Without it | With it |
|---|---|---|
| Novelty | We are a faster Cornucopia, competing with ASan/MTE/MarkUs/FFmalloc/PoisonCap on their turf | We address the case where all of them are structurally blind, because their metadata is domain-local |
| Cost | 22.8× per borrow is indefensible; SPEC2006 ends the discussion | Crossings are rare — 1 per 21,423 instructions in SQLite — so the same mechanism is ~1% |
| Mechanism shape | "Lender-controlled duration" is meaningless in one program | Two parties, so the loan contract is the natural abstraction |

**Never write "cross-domain bugs" as a category.** Write "memory-safety bugs at a sharing
boundary". The paper title already has the grammar right: cross-domain modifies *sharing*,
not *bugs*.

### Two boundary types, and we have only one of them

* **(a) enforced isolation boundary** — SFI, wasm, process, hardware domain. What the drafted
  threat model describes.
* **(b) memory-management boundary** — two components in one address space with different
  lifetime regimes (a GC on one side, `malloc` on the other; an engine and its host binding).

**Every bug in all three corpora is (b).** The threat model should be rewritten for (b): it
needs no deployment assumption, the surface is enormous (every C extension, every JNI call,
every language binding), and it is where the GC-root-set argument actually applies.

## Taxonomy — by which dimension of the loan is violated

| # | Class | Dimension | Why existing mechanisms miss it |
|---|---|---|---|
| 1 | Cross-domain UAF / use-after-close | **Duration** | Allocator metadata is domain-local; the borrower cannot learn the object died |
| 2 | Use-after-reallocation | Duration (weaker) | This is what CHERI *does* provide — dead but not yet recycled |
| 3 | **Reuse-not-free** | Duration, no `free()` | No `free()` ⇒ revocation never triggers. Quarantine is structurally blind |
| 4 | **Hierarchical lifetime violation** | Duration, transitively | Mechanisms see the raw `free()`, not the parent→child link `close()` implies |
| 5 | Double-free / return-twice | **Exclusivity** of the handle | Caught by allocator heuristics, not by the safety mechanism |
| 6 | **TOCTOU / double-fetch** | **Exclusivity** | Every access is individually legal; nothing represents "not while I'm reading" |
| 7 | Callback re-entrancy | Exclusivity across control transfer | The lender re-enters while the loan is live |
| 8 | **Confused deputy / authority amplification** | **Permissions** | Nothing narrows authority at the crossing |
| 9 | Over-wide loan / sub-object confusion | **Extent** | Spatial mechanisms bound the allocation, not the shared sub-object |
| 10 | Uninitialised read across the boundary | **Validity** | Type systems catch it in-language, nothing across an ABI |
| 11 | Provenance forgery | **Identity** | Integer↔pointer round-trips re-materialise revoked authority |

### Second axis: WHO ends the lifetime

This is the sharpest cut, because it explains the blindness mechanically —
**quarantine-and-sweep keys on `free()`, so every row where the lifetime ends by other means
is a structural miss:**

lender frees · lender **reuses without freeing** (class 3) · a **collector** reclaims (the
borrower is invisible to it) · a **parent destroy** cascades (class 4) · an **internal
allocator** recycles (invisible to malloc-granularity quarantine).

Class 1 should be **split by lifetime-ender**. Its GC-ended and internal-allocator-ended
sub-cases belong with 3 and 4 — they are structural misses, not timing misses.

## The blindness map — the novelty argument

| Class | ASan/MTE | GC | Rust | CHERI spatial | CHERI async *(deployed)* | CHERI eager | **Capstone** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 UAF / use-after-close | partial¹ | ✗ across FFI | ✗ at FFI | ✗ | ✗ *at contract point* | ✓ late | **✓ sync** |
| 2 Use-after-realloc | partial | ✗ | ✗ | ✗ | **✓** | ✓ | ✓ |
| 3 **Reuse-not-free** | ✗ | ✗ | ✗ | ✗ | **✗** | **✗** | **✓** |
| 4 **Hierarchical** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (H)** |
| 5 Double-free | allocator | ✗ | ✓ in-lang | ✗ | allocator | allocator | **✓ (L)** |
| 6 **TOCTOU / double-fetch** | ✗ | ✗ | ✓ in-lang | ✗ | ✗ | ✗ | **✓ (L)** |
| 7 Callback re-entrancy | ✗ | ✗ | partial | ✗ | ✗ | ✗ | **✓ (L+R+S)** |
| 8 **Confused deputy** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (tighten)** |
| 9 Over-wide / sub-object | partial | ✗ | ✓ | partial² | partial² | partial² | partial² |
| 10 Uninitialised | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | **✓ (U)** |
| 11 Provenance forgery | ✗ | ✗ | ✓ | ✓ tag | ✓ tag | ✓ tag | ✓ tag |

¹ intra-domain only  ² allocation-granular, not sub-object

**Read the columns.** Rust covers a lot — inside one language, with escape hatches exactly at
the FFI seam. CHERI covers extent and identity, and its deployed configuration covers nothing
temporal at the contract point. Capstone is the only column spanning duration, exclusivity,
permissions and validity *across* a boundary.

**Rows 3, 4, 6, 8 are where every other column is ✗** — capability gaps, not cost gaps. On
class 1, CHERI can match with a knob turn (eager), so the argument there is cost.

## Evidence we actually have

| Class | Evidence | Apps |
|---|---|---|
| 1 UAF / use-after-close | **Strong** — 3 corpora, ~40 reproduced rows, real-library confirmation on Lua | SQLite bindings (7 ecosystems), mruby, 10 Lua libraries |
| 3 Reuse-not-free | **1 row** — but the clean "CHERI cannot" | diesel (Rust) |
| 4 Hierarchical | 6 rows; mechanism in spec + QEMU + RTL | PHP, CPython, sqlite3-ruby, expo |
| 5 Double-free | 3 rows | go-sqlite3, luaossl, lua-openssl |
| 6 **TOCTOU / double-fetch** | **NONE** — a figure only | — |
| 7 Callback re-entrancy | 4 rows | CPython, rusqlite, PHP, datasette |
| 8 **Confused deputy** | **NONE** in a corpus | — |
| 9 Over-wide / sub-object | 2 rows, both *intra*-domain (mruby) | — |
| 10 Uninitialised | 1 row | CPython |
| 11 Provenance forgery | threat-model analysis only | — |

Class 1 is the volume; 3/4/6/8 are the argument. **Class 6 is the gap worth closing** — second
largest class, real CVEs at the kernel/user boundary, and it needs **exclusivity (L)**, a
different mechanism from everything currently evaluated.

## What we CAN and CANNOT claim

**CAN — performance is claimable, and it is the primary axis.** "To match our temporal
security CHERI must run eager, which costs ~14 M instructions per `free` (microbench) or
~16.8 M (BST), against our +5" is a legitimate, measured, QEMU-to-QEMU claim on a shared
workload. Four repairs make it hold:

1. **Drop "O(1)".** The RTL never prunes the revocation list; measured `borrow(N) ≈ 75 + 3N/2`.
   Say "orders of magnitude, growing slowly with the live tree". The *quantitative* claim
   survives easily — even at 10,000 live nodes our revoke is ~15 k cycles against 14 M
   instructions — it is the *complexity* claim that does not.
2. **Frame it as "the configuration that matches our security costs X"**, never "CHERI costs X".
   Eager is not the deployed default and a reviewer will say so first if we do not.
3. **Re-measure on the current bitstream.** Every silicon number predates the 2026-08-04
   reflash; the measurements doc says so itself.
4. **Re-run the BST arm with umm_malloc.** Whole-workload Capstone is currently ~105× (−O2)
   against CHERI async's 1.91×, entirely from the Phase-0 O(n) allocator. The paper prints only
   the +5 mechanism row and omits that denominator — the single most attackable gap in the
   perf story.

**CANNOT:**

* *"CHERI does not solve temporal safety"* — false. Cornucopia Reloaded (ASPLOS'24) ships
  use-after-reallocation safety; PoisonCap (preprint, ISA-extension proposal, not implemented)
  targets at-access UAF at ~0.1% overhead. Cite and distinguish on mechanism: they close the
  access window but still need the sweep; we do not sweep.
* *"O(1) revocation"* on silicon — see repair 1.
* Any end-to-end application-level silicon performance number — none exists.
* Silicon-enforced spatial safety for plain integer loads/stores — the LSU check is inert in
  our domains.
* Linear exclusivity as enforced on current silicon — R-21/R-22 are open spec violations, and
  our own domain glue depends on the non-conformance.

**"CHERI can, expensively and late" is exactly what we should claim.** The earlier ranking of
3/4/6/8 above class 1 was about which claims are *most robust*, not which are permitted.

## Concrete implications

* **§3** — organise by loan dimension, cross-cut by lifetime-ender. Cross-domain stated once,
  in the threat model, as the setting.
* **Threat model** — rewrite for boundary type (b).
* **§7** — class 1 carries volume; the structural argument is 3/4/6/8 plus the GC-ended and
  allocator-ended sub-cases of 1.
* **§5** — revocation is not a derivation tree. Spec says aliasing + creation order; both
  implementations say depth-tagged DFS-linearised list.
