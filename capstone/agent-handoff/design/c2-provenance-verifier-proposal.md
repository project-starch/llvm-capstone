# Proposal: C2 — capability provenance verifier (MIR dataflow checker)

*Status: PROPOSAL for discussion with the reviewer. Not yet approved or implemented.
This is the candidate "lead" contribution (C2) for the first paper; C1
(granularity) has initial slices implemented as the supporting (not yet measured)
systems result.*

> **Audit response (2026-06-29) — revise before implementing; do NOT build verbatim.**
> The 2026-06-29 audit is correct that, as written, this is a **hygiene checker,
> not a provenance proof**: the `UNKNOWN`-accepting, opcode-only lattice proves
> only "no definitely-INT chain reached a checked operand among modeled opcodes,"
> not "every tagged capability derives from a trusted root." Three concrete fixes
> are agreed and must land before/with implementation:
> 1. **Don't accept `UNKNOWN`.** A security checker must minimize **false
>    negatives**, so "sound against false positives" (in the original §"what it
>    is") is the *wrong* notion — drop it. Carry pointer/capability **intent from
>    IR into MIR** (virtual-reg class / operand annotation / MF metadata), seed
>    **arguments and returns from the calling convention** (not `UNKNOWN`), and
>    require an explicit trusted annotation for genuinely-unknown capability
>    consumers.
> 2. **Sharper opcode semantics.** `LDC` only *propagates* the memory tag (not a
>    guaranteed-tagged result); `SHRINK`/`DELIN` have tied in/out operands and must
>    *inherit + validate* their cap input; `CCSRRW` is a source only for reviewed
>    CSRs; `SCC`/`INIT`/`TIGHTEN`/`MREV`/`SEAL` have distinct input roles; an
>    integer used as a memory base is a **tag-faulting invalid access**, not
>    evidence of a forge.
> 3. **Separate two properties** and define transfer functions for COPY/PHI/SELECT/
>    call/return/spill/reload/inline-asm: **non-forging** (no tag from scalar bits)
>    vs **preservation** (a source pointer is never accidentally demoted before a
>    required use). Add a **small formal machine model**; use the corpus run as
>    *validation*, not the proof.
>
> **Revision done (2026-07-01):** the three fixes are now folded into
> "## Design (v2 — the version to implement)" below; the v1 design is retained as
> "## Design (v1, superseded)" for the reasoning trail. Implement v2, not v1. The
> audit also reframes the research position (see Open questions §4).

## Why this, and why now

The granularity work (C1) is implemented and validated — object-granularity
`SHRINK` for globals + heap (default on) and stack (gated) — see
`capability-bounds-model.md` and `../../tests/capstone-authority/`. But object
bounds for globals/heap/stack are **not novel vs CHERI**, which has done all
three since ~2015. Leading the paper with C1 invites the reviewer response "CHERI
already does this."

The defensible, reviewer-requested core is **C2 — provenance**:

> A tagged capability can only be produced by deriving it from an existing
> capability (a root, or a legal constructor applied to a capability); no integer
> computation can become authority.

Today we have only negative *examples* (the authority suite:
`forge_inttoptr`, `ptr_int_ptr_roundtrip` trap at runtime). This proposal turns
that into a **general, checkable invariant over all generated code**, plus an
empirical artifact:

> Across CoreMark + BEEBS-82 + RV8, the compiler emits **zero**
> integer→capability forging paths.

### Why MIR, not LLVM IR
The live threat class (T2 in `capability-provenance-threat-model.md`) is a
*backend lowering* defect — a capability demoted to an integer via `ISD::ADD`, a
value then re-used as a pointer (we hit exactly this earlier: the stack-passed
9th+ argument tag-loss bug). LLVM IR still shows a pointer there; only MIR shows
the demotion. So the verifier must work on machine instructions. We run **pre-RA**
(SSA vregs, single def) so provenance is a simple backward def-walk.

### Why dataflow (not a type check)
There is no separate capability register class — `GPR` holds both i128 caps and
i64 ints (`CapstoneRegisterInfo.td:247`). Cap-vs-int is not statically typed at
MIR, so we classify each value by the opcode that defines it.

### What it is / isn't
A read-only MIR analysis proving two invariants (below) over generated code,
resting on a small formal model with per-opcode transfer functions proven to
preserve them; the corpus run is **validation of the implementation against the
model**, not the proof. It is **not** a mechanized (Coq/Isabelle) soundness proof.
It either passes everywhere (the strong claim) or flags a path that is itself a
result (a forge or a T2 demotion) — the same shape of finding as the rijndael
out-of-bounds write was for C1.

---

## Design (v2 — the version to implement)

*Supersedes the v1 design block retained below (§"Design (v1, superseded)") for
history. v2 folds in the three audit fixes: no permissive `UNKNOWN`; precise
per-opcode semantics; two separate properties + a formal model.*

A read-only `MachineFunctionPass` `CapstoneProvenanceVerifier`, scheduled in
`addPreRegAlloc` (`CapstoneTargetMachine.cpp`), gated by
`-capstone-verify-provenance` (default **off**); reports + counts violations, with
`-capstone-verify-provenance-fatal` to hard-error instead of warn.

### Two separate properties (audit fix #3)
A single lattice cannot express both; conflating them is why v1 was only a
hygiene checker.

- **P1 — Non-forging (authority origin).** Every value *used as authority*
  (memory base, indirect-call target, cap-producer cap-input) must have a
  provenance chain terminating at a **trusted root**, never at an integer-only
  computation. This is the security property. It is a **backward** "does origin
  trace to a root?" question.
- **P2 — Preservation (no accidental demotion).** A value that *is* a capability
  at its def is not silently narrowed to a 64-bit scalar (integer `ADD` on a cap
  operand, a scalar `SD`/`LD` spill instead of `STC`/`LDC`, an i64 truncation) and
  then re-used where a capability is required. This is the T2 backend-defect
  property — exactly the stack-passed-9th-arg tag-loss bug. It is a **forward**
  "is this cap ever demoted before a required use?" question.

### Lattice — no permissive UNKNOWN (audit fix #1)
Replace v1's `{CAP, INT, UNKNOWN}` with a lattice that carries **intent** and
treats genuine unknowns as **must-justify**, not **assume-ok**:

- `ROOT` — a trusted capability root: `gp`/`sp`/`fp` live-in, a
  calling-convention capability argument/return, a **reviewed** cap-CSR read.
- `CAP` — derived from a `ROOT`/`CAP` through a legal cap constructor (provenance
  intact).
- `INT` — an integer/scalar value.
- `TAINTED` (⊥) — capability-ness cannot be justified from a root: a PHI/SELECT
  merging CAP with INT, an unmodeled def, or an incoming value with no
  CC/annotation basis. **`TAINTED` used as authority is a violation** — unless the
  source carries an explicit, reviewed `!capstone.trusted` annotation (an opt-in
  escape hatch, not the default).

The key inversion vs v1: v1's `UNKNOWN` was "assume legitimate, never flag"
(false-negative-prone — the wrong bias for a security checker). v2's `TAINTED` is
"flag unless justified" (false-positive-prone, the correct bias). "Sound against
false positives" is dropped as a goal.

### Carrying capability intent from IR into MIR (audit fix #1)
MIR has no cap register class, so v1 *guessed* class from opcodes. v2 propagates
IR-level intent and uses opcodes to *validate* it:

- Tag IR pointer values (`addrspace(200)`) so isel carries a def flag /
  `!capstone.cap` MI-metadata / vreg attribute; likewise mark scalar defs.
- **Seed arguments and returns from the calling convention**: a param/return in a
  cap-typed ABI slot is `CAP` (root-equivalent intra-procedurally); an integer
  slot is `INT`. No blanket `UNKNOWN` for physreg live-ins.
- Where intent genuinely can't be recovered (inline asm producing a base),
  require an explicit trusted annotation; absent it → `TAINTED` → flagged.

### Per-opcode transfer functions — P1 (audit fix #2)
Def-class as a function of operand classes (from `CapstoneInstrInfo.td`):

- `CIncOffset[Imm]`, `MOVC`: def = class(cap_in); **require** class(cap_in) ∈
  {ROOT,CAP}.
- `SHRINK`, `TIGHTEN`, `DELIN`, `INIT`, `SCC`, `SEAL`, `MREV`: **tied in/out** —
  def *inherits* the cap input's class and **validates** it ∈ {ROOT,CAP}; scalar
  operands (bounds/perms) contribute **no** provenance. (v1 wrongly treated these
  as unconditional CAP producers.)
- `LDC`: def-class = **the propagated memory tag**, i.e. `CAP` *iff the loaded
  slot is known cap-typed* (pointee IR type / a cap-typed store reaches it),
  else the loaded value is data → `INT`. `LDC` is **not** a guaranteed-cap
  producer.
- `CCSRRW`: `ROOT` **only** for the reviewed cap-CSRs (domain entry/return);
  other CSR indices are not a trusted source.
- `CJALR` rd (link cap), domain-entry defs: `ROOT`.
- scalar ALU/loads (`ADD`/`ADDI`/`SUB`/`LUI`/`AUIPC`/shift/logic/`LD`/`LW`/`LB*`/
  `LCC`/`PseudoLLA`): `INT`.
- `COPY` inherits source. `PHI`/`SELECT`: {ROOT,CAP} iff *all* inputs ∈
  {ROOT,CAP}; `INT` iff all INT; **else `TAINTED`** (not silently accepted; PHI
  cycles resolve pessimistically to `TAINTED`).

**Crucial semantic correction (audit fix #2):** an *integer used as a memory
base* is a **tag-faulting invalid access at runtime**, not itself a forge. So P1
flags "authority from integer" only when the base's chain has **no** capability
origin at all. A cap→int demotion re-used as a base is a **P2** finding, not a P1
forge. Keeping these distinct stops the T2 bug from being mislabeled a forge (and
vice-versa).

### Per-opcode transfer functions — P2 (demotion)
Forward-track whether a value *originated* as a capability. Flag when a
cap-origin value passes through a **scalar-narrowing** def (integer `ADD`/`SUB`
on a cap operand, scalar `SD`/`LD` spill/reload of a cap value, i64 truncation)
and the result later reaches a cap-required operand. Transfer functions must
cover `COPY`, `PHI`, `SELECT`, **spill/reload** (a cap value must spill via
`STC`/`LDC`; a scalar spill of a cap value is a P2 violation), **call/return** (a
cap-typed value passed/returned in an integer ABI slot), and **inline asm**
(def/clobber constraints).

### The invariant (what we flag)
- **P1 violation:** an authority operand — memory **base** of
  `LDC`/`STC`/`LD`/`SD`/`LW`/`SW`/`LH`/`SH`/`LB*`/`SB`, the **target** of an
  indirect call (`PseudoCALLIndirect`/`PseudoTAILIndirect` pre-RA; `CJALR` rs1
  later), or the **cap input** of a cap constructor — whose class is `INT` (pure
  integer origin) or `TAINTED` (unjustified).
- **P2 violation:** a cap-origin value reaching a cap-required operand along a
  path containing a demoting def.

Each emits a diagnostic (function, instruction, operand, and the classifying def
chain) and increments a per-property counter.

### Small formal model (audit fix #3)
Abstract machine: values `v` with abstract class `α(v) ∈ {ROOT,CAP,INT,TAINTED}`
and concrete tag `tag(v) ∈ {0,1}`. The transfer functions must satisfy:

- **P1 soundness (under-approximation of "rooted"):** if `α(v) ∈ {ROOT,CAP}` then
  in every concrete execution `tag(v)=1` and `v`'s bounds derive, by the
  constructor rules, from a root. Contrapositive (the useful direction): any
  authority use whose value is a forged/demoted integer has `α ∈ {INT,TAINTED}`
  and is therefore flagged. (So the checker admits false positives but **no**
  false negatives for the modeled opcodes.)
- **P2 soundness:** if a cap-origin value reaches a cap-required use, the forward
  analysis reports it unless every def on the path preserves the tag (STC/LDC
  spill, cap move).

Discharge by case analysis over the ~20-opcode table (a paper-grade hand proof
for v1; mechanizable later). The corpus run then *validates the implementation*
against the model, and is not itself the proof.

### Result framing
"Verifier" is now defensible: no false negatives on modeled opcodes
(`TAINTED`-as-authority is flagged), real carried intent (not opcode guessing),
two separated properties, and a stated model with proven transfer functions. The
corpus claim: across CoreMark + BEEBS-82 + RV8, **every** authority use is
`ROOT`/`CAP` (P1) and **no** cap-origin value is demoted before a required use
(P2) — zero `TAINTED`/`INT`-as-authority, zero P2 demotions — or any hit is a
documented backend finding.

## Design (v1, superseded — kept for history)

*The original opcode-only, `UNKNOWN`-accepting design. Do not implement; see the
audit banner and the v2 design above for why. Retained so the reasoning trail is
visible.*

- **CAP producers** (def is a capability): `CIncOffset`, `CIncOffsetImm`, `MOVC`,
  `DELIN`, `SHRINK`, `SCC`, `INIT`, `TIGHTEN`, `MREV`, `SEAL`, `LDC` (tagged
  load), `CCSRRW`, `CJALR` rd.
- **INT producers**: scalar ALU + scalar loads + `LCC` + `PseudoLLA(Imm)` + any
  other GPR-defining op not in the CAP set.
- **COPY** inherits source; **PHI** = CAP iff all CAP, INT iff all INT, else
  `UNKNOWN`. Roots: `COPY` from `X3`/`X2`/`X8` ⇒ CAP; other live-ins ⇒ `UNKNOWN`
  (never flagged — the false-negative hole v2 closes).
- Invariant: for each cap-required operand assert `classOf ≠ INT`; `UNKNOWN`
  passed silently.

### Files
- New: `llvm/lib/Target/Capstone/CapstoneProvenanceVerifier.cpp`.
- `Capstone.h` (declare create fn + init), `CapstoneTargetMachine.cpp`
  (`addPreRegAlloc`, gated), `CMakeLists.txt`. Scaffolding modeled on
  `CapstoneCapGlobalInit.cpp`.

## Verification
- **Unit lit** `cap-provenance-verify.ll`, split by property:
  - *P1:* clean capability code ⇒ 0 violations; an `inttoptr`-of-integer used as
    a load base (no cap origin) ⇒ 1 **P1** violation; a `TAINTED` PHI (CAP∪INT
    merge) used as a base ⇒ 1 **P1** violation (proving `TAINTED` is flagged, not
    passed as v1 did); a value with a `!capstone.trusted` annotation ⇒ 0.
  - *P2:* a capability spilled with scalar `SD`/reloaded with `LD` then
    dereferenced ⇒ 1 **P2** violation; the correct `STC`/`LDC` spill ⇒ 0; the
    stack-passed-9th-arg tag-loss shape ⇒ 1 **P2** violation.
- **Negative controls:** authority `forge_inttoptr` / `ptr_int_ptr_roundtrip`
  flagged (P1); the other 18 domains report 0 on both properties.
- **Corpus artifact (headline):** build CoreMark + BEEBS-82 + RV8 with
  `-mllvm -capstone-verify-provenance`; collect per-function P1/P2 counts.
  Expected: 0 on both (or any hit is a documented backend finding). Read-only, so
  all benchmarks must still pass.

## Scope / non-goals
- Not a mechanized proof — a MIR analysis backed by a hand-proved model + an
  empirical corpus. Bias is **false-positive-tolerant, false-negative-free** for
  the modeled opcodes (the security-appropriate bias; the v1 "false-positive-
  sound" goal is dropped).
- Capability arguments/returns are seeded from the calling convention (`CAP`), so
  intra-procedural provenance *is* proven; genuine cross-module unknowns need an
  explicit `!capstone.trusted` annotation. Inline asm and cap-effecting
  intrinsics without recoverable intent are `TAINTED` (flagged) unless annotated.
- Out of scope: stack-shrink default-on, BEEBS dtoa/trio heap narrowing.

## Risks
- **Pre-RA pseudo calls**: indirect calls are `PseudoCALLIndirect`/
  `PseudoTAILIndirect` pre-RA (not `CJALR`); handle those operand forms.
- **Cap-argument seeding**: handled by UNKNOWN (no false positives), at the cost
  of not proving provenance through args — stated explicitly.
- **Opcode-table drift**: a new cap instruction missing from the CAP set →
  misclassified INT → false positive; the unit lit test + corpus run surface this.

## Open questions for the review discussion
1. *(v2 addresses this)* The design now includes a small formal model + hand-proved
   transfer functions (§"Small formal model"); the corpus is validation. Open:
   is a **hand** proof enough for the lead contribution, or does the reviewer want
   it mechanized (Coq/Isabelle)?
2. *(v2 partially addresses this)* Arguments/returns are now seeded from the
   calling convention, so intra-procedural provenance is proven and cross-module
   unknowns must be `!capstone.trusted`-annotated. Open: do we want full
   inter-procedural provenance (propagate `CAP`/`INT` across call boundaries via a
   summary), or is CC-seeding + annotation acceptable for v1?
3. Relationship to C1 in the writeup: provenance ("where authority came from") +
   granularity ("how much it carries") as two halves of one story, or two
   separate contributions?
4. **(Audit) Research position.** Object bounds re-derive CHERI; Capstone's
   distinctive mechanisms are **linearity / revocation / `SPLIT` / root
   elimination**, and our path currently *delinearizes* `gp`/`sp` and re-derives
   CHERI-style bounds. The audit's stronger framing is **"provenance +
   attenuation + root-elimination (ordinary code cannot bypass the broad
   root)."** Do we pivot the paper to that (trusted `SPLIT` partitioning removes
   the ambient root from application state; verified attenuation; capability-spill
   isolation), with provenance+granularity as components? This is a
   research-direction decision for the reviewer, not a doc edit.
