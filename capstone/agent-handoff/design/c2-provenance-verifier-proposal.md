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
> Until revised, describe this as an **experimental diagnostic**, not a verifier.
> The audit also reframes the research position (see below).

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
A structural + dataflow **checker** (read-only), sound against false positives
(see "conservatism"). It is **not** a formal/mechanized soundness proof. It
either passes everywhere (the strong empirical claim) or flags a path that is
itself a result (a forge or a T2 demotion) — the same shape of finding as the
rijndael out-of-bounds write was for C1.

## Design

A read-only `MachineFunctionPass` `CapstoneProvenanceVerifier`, scheduled in
`addPreRegAlloc` (`CapstoneTargetMachine.cpp`), gated by
`-capstone-verify-provenance` (default **off**); reports + counts violations, with
`-capstone-verify-provenance-fatal` to hard-error instead of warn.

### Opcode taxonomy (from `CapstoneInstrInfo.td`)
- **CAP producers** (def is a capability): `CIncOffset`, `CIncOffsetImm`, `MOVC`,
  `DELIN`, `SHRINK`, `SCC`, `INIT`, `TIGHTEN`, `MREV`, `SEAL`, `LDC` (tagged
  load), `CCSRRW` (cap CSR / domain entry), `CJALR` rd (return/link cap).
- **INT producers** (def is an integer): `ADDI`/`ADD`/`SUB`/`LUI`/`AUIPC`/shifts/
  logic, scalar loads `LD`/`LW`/`LH`/`LB*`, `LCC` (field query), `PseudoLLA(Imm)`,
  and any other GPR-defining op not in the CAP set.
- **COPY** inherits the source class. **PHI**: CAP iff all inputs CAP; INT iff all
  INT; else UNKNOWN.
- **Roots / seeds**: `COPY` from `X3`(gp), `X2`(sp), `X8`(fp) ⇒ CAP. Other
  physreg live-ins (may be capability arguments) ⇒ **UNKNOWN**.

### Classification
Per function, memoized backward `classOf(vreg) ∈ {CAP, INT, UNKNOWN}` via the def
opcode (recurse through COPY/PHI; PHI cycles resolve to UNKNOWN). UNKNOWN is the
"could legitimately be a capability" escape hatch — never flagged.

### The invariant (what we flag)
For every operand that **requires a capability**, assert `classOf` is not `INT`:
- memory-access **base** of `LDC`/`STC`/`LD`/`SD`/`LW`/`SW`/`LH`/`SH`/`LB*`/`SB`
  (PureCap: every access is through a capability),
- **target** of an indirect call (`PseudoCALLIndirect`/`PseudoTAILIndirect`
  pre-RA; `CJALR` rs1 if run later),
- the **capability input** (rs1 / `$cap_in`) of each CAP producer above.

`classOf == INT` in any of these = **violation**: a value built purely from
integer arithmetic used as authority (an `inttoptr`-style forge that would fault,
or a T2 cap→int demotion re-used as a pointer). Emit a diagnostic (function,
instruction, operand, integer-origin def chain); increment a counter.

### Files
- New: `llvm/lib/Target/Capstone/CapstoneProvenanceVerifier.cpp`.
- `Capstone.h` (declare create fn + init), `CapstoneTargetMachine.cpp`
  (`addPreRegAlloc`, gated), `CMakeLists.txt`. Scaffolding modeled on
  `CapstoneCapGlobalInit.cpp`.

## Verification
- **Unit lit** `cap-provenance-verify.ll`: clean capability code ⇒ 0 violations;
  an `inttoptr`-of-integer used as a load base ⇒ 1 violation (FileCheck the
  diagnostic); a demotion case (cap address forced through an integer then
  dereferenced) ⇒ flagged.
- **Negative controls:** authority `forge_inttoptr` / `ptr_int_ptr_roundtrip`
  flagged; the other 11 domains report 0.
- **Corpus artifact (headline):** build CoreMark + BEEBS-82 + RV8 with
  `-mllvm -capstone-verify-provenance`; collect per-function counts. Expected: 0
  forging paths (or any hit is a documented finding). Read-only, so all
  benchmarks must still pass.

## Scope / non-goals
- Not a formal proof — a false-positive-sound MIR checker + empirical corpus
  result.
- Physreg live-ins are UNKNOWN (capability arguments not flagged); inline asm and
  cap-effecting intrinsics treated conservatively. We do not prove provenance
  *through* incoming arguments.
- Out of scope: stack-shrink default-on, BEEBS dtoa/trio heap narrowing.

## Risks
- **Pre-RA pseudo calls**: indirect calls are `PseudoCALLIndirect`/
  `PseudoTAILIndirect` pre-RA (not `CJALR`); handle those operand forms.
- **Cap-argument seeding**: handled by UNKNOWN (no false positives), at the cost
  of not proving provenance through args — stated explicitly.
- **Opcode-table drift**: a new cap instruction missing from the CAP set →
  misclassified INT → false positive; the unit lit test + corpus run surface this.

## Open questions for the review discussion
1. Is the checker+empirical-corpus framing strong enough as the *lead*
   contribution, or does the paper need a mechanized/proof element (e.g. a
   small formal model of the lowering invariant)?
2. Should provenance be proven *through* capability arguments and returns
   (inter-procedural), or is intra-procedural + UNKNOWN-args acceptable for v1?
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
