# Audit: capability granularity and provenance update

Date: 2026-06-29

Audit range:

- discussion and design:
  - `design/granularity-provenance-discussion.md`
  - `design/c2-provenance-verifier-proposal.md`
  - `design/capability-bounds-model.md`
  - `design/capability-provenance-threat-model.md`
- implementation commits:
  - `01e45b0ffbb9`
  - `3d38a814bf73`
  - `7c1efedabe85`
  - `d61d8aff89d9`
  - `a2d6c2ce4c4`
  - `3e0e11ddf943`
  - `b22eaf7da0bb`
  - `d7ffc99bc5b3`
  - `3d691c7291e6`
  - `aaa3d6c66fc8`
  - `7718948cc206`
  - `3cd5d1c41752`

This is a review, not an implementation commit. It distinguishes current
behavior, engineering evidence, and claims that would require a proof or a
stronger implementation.

## Executive verdict

The update made useful progress:

- it identified the real pre-existing broad-bounds problem;
- it added object-bound narrowing for common global materializations;
- it narrowed allocations in two benchmark-local allocators;
- it built a reproducible runtime authority suite;
- it conservatively kept stack narrowing behind a flag;
- it found a genuine RV64 out-of-bounds write in BEEBS rijndael.

Those are good engineering results. They are not yet a proof of spatial safety
or maximum capability granularity.

The current accurate claim is:

> The SelectionDAG path narrows common sized data-global materializations by
> default, two benchmark allocators return narrowed capabilities, and an
> experimental flag narrows some fixed stack-frame objects. Focused runtime
> probes show that these cases catch selected cross-object accesses.

The following stronger claim is not currently supported:

> Every compiler-generated pointer is bounded to its C object, all accesses are
> spatially safe, and all pointer provenance is verified end to end.

Three findings need attention before the design documents can be used as paper
ground truth:

1. QEMU does not actually model compressed bounds as described in the bounds
   document. It preserves full fat bounds in side metadata.
2. Negative pointer differences are miscompiled at `-O0`; the positive-only
   authority test masks the defect.
3. The proposed MIR verifier is a useful hygiene checker, but its `UNKNOWN`
   policy and opcode-only model cannot establish the stated provenance theorem.

## Findings

### High: the QEMU model does not exercise 128-bit compressed-bound representability

Commit `01e45b0ffbb9` describes Capstone as precise in registers and compressed
to 128 bits in memory, with exact bounds below 4 KiB and outward rounding above
that threshold. The compression code does contain such an encoding:

- `capstone-qemu/target/riscv/cap_compress.c:39-118`

However, tagged stores also put the complete `CapBoundsFat` into `cm_map`:

- `capstone-qemu/target/riscv/capstone_helper.c:47-55`
- `capstone-qemu/target/riscv/capstone_helper.c:74-87`

Tagged loads decompress the 128-bit payload and then overwrite the decompressed
bounds with the full bounds from `cm_map`:

- `capstone-qemu/target/riscv/capstone_helper.c:57-68`

Therefore, observable QEMU behavior is exact fat-bound preservation, not
compressed-bound rounding. A focused runtime probe narrowed an unaligned
5,000-byte range, stored and reloaded it with `stc`/`ldc`, and observed the
exact requested base and end (`retval = 1`).

This has two consequences:

1. Current QEMU experiments cannot substantiate claims about representability
   loss, required allocation alignment, or authority rounding in a real
   128-bit-only implementation.
2. If `cap_compress.c` were used without the fat-bounds side table, it rounds
   base down and end up. That can produce a more powerful capability than exact
   `SHRINK`, conflicting with the current specification condition that a
   compressed implementation must preserve values and must not make an
   operation more powerful:
   `capstone-spec/parts/prog-model.adoc:149-165`.

There is also undefined C behavior in `cap_uncompress`: expressions such as
`1 << (E + 14)` use an `int` left operand for shifts that can exceed 31
(`cap_compress.c:102-103`). This needs independent unit coverage.

Required decision:

- either define the side table as part of the architectural representation and
  stop claiming a self-contained 128-bit compressed capability;
- or make the 128-bit encoding authoritative, define representable `SHRINK`
  semantics, align/pad objects and allocations accordingly, and test
  encode/decode without hidden fat-bound restoration.

Until then, use “exact bounds in the current QEMU model” in results. Do not use
the current <4 KiB / rounded-above rule as measured experimental evidence.

### High: C1 is a partial coverage implementation, not a spatial-safety theorem

Commit `3d38a814bf73` adds a sensible first slice: after `LGA` derives a sized
data-global capability from `gp`, it emits `SHRINK` to the LLVM allocation
size. The implementation is clear and the default-off comparison flag is
useful.

Its coverage is deliberately limited:

- only `GlobalVariable` values with offset zero and a sized type are narrowed;
- functions, aliases, constant pools, block addresses, TLS, non-zero
  `GlobalAddress` offsets, and unsized objects are not covered;
- function capabilities remain broad and unsealed;
- narrowing changes bounds only, not permissions;
- subobjects are not assigned independent bounds.

Commit `a2d6c2ce4c4` is even narrower by design:

- stack narrowing is default off;
- only non-spill, fixed-size, non-variable `FrameIndex` objects with `FI >= 0`
  are eligible;
- interior `ADD(FrameIndex, offset)` shapes, dynamic `alloca`, varargs areas,
  and spills are not covered.

The “heap” result is not a compiler-wide heap policy:

- `7c1efedabe85` modifies only `rv8_malloc.c`;
- `d7ffc99bc5b3` patches only BEEBS dtoa's `malloc_beebs`;
- trio remains intentionally unbounded because its `realloc_beebs` reads
  `size` bytes from an old allocation of unknown size;
- CoreMark uses `MEM_STACK`; its `portable_malloc` returns null.

Calling heap narrowing “default on” is therefore misleading. There is no
general domain libc allocator whose contract is enforced for all programs.

The current implementation also retains ambient broad roots. `start.S`
delinerizes both `sp` and `gp`, and `selectLGA` derives every global from `gp`.
Compiler-inserted `SHRINK` protects code that follows the intended lowering,
but it does not remove broad authority from the domain. Inline assembly,
uncovered lowering, or attacker-controlled machine code can still use the
ambient root.

For a defensible C1 theorem, define and verify a coverage map over at least:

- every static-storage object materialization;
- every fixed and dynamic automatic-storage object;
- every allocation API recognized as creating a fresh object;
- ABI-created objects such as vararg save areas and by-value copies;
- all escapes, PHIs, selects, calls, and returns of those capabilities.

The theorem must state explicit exclusions for subobjects, unions,
`container_of`, flexible-array members, custom allocators, and inline assembly.

### High: negative pointer differences are miscompiled

The design documents state that pointer difference is correctly implemented by
cursor extraction, subtraction, and scaling. The existing runtime test checks
only:

```c
long d = b - a; /* +7 */
```

The codegen regression also explicitly expects `srli` for an LLVM
`sdiv exact` pointer difference:

- `llvm/test/CodeGen/Capstone/i128-xlen-lowering.ll:136-153`

A focused `-O0` probe for:

```c
long difference = low - high; /* expected -7 */
```

emits:

```asm
lcc   ...
lcc   ...
sub   a0, a0, a2
srli  a2, a0, 2
```

and returns failure (`retval = 0`) with the current loader. Logical right shift
does not implement signed exact division for a negative difference. The
positive-only test in
`capstone/tests/capstone-authority/domains/pointer_diff.c` cannot detect this.

Smallest fix:

- preserve signedness when lowering `sdiv exact` by a power of two;
- use `srai` for the signed case and `srli` only for logical/unsigned shifts;
- add positive and negative C runtime cases and signed/unsigned lit cases.

This bug predates the C1 commits, but the new discussion makes the incorrect
pointer-difference claim part of the audited context.

### High: bounds are narrowed, but permissions and control-flow authority remain broad

The domain linker script emits one `PT_LOAD` with `FLAGS(0x7)`:

- `capstone/my_first_domain/link.ld:3-5`

The QEMU domain genesis capabilities are `CAP_PERMS_RWX`:

- `capstone-qemu/target/riscv/op_helper.c:1220-1235`

`SHRINK` changes bounds, not permissions. Consequently:

- data, stack, and heap capabilities retain execute authority;
- code capabilities retain write authority;
- function capabilities are not bounded or sealed;
- the image does not provide W^X separation.

This is not a direct refutation of a narrowly scoped data-spatial-safety
experiment, but it materially weakens a general security claim. A future system
should use separate RX code and RW data roots/segments, permission tightening,
and sealed or otherwise constrained call targets.

The more fundamental research direction is to remove the omnipotent `gp` root
from ordinary application code. The trusted loader can use Capstone `SPLIT` to
partition an image and provide only attenuated object/table capabilities. A
compiler that merely re-derives narrow pointers from a permanently available
broad root demonstrates good lowering discipline, but not least authority
against arbitrary in-domain code.

### Medium: the C2 proposal is a hygiene checker, not a provenance proof

Commit `3e0e11ddf943` correctly chooses a machine-code layer to detect backend
demotion bugs. That is the proposal's strongest decision: LLVM IR alone cannot
show whether a pointer operation became scalar `ADD`/`LD`/`SD`.

The current proposed lattice is too permissive for the stated theorem:

- all non-special physical live-ins become `UNKNOWN`;
- `UNKNOWN` is never rejected;
- PHI cycles become `UNKNOWN`;
- inline assembly and cap-effecting intrinsics are conservative escapes;
- incoming arguments and returns are explicitly not proved.

Passing such a checker means only:

> No definitely-INT def-chain reached a checked capability operand among the
> modeled opcodes.

It does not mean:

> Every tagged capability in the program is proven to derive from a trusted
> root.

The opcode taxonomy also needs more semantic precision:

- `LDC` does not guarantee a tagged result; it propagates the memory tag;
- `SHRINK` and `DELIN` have tied input/output operands and must inherit and
  validate their capability input;
- `CCSRRW` is a capability source only for reviewed capability CSRs/contexts;
- `SCC`, `INIT`, `TIGHTEN`, `MREV`, and `SEAL` have different input roles and
  preconditions;
- an integer register used as a memory base is a tag-faulting invalid access,
  not evidence that the compiler forged a tagged capability.

The proposal also uses “sound against false positives,” which is not the
relevant security notion. A proof-oriented checker must minimize false
negatives; treating unknown values as acceptable does the opposite.

Recommended redesign:

1. Preserve pointer/capability intent from IR into MIR with an explicit virtual
   register class, operand annotation, or machine-function metadata even if the
   physical register file is unified.
2. Seed function arguments and returns from the calling-convention assignment,
   not as `UNKNOWN`.
3. Define transfer functions for every capability instruction, COPY, PHI,
   SELECT, call, return, spill, reload, and inline-assembly boundary.
4. Reject or require an explicit trusted annotation for unknown capability
   consumers.
5. Separate two properties:
   - non-forging: no instruction can set a tag from scalar bits;
   - preservation: source-level pointer values are never accidentally demoted
     before a required use.
6. Add a small formal machine model and prove the transfer rules preserve the
   authority invariant. Use the corpus run as validation, not as the proof.

Implemented in this form, a MIR checker would be valuable. In its current
proposed form, it should be described as an experimental diagnostic.

### Medium: `uintptr_t` policy is internally inconsistent

The current target reports:

```text
__SIZEOF_POINTER__ = 16
__UINTPTR_TYPE__   = unsigned long
__UINTPTR_WIDTH__  = 64
```

`Capstone64TargetInfo` sets `PointerWidth = 128` but does not define a
capability-preserving `IntPtrType`. The design then deliberately treats
`ptr -> uintptr_t -> ptr` as authority-destroying.

That can be a deliberate research-language policy, but it should not silently
reuse the standard `uintptr_t` name and expected round-trip role. CHERI C uses a
provenance-carrying `uintptr_t` representation and a separate address-only
`ptraddr_t` concept.

Choose one explicit ABI:

- capability-preserving `uintptr_t` plus a 64-bit cursor/address type; or
- no standard round-trip `uintptr_t`, with a documented non-conforming
  address-only extension and diagnostics for pointer round trips.

The current accidental middle ground will break real code and weakens any
formal source-language claim.

### Medium: the spilled-capability question remains open

The PI's concern is valid. `stc`/`ldc` deliberately preserves a real tagged
capability in a spill slot. Anyone who can address that slot and execute an
`ldc` can acquire the authority.

The current implementation gives:

- likely inter-domain isolation, subject to a separate loader/root audit;
- no general intra-domain spill confidentiality;
- stack-object narrowing only under an opt-in flag;
- broad, non-linear `sp` and `gp` roots at domain entry.

A MAC does not solve this problem by itself. A MAC provides integrity and
authenticity, not confidentiality; an attacker who can read a valid
capability-plus-MAC can replay it. The out-of-band hardware tag already provides
strong in-memory non-forgeability.

Relevant defenses depend on the attacker model:

- for memory corruption in compiled C: complete object bounds can prevent a
  buffer capability from reaching unrelated spill slots;
- for arbitrary in-domain code: use a dedicated protected capability-spill
  region/root that ordinary code cannot address;
- for serialization to untagged storage: use opaque handles or
  context-bound cryptographic sealing, not a raw pointer plus a generic
  checksum;
- for unique authority: retain linear capabilities in a trusted subsystem
  rather than immediately applying `DELIN`.

The current ordinary C path delinerizes `sp` and `gp`, so linearity is not yet a
deployed answer to spill theft.

### Medium: the evidence suite is useful but not proof-grade yet

Commit `01e45b0ffbb9` created a valuable, reproducible source/assembly/runtime
suite. Commit `a2d6c2ce4c4` expanded it to 12 cases and added one retry for boot
flakes. These are good decisions.

Limitations:

- out-of-bounds C probes execute undefined behavior and are intentionally
  compiled at `-O0`;
- the suite does not check the same property at `-O1`, `-O2`, and `-O3`;
- positive pointer difference hid the signed bug;
- faulting domains terminate QEMU through an assertion rather than an
  architectural guest exception;
- the suite checks selected examples, not all materialization paths;
- benchmark “green” results establish compatibility, not overhead.

For the paper artifact:

- add LLVM IR or assembly-level negative tests where C UB would weaken the
  interpretation;
- run an optimization-level matrix;
- make capability faults architectural exceptions instead of QEMU-process
  assertions;
- report instruction count, code size, dynamic `SHRINK` count, cycles, and
  memory overhead separately from pass/fail correctness;
- add explicit before/after variants rather than relying only on hidden flags.

The existing statement “overhead green” should be removed. No performance
overhead measurement was found in these commits.

### Low: benchmark allocator changes are appropriate prototypes, not production malloc

`7c1efedabe85` made the right local design choice for `realloc`: because the
returned capability cannot reach its header at `p - 16`, the allocator recovers
metadata through its retained wide arena capability and copies
`min(old, new)` bytes.

Remaining limitations:

- `n + 15`, `payload + header`, and `rv8_off + need` are not checked for
  integer overflow;
- `free` is a no-op, so there is no temporal safety or reuse;
- invalid `realloc` pointers are not validated;
- the allocator-wide root remains powerful;
- large-allocation representability is not modeled by QEMU as documented.

`d7ffc99bc5b3` narrows dtoa to the 16-byte-rounded size rather than the original
request. This grants up to 15 bytes of extra authority. That is reasonable for
an allocator-granule experiment, but it is not maximum byte granularity.

Trio's allocator is a more important open item: its `realloc_beebs` copies the
new size from the old pointer without knowing the old size. Keeping it
unbounded avoids a fault but preserves a known over-read. Add size metadata and
copy `min(old, new)` before claiming suite-wide heap narrowing.

### Low: canonical documentation is duplicated and stale

`design/thoughts-from-gpt.md` is a pre-implementation planning note that still
says automatic narrowing does not exist. It duplicates the canonical
discussion and should be archived or replaced by a short historical pointer.

`ref/capstone-purecap-pointer-model.md:46-49` also says the compiler does not
generally synthesize tight bounds. That is now incomplete: common globals are
narrowed by default, selected heaps are narrowed locally, and stack narrowing
is opt-in.

The design documents added “now” annotations rather than rewriting every old
claim. This preserves history but makes it easy to quote superseded text. The
paper-facing document should contain only current claims, with historical
behavior moved to an explicit before/after section.

## Commit-by-commit assessment

### `01e45b0ffbb9` — bounds note and authority suite

Decision quality: good experimental direction. The source/assembly/runtime
triangulation is the right method.

Issues: the compressed-bounds model does not match observable QEMU behavior;
the initial suite was too small for a general provenance claim; C UB and
positive-only ptrdiff need stronger controls.

### `3d38a814bf73` — global `SHRINK`

Decision quality: good minimal C1 slice. Late SelectionDAG insertion is easy to
test and immediately exposed a real bug.

Issues: it is a materialization-pattern patch, not a complete object-provenance
system. Default-on is acceptable for the benchmark branch, but the paper must
state the uncovered paths. No permissions are tightened.

The rijndael `r[4] -> r[8]` adaptation is correct for LP64: the source performs
an 8-byte `unsigned long` store and only consumes the first four bytes.

### `7c1efedabe85` — RV8 heap narrowing

Decision quality: correct prototype allocator design, especially recovering
metadata via the retained arena root.

Issues: benchmark-specific, no overflow hardening, no free/revocation, and not
evidence of a general `malloc` implementation.

### `d61d8aff89d9` — global narrowing lit test

Decision quality: necessary and well scoped. It checks default-on/default-off,
sized data, function, and unsized external cases.

Issues: add aliases, non-zero offsets, weak/unresolved symbols, large objects,
permissions, and multiple optimization/code models.

### `a2d6c2ce4c4` — gated stack spike

Decision quality: conservative and technically honest. Keeping it default off
was the correct choice.

Issues: its limited selector shape must not be generalized to “stack safety.”
The runtime retry and build flag support are useful but make the commit broader
than the compiler change.

### `3e0e11ddf943` — MIR provenance verifier proposal

Decision quality: correct compiler layer, insufficient invariant. Pre-RA MIR is
a useful observation, but opcode-only classification plus accepted `UNKNOWN`
values cannot prove provenance.

Recommendation: revise before implementation; do not implement the proposal
verbatim.

### `b22eaf7da0bb` — state refresh

Decision quality: useful pivot from benchmark bring-up to security work.

Issue: “C1 implemented” is too broad. Use “initial C1 slices implemented.”

### `d7ffc99bc5b3` — dtoa heap narrowing

Decision quality: safe benchmark-local extension and honest documentation of
the trio blocker.

Issue: rounded-size authority is not byte-minimal, and leaving trio unbounded
prevents suite-wide heap claims.

### `3d691c7291e6` — documentation reconciliation

Decision quality: necessary correction of stale pre-narrowing claims.

Issues: it retained unsupported compressed-bound behavior and still moves too
quickly from selected examples to spatial-safety language.

### `aaa3d6c66fc8`, `7718948cc206`, `3cd5d1c41752`

Documentation-only neutralization/rename commits. No technical concerns.

## Answers to the PI's questions

### Can an attacker steal a spilled capability?

Yes, if the attacker can address the spill slot and execute a tag-preserving
load. Tags prevent fabrication; they do not make a genuine capability secret.
Current default stack behavior does not provide a general intra-domain
confidentiality guarantee.

### Should integers and capabilities be disjoint, and should a MAC be used?

They are architecturally disjoint through an out-of-band validity tag. This is
the right primitive for in-memory non-forgeability. A MAC is relevant for
serialization or cryptographic sealing, but does not prevent reading or replay
of a valid spilled capability.

### What should `ptr -> int -> ptr` do?

The project must choose a source-language contract. Arbitrary integer values
must not create authority. A provenance-carrying `uintptr_t` may preserve an
existing capability through controlled operations; an address-only integer
must require combination with an existing capability. The current 64-bit
`uintptr_t` silently loses authority and should not remain an accidental ABI.

### How should pointer difference work?

Extract cursors, subtract, and perform signed scaling for `ptrdiff_t`. The
current positive case works; the negative case is miscompiled by logical shift.
Subtraction across different C objects remains undefined at the language level.

### How does malloc work now?

There is no general domain libc heap. RV8 and dtoa use static bump allocators;
trio has a separate unsafe bump/realloc implementation; CoreMark uses stack
storage. A proper allocator needs overflow checks, representable
alignment/padding, bounded returns, protected metadata, validated `free`, and a
temporal-safety/revocation policy.

### How are capabilities created and bounded?

Current compiled code derives from runtime roots (`gp`, `sp`, imported shared
caps), loads tagged caps with `ldc`, and applies monotonic capability
operations. Common globals and selected heap/stack objects are then narrowed
with `SHRINK`. The broad roots remain available.

### Do we perform high-quality splitting?

Not yet in the ordinary C compiler path. The ISA/runtime uses `SPLIT` in some
trusted components, but LLVM object materialization uses repeated derivation
from broad roots followed by `SHRINK`. True least-authority partitioning would
split in trusted code and remove the broad root from untrusted application
state.

### What happens for `p + 5` and `p + 25`?

Both expressions compile. Pointer arithmetic derives from the original
capability. Dereferencing the out-of-range result traps only when the source
capability is correctly bounded and the access remains present after C
optimization. This is true for the covered global case and opt-in covered stack
case, not yet universally.

## Research-position audit

Object-level bounds alone are not a strong novelty claim. CHERI has long
implemented bounded global, stack, and heap pointers, and CheriSH explored
subobject bounds and compatibility:

- https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-949.pdf
- https://www.cl.cam.ac.uk/~pffm2/sosp2023_cheri_tutorial/exercises/4_subobject-bounds/
- https://www.cl.cam.ac.uk/~pffm2/sosp2023_cheri_tutorial/exercises/7_cheri-allocator/

Formal CHERI C semantics already covers capabilities, undefined behavior, and
provenance:

- https://doi.org/10.1145/3617232.3624859

Compressed-capability correctness and representability also have substantial
prior work:

- https://doi.org/10.1109/TC.2019.2914037

Capstone's own architectural novelty is linearity, revocation, splitting, and
trustless memory delegation:

- https://arxiv.org/abs/2302.13863

The current compiler path largely de-linearizes ordinary roots and then
recreates CHERI-like object bounds. That is valuable infrastructure but does
not yet exploit Capstone's distinctive mechanisms.

More defensible paper directions are:

1. **Verified authority attenuation for a Capstone C compiler.**
   Give every compiler-generated capability an object identity and prove that
   lowering preserves provenance and never increases authority.
2. **Root elimination through trusted partitioning.**
   Use `SPLIT` to partition image, stack, and heap authority, then remove broad
   roots from application-visible state.
3. **Capability spill isolation.**
   Combine object bounds with a protected capability-spill region and evaluate
   resistance to intra-domain disclosure.
4. **Capstone-specific temporal/ownership integration.**
   Use linear and revocation capabilities in allocator/compiler contracts
   rather than immediately de-linearizing all ordinary authority.
5. **A formal end-to-end refinement result.**
   Prove that source-level object/provenance rules refine to Capstone
   instructions, and validate the implementation with a strict MIR checker and
   corpus measurements.

The best unified framing is:

> Provenance determines where authority came from; attenuation determines how
> much authority remains; root elimination ensures ordinary code cannot bypass
> the attenuation.

That is stronger than the current two-part framing because it addresses the
ambient-root bypass explicitly.

## Recommended order

1. Fix and test negative pointer differences.
2. Correct the bounds-model document to describe the QEMU fat-bounds side table;
   decide the intended real compressed representation.
3. Rewrite the C1 claim as a coverage matrix and remove “overhead measured”
   language until performance is measured.
4. Decide the `uintptr_t` / address-only integer ABI.
5. Replace the C2 proposal with a strict typed-MIR invariant and a small formal
   model; implement only after review.
6. Make stack coverage complete enough for default-on, including varargs,
   dynamic alloca, interior frame addresses, and optimization-level testing.
7. Replace trio's realloc with size-aware metadata and introduce one canonical
   bounded domain allocator.
8. Separate RX code and RW data authority; tighten permissions and constrain
   function capabilities.
9. Design root elimination / trusted `SPLIT` partitioning as the likely
   Capstone-specific research contribution.

## Validation performed during this audit

- `llvm/test/CodeGen/Capstone`: 31/31 passed.
- Focused negative pointer-difference probe:
  - `-O0` assembly contains `sub` + `srli`;
  - current host loader/QEMU returns `retval = 0`, expected `1`.
- Focused 5,000-byte bounds store/reload probe:
  - exact requested base/end survive;
  - current host loader/QEMU returns `retval = 1`;
  - source inspection confirms full bounds are restored from `cm_map`.
- `git diff --check e6f35b2b95e9..HEAD`: clean.

The full benchmark suites were not rerun: this audit did not alter compiler or
runtime behavior, and the relevant commits already record full functional
certification. Functional benchmark success would not resolve the findings
above.
