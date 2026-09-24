# A capability-carrying `uintptr_t` for Capstone: design and cost

Status: ANALYSIS, 2026-09-24. Decision owner: the project lead (it is an ABI change). Branch:
`compiler/intcap-design`. This answers the open item in
`design/granularity-provenance-discussion.md` §3 and §"next steps" 5 ("Decide the
`uintptr_t`/`__intcap` model").

## Summary

**Recommendation: adopt CHERI C's model** -- `intptr_t`/`uintptr_t` become `__intcap`/`__uintcap_t`,
capability-carrying integers whose arithmetic acts on the address and keeps one operand's
capability, plus an address-only `ptraddr_t` -- with CHERI's exact semantics and one Capstone rule
(an intcap never holds a linear capability). It is the C dialect the capability ecosystem already
uses (CheriBSD, Morello, CHERIoT), so ports and upstream fixes carry over; Capstone's LLVM already
uses CHERI's IR conventions; and no instruction is added.

**But it is not a compiler-only project.** Three things come first, none of them compiler work:

1. **An ISA semantics decision for untagged values in capability registers.** Intcap puts plain
   integers (hashes, sizes, constants) into capability registers everywhere. Today the spec has
   `MOVC` null a source that is not a non-linear capability -- integers included; the RTL does it
   (C-32, live on silicon) and whether the spec means it is open (Q-04) -- and `SCC` raises
   *unexpected operand type* on an untagged source (`cap-man-insn.adoc` 132-137), where CHERI's
   set-address takes any value. Register copies cannot be guarded in software, so this has to
   change in the ISA (and QEMU, and the RTL) before intcap code can run on silicon.
2. **R-33** (a moved cursor widens authority on silicon for non-representable sizes) and **R-29**
   (an 8-byte store into the high half of a granule, then a 16-byte `ldc`, reads zeros), both
   OPEN, both hit harder by 16-byte integers.
3. **The lead's decision** (ABI, and the paper's provenance framing; see
   `design/granularity-provenance-discussion.md` §3).

**Cost: roughly 10-17.5 engineer-weeks** of compiler, runtime and port work, plus the ISA/RTL change
(1-3 weeks and 1-3 synthesis cycles) -- 11-20 weeks in all, with medium confidence on the compiler
front end (measured against CHERI's implementation) and low-medium on the back end and the ports.
Everything can be built and validated under QEMU before the silicon prerequisites land.

**Payoff, CPython:** of the 232 changed code lines in its 13 patches (counted with comments
stripped from the whole files), 176 are patches 0008 (GC links, 138) and 0009 (pymalloc arenas,
38); under intcap 0009 disappears and 0008 shrinks to a few lines. Elsewhere: musl's `atexit`
override (#86), nine MicroPython patches, the CoreMark rewrites and a libc-test patch become
unnecessary. Costs added: nginx and PostgreSQL use `intptr_t`/`uintptr_t` as ordinary integers,
and CPython and FFmpeg use 16-byte atomics on them.


## 1. What exists today

- **`uintptr_t` is the address.** `capstone64` leaves `IntPtrType` at the 64-bit default and says so
  (`clang/lib/Basic/Targets/Capstone.h`, `getMaxAddressWidth()`): "intptr_t, size_t and ptrdiff_t
  describe the ADDRESS". A pointer taken out through `uintptr_t` and cast back is untagged and
  faults on use. `-Wcapstone-pointer-roundtrip` (on by default) names the casts.
- **musl ties them together.** `arch/capstone64/bits/alltypes.h.in` derives `size_t`, `ssize_t`,
  `ptrdiff_t`, `intptr_t`, `uintptr_t` and `regoff_t` all from one `_Addr` (= `long`).
- **The IR conventions are CHERI's.** Capabilities are address space 200 with a 64-bit index
  (`p200:128:128:128:64`, `A200-P200-G200`), the same numbering CHERI LLVM uses, but the machine
  type is Capstone's own `MVT::c128`, and our LLVM 22 fork has no `IntCap` type anywhere.
- **Two stopgaps are in review**: `CapstoneRecoverProvenance` (a round trip computed from one
  pointer inside one function keeps that pointer's capability) and
  `-Wcapstone-capability-alignment`. Neither covers a pointer stored in integer-typed MEMORY,
  which is what CPython's patches 0008 (GC links) and 0009 (pymalloc arenas) and musl's
  `atexit` override exist for.

## 2. CHERI's model

Measured in CHERI's LLVM 17 (`/home/biecho/cheri/llvm-project`, a shallow clone, so by identifiers,
not by diff; file:line references are to that tree):

- **Types.** `__intcap`/`__uintcap_t` are builtin types (`BuiltinTypes.def:96,125`), capability
  sized; their VALUE range is the address width (`ASTContext.cpp:11225-11229`); their integer rank
  is above every other integer, so `int op intcap` converts the int. A purecap target sets
  `IntPtrType = SignedIntCap` (`Basic/Targets/RISCV.h:61-63`). `ptraddr_t` is the address type
  (`stddef.h`); `long`/`size_t` stay 64-bit.
- **Provenance.** Arithmetic takes the address of each operand, does the integer operation, and
  sets the result's address on ONE operand's capability: the left for compound assignment and
  non-commutative operators, otherwise the one not marked as provenance-free (a cast from a plain
  integer, a constant, NULL). Both candidates -> `-Wcheri-provenance`
  (`SemaExpr.cpp:11475-11504`); `__attribute__((cheri_no_provenance))` opts a value out.
  `(uintptr_t)42` is a null-derived (untagged) capability with address 42.
- **Codegen.** An intcap is `ptr addrspace(200)` in IR; the arithmetic is
  `llvm.cheri.cap.address.get`/`.set` around an integer op (`CGExprScalar.cpp:801-894`, target
  hooks in `CommonCheriTargetCodeGenInfo.h`, 135 lines). Comparisons compare addresses.
- **Porting practice** (CheriBSD, measured): of its 691 per-file "CHERI CHANGES" annotations, 173
  are in userland; `pointer_as_integer` 29, `pointer_alignment` 27, `subobject_bounds` 22,
  `integer_provenance` 20, `intcap_arithmetic` 4. Lines inside CHERI conditionals: 0.92 % of
  `lib/libc`, 6.7 % of `rtld-elf`, 0.72 % of `contrib/jemalloc`, 0.02 % of `libarchive`; only 34 of
  libc's 186 `(u)intptr_t` lines needed a CHERI conditional. `share/man/man7/arch.7` states the
  rule ports follow: pointers-as-integers use `uintptr_t`, never `long`; addresses use `ptraddr_t`.
  CHERI's own CPython build turns pymalloc off (`cheribuild .../python.py:61`), so there is no
  obmalloc port to compare with; no CHERI musl is available locally.


## 3. Where Capstone has to differ

- **Linear capabilities.** C integers are copied freely -- by the source, and by the compiler
  (register copies, spills). A linear capability is consumed by `movc`, so an integer holding one
  would either be duplicated or destroyed by an ordinary copy. Our clang has no linear-pointer
  type at all (linearity in C comes only from how the backend materialises a capability, and it
  `delin`s what it creates), so a compile-time "no linear capability in an integer" rule would first
  need a linear qualifier. Until then an intcap has exactly the exposure a C pointer copy already
  has: C-46, observed live 2026-09-24 for call targets.
- **Sealed, uninitialised and revocation capabilities** need defined intcap behaviour (CHERI: an
  operation on a sealed capability clears the tag).
- **Silicon prerequisites, not compiler work** (both OPEN in `ISSUES.md`):
  - **R-33**: moving the cursor of a capability whose size the compressed encoding cannot represent
    WIDENS its authority on silicon (measured). Intcap arithmetic moves cursors constantly; CHERI's
    model assumes the opposite guarantee (bounds kept, or tag cleared, when the cursor leaves the
    representable window).
  - **R-29**: an 8-byte store into the high half of a 16-byte granule followed by a 16-byte `ldc`
    returns that half zeroed. Sixteen-byte integers put more code on exactly that mix.
  QEMU has neither defect, so the software can be built and validated there first.

## 4. Work breakdown

Estimates are focused engineer-weeks, ranges, for someone who knows this fork. "Measured" means
counted in this session's research; everything else is judgement and labelled so.

| WP | What | Size | Estimate | Confidence |
|---|---|---|---|---|
| 0 | **Decisions** (lead, spec owners): adopt CHERI C semantics; the untagged-value ISA rule; the linear rule; a printf length modifier for intcap | -- | days to weeks elapsed | -- |
| 1 | **ISA/QEMU/RTL: untagged values are data.** `MOVC` leaves an untagged source alone and still consumes every capability it may not duplicate, as STC already does (resolves Q-04 and C-32; the recommendation and its evidence are in `plans/2026-09-24-q04-movc-integer-source.md`, branch `plans/q04-movc-decision`); `SCC` and `CINCOFFSET` on an untagged value give an untagged value with the new address (CHERI's rule) instead of trapping. QEMU: its `scc` asserts today (`op_helper.c:798`). RTL: execute-stage logic, then synthesis and a board pass | small logic, high verification cost | 1-3 wk + 1-3 synthesis cycles | low-medium |
| 2 | **Clang front end**, ported from CHERI: the types, rank and promotions; provenance marking on casts (the intrusive part: CHERI threads an `ASTContext` into every CastExpr constructor, which LLVM 22 has also changed); Sema rules and the provenance diagnostics; codegen through `cap_get_cursor`/`scc` in place of `llvm.cheri.*`; TargetInfo (`IntPtrType`), macros (`__UINTPTR_MAX__` = address range, `__SIZEOF_INTCAP__`), `stdint.h`/`stddef.h` (`ptraddr_t`). Hybrid-mode code (`__cheri_tocap` etc., ~870 lines of SemaCast) is not needed | measured: ~1,800 lines in CHERI, ~1,200 for a purecap-only port (+-30 %), ~60 files | 3-5 wk | medium |
| 3 | **Back end.** Set-address for any value (with WP1: `SCC`; without it a tag/type dispatch that cannot cover register copies); null-derived values through the existing bridge pseudo (move the C-40 path onto `PseudoBRIDGE_CAP`); 16-byte capability atomics for `_Atomic uintptr_t` (builds on C-54, #78); intrinsic selection in C++ (TableGen cannot match `anyptr` to `c128`). Optional for code quality: CHERI's get/set-address folds (~560 lines in InstSimplify/InstCombine/ValueTracking) as a Capstone IR pass | 300-700 lines | 2-4 wk | low-medium |
| 4 | **Tests.** Retarget CHERI's intcap-named clang tests (40 files, 7.8k lines; 41 of the 109 intcap-touching files are generated and can be regenerated) from MIPS/RISC-V to capstone64; Capstone-specific lit (linear rule, sealed/uninit behaviour, `ptraddr_t`); a QEMU domain test with a control | measured test base | 1-2 wk | medium |
| 5 | **musl and the runtime.** Split `_Addr` in `alltypes.h.in` (`uintptr_t`/`intptr_t` from `size_t`/`ptrdiff_t`); the six `#if` sites on `UINTPTR_MAX` (`PRIxPTR`, `stdint.h`, `link.h`, `procfs.h`, `dynlink.h`, `__stack_chk_fail.c`); `syscall_arg_t` becomes `__intcap` (about 504 syscall-argument casts become clean); timer ids (5 lines, unreachable in a domain), the stack canary (3), three ambiguous-provenance subtractions, `runtime/fputwc_null_safe.c`; rebuild and run libc-test and the probes. The hostcall wire protocol is fixed-width throughout: no layout change (measured) | measured: 22 lines to review, 9 sites fixed for free | 1-1.5 wk | medium |
| 6 | **Ports.** CPython: re-key patch 0001 on the `ptraddr_t` width (as written it turns into an `#error` everywhere), 0004 hashes the address, 0008 shrinks to the flag and refcount updates (CHERI takes the left operand's capability there), 0009 goes, 34 `_Py_atomic_*_uintptr` sites become 16-byte atomics, `lv_tag` grows every int object. nginx: `ngx_int_t`/`ngx_uint_t` are `intptr_t`/`uintptr_t` (`ngx_shim.h:11-12`) -- redefine as `long`. PostgreSQL: `Datum` is `uintptr_t`, and its config says `SIZEOF_VOID_P 8`. MicroPython: nine patches drop. FFmpeg: 16-byte atomics. Each port rebuilt and rerun | measured site counts; effort judged | 2-3 wk | low-medium |
| 7 | **Validation and measurement.** Full lit, QEMU nightly, libc-test, every port; memory and time cost of 16-byte integers (objects that grow, atomics); then the FPGA once WP1, R-33 and R-29 are in | -- | 1-2 wk + board time | medium |

**Total WP2-WP7: 10-17.5 engineer-weeks; with WP1, 11-20 weeks of effort, 3-5 months elapsed for
one person.** For scale, from this session: the thread-local storage change (lowering, linker
script, runtime, lit and QEMU tests) went from branch creation (08:22) to a passing QEMU test
(10:21) in about two hours of agent time, including finding and fixing a pre-existing compiler
bug (C-46) that the test exposed; the full nightly gates take longer than that. Intcap is an order of magnitude larger and crosses the ISA, so expect several such finds.


## 5. Where the estimate could break

- **LLVM 17 -> 22 drift** in the files CHERI touched (Sema, CGExprScalar, the CastExpr
  constructors). The port is mechanical in principle and conflict-heavy in practice.
- **Capstone's capability types.** Linear capabilities cannot live in an integer; there is no
  linear qualifier in our clang to reject them at compile time, so an intcap has the exposure a C
  pointer copy has today (C-46, observed live 2026-09-24). Sealed and uninitialised values: CHERI
  clears the tag on arithmetic; Capstone's `SCC` traps -- WP1 must cover them too.
- **More `MOVC` everywhere.** C-46 and C-32 are both copy-destroys-source defects; intcap moves a
  large amount of ordinary integer traffic into capability registers. WP1 removes the untagged half;
  the linear half stays as it is today.
- **Code that relies on a round trip dropping the tag.** The whisper port returned
  `(void *)(uintptr_t)p` so a zero-byte allocation carries no authority; intcap (and, already,
  `CapstoneRecoverProvenance`) would hand back `p`. The explicit form is to take the address
  (`__builtin_capstone_cap_get_cursor`, or `ptraddr_t` under intcap). A tree-wide search found no
  other such code, but new ports may carry it.
- **Integers that are really integers.** Any code base that uses `uintptr_t` as its general
  unsigned type (nginx) pays capability-sized storage and capability instructions for plain
  numbers, and needs a type change.


## 6. Staged plan

1. **Decide** (WP0). Nothing below is worth starting without the ISA rule and the lead's go-ahead.
2. **QEMU first.** Make QEMU implement the WP1 semantics (small), so all software work can proceed
   and be validated without silicon.
3. **Front end behind a flag** (`-mcapstone-intcap` or a triple variant), WP2 + WP4: `uintptr_t`
   stays 64-bit by default until everything below is green.
4. **musl, then CPython** (WP5, WP6): measure which CPython patches actually disappear -- the
   number that justifies the switch -- before touching the other ports.
5. **The other ports and the full gates** (WP6, WP7), then flip the default.
6. **Silicon** once WP1's RTL change, R-33 and R-29 are in: the FPGA pass is the last gate.

The stopgaps in review stay useful either way: `CapstoneRecoverProvenance` remains the backstop for
pointers cast through `long` (which CHERI leaves untagged), and `-Wcapstone-capability-alignment`
is independent of the pointer-integer model.


## 7. Sources

Research for this document (2026-09-24), read-only, with file:line evidence checked in the main
session: an inventory of CHERI LLVM's clang `__intcap` implementation; a mapping of what its codegen
needs onto the Capstone back end (`CapstoneISelLowering.cpp`, `CapstoneInstrInfo.td`,
`CapstoneISASemantics.md`, `capstone-spec/parts/cap-man-insn.adoc`); a survey of `uintptr_t` use in
musl 1.2.5 (1,355 compiled sources), our runtime, the hostcall headers, CPython 3.13.7 (249 objects
in the Makefile's lists) and the other ports; and CheriBSD's CHERI annotations and markers.

