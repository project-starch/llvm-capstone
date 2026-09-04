# Capstone backend compiler fixes

Known backend bugs surfaced during CoreMark PureCap bring-up. The prologue
frame-lowering bug is fixed and validated; the remaining active workarounds are
still documented in `capstone/benchmarks/coremark/build-coremark-capstone.sh`
with comments pointing to the root cause. These need proper fixes in the LLVM
backend.

## Active bug catalog

| Bug | Current workaround | Root fix needed |
|-----|--------------------|----------------|
| `cincoffset rd≠rs1` with LINEAR rs1 consumes the cap; compiler hoists the offset computation above a loop, so the first iteration consumes the LINEAR cap and subsequent iterations crash | `CAPSTONE_DELIN(ptr)` applied manually before each affected pointer; `-O0` for files where the compiler would hoist | Backend: after instruction selection, identify `cincoffset` uses where rs1 is a LINEAR cap and rd≠rs1; insert an automatic `delin` on rs1 before the first such use |
| Tail-call lowering emits `cjalr ra, imm(rs1)` (a CALL that sets `ra = pc+4`) instead of restoring `ra` and jumping | `-fno-optimize-sibling-calls` on affected files | `CapstoneISelLowering`: detect sibling-call scenario; emit `ldc ra, N(sp)` + `cincoffsetimm sp, sp, frame` + `cjalr zero, 0(ra)` instead of a CALL |
| Jump tables use scalar `lw` base load in cap_mem mode; `lw` requires `rs1.tag=1` but the GP-derived table pointer is scalar | `-fno-jump-tables` on affected files | `CapstoneISelLowering` / `CapstoneAsmPrinter`: lower jump tables through `ldc` (capability load) or materialize table entries as PC-relative capabilities so the table base is tagged |

## Resolved

- **`va_list` capability-tag loss.** `va_start`/`va_arg` (and `va_copy`) were
  lowered with the AS0 pointer type (i64): the variadic-argument pointer was
  stored/reloaded with scalar `sd`/`ld`, dropping the capability tag (any later
  argument dereference faulted in cap_mem mode), and `va_arg` advanced the cursor
  by the raw type size (8 for `long`) instead of the 16-byte (CLEN) save-area slot
  stride. The bug was entirely in the **LLVM Capstone backend** — Clang merely
  emits the raw `va_arg` IR instruction (`capstone64` has no case in
  `CodeGenModule::createTargetCodeGenInfo`, so `DefaultABIInfo::EmitVAArg` defers
  to the backend; `clang/.../Targets/Capstone.cpp::CapstoneABIInfo::EmitVAArg` is
  unwired dead code for this triple).
  Fix (`CapstoneISelLowering.cpp`): `VAARG`/`VACOPY` made `Custom`; `lowerVASTART`
  types the save-area frame index with the capability (AS200, i128) pointer so the
  store selects `stc`; new `lowerVAARG` loads via `ldc`, advances by
  `alignTo(typeAllocSize, 16)` with `CapstoneISD::CIncOffset`, stores back via
  `stc`, and loads the argument through the tagged capability; new `lowerVACOPY`
  copies the va_list via `ldc`/`stc`. (SelectionDAG path; GISel
  `G_VASTART`/`G_VAARG` has the same latent bug but is not exercised by the
  benchmarks — parallel follow-up.)
  Regression test: `llvm/test/CodeGen/Capstone/vararg.ll`.
  This removed the CoreMark `ee_printf` workaround: `ee_printf` now uses a standard
  C `va_list` (`port/core_portme.c`); the `ee_printf_asm.S` trampoline and the
  `ee_printf_impl`/`pick_arg` indirection are deleted and CoreMark still validates.
  Unblocks the `va_list` prerequisite for `trio`.

- **16-byte non-capability aggregate copy miscompile (was "Bug #10").** A `memcpy`
  of a 16-byte `{i64,i64}` value that is only 8-byte aligned (e.g. the by-value
  struct copy emitted for `range = MakeRange(...)`, assignment of a struct return
  into a pre-declared local) was lowered as an 8-byte `ld` feeding a 16-byte
  `stc`: only the low 8 bytes were loaded and the upper field was overwritten
  with the sign/zero-extension. The original "all 16-byte aggregates are coerced
  to i128 and stored by `stc`" theory was wrong — by-value args/returns, struct
  inits, and pointer-to-pointer copies were already correct; only the misaligned
  i128 `memcpy` path mis-legalized (the unaligned i128 *load* narrowed to one
  i64, the paired i128 *store* stayed `stc`).
  Fix: `CapstoneTargetLowering::findOptimalMemOpLowering` now copies a memcpy
  that is not 16-byte (capability) aligned as matched XLen (`i64`) chunks, never
  forming a misaligned i128 (capability) memory op; capability-aligned copies
  still use `i128` (`ldc`/`stc`) to preserve tags.
  Regression test: `llvm/test/CodeGen/Capstone/aggregate-memcpy-align.ll`.
  This also removed the BEEBS wikisort source workarounds (upstream `Range
  { long; long; }` restored; the manual final-merge / early-`return` deleted —
  the "final-level hang" was a downstream symptom of the corrupted ranges).

## Pointer/integer cast policy

See also `ref/capstone-purecap-pointer-model.md`.

Pointer-to-integer casts may preserve numeric address bits, but they do not
preserve capability provenance or dereference authority. Integer-to-pointer casts
may produce a pointer-shaped value, but unless the implementation has a specific
valid provenance-restoration rule, the resulting pointer must not be assumed safe
to dereference. Runtime trap/failure is expected if such a pointer is used for
load/store/call.

Valid pointer arithmetic should stay in the capability domain, for example
`p + offset`, rather than round-tripping through `uintptr_t`. For benchmark
bring-up, do not fix failures by fabricating authority from raw integers. Prefer
benchmark-local rewrites that keep accesses derived from valid capabilities, or
add a focused backend/runtime test if a real benchmark exposes this pattern.

## Related files

- `capstone/benchmarks/coremark/build-coremark-capstone.sh` — remaining active workarounds have comment blocks
- `llvm/lib/Target/Capstone/CapstoneISelLowering.cpp` — tail-call and jump-table lowering; `va_list` lowering (`lowerVASTART`/`lowerVAARG`/`lowerVACOPY`)
- `llvm/lib/Target/Capstone/CapstoneFrameLowering.cpp` — prologue/epilogue emission (prologue LINEAR-sp consumption fixed)
