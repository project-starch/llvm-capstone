# C-69 — an under-aligned capability load/store is lowered to a BYTE COPY, silently dropping the tag

A capability read or written through `*(T **)((char *)&lvalue + runtime_offset)` carries
`align 1` in the IR, and the backend legalizes that into a byte-by-byte copy through a stack
temporary. The address and the metadata bits survive; the **tag does not**. The result is a
pointer-shaped value that is not a capability, and the fault appears later, at the first
dereference, somewhere else entirely.

    field_at       lbu=16  ldc=3  sd=2     <- the capability is reassembled from 16 bytes
    field_store     sb=16  stc=4  ld=2     <- and scattered back out as 16 bytes
    field_direct   ldc=3   lbu=0           <- the naturally-aligned control: one ldc

## Status: fixed on lane `capstone-misaligned-cap-access` (2026-11-15), default on

`CapstoneMisalignedCapAccess.cpp`, an IR pass after CodeGenPrepare at every optimisation level, gives each
under-aligned capability load or store a run-time alignment test:
- if the address is 16-aligned, the same access at align 16 runs: `ldc`/`stc`, so the tag survives;
- otherwise the old scalar expansion runs: a misaligned granule holds no tag, and `ldc`/`stc` there would trap.

That is musl `memcpy`'s result for a 16-byte copy into or out of an aligned object. The pass is off with
`-mllvm -capstone-misaligned-cap-dispatch=false`. `-capstone-misaligned-cap-log=<file>` lists the functions it
changes, or with the dispatch off, would change.

**Second producer.** The same access comes from the C idiom `memcpy(&val, void_arg, sizeof val)` once SROA has
folded the copy into an align-1 load: FFmpeg's `av_freep`. That is the fault the nested lane hit with builtins on
(cause 24 in `__libc_free`, an untagged pointer with its metadata intact). With `-fno-builtin` the copy stays a call
to musl, which tests the alignment itself. C-81's memcpy guard never sees it, because no memcpy is left.

**Not covered.** A capability member of a PACKED aggregate. The test is on the access's base address. A member at an
offset that is not a multiple of 16 stays scalar in both arms, so it still loses its tag when the base is
misaligned by exactly the complement of that offset. Natural layout puts every capability member at a multiple of
16, where the base test decides it.

**Run-time test (`c69-run.c`).** Five shapes in noinline functions:
- `av_freep`'s load (`load_void`), its store half (`store_void`), and the same load through an `unsigned long *`
  (`load_word`, align 8);
- `FF_FIELD_AT`'s load and store (`load_field`, `store_field`).

Each runs at a 16-aligned address holding a tagged capability, where the tag must survive, and at a misaligned
one, where the bytes must equal the runtime memcpy's and the tag must be clear. A plain-word copy is the control
and must lose the tag. Every read-back the checks make is 16-aligned in the IR. An earlier version read `buf + off`
directly, which is itself the C-69 shape, so its dispatch-off arm reported a store failure the code under test had
not made. At -O0 there is no SROA, so the three memcpy shapes stay calls, and only the `FF_FIELD_AT` shapes are C-69
there.

Native twin (qemu55, images-a0ed, the SPEC arms' FRCDKMSa3 flags); judge `perf/c69/judge-c69.py` on the build host:

| compiler | level | dispatch | result |
|---|---|---|---|
| llvm-c15 (29027126f39b + this fix) | -O2 | on | 10/10 |
| llvm-c15 | -O2 | off | the 5 aligned-address checks FAIL, the other 5 pass (image identical to llvm-c9's) |
| llvm-c15 | -O0 | on | 10/10 |
| llvm-c15 | -O0 | off | `load_field`, `store_field` FAIL, the other 8 pass |
| llvm-c9 (29027126f39b) | -O2 | | the 5 aligned-address checks FAIL |

`check.py` now takes an optimisation level. A byte copy behind the run-time `andi ..., 15` test counts as gone: it
is the arm a misaligned address takes. With llvm-c15 it exits 1 (gone) at -O2 and at -O0; with llvm-c9 it exits 0
(present) at both.

## Reproducing

    python3 check.py <path-to-clang>

Exit 0 means the defect is present, 1 means it is gone, 2 means the check could not run (which is
reported as an error rather than as a clean result). Verified two-sided on
clang 22.0.0git @ 08ff5d0702c3:

* **present** on `repro.c` as committed — both the load and the store lower to byte copies;
* **absent** when `FF_FIELD_AT` is wrapped in `__builtin_assume_aligned(p, _Alignof(type))`,
  which takes `lbu` and `sb` to 0 and restores `stc`/`ldc`. That is both the negative control for
  this check and independent confirmation that the port-side workaround does what it claims.

The check refuses to pass on a broken control: if `field_direct` ever stops lowering to a plain
`ldc`, it exits non-zero and says so, because the comparison is the whole evidence.

## Where the `align 1` comes from — NOT a fork defect

Upstream clang, not this fork and not a Capstone-specific alignment choice:
*"[Clang][CodeGen] Preserve alignment information for pointer arithmetics (#152575)"*, present in
the fork at 5569bf26f009. The same compiler emits `align 1` for `--target=x86_64-linux-gnu` as
well; Ubuntu clang 18.1.3 emits `align 8`. On an ordinary target an under-aligned pointer load is
merely slow. **The IR is correct; the Capstone-specific defect is what the backend does with it.**

That distinction matters for the fix: reverting or special-casing the clang change would be
treating the symptom on one target and would leave every other path that can produce an
under-aligned capability access — a packed struct, a memcpy-shaped idiom, an explicit
`__attribute__((aligned(1)))` — still silently dropping tags.

## Why it is worse than a wrong answer

A byte copy of a capability is indistinguishable, at the point it happens, from a legitimate copy
of data. Nothing faults, nothing warns, and the value keeps its address, so it reads correctly in
a debugger and in any check that looks at the pointer value. It fails only when something later
tries to use it as a capability, in a different function.

## Exposure

Every port built with this compiler, wherever `(T *)((char *)&lvalue + runtime_offset)` reads or
writes a pointer field. It is not an FFmpeg-specific shape; FFmpeg is simply where it was first
hit, via `FF_FIELD_AT` (`libavutil/internal.h`) in libavfilter's format negotiation
(`avfiltergraph.c`). A disassembly scan for the two lowering shapes found 10 sites in the
pre-workaround `avfilter_graph_config` and 0 after the port-side workaround.

## Suggested directions (as filed; the fix took neither, see Status)

1. **Refuse to lower it.** Diagnose an under-aligned capability-typed access rather than emitting a
   byte copy. Loud and cheap, and correct in the sense that the byte copy is never what was meant.
2. **Treat capability-typed accesses as naturally aligned** when the target cannot split them.
   Silently correct, but it makes a genuinely misaligned access UB rather than an error.

Both are backend changes and neither has been validated. Filed with a reproducer rather than fixed
speculatively, following C-68.

## Provenance

Found by the FFmpeg/wmem port lane, 2026-09-29, on compiler 3979abd8; their first report placed the
defect in clang's alignment choice and a claim audit corrected it to the backend lowering. The
reduction here was written and verified independently by the compiler lane on 08ff5d0702c3 — a
different compiler commit, which is part of why the finding is solid rather than a property of one
build.
