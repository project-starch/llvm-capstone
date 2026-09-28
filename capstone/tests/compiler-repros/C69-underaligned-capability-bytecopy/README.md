# C-69 — an under-aligned capability load/store is lowered to a BYTE COPY, silently dropping the tag

A capability read or written through `*(T **)((char *)&lvalue + runtime_offset)` carries
`align 1` in the IR, and the backend legalizes that into a byte-by-byte copy through a stack
temporary. The address and the metadata bits survive; the **tag does not**. The result is a
pointer-shaped value that is not a capability, and the fault appears later, at the first
dereference, somewhere else entirely.

    field_at       lbu=16  ldc=3  sd=2     <- the capability is reassembled from 16 bytes
    field_store     sb=16  stc=4  ld=2     <- and scattered back out as 16 bytes
    field_direct   ldc=3   lbu=0           <- the naturally-aligned control: one ldc

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

## Suggested directions, none of them taken here

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
