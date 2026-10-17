# Pre-registration — the narrow-decline libcall prototype

**Written and pushed BEFORE the run.** Cite this commit's hash from whatever reports the result.
Nothing below may be edited after the run; corrections go in the result, not here.

## What is being tested

The wholesale flag `-capstone-lower-memops-via-libcall` declines **every** `mem*` op and a domain
built with it does not run (mruby `f87f9cbf`: cause 24, `pc=0xe030c96c`, `address=0xe2246e90`,
inside `memcpy+0xc8`, a heap address, before any case code; and the 2026-08-11 SQLite record). That
disqualifies the wholesale flag as a workaround. It does **not** settle whether the *narrow* fix is
sound, which is what this prototype isolates.

**The prototype:** in `findOptimalMemOpLowering`'s final scalar branch — the one guarded by
`!(Op.isAligned(Align(16)) && (Op.size() % 16) == 0)` — decline (return false, so a libcall is
emitted) when the copy is at least 16 bytes AND its effective alignment is below 16. Everything
else keeps today's lowering. Under a flag, default off.

## Predicted readings IF "the narrow libcall is sound" is TRUE

| arm | prediction if sound |
|---|---|
| mruby, prototype | **runs**: `smoke.rb` and all 20 rows reach their verdicts, no cause 24 in `memcpy` |
| mruby, unmodified compiler, same images | runs (the control: it runs today) |
| mruby, wholesale flag | faults as recorded — the known-positive, proving the arms differ |
| SQLite arms, prototype | **run** to their recorded verdicts |
| SQLite arms, unmodified | run (control) |
| C-81 `run.sh` | `copy32_unknown` and `copy32_align8` become **calls**; `copy32_known` unchanged `ldc`/`stc` |
| `hosted-contract.c` | `memcpy_constant_size_unknown_alignment` **PASSES** without `-fno-builtin` |

**Predicted readings if it is FALSE** (the fault is intrinsic to compiler-generated `mem*` libcalls
on this ABI): mruby and/or SQLite fault under the prototype too, and the fault signature matches the
wholesale one — cause 24 inside `memcpy`, during startup, before case code.

**If the prototype reproduces the wholesale fault, that fault becomes the next thing to explain,
not a reason to abandon the direction.** Its mechanism is currently unexplained: my probe found the
libcall's argument setup correct in every shape tested, so "compiler-generated libcalls are broken
here" is not established and the `gp` hypothesis does not apply under the v6 flags.

## THE VOID GATE, which binds the verdict

**A pass on an image where the prototype converts nothing is VOID, not a pass.** Before reading any
run, compute per image the number of sites the prototype converts, as the `mem*` call references the
prototype build gains over the unmodified build (calls-gained, the method that measured 48 where my
scanner said 2). Record it beside each verdict.

* **converted sites == 0** → that image's result is VOID and says nothing about the prototype. It
  must not be counted as a pass, and it must not be counted as a failure.
* **converted sites > 0 and the image runs** → informative pass.
* **converted sites > 0 and the image faults** → informative failure; record the signature and
  compare it with the wholesale one.

This gate exists because the whole point of a narrow decline is to convert few sites, so "it ran"
is exactly what an image with zero conversions will say.

## Named hazards, written down before the build

1. **`llvm.memcpy.inline` must NOT be declined.** `findOptimalMemOpLowering` returning false for a
   forced-inline copy is a documented SILENT MISCOMPILE on this target: `getMemcpy` retries with
   `AlwaysInline`, that returns null too, and the null `SDValue` destroys the DAG root — the copy,
   the following stores, the stack teardown and the return all vanish, exit code 0, no diagnostic.
   The prototype must therefore fire only when the copy is not forced-inline; `Limit` is `~0U` in
   the `AlwaysInline` case, which is the available discriminator. **A build that does not guard
   this is not to be run.**
2. **`isDstAligned` can read TRUE spuriously** when `DstAlignCanChange` (a fresh alloca whose
   alignment the expander may still raise), so an alignment test can under-fire and miss cases.
   That costs coverage, not correctness, but it means a zero conversion count may be the test's
   fault rather than the image's — which is why hazard 1 and the void gate are separate.
3. **The S-06 handshake must not be disturbed.** The first block pairs with
   `capstoneShouldExpandCapMemcpyInline` and its condition must keep matching the hook's exactly;
   the prototype goes after it, not into it.

## What would make me abandon the direction

Only this: the prototype converts a non-zero number of sites, the images fault, **and** the fault
is shown to arise from the libcall mechanism itself rather than from an untagged pointer that
already existed before the call. Absent that last part, a fault is an unexplained defect to chase,
not a verdict on the direction.
