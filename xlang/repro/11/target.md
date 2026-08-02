# target.md — CVE-2018-10191 (Row 11)

* **CVE:** CVE-2018-10191
* **Product:** mruby ≤ 1.4.0
* **Vulnerability Type:** Operand-width (integer) truncation of the `OP_GETUPVAR`
  scope level, yielding an out-of-bounds read of an environment's register array
* **Status:** REPRODUCED — but **not as a use-after-free**; see below
* **Vulnerable Tag/Commit:** `e340b1725260e70a4b5b7b96c53d5015b8f9c1b0` (1.4.0-96)
* **Crash Site:** `src/vm.c:1208` — the `*regs_a = e->stack[b]` read in `CASE(OP_GETUPVAR)`
* **ASan Verdict:** `heap-buffer-overflow`, `READ of size 16`, 528 bytes past a
  4096-byte region
* **Determinism:** aborts on 10/10 consecutive native-ASan runs; the trigger is a
  pure function of nesting depth and local count — no GC timing, allocation
  layout, or randomness.
* **RISC-V QEMU:** **SIGSEGV, exit 139, on 3/3 runs.** On the riscv64 build the
  out-of-range read reaches an unmapped page, so the plain non-ASan binary faults
  outright — a full QEMU reproduction per task spec §4.2.

## Mechanism

mruby packs `OP_GETUPVAR`'s operands into one instruction word
(`include/mruby/opcode.h`):

* `B` = the local's index inside the target scope — **9 bits** (`& 0x1ff`, max 511)
* `C` = how many scopes to walk outward — **7 bits** (`& 0x7f`, max 127)

`mrbgems/mruby-compiler/core/codegen.c:2191` emits
`MKOP_ABC(OP_GETUPVAR, cursp(), idx, lv)` with **no check that `lv` fits in 7
bits**. At nesting depth ≥ 129 the level silently truncates (`129 & 0x7f == 1`).

`uvenv()` (`src/vm.c:229`) then walks only 1 scope outward instead of 129 and
returns the wrong — much smaller — environment. `src/vm.c:1208` reads
`e->stack[b]` on it while `b` is still the large index computed for the outer
scope, so the read lands far past the end of that environment's register storage.

Both knobs are required: ≥129 nesting for the truncation, and ≥~80 locals in the
outer scope so `b` is large enough to overshoot. Below ~80 the stray read stays in
bounds and returns a wrong-but-valid value (visible as the trigger printing the
`instance_eval` receiver instead of the intended local).

## Bug-class discrepancy — needs a decision

The task spec §6 and the benchmark table both classify this row as a
**use-after-free / temporal borrow**. It does not reproduce as one. ASan reports
**`heap-buffer-overflow`** — a *spatial* violation.

This was checked rather than assumed. The plausible temporal path is a dangling
`REnv::stack` after the VM stack is reallocated, but mruby closes that at this
pin: `stack_extend_alloc` calls `envadjust()` (`src/vm.c:143`) on every realloc,
which walks all live callinfo and rewrites `e->stack` for both `ci->env` and
`MRB_PROC_ENV(ci->proc)` on stack-shared environments. Two further attempts to
force a temporal shape also failed:

1. Letting a `Proc` escape all 129 scopes and calling it after GC and stack churn
   — mruby *closes* the environment on escape (copying registers to heap-owned
   storage), so nothing dangles; the read returns a wrong value, no fault.
2. Driving stack reallocation between capture and read — neutralised by
   `envadjust`.

So at this pinned version the defect is an operand-truncation OOB read, not a
temporal borrow. **This matters for the benchmark**: the companion note claims
"every defect here is a temporal borrow" and "There is no spatial or
single-domain row, by selection". Including this row as reproduced breaks that
uniformity claim.

Options, for the core team to choose:

* **Keep it, reclassified as spatial.** Honest, and the artifact is complete and
  deterministic — but the "all temporal" framing has to be softened, and the
  linearity argument for this row becomes bounds-checking rather than revocation.
* **Drop it from the temporal subset.** Preserves the uniformity claim. The
  artifact stays here as evidence of *why* it was dropped.

Either way the CVE's "UAF in the upvalue/environment stack" characterisation does
not hold at `e340b172`. If a temporal variant exists it is in a different mruby
version or a different code path than the one this trigger exercises.

> **Supersedes an earlier SKIPPED filing.** The previous rationale said deeply
> nested scopes "are rejected by the parser's compilation engine (throwing 'too
> complex expression' or stack overflow memory limits) before the runtime
> integer overflow can be executed". That is wrong on the numbers: depths 129–254
> compile and run fine, and `codegen error: too complex expression` first appears
> at depth **255**, not 128. The old `trigger.rb` was also written in Python, so
> it died with a Ruby `SyntaxError` at line 2 and never exercised nesting at all.
