# CVE-2018-10191 (Row 11) — `OP_GETUPVAR` scope-level truncation in mruby ≤ 1.4.0

Minimal, deterministic reproduction of `CVE-2018-10191`: nesting Ruby scopes past
128 levels silently truncates the `OP_GETUPVAR` scope-level operand, so the VM
resolves the wrong environment and then indexes it with an index computed for a
much wider scope — reading past the end of its register array.

> ### ⚠ This is a spatial bug, not a use-after-free
> The spec and the benchmark table list this row as a temporal borrow (UAF). It
> **does not reproduce as one** — ASan reports `heap-buffer-overflow`. The
> temporal path is closed at this version by `envadjust()`. This needs a decision
> before the row goes into a temporal-only benchmark; the evidence and the options
> are in **`target.md` → "Bug-class discrepancy"**.

## Vulnerability overview

`OP_GETUPVAR` carries two operands in one instruction word: `B`, the local's index
within the target scope (**9 bits**, max 511), and `C`, how many scopes to walk
outward (**7 bits**, max 127). `codegen.c:2191` emits the level with no range
check, so at depth ≥ 129 it wraps — `129 & 0x7f == 1`.

`uvenv()` then walks 1 scope instead of 129 and hands back a small, wrong
environment. `vm.c:1208` reads `e->stack[b]` on it with `b` still the large outer
index, overshooting that environment's storage.

Two independent bugs have to line up, and either check alone would stop it: the
compiler emits an out-of-range level without a diagnostic, and the VM indexes the
resolved environment without bounding `b` against its register count.

## Classification: spatial, not temporal

The benchmark table lists this row as a use-after-free. It is not one. The defect is
an operand-width truncation: the compiler emits a 7-bit scope level that wraps at
depth ≥ 129, so `uvenv()` resolves the wrong — much nearer — environment while the
register index `b` is still the large one computed for the intended outer scope. The
read at `vm.c:1208` then runs off the end of a live, correctly-sized register array.
The temporal reading was tested and closed rather than assumed: `envadjust()`
rewrites `REnv::stack` on every VM-stack reallocation, and a `Proc` that escapes its
scopes gets its environment *closed* onto heap-owned storage. Neither path leaves a
reference outliving its referent.

Bounds, not revocation, are what stop it. A capability bounded to the resolved
environment's register array rejects `e->stack[b]` as soon as `b` exceeds that
array's length, independently of whether the overshoot reaches unmapped memory —
which matters here, because with fewer than ~80 outer locals the same truncation
reads *in bounds* and silently returns the wrong value (see "Tuning the trigger").
Revocation has nothing to act on: the environment it would revoke is alive and
intact. The error is that the wrong environment was selected and then indexed past
its end, and only a bound on that index catches it.

## Contents

| File | What it is |
|---|---|
| `target.md` | Pinned commit, mechanism, ASan verdict, **bug-class discrepancy** |
| `build_config.rb` | host+ASan and riscv64 cross build |
| `build.sh` | Clean checkout of `e340b172` → both builds |
| `trigger.rb` | **The trigger** (generated, self-contained): 80 outer locals + 129 nested blocks |
| `run.sh` | Runs both, asserts the native abort |
| `asan.txt` | Captured ASan report (scrubbed) |
| `boundary.md` | Annotation per §8, with the caveat that nothing crosses an FFI here |

## How to build and run

```bash
chmod +x build.sh run.sh
./build.sh
./run.sh
```

## Expected outcome

**Native + ASan** — aborts, exit 1:

```
==NNNN==ERROR: AddressSanitizer: heap-buffer-overflow
READ of size 16 at 0x... thread T0
    #0 ... in mrb_vm_exec .../src/vm.c:1208
0x... is located 528 bytes after 4096-byte region
SUMMARY: AddressSanitizer: heap-buffer-overflow .../src/vm.c:1208 in mrb_vm_exec
```

Deterministic — 10/10 runs.

**RISC-V QEMU** — **segmentation fault, exit 139**, on 3/3 runs. On the riscv64
build the out-of-range read reaches an unmapped page, so the plain non-ASan binary
faults outright. Both legs of the deliverable are therefore present for this row.

`PASS = native ASan shows heap-buffer-overflow at vm.c:1208 in mrb_vm_exec`
(and RISC-V QEMU segfaults on the same trigger)

## Tuning the trigger

Both knobs are load-bearing:

- **Nesting depth ≥ 129** — below that the level fits in 7 bits and resolution is
  correct. Depths 129–254 all truncate; **255+** is rejected by the compiler with
  `codegen error: too complex expression`, so the usable window is 129–254.
- **Outer locals ≥ ~80** — sets how large `b` is. Below ~80 the stray read stays
  in bounds and quietly returns the wrong value instead of faulting, which is the
  same bug with no sanitizer-visible symptom.
