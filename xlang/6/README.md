# CVE-2026-1979 / mruby #6701 (Row 6) — pattern-matching bytecode corruption

Minimal, deterministic reproduction of `CVE-2026-1979`: a peephole optimization in
mruby's pattern-matching codegen overwrites a byte of the **wrong instruction**,
corrupting a register operand. The corrupted instruction then stores outside its
frame.

> ### ⚠ This is a spatial bug, not a use-after-free
> The spec and the benchmark table list this row as a temporal borrow ("UAF in
> `mrb_vm_exec`"). It reproduces as `heap-buffer-overflow` — a **WRITE** past the
> end of the VM stack. Nothing is freed and reused. Same finding as Row 11; see
> **`target.md` → "Bug-class discrepancy"** for the options.
>
> Also: **Row 6 and Row 7 are not the same bug**, despite an earlier note in this
> directory saying so. See `target.md` → "Relationship to Row 7".

## Vulnerability overview

`expr in pattern` codegen tries to fold a single-failure pattern by inverting its
`JMPNOT` to a `JMPIF` and dropping a redundant `JMP`:

```c
s->iseq[fail_pos - 2] = OP_JMPIF;   /* codegen.c:6632 */
```

`fail_pos - 2` is only the opcode position if a **4-byte** `OP_JMPNOT` sits there.
When the pinned variable is **undefined**, `NODE_PAT_PIN` emits a **3-byte**
`OP_JMP` instead, so the store lands one byte earlier — on the last byte of the
*preceding* instruction — writing `38` (`OP_JMPIF`) over it.

In this trigger the preceding instruction is `LOADI_5 R2` (2 bytes: opcode +
destination register), so it becomes **`LOADI_5 R38`** in a frame with `nregs=4`:
an out-of-bounds register write. `bytecode-diff.txt` shows this against a compiler
with the upstream fix applied.

## Contents

| File | What it is |
|---|---|
| `target.md` | Pinned commit, full mechanism, ASan verdict, **bug-class discrepancy**, Row 7 relationship |
| `build_config.rb` | host+ASan and riscv64 cross build |
| `build.sh` | Clean checkout of `cda2567c` → both builds |
| `trigger.rb` | **The trigger**, with each of its three preconditions explained inline |
| `run.sh` | Runs both legs, asserts the native abort |
| `asan.txt` | Captured ASan report (scrubbed) |
| `bytecode-diff.txt` | Side-by-side disassembly, vulnerable vs. fixed compiler |
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
WRITE of size 8 at 0x... thread T0
    #0 ... in mrb_vm_exec .../src/vm.c:1788
0x... is located 32 bytes after 1024-byte region
```

Deterministic — 10/10 runs.

**RISC-V QEMU** — **SIGSEGV, exit 139**, 3/3 runs. The stray store reaches an
unmapped page on the riscv64 build. Note it prints `done` first: the corrupted
write happens during the recursion and the fault surfaces slightly later.

`PASS = native ASan shows heap-buffer-overflow WRITE at vm.c:1788 in mrb_vm_exec`
(and RISC-V QEMU segfaults on the same trigger)

## Inspecting the corruption directly

```bash
./mruby/build/host/bin/mrbc -v -o /dev/null trigger.rb | grep -A6 ENTER
```

`LOADI_5 R38` is the corrupted instruction. A compiler with `e50f15c1` applied
emits `LOADI_5 R2` and keeps the second `JMP`.

## Tuning the trigger

All three preconditions are load-bearing — remove any one and nothing corrupts:

- **`^u` must be undefined.** Define `u` and `NODE_PAT_PIN` takes the 4-byte
  `OP_JMPNOT` path, where `fail_pos - 2` is correct.
- **The match must be in statement position.** Drop the trailing `nil` and
  `5 in ^u` becomes the method's return value; codegen then emits `OP_LOADT`
  first, the optimization's `fail_pos + 2 == s->pc` guard fails, and no store
  happens.
- **Recursion depth ≥ 29.** At top level the frame has ~128 slots of slack so the
  stray write stays inside the allocation and nothing faults. Depths 29–31, 80 and
  118 all fault; the bug is present either way, only its observability changes.
