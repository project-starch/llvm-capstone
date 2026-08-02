# CVE-2022-1106 (Row 10) — Use-After-Free in `mrb_vm_exec` (`OP_RANGE_INC`)

Minimal, deterministic reproduction of `CVE-2022-1106`: the mruby VM caches a
pointer into its register stack across a call that re-enters Ruby, and the Ruby
callback grows the stack — moving and freeing the old allocation. The VM then
writes the result through the stale pointer.

This is the **template row** the task spec (§5) designates as the pattern for the
rest of the corpus.

## Vulnerability overview

Constructing a `Range` executes `OP_RANGE_INC`, which assigns into `regs[a]` while
the right-hand side calls `<=>` on the endpoints to validate them. `regs` is a macro
for `mrb->c->ci->stack`, and in C the destination lvalue and the right-hand call are
unsequenced — so the compiler is free to materialise `&regs[a]` *before* the call,
and does.

The trigger's `Bad#<=>` recurses 150 frames deep on its first invocation. That
exceeds the stack's capacity, so `mrb_stack_extend` reallocs the register array; it
moves and the old block is freed. When `OP_RANGE_INC` resumes it writes the new
`Range` to the cached, now-dangling address.

Same shape as rows 4, 5, 8, 13 and 15: **native C code holds a raw pointer into the
VM stack across a callback into Ruby.** See `boundary.md`.

## Contents

| File | What it is |
|---|---|
| `target.md` | Pinned commits (vulnerable `bf5bbf0a`, fix `7f5a490d`) |
| `build_config.rb` | host-asan and riscv64 cross build |
| `build.sh` | Clean checkout → both builds |
| `trigger.rb` | **The trigger**: `Bad#<=>` recursing to force a stack realloc |
| `run.sh` | Runs both legs |
| `asan.txt` | Captured ASan report (scrubbed) |
| `boundary.md` | Boundary annotation per task spec §8 |

## How to build and run

```bash
chmod +x build.sh run.sh
./build.sh
./run.sh
```

`build.sh` is not idempotent against a dirty `mruby/` tree — a modified `Rakefile`
will abort the checkout. If that happens, `git -C mruby checkout -- .` first, or
delete `mruby/` for a clean clone.

## Expected outcome

**Native + ASan** — aborts, exit 1:

```
==NNNN==ERROR: AddressSanitizer: heap-use-after-free
WRITE of size 8 at 0x... thread T0
    #0 ... in mrb_vm_exec .../src/vm.c:2822:17
SUMMARY: AddressSanitizer: heap-use-after-free .../src/vm.c:2822:17 in mrb_vm_exec
```

Freed by `mrb_stack_extend` → `stack_extend_alloc` → `mrb_realloc_simple`, reached
from the recursion inside `Bad#<=>`.

**RISC-V QEMU** — runs to completion, **exit 0**. Without sanitizer instrumentation
the stale write lands inside the old, still-mapped stack allocation, so nothing
traps. Task spec §4.2/§9 accept the native ASan run as the authoritative evidence.

`PASS = native ASan shows the UAF at vm.c:2822 in mrb_vm_exec`

## Note on compiler sensitivity

This row's reproduction depends on the compiler caching `&regs[a]` across the call.
Nothing in the C standard *requires* that — a build that reloads
`mrb->c->ci->stack` after the callback returns writes to the *new* stack and the bug
silently does not fire. The pinned host toolchain does cache it, which is why the
build config matters as much as the trigger here.

Rows 8, 13 and 15 are more robust in this respect: they hold the stale pointer in an
explicit C local across a loop, not in a compiler-chosen temporary.
