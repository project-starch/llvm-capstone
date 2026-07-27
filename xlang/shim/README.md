# shim — the Phase-2 seam

Phase 1 proves each row's defect with a stock toolchain. Phase 2 has to re-run the
same defects against capability-protected allocations. The gap between them is not
conceptual, it is plumbing: **the corpus has nowhere to substitute an allocator.**

Every mruby row (4–15) reproduces by running the stock `bin/mruby` binary on its
`trigger.rb`. The allocate → free → use sequence under test happens entirely inside
the VM, behind an allocator the row cannot reach. Swapping in a capability
allocator would mean editing twelve vendored mruby trees.

This directory closes that gap without touching any of them.

## What it is

mruby already exposes the seam — `mrb_open_allocf()` takes a custom allocator. Only
the stock `main()` never uses it. `mruby_host.c` *is* that `main()`: it opens the VM
through a custom allocator and then runs the same trigger script unchanged.

`<row>/xlang-host` is therefore a drop-in replacement for
`<row>/mruby/build/<build>/bin/mruby`. Today it forwards to the C allocator, which
is byte-for-byte what `mrb_default_allocf` does, so every row behaves exactly as it
does now.

Phase 2 replaces three function bodies:

```c
void *xlang_alloc(size_t size)            { return malloc(size); }
void *xlang_realloc(void *p, size_t size) { return realloc(p, size); }
void  xlang_free(void *p)                 { free(p); }
```

They are three and not one because a capability allocator has to distinguish cases
mruby's single realloc-shaped hook conflates: mint a bounded capability, derive one
for a moved block and revoke the old, revoke outright. Row 4 is the worked example —
its use-after-free is a write through a register-stack pointer cached across exactly
the `xlang_realloc` call that frees the old stack.

## Using it

```bash
./build-mruby-host.sh ../4         # -> ../4/xlang-host
../4/xlang-host ../4/trigger.rb    # same output as bin/mruby
```

The build name is auto-detected (rows 4, 5, 10 call their ASan build `host-asan`;
the other nine call it `host`). Host binaries are gitignored and rebuilt on demand.

### Checking the seam is actually live

At `-O1` the three functions inline into the allocator callback, so they never
appear in an ASan backtrace — a trace alone cannot tell "routed through the seam"
from "mruby used its default allocator". Hence:

```bash
XLANG_SEAM_STATS=1 ./xlang-host any_script.rb
# xlang-seam: alloc=1668 realloc=3 free=1883
```

Non-zero counts are the proof. **Keep this in Phase 2** — it is the check that
fails loudly if a capability allocator is silently bypassed. It cannot print on a
row that aborts under ASan (the process dies inside the VM), so verify on any
non-crashing script; whether the seam is live is a property of the host, not of
the trigger.

## Status

Verified on two rows, chosen as the extremes of the corpus's mruby range:

| Row | mruby | Stock `bin/mruby` | Via `xlang-host` |
|---|---|---|---|
| 4 | 3.x | `heap-use-after-free`, WRITE size 8, `vm.c:1426` in `mrb_vm_exec`, exit 1 | identical |
| 11 | 1.4.0 | `heap-buffer-overflow`, READ size 16, `vm.c:1208` in `mrb_vm_exec`, exit 1 | identical |

The same source compiles unmodified against both, so the API surface used here
(`mrb_open_allocf`, `mrb_load_file`, `mrb_print_error`) is stable across the whole
range the corpus spans.

### Remaining

- **Nine more reproducing mruby rows** (5, 6, 8, 9, 10, 12–15). Mechanical: run the
  build script, then point `run.sh` at `xlang-host`. Each needs its documented
  verdict re-confirmed, since that is the only thing proving the swap was
  behaviour-preserving. Row 7 does not reproduce, so it needs no seam.
- **The riscv64 leg.** This host is built for the ASan target only. The cross
  target needs the same treatment with `riscv64-linux-gnu-gcc` and no ASan flags.
- **Rows 1–2 (Rust/rlua).** Different seam, same idea: Rust's
  `#[global_allocator]`. The allocation under test is `String`'s buffer, freed by
  rlua's `destructor<T>`.
- **Row 3 (Rust→C).** Hardest. The allocation lives inside prebuilt `libpulse.so`,
  so there is no source-level seam at all — it needs malloc interposition
  (`LD_PRELOAD`). This is the same structural blindness that already forces row 3
  onto valgrind instead of ASan.
