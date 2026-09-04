# Monitor regen is NOT broken (B tree) — the "toolchain gap" does not reproduce

**Date:** 2026-07-25 · **Lane:** B · **Board-free** (QEMU only) · Task: monitor-regen
fast path (`plans/monitor-regen-audit-task-B.md`, "Answer A").

## Bottom line

**The OpenSBI monitor rebuilds from source and boots — in B's tree, with *either*
compiler.** The premise the audit task was written on ("the working `fw_jump.elf` is an
unreproducible prebuilt; regenerating from the current `capstone-c` yields a
boot-hanging image") **does not reproduce here.** Monitor-side work — the domain-boundary
`fence.i` fix, and any large-`.rodata` monitor change — is unblocked as of now, and it is
**not** gated on pinning `508342a`.

## What was run

Full regen each time: delete `components/opensbi/lib/sbi/{sbi_capstone_dom,capstone_int_handler}.c.S`,
regenerate with the Capstone-C compiler, then `make -C buildroot ... opensbi-rebuild`,
then `tests/runtime-qemu/run-smoke.sh` (boot → create domain → call → `retval = 42`).

| # | image | compiler | `.gct` align | `.text` size | boots + domain call |
|---|---|---|---|---|---|
| 1 | fresh regen | `4899cf9` (our tree's pin) | `.align 16` | `0x3fff8` | ✅ PASS |
| 2 | fresh regen | `508342a` (caplifive-system pin) | `.align 4` | `0x1e358` | ✅ PASS |
| 3 | B's resident Jul-19 build | old | `.align 16` | `0x3fff8` | ✅ PASS |

All three print `OpenSBI v1.3 for Capstone` → `Created domain ID = 0` →
`Called dom (1-th time) retval = 42`.

## What `508342a` actually changes

The **entire** `4899cf9` → `508342a` codegen delta for the monitor is **two lines**, one
per generated file:

```
-.align 16
+.align 4
.section .gct
```

That is commit `3780447 "Fixed overly large alignment for gct"`. On RISC-V `.align n` is a
**power-of-two exponent**, so `.align 16` = 2^16 = **64 KiB**, not 16 bytes — and it sits
*before* `.section .gct`, so it pads the **preceding** section. Effect on the linked image:
`.text` shrinks `0x3fff8` → `0x1e358` and the firmware loses ~192 KiB of pure padding
(1,714,320 → 1,517,952 bytes).

**It is a real fix and worth adopting — but it is not a boot fix.** The known-good
prebuilt (`6724bcb3`) has the *padded* `0x3fff8` layout, i.e. the working firmware was
itself built with the "bad alignment" compiler. Alignment was never the boot blocker.

## Correcting the audit's stated evidence

The audit doc reasoned from a codegen signature: good prebuilt `s0–s6`/frame −368 vs
regen `s0–s11`/frame −464, read as "the current compiler allocates bigger frames."
**Both `4899cf9` and `508342a` emit byte-identical register allocation and frame sizes**
here (the only diff is the `.align` line above), so that signature does not distinguish the
compilers. Whatever produced the frame difference, it is not the `4899cf9`→`508342a` gap.

## Scope of the claim (deliberately narrow — do not over-read)

**Proven:** B's tree, B's current monitor sources, QEMU boot + domain create/call, both
compilers.
**Not tested:** the A lane's tree; the A lane's large-`.rodata`-modified monitor sources;
the FPGA firmware build (`caplifive-system`, `PLATFORM=fpga/ariane`) — a *different* build
from this QEMU `fw_jump.elf`.

**Leading hypothesis for the A-lane hang (UNPROVEN — do not treat as established):** the
hang came from the large-`.rodata` **source change**, not from the compiler. Circumstantial
only: a `/tmp/capstone/fw_jump.elf.largero-broken` image exists (Jul-24 16:49). Its md5
(`c7076ed0`) is *not* the `788f8a1a` the audit cites, so the actual hanging image is gone
and this was **not** verified. Cheap test if it ever matters: apply the large-RO change in
B's tree and re-run `run-smoke.sh`. Per "Answer B" the large-RO copy is moving to host
userspace anyway, so this may never need answering.

## State left behind

- Resident `capstone/caplifive-buildroot/build/images/fw_jump.elf` = **restored to the
  status quo** (`2311c0b9`, B's Jul-19 build). Nothing adopted yet.
- Verified-good `508342a` image kept at `/tmp/capstone-b/monitor-regen/fw_jump.elf.508342a-GOOD`
  (`a56492c4`), plus the pre-existing `.c.S` at `*.old-4899cf9` and the Jul-19 image.
- Both submodules clean — the generated `.c.S` are gitignored build products.
- **A lane untouched:** B's `caplifive-buildroot/build/` is a separate tree from A's
  (different inodes), so a B monitor rebuild cannot clobber A's `fw_jump.elf`. Verified A's
  copy is still `6724bcb3` after all three rebuilds. The "shared `fw_jump`" warning in the
  audit doc is **not** true across lanes.

## Build gotcha

`caplifive-system/sw/capstone-c/target/` is **root-owned** (from an earlier root/docker
build), so `cargo run` fails with `Permission denied` on `.cargo-build-lock`. Do **not**
chown or delete it — export `CARGO_TARGET_DIR=/tmp/capstone-b/capstone-c-target-508342a`
instead. Also, the Makefile's `.c.S` rule targets `$(CURDIR)`-absolute paths, so
`make components/...` fails with "No rule to make target"; pass the absolute path.

## Recommended next

1. **Adopt `508342a`** for monitor builds (strictly better: same codegen, −192 KiB of
   padding) after one broader QEMU regression pass, not just `run-smoke.sh`.
2. **Land the `fence.i` domain-boundary fix** (`plans/curried-crunching-gizmo.md`) — now
   unblocked. This is the change that kills the ~2.5 min/rung board cost.
3. Confirm the **FPGA firmware** build (`caplifive-system`, `PLATFORM=fpga/ariane`) rebuilds
   too — that is the one that actually has to change for board throughput, and it was not
   exercised here.
