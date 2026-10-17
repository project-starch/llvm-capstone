# vp2 flat lifetime table — the module's CSR surface and the encoding control

Companion to `vp2-kit-2026-10-17/` (the kit builder) and to the runtime lane's
`2026-10-17-vp2-corpus-on-silicon.md`. This folder holds the two records that say the flat-mode
module is safe to put on vp2 silicon: the CSRs it touches, and the proof that the two CSRs vp2
lacks are not in the flat binary at all.

## Why a compile-time define and not a module parameter

vp2 has neither `scapctl` (0x5c0) nor `urevavail` (0xcc0); a `csrr` of either is an illegal
instruction in every mode, and the module read both at context creation. With a runtime parameter
both encodings sit in one binary and the only guarantee is "the branch is not taken" — which cannot
be checked by inspecting the module, and which depends on the compiler not keeping an
`asm volatile` under `if (0)`. `-DCAPSTONE_VM_FLAT_TABLE=1` makes the flat build not *contain*
them. For a trap that kills a context at creation, absence beats an untaken branch.

## The encoding control

Both modules built for the board kernel (6.4.14) from one source, differing only by the define:

| build | `0x5c0` | `0xcc0` | `0x5c1` | 16-bit (RVC) insns | warnings |
|---|---:|---:|---:|---:|---:|
| flat (`-DCAPSTONE_VM_FLAT_TABLE=1`) | **0** | **0** | 10 | 0 | 0 |
| paged (no define) | 1 | 1 | 13 | 0 | 0 |

The paged row is the positive control: it proves the grep finds these encodings when they are
there, so the flat row's zeros are a reading and not a silent miss. `0x5c1` (srevroot) is present
in both and must be — it is how the table root is published. Its count differs because the flat
build has no `select_root`/`restore_root` pair around the removed `urevavail` read.

    OD=<buildroot>/host/bin/riscv64-buildroot-linux-gnu-objdump
    "$OD" -d capstone_vm.ko | grep -cE 'csr[a-z]*[^#]*0x5c0'
    "$OD" -d capstone_vm.ko | grep -cE 'csr[a-z]*[^#]*0xcc0'
    "$OD" -d capstone_vm.ko | grep -cE 'csr[a-z]*[^#]*0x5c1'
    "$OD" -d capstone_vm.ko | grep -cE '^\s+[0-9a-f]+:\s+[0-9a-f]{4}\s+[a-z]'   # RVC

## The module's CSR surface, enumerated from source

| CSR | access | sites | on vp2 |
|---|---|---|---|
| `0x5c1` srevroot | read + write | `select_root`, `restore_root` | exists, S and M RW; the write flushes the pipeline |
| `0xcc0` urevavail | read, once, at context creation | the older-processor probe | **does not exist** — paged builds only |
| `0x5c0` scapctl | **read only**, once, at context creation | the profile word; bit 8 is exact-bounds | **does not exist** — paged builds only |

The module never **writes** `0x5c0`: one access in the whole file, a read. Under option (iii) the
exact-bounds answer is false, so the flat build sets `c->exact_bounds = false` and loses nothing.
Custom instructions used: `.insn r 0x5b, 1, {0x24, 0x50, 0x53}` (0x53 is CSRETIRE).

## U outside a context never has capabilities

The record that this case does not arise, since on silicon such a path would have no table:

    CROSS=<buildroot>/host/bin/riscv64-buildroot-linux-gnu-
    "${CROSS}objdump" -d capstone-vexec \
      | grep -cE '\b(ldc|stc|movc|cincoffset|scc|lcc|shrink|delin|cjalr|csrunv|csretire)\b'   # -> 0
    "${CROSS}objdump" -d capstone-vexec | grep -cE 'csr[rw]+[^#]*0x(5c0|cc0)'                 # -> 0

Zero, and by construction rather than by luck: `build-adapter.sh:18-22` builds `capstone-vexec`
with `${CROSS_COMPILE}gcc`, the plain Linux cross compiler, not the Capstone clang. The launcher's
own code cannot contain a capability instruction, and every domain instruction runs inside the
context.

## Not yet done

`run.py` 41/41 and bundle 5 13/13 on the flat QEMU kit, the paged build as the null control of what
was not changed, and a collection row (moving the retired count out of the table changes collection
triggering, so it cannot ride on `run.py`'s pass). Then a context's whole 1 MiB table dumped after a
run, with the allocation/retirement sequence, for the RTL lane to feed as `.data` into an arm whose
readings are QEMU's. No module hash goes to the board lane before those pass.
