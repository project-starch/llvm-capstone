# WAMR executes WebAssembly in a Capstone domain

Stage 4 of the ladder returns `0x5741002A`. The 0x2A is the 42 that
`i32.const 7; i32.const 35; i32.add` computes, so the interpreter runs its opcode
loop and the runtime is up end to end: init, load, instantiate, call. WAMR is the
third nested allocator in the study after SQLite and MicroPython, and the first
that is an interpreter with its own GC heap.

Branch `capstone-wamr`. Nothing on `capstone-bootstrap`.

## Two defects, and neither fix alone is enough

All four cells are stage 4 under QEMU, same source, same flags apart from the one
variable:

| | computed goto | switch dispatch |
|---|---|---|
| **no align fix** | cause 24 | cause 4 |
| **align fix** | cause 1 at an unrelocated pc | **42** |

### 1. The label stack is under-aligned (patch 0004)

`wasm_interp_classic.c` builds the label stack by casting the operand stack's end:

```c
frame_csp = frame->csp_bottom = (WASMBranchBlock *)frame->sp_boundary;
```

`sp_boundary` is a `uint32 *`, so 4-byte granular. A `WASMBranchBlock` is made of
pointers, which here are 16 bytes and need 16-byte alignment, so the first
capability store into the label stack is misaligned. The patch pads
`max_stack_cell_num` until `param + local + max_stack` is a whole number of
pointer-sized cells; those cells are counted in `all_cell_num`, which is what sizes
the frame, so nothing overflows. Exactly one site performs the cast.

### 2. The computed-goto dispatch table is not relocated

`core/config.h` enables `WASM_ENABLE_LABELS_AS_VALUES` for any GCC or Clang. The
resulting static table of `&&label` addresses is not relocated for the domain's
load slide, so the first table-driven dispatch jumps to a link-time address --
cause 1 at pc 0x190ac, a value with no slide applied. Built with 0, the portable
`switch` has no address table. That is now the default in
`build-wamr-silicon.sh`; `WAMR_LABELS_AS_VALUES=1` restores the original, because
the pair is the A/B test for anything that looks like a spill bug.

## The pc named the wrong instruction, for the third time

At the reported 0x1BF24 stands `lhu a0, 0x6(s5)`, a two-byte load.
`capstone-qemu/target/riscv/op_helper.c:1250` raises cause 4 **only** inside
`if (size == 16)`, and for loads and stores alike. A two-byte load therefore
cannot be what faulted, and the `stc` after it can. The cause code identified the
instruction; the pc named its neighbour.

**RETRACTED:** the earlier reading of this same address as "an untagged frame
pointer inside `memset`". `memset` here is beebs' plain byte loop, and the test
module's function has no parameters and no locals, so its length argument was zero
and it could not have faulted at all. That should have been checked against the
memset source before the theory was written down; it took one grep.

Also retracted on the way: a stack-slot scan that flagged 15 slots written with
`stc` and read with `ld`. It cannot distinguish a destroyed tag from ordinary
slot reuse across disjoint live ranges, which LLVM does routinely. Not a finding.

## The control is the part worth keeping

A domain that reports a number proves nothing until the number is shown to follow
its input, so `gen-test-module.py` now takes `--a` and `--b`. Rebuilt as 11 + 88 it
returned **-29**, which read as a failure and was not: `i32.const` takes SIGNED
LEB128 and the generator was writing the operand as a raw byte, so one byte 0x58
means -40, not 88. 11 + -40 = -29. The interpreter had computed exactly what the
module said, including the generator's own mistake. The encoder now emits real
sleb128, and 11 + 88 returns 99.

That miss is better evidence than a clean pass would have been: nothing that
returns a constant can produce -29 from those bytes. Three points now lie on the
line -- 42, 99, -29 -- and all three follow the module.

## Instrument notes, which cost time today

- **`pgrep -x <name>` silently matches nothing** when the name exceeds 15
  characters, and prints a warning to stderr that is easy to miss while the exit
  status reads as "none found". `qemu-system-riscv64` is 21. Use
  `ps -eo args | grep -c '[q]emu-system-riscv64'`.
- **Two QEMU runs overlapped for most of an hour**, because a background script
  that looked dead was still in its retry loop, and both wrote the same log file.
  That produced a string of "boot flakes" that were nothing of the kind. Check
  `ps` for a live run before starting another; the project already requires QEMU
  suites to be serialized, and this is what ignoring it looks like.
- **`--timeout-multiplier` exists** on `run-domain-smoke.py` and is the right
  answer when the host is loaded, rather than retrying.
- A boot that produces **no guest timestamps at all** never reached the kernel and
  is a different failure from a slow one.

## Where it stands

`run-wamr.sh` is the nightly gate. It derives the expected value from the
generated module rather than hard-coding it, so changing the summands moves the
gate with them. Registered in `run-nightly.sh` as `wamr`.

The default build with no environment variables produces an image **byte-identical**
to the one that returned 42, which is what verifies the default flip.

Open: the ladder still has stage 6 and the `BEEBS_TAGCHECK` knob as diagnostics
rather than tests; nothing exercises WAMR's EMS allocator under load yet, which is
the reason WAMR is in the study at all.
