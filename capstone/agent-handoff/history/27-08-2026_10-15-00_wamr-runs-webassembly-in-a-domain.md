# WAMR executes WebAssembly in a Capstone domain

Stage 4 of the ladder returns `0x5741002A`. The 0x2A is the 42 that
`i32.const 7; i32.const 35; i32.add` computes, so the interpreter runs its opcode
loop and the runtime is up end to end: init, load, instantiate, call. WAMR is the
third nested allocator in the study after SQLite and MicroPython, and the first
that is an interpreter with its own GC heap.

Branch `capstone-wamr`. Nothing on `capstone-bootstrap`.

## Two defects, and neither fix alone is enough

All four arms are stage 4 under QEMU, built back to back from ONE tree with one
variable changing, through the `WAMR_LABEL_STACK_PAD` and `WAMR_LABELS_AS_VALUES`
knobs:

| | computed goto | switch dispatch |
|---|---|---|
| **no pad** | cause 4, `addr = 0x1022d1e08` | cause 4, `addr = 0x1022d1e28` |
| **pad** | cause 1 at an unrelocated pc | **42** |

Both no-pad arms give the same fault whatever the dispatch, which is what attributes
it to the pad. Both addresses are congruent to 8 mod 16, and 8 is what the layout
predicts here: `param = 0, local = 0, max_stack = 2`, so `sp_boundary = lp + 8` with
`lp` 16-aligned. Each knob has a positive control in the artifact -- `handle_table`
is a symbol in exactly the two `labels=1` images, and the pad's
`neg`/`sub`/`andi 0x3`/`add` sequence appears in exactly the two padded ones. The
padded and unpadded arms have identical captable global counts.

**RETRACTED.** The first version of this table gave cause 24 for the top-left cell.
That came from a log written the previous day by a binary built before patches 0002
and 0003; the image of that name on disk gives cause 4. Read by filename instead of
checked against the artifact under test. An adversarial audit caught it after the
claim had already been committed and pushed.

**Also retracted:** the first run of this 2x2 was not single-variable. The four
images had been built from different working trees and differed by five captable
globals, because `capstone_mcp_*` in `port/capstone_libc_extra.c` were defined
UNGUARDED and landed in every build regardless of `BEEBS_MEMCPY_TAGCHECK`. They are
now behind the same `#if` as their use, and the pad is a build knob so the arms can
come from one tree.

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

The mechanism is now understood and it applies to EVERY capability fault this
emulator reports. `_helper_access_with_cap` in `op_helper.c` is `static` and is not
inlined -- `nm build/qemu-system-riscv64 | grep with_cap` shows it as a local `t`
symbol -- so `GETPC()` inside it returns an address outside the code-gen buffer,
`cpu_restore_state` takes its documented early exit, and `env->pc` is never
restored. The printed pc is the translation block's ENTRY, which in all three runs
happened to be the return address of a `jalr`. The real faulting instruction was +4
in two runs and +12 in the third, so "the next instruction" is not a rule either.

`badaddr`/`tval` are stale on this path: the raise sites never assign `env->badaddr`
while the reporter prints it. One run printed `addr = 0x1022d1e28` alongside
`badaddr = 0x1018ad346`, which is not 4-aligned and so cannot be a 16-byte access.
The `[CAPSTONE] Unaligned cap access (addr = ...)` line is the only trustworthy
address.

What identified the instruction was the cause code, which narrows by WIDTH:
`op_helper.c:1250` raises cause 4 from the capability path only inside
`if (size == 16)`, for loads and stores alike, so the two-byte `lhu` at the reported
pc cannot be what faulted and the `stc` after it can.

Written up for everyone in `ref/HOW-TO-RUN-ON-QEMU.md`. This plausibly explains the
project's two earlier retracted trap-pc conclusions as well.

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

## RETRACTED: there was no residual

An earlier version of this note reported five un-relocated pointers to
`invokeNative` in `.data` and downgraded the claim to "runs this module". Wrong, and
the mistake was reading the FILE as if it were the runtime state. A tag cannot live
in an ELF image, so NO pointer in static data is correct on disk on this ABI; the
compiler synthesizes `__capstone_cap_init` per module and the entry glue runs it
before main. Same job as CHERI's `__cap_relocs` plus `crt_init_globals`, except
theirs is a data table a loader interprets and ours is compiled code.

Counted:

| | pointers in static data | stores in `__capstone_cap_init` |
|---|---|---|
| switch dispatch (works) | 34 `exception_msgs` + 5 `invokeNative_*` = **39** | **39** |
| computed goto | + 256 `handle_table` = **295** | 39 |

Fully covered in the working image. The 256 missing entries are the `&&label` block
addresses, which IS the cause-1 fault. So computed-goto dispatch fails because of a
gap in the cap-init synthesis, not a limit of the ABI: the backend emits
initializers for function pointers, data pointers and string tables, and nothing for
a block address. Closing that would retire the `WASM_ENABLE_LABELS_AS_VALUES=0`
knob.

Third retraction of the day, and the first two share a shape with this one: a stale
log read as current, a trap pc read as an instruction, a file read as memory. In
each case the artifact was real and I asked it the wrong question.

Two defects introduced by this session's own instrumentation, both fixed:
`capstone_mcp_*` were unguarded, and `*res = 0x57410000u | argv[0]` lost the 0x5741
tag whenever the result had high bits set, which is what made the -29 control read
like a crash.


Open: stage 6 and the `BEEBS_TAGCHECK` knob are diagnostics rather than tests;
nothing exercises WAMR's EMS allocator under load yet, which is the reason WAMR is
in the study at all. Every cell of the 2x2 is still N=1, though the alignment
mechanism reproduces across three separate builds.
