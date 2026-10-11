# Lifetime controls: parent, sibling, successor

Not cases, and not the arm controls one directory up. These are Wireshark's row of the
pre-registration `docs/plans/2026-10-11-nested-lifetime-controls.md`. The corpus shows that patch
0001 rejects stale accesses; these controls show that it leaves PERMITTED accesses alone. They drive
wmem's real `block` allocator through its own API: a fresh `wmem_allocator_new(WMEM_ALLOCATOR_BLOCK)`,
`wmem_alloc`, and `wmem_free`, which is the release operation patch 0001 revokes at
(`wmem_block_sublet_free`). `block_fast` is not used, because its free is a no-op.

| program | what it does | predicted, protected | predicted, release |
|---|---|---|---|
| `00_control_parent_block` | allocates A (64 bytes) and a guard, frees A, then reads and writes A's former range through the allocator's own reference to the block; on the protected build, also through `generation` | completes; the read returns what the allocator left | completes |
| `01_control_sibling_intact` | allocates A and B next to each other in one block, frees A, then reads B, checks it byte for byte, overwrites it and reads it back | completes; B intact | completes; B intact |
| `02_control_successor_reuse` | allocates A and a guard, frees A, allocates at A's size until wmem returns A's address as C (printed, and required to be equal), then writes and reads C | C == A; completes | C == A; completes |
| `03_control_successor_stale` | 02's sequence exactly, then reads A's whole former range through the OLD A pointer at the corpus's labelled read (`wm_probe` / `wm_defect_probe`), after the ready mark | **fault, cause 25, at `wm_defect_probe`** | completes and reads C's bytes |

`03` is the positive control. It differs from `02` by exactly one thing, the stale read, and its
two arms differ only in patch 0001. The successor control runs as two programs so that the stale
read is a separate run: on the protected build its fault ends the process.

## Which reference is "the parent" (control 00)

* **`via=block`**: the `wmem_block_hdr_t *` on `allocator->block_list`. It is the capability the
  system allocator (virtual mallocng) returned for the 8 MiB block (`wmem_block_new_block`). Every
  consumer function of the allocator works through it: chunk headers and free-list links are
  reached by address arithmetic from the block. Patch 0001's `wmem_sublet_release` also hands
  upstream's free `block + offset`, which is this capability, not the caller's. Control 00 moves it
  to A's offset. It does so on both builds.
* **`via=generation`**: patch 0001's `block->sublet.generation`, A's DIRECT parent lifetime.
  `wmem_sublet_issue` derives every object from it with CDERIVE, and `wmem_free` ends the object
  with `CREVOKE(generation, object)`. This path exists on the protected build only.

Before writing, control 00 checks what the allocator itself left in the range, reading through the
block reference:

* A's chunk header is marked free and carries A's length (80 = 64 + the 16-byte header).
* The free-list links in A's first 32 bytes both name A's own chunk, so the recycler holds A alone.
  These are read as pointer loads, not at a labelled probe.
* Bytes 32..63 still hold A's pattern, because the allocator writes only the links into a free
  chunk.

The range is then overwritten and read back. Nothing is allocated afterwards, because the overwrite
destroys the links on purpose.

## The labelled accesses

Every PERMITTED access is one byte at a time through `wm_ctl_load` / `wm_ctl_store`
(`lifetime.h`, labels `wm_ctl_load_insn`, `wm_ctl_store_insn`). The stale read uses the corpus's
own `wm_probe` (`shared/driver.c`, label `wm_defect_probe`), so its attribution is the one the
corpus arms use. Each access prints an `access` line naming the reference, the address, the length,
the probe and the call site (`case.c:N` or `lifetime.h:N`) BEFORE it runs, and a `done` line after.
A fault is therefore named by the last `access` line that has no `done`.

Each program refuses (exit 75) when the wmem it links is not the build it claims. The protected
build is recognised by the patched `wmem_allocator.h`, the only header in the build that includes
`<capstone/capability.h>`. The program then checks the bounds wmem handed out: bounded to the
request on protected, the whole block on release.

## Build and run

The port builds these exactly as it builds the corpus, from this directory:

    cmake --preset capstone-application -S <repo>/capstone/ports/wireshark/wmem -B <out> \
        -DCAPSTONE_SDK=<virtual SDK> -DWM_CORPUS_DIR=<this directory> [-DWM_SUBLET=ON]

`run-lifetime-controls.py` runs the four programs on one arm in one boot of the virtual platform
(`capstone/runtime/virtual/run-staged.py`: one hart, exact bounds). Each program runs as its own
process under `capstone-vexec`. The runner resolves a fault's pc to the labelled instruction from
the image's own symbols, and judges each program against the prediction above. A program with no
result is an ERROR, never a match. The record of 2026-10-11 is
[`../../results/2026-10-11-controls/`](../../results/2026-10-11-controls/README.md).

This directory adds no case to the corpus. The CMake glob and `shared/run-defects.py` both scan
only `NN_*` directories, so neither reaches `controls/lifetime/`.
