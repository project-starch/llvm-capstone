# The nested layer on the one-operand CREVOKE, 2026-11-11

`zend-cap-mode.c` ends the caller's payload with `heap_end(ptr)` where it called
`heap_end(block, ptr)`: the release names the object alone (CREVOKE, funct7 0x52, rd and rs2
x0; `docs/design/virtual-capstone/isa.md`). `probe-derive-split.c` and `probe-nested.c` moved
with it (`capstone_cap_release`). These are the two probes run on the QEMU that implements
that instruction, each stage its own `capstone-vexec` process in one boot through
`../../../app/run-virtual.py` (its `-cpu` string and module arguments, a pinned load).
Images: `probe-derive-split.dom` `663bfe0f...`, `probe-nested.dom` `499c3817...`, built by
`app/build-virtual.sh probe` and `nested-probe` on an SDK with an 8 MiB stack, line tables in
the musl objects and the corpus codegen gate (`-fno-builtin`).

## probe-derive-split.json -- the primitives

| stage | question | result | 2026-10-26 |
|---|---|---|---|
| 0 | control: carve, write both children, free | exit 0 | exit 0 |
| 1 | geometry | `object_off=0`, header `[obj,obj+48)`, payload `[obj+48,obj+256)` | `object_off=48` |
| 2 | is the header still writable after the payload is released? | exit 0 -- yes | exit 0 |
| 3 | does a released payload deny? | exit 139, cause 25 | exit 139, cause 25 |
| 4 | re-derive over the same bytes (the `AG(cache)` hit) | fresh child works, old alias exit 139, cause 25 | the same |
| 5 | cursor read of a released capability | exit 0 -- does not trap, returns the cursor | exit 0, a bogus value |

Every exit is the recorded one. Two readings moved and neither is the release's: the object
now starts at its parent's base (`object_off=0`: the capability `malloc` returns begins at
the object, where the 2026-10-26 heap's began 48 bytes below it), and a released non-linear
capability keeps its tag and cursor (isa.md: a stale reference loads tagged and faults with 25
at the dereference), so stage 5 reads the true address. The adapter computes the payload
offset from the capability and never assumed 48, and it never dereferences what stage 5 reads.

## probe-nested.json -- the real adapter code

| stage | expectation | exit | cause |
|---|---|---|---|
| 0 | positive control, must complete | 0 | |
| 1 | write through a reclaimed payload alias | 139 | 25 |
| 2 | read through a reclaimed payload alias | 139 | 25 |
| 3 | `AG(cache)`: reclaim, re-issue to a new tenant, use the old alias | 139 | 25 |
| 4 | the intrusive list: reclaim the middle block, walk the neighbours' headers | 0 | |
| 5 | realloc moves the payload: contents preserved, old alias dead | 139 | 25 |
| 6 | the stock arm, revoke off: the same stale use must complete | 0 | |
| 7 | the bound: one byte past the request on the exact arm | 139 | 28 |
| 8 | the system-allocator arm on a cache-absorbed free: must not fault | 0 | |
| 9 | the system-allocator arm on a real free: must fault | 139 | 25 |
| 10 | a double efree: the second reclaim faults at the liveness probe | 139 | 25 |
| 11 | a grandchild the caller derived, handed to efree | 139 | 25, at the use of the grandchild |
| 12 | an interior pointer handed to efree | 134 | `ZEND_CAP_FAIL code=4` |

Stages 0 to 9 exit as `derive-rebase-2026-10-26/probe-seam.json` records them. Stage 0 prints
`issues=2 retires=2 revokes=2`: one release per retire (the 2026-10-26 layer also revoked the
header at block death and printed three; the count is two on unmodified `dev` as well).

**Stage 11 is the one behaviour the instruction change removes, and the probe now states the
new one.** With the two-operand form the reclaim passed the block and the caller's pointer,
and the processor refused a grandchild. `control-dev-probe-nested.json` is that reading, taken
in the same session: the probe as unmodified `dev` has it (image `d13a7dbd...`), on `dev`'s
QEMU (`35529815...`), where stage 11 faults with cause 29 at the CREVOKE inside the reclaim
(`S11 handing a GRANDCHILD to efree (expect a fault at CREVOKE, cause 29)`, then the fault).
Its other twelve stages exit as these do, stage 0 with the same `revokes=2`. The one-operand
release has no parent operand to check against. The transcript here:

    S11 handing a GRANDCHILD to efree (expect NO fault: the grandchild alone ends)
    S11 RESULT: no fault, and the issued payload is still live. The release cannot tell a
                grandchild from the issued child; using the grandchild now (expect a FAULT)
    Cap mem access on revoked capability: ... cursor = 3fa4720090, bounds = (3fa4720090, 3fa47200a0)
    domain fault cause=25

So `efree` of the grandchild returns, a write through the issued payload afterwards succeeds
(the line is printed after it), and the write through the grandchild faults with cause 25 on
the grandchild's own 16-byte bounds. The run ends in that fault, exit 139, by design. What the
heap keeps is the start-of-object test (stage 12) and the liveness probe (stage 10); it cannot
tell a grandchild that starts at the object's start from the issued child. No stale reference
gets through: the grandchild is dead, and the issued child dies at its own `efree` or with its
block.
