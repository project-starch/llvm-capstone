# Class 3 (reuse-not-free) on memcached's real bipbuffer: revoke-on-free is blind, as predicted (2026-10-03)

**Question.** Hooking memcached's small core allocators began with reading the bipbuffer — the
logger's per-watcher output buffer, which also carries `items.c`'s `lru_bump_entry` records. It is not
the class expected. **Does revoke-on-free catch a lifetime that ends with no allocator event at all?**

**This is the first MEASUREMENT of a prediction the project already wrote down.**
[`xlang/reuse-not-free/README.md:90-97`](../../../../../../xlang/reuse-not-free/README.md) states it
plainly: *"does revoke-on-free catch it? **It should not** — nothing is freed, so revocation is never
triggered… this class is the one where **our own revoke-on-free is also blind**, and only explicit
lender-driven revocation at the contract point works."*
[`docs/ref/paper-bug-inventory.md`](../../../../../docs/ref/paper-bug-inventory.md) says the same in
its cheapest-high-value-work list, and
[`docs/ref/table6-cheri-vs-capstone-explained.md:174`](../../../../../docs/ref/table6-cheri-vs-capstone-explained.md)
assigns class 3 the mechanism **`sync (R)`** — *"the borrow is revoked at the `step()` that reuses the
buffer — the contract point — regardless of free"* — with `R` defined in the legend at `:51-57` as the
revocation primitive. So the result below confirms documented expectation; it does not overturn it.

**Pre-registration.** Fixture 19 and all five predictions were pushed in `4a30b3f34dbc` at 17:21:41,
before the build source was staged (17:22:00) and the images built (17:22:51).
`git diff 4a30b3f34dbc 707e7a8a93de` over the fixture and the expect file is **empty** — neither
changed after registration. What was registered was a predicted **blindness**.

## Verdict

**Every arm returns, on all five arms, in two independent runs.**

| arm | fixture 19 — class 3 | matched in-boot control, **after** it, same arm |
|---|---|---|
| `level0` | RETURN `130015b` | fx17 RETURN, fx18 RETURN |
| `shrink` | RETURN `130015b` | fx17 RETURN, fx18 RETURN |
| `sublet` | **RETURN `130015b`** | **fx17 FAULT** cause 24, **fx18 FAULT** cause 24 |
| `slabsublet0` | **RETURN `130015b`** | **fx17 FAULT**, **fx18 FAULT** |
| `slabsublet1` | **RETURN `130015b`** | **fx17 FAULT**, **fx18 FAULT** |

Exit status corroborates every RETURN: `91 = 0x5b = 130015b & 255`. **Five boots per run across four
images** — `slabsublet0` and `slabsublet1` are **one** image (`3700bc7b98b56de9`) selected by
`MC_SLAB_SUBLET_MODE`, and for fixture 19 they produced byte-identical output at the same address, so
they are two configurations rather than two independent measurements. Three distinct heap behaviours.

## Why this is class 3, and how the absence of a free is established

Read from the port's own source tree (`bipbuffer.c`, `bipbuf_request:77-89`, `bipbuf_push:91-107`,
`bipbuf_poll:151-180`):

    bipbuf_new      malloc(sizeof(bipbuf_t) + size)                 -- ONE allocation, flexible data[]
    bipbuf_request  return (unsigned char *)me->data + me->a_end    -- a pointer INSIDE it
    bipbuf_poll     void *end = me->data + me->a_start;
                    me->a_start += size;
                    ... me->a_start = me->a_end = 0;  return end;   -- CURSORS ONLY

`bipbuf_poll` performs no free. It advances cursors and, when the buffer empties, resets them to zero —
so the next `bipbuf_request` hands out the very same bytes. The consumer's polled pointer **was never
freed, is still tagged, and is still in bounds of the single `malloc`**. Only the data's identity
changed. That is class 3 in
[`docs/design/sharing-bug-taxonomy-and-novelty.md`](../../../../../docs/design/sharing-bug-taxonomy-and-novelty.md).

**How "no free occurred" is established — and how it is NOT.** An earlier version of this file cited
the fixture's printed `frees-performed=0` as evidence. **That was wrong: it is a string literal inside
the `printf` format at `src/mcapp-safety.c:664`, with no counter variable anywhere in the file.** It
cannot read anything but `0`. Withdrawn. The same applies to the printed `polled-same=1
reissued-in-place=1`: by the time they print, the guard at `:656-660` has already returned
`MCAPP_MARK(n, 0xE0019)` unless both are 1, so at the print site they are tautological.

What actually establishes it:

- **The guard, not the print.** `:656-660` exits `0xE0019` (exit status 25) if the buffer did not
  recycle in place. Every `out-19.txt` reads `exit=91`, so the guard passed on real values.
- **The disassembly.** In the built image, `bipbuf_request`, `bipbuf_push` and `bipbuf_poll` contain
  **no `auipc` and no external symbol reference at all** — they cannot call any function. Positive
  control: `bipbuf_new` contains `auipc` → `<malloc>`. Case 19's own compiled range references only
  `bipbuf_new/request/push/poll`, the touch helpers and rodata — **no `free`/`realloc`/`calloc`**.
- **Limit of that evidence, stated:** the scan of musl's stdio path is depth-1. So the supportable
  claim is *"no free on the fixture's own path, and the three bipbuffer functions make no calls at
  all"* — **not** "zero frees in the process".

## The matched in-boot positive control

A clean RETURN on a protection arm looks exactly like a dead arm, a mis-built image, or a boot that
never ran the fixture. So a control ran **in the same boot, on the same arm, immediately after 19**:

- **fixture 17 — a LOAD, like fixture 19** — faults on `sublet`/`slabsublet0`/`slabsublet1`, at
  `pc=0xc02478c4` → `mcapp_fix_touch+0x14`, the same helper and offset fixture 19 uses. It is the same
  stale-pointer-then-reuse shape, but its release reaches the runtime allocator.
- **fixture 18 — a STORE** — faults on the same three arms.
- **fixture 1** ran first in all boots and returned `100001`.

Fixture 17 is the access-type-matched one and it is why the control is airtight: a *load* through
released storage faults on that arm, in that boot, while a *load* through recycled-in-place storage
returns. (The direction never mattered mechanically — cause 24 is raised by `cscincoffset` on an
untagged operand, in `capstone-qemu target/riscv/op_helper.c:746-749`, **before** either access; the
identical instruction word sits at `+0x14` in both `mcapp_fix_touch` and `mcapp_fix_poke`. But
matching it removes the question.)

**And a matched pair inside this bundle pins the variable.** `shrink` and `sublet` have identical
bounds width and differ *only* in revocation. They split RETURN/FAULT on fixtures 17 and 18, and both
RETURN on 19. So revocation is the discriminating variable, localised by a pair rather than argued
from a ladder.

## What the capability shows, and what it does not

On the per-object-bounding arms:

    record1  cursor=c86cc020  bounds=[c86cc000,c86cd020)  len-from-cursor=4096
    record2  cursor=c86cc020  bounds=[c86cc000,c86cd020)  len-from-cursor=4096

Identical cursor and identical bounds: both records are sub-allocations inside one `malloc`, and the
arms narrowed that `malloc` to exactly its own size (`0x1020 = sizeof(bipbuf_t) + 4096`, the same width
on `shrink`, `sublet` and the slab arms — `sublet` is **not** the narrowest; only `level0` differs, at
the 64 MiB arena). So the alternative explanation "the allocation was never on a bounded heap" is
refuted: it was bounded, exactly.

**But the blindness is temporal, not spatial.** Both records occupy the *same bytes*, so even
hypothetical per-record bounds could not separate them. The identical bounds **illustrate** that
per-object bounds on the outer allocation cannot help; the reason nothing fires is that **no
revocation event ever occurs**.

## What this does and does not say

- **It fills the revoke-on-free arms' cell for class 3, and the answer is negative.** No class-3
  specimen had a Capstone measurement: `paper-bug-inventory.md:211` records row 3 "Have (measured
  both)" = **0**, both xlang specimens are "REPRO — Capstone/CHERI columns TODO", and
  `xlang/reuse-not-free/corpus.json:18` says "Neither has a Capstone or CHERI column yet". The one
  candidate counterexample, `tests/cheri-baseline/row3_reuse.c`, was run on **CHERI only** — its
  `RESULTS.md:148-149` cites the paper's column rather than measuring ours.
- **It does NOT close row 3's Capstone TODO.** The mechanism the project assigns class 3 is **`(R)`**,
  lender revocation at the contract point. What is measured here is that **revoke-on-free** — which is
  what all five arms implement — is blind. The `(R)` measurement is still unbuilt, and the remedy this
  bundle points at (making `bipbuf_poll` a revocation event) *is* `(R)`.
- **The notation defect worth one line.** `sharing-bug-taxonomy-and-novelty.md:76` gives class 3 a bare
  **✓** with no mechanism letter, where rows 4–7 and 10 carry `(H)`, `(L)`, `(L)`, `(L+R+S)`, `(U)`, and
  that file has no legend. A reader of that table alone would expect the heap arms to catch this. The
  gap is in the notation, not in the project's claim — which `table6` and `xlang` both scope correctly.
- **It is not a defect report.** `bipbuffer.c` has exactly **two** commits in upstream's entire history
  — the initial import and a Solaris compiler-warning fix. The fixture reproduces the allocator's
  **documented, intended** recycling contract. That is why class 3 is underreported: nothing is logged
  because nothing is locally wrong.
- **ASan was not run here.** Its blindness to this shape is structural and is asserted from the
  taxonomy, not measured in this bundle.
- **QEMU only**, and **N = 2** for fixture 19 (two independent runs over the same four images, both
  `130015b` on all five arms). Fixture 18's faults also now reproduce on a second image
  (`sublet f61c16a447dcd794` here against `1e3df2b1edefeb31` in
  `../2026-10-03-qemu-plain-heap-contrast/`).
- **The port's own oracle did not run** — `run-safety.py` needs `capstone-vm`, which this rootfs cannot
  support. The guest command is that runner's, replicated over serial/9p with `mc-harness` cross-built
  for the guest; the comparison against the expect file was done by hand.

## What the next step is, and what it costs

Hooking the bipbuffer means giving `bipbuf_request`/`bipbuf_push` a per-record sublet region and making
`bipbuf_poll` **revoke** it — i.e. implementing `(R)` for this allocator. The existing slab adapter is
the pattern and the yardstick: `src/slab-sublet/` is **137 lines** across three files with 14 hook
functions, plus one patch. A bipbuffer adapter is a smaller surface — three call sites, one buffer
kind, no page/chunk split. Measured build cost for the port as it stands: libevent 199 s once, arm SDKs
14 s, safety images 9 s, slab image 19 s.

Also recorded while reading, as the residual of a defect already built: `e779381` (fixture 18) took
**three** commits to fix — `e779381`, then `3bc58f6` *"prior fix missing continue, still broke"* and
`3a66ebe` *"fix null watcher more"*, all ancestors of our pin. The final form's loop condition is
`while (!w->failed_flush && (skip_scr = bipbuf_request(w->buf, ...)) == NULL)`, so the unfixed version
reads the freed watcher twice more — including `w->buf` handed **straight into the bipbuffer
allocator**. Fixture 18 reduces only the write half; that second shape is available and not built.

### Two instrument traps, for whoever re-checks this

1. `objdump -d` on a `.dom` without `-m riscv:rv64` prints **nothing but**
   `can't disassemble for architecture UNKNOWN!` — `e_machine` reads as something else entirely. A
   "no calls found" result from that is a vacuous zero.
2. Even with the right `-m`, **a call in this ABI is not `jal`.** It is `auipc`+`addi` followed by
   capability `.insn` words (opcode `0x5b`). Grepping `jal|jalr|call|tail` finds zero and means
   nothing. Grep for `auipc` and for the `# <addr> <symbol>` comments.

Files: `result-lines.txt` (every line above).
