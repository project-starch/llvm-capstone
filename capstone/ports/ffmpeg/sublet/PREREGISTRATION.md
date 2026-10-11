# A Sublet port of FFmpeg's own pools — pre-registration

Written and pushed **before the first ported line**, because `capstone/tests/port-effort.py` states
the rule it enforces: the counting rule "has to stand before the first ported line and must never
be adjusted at the result".

## What exists, and what this adds

The app port's pool arms (`FFAPP_POOL=0|2`) do not port FFmpeg's pools, they **substitute** them:
buffer-pool patches 0001 and 0002 divert every pool payload into the buffer-pool port's own
`pool-allocator.c` (2048 fixed exact-size blocks in a separate host region). `pool->alloc` is never
called, custom allocators are refused (`ff2_fail 125/126`), and there is no per-pool handle, so a
pool's teardown frees its entries one at a time. Those arms and every result built on them (cases
36–38, the replay, PoisonCap) are **left untouched**.

This port is a third arm, `FFAPP_POOL=sublet`: FFmpeg's pools keep their own policy and take their
storage from the level below, the way SQLite's lookaside takes a block from memsys5
(`ports/sqlite/sublet/README.md`).

| operation | this port |
|---|---|
| storage from the level below | `__capstone_sublet_malloc_linear` on the app's Sublet heap: the heap keeps its handle, the pool gets the block LINEAR |
| a new entry | `sublet_carve` from the pool's current block |
| `av_buffer_pool_get` of a free entry | `sublet_take`: the entry's slot keeps the handle, the caller gets an alias of exactly the pool's size |
| the buffer's last unref | `sublet_give`: one revoke, the entry is free again, FFmpeg's LIFO freelist unchanged |
| the pool's destruction | the heap's free of each block: **one revoke per block, every entry dies** |

## The classification rule, fixed now

The same vocabulary as every other port: `hierarchy|metadata|workaround` × `allocator|application`,
one line per hunk in `<patch>.classes`.

- **hierarchy** — taking a block from the heap, carving an entry, the take on a get, the give on a
  return, the heap's free of a block at destruction, and any signature change that carries them.
- **metadata** — anything moved out of memory that is handed away or revoked.
- **workaround** — only for a compiler or RTL defect; predicted zero.
- **level** — `allocator` for `libavutil/buffer.c`, `buffer_internal.h` and `refstruct.c`;
  `application` above their interfaces. **H1 predicts zero application lines.**

## Predictions

**Phase A — `AVBufferPool`** (`libavutil/buffer.c`, `buffer_internal.h`).

- **A1.** Fixtures on the new arm: 11 `RETURN b00001`, `LEN 64`; **12 FAULT temporal** (read after
  the buffer's return); **13 FAULT temporal** (read after the entry went to a new owner);
  14 `FAULT oob`. On the heap arms 12 and 13 complete, because a pool never calls `free`.
- **A2.** Behaviour unchanged: mpeg4 M1–M5 reached, 30 frames, 0 hash mismatches against the native
  reference, and the flipped-input control fires in the same boot.
- **A3 — a prediction AGAINST H4, registered because it is the interesting case.** Upstream already
  keeps an `AVBufferPool` entry's bookkeeping — `BufferPoolEntry`, including the freelist link
  `next` — in its own allocation, outside the payload. Nothing the discipline forbids lives in
  handed-away memory, so **Phase A should be mostly hierarchy: metadata < hierarchy.** If H4 is a
  law, this is where it should fail.
- **A4.** A pool's destruction costs one revoke per block it took from the heap, not one per entry.

**Phase B — `AVRefStructPool` and refstruct objects** (`libavutil/refstruct.c`).

- **B1.** 15 **FAULT temporal**; 16 **FAULT oob** (the byte below a refstruct object is no longer its
  header, because the header has left the object); 17 **FAULT temporal**: a stale pointer handed to
  `av_refstruct_unref` faults on its own authority before any header is looked up, so the live owner
  keeps its reference.
- **B2.** Behaviour unchanged, as A2.
- **B3 — the H4 case.** Upstream keeps a refstruct object's `RefCount` **in** the allocation, just
  below the user's pointer, and a pool's freelist link in that header. Both must leave. **Phase B
  should be metadata-dominated: metadata > hierarchy.**

## Scope, and what is deliberately not attempted

- Pools with a custom allocator (`av_buffer_pool_init2`, or an `alloc` other than
  `av_buffer_alloc`/`av_buffer_allocz`) keep upstream's path. In this build none exists: every such
  caller is a hwaccel or hwcontext, which the minimal configure disables.
- `av_buffer_pool_buffer_get_opaque` returns NULL for a ported pool's buffers; its only callers are
  hwcontext files.
- The Sublet heap now holds the pools' storage too, so its granted region may need to grow; that is
  sizing, not policy, and is reported rather than predicted.
- QEMU only (**Q-11**).

## Addendum, 2026-09-29: v2, registered before v2's first build

A claim audit of the A+B run found that A4's evidence could not fire. The `destroy-revokes` counter
bracketed the heap's own, per-file revoke count around `__capstone_sublet_free_linear`, which by
that function's code is always one revoke plus one per buddy merge. It also showed that refstruct's
pool end was NOT one revoke per block: `pool_free_entry` took and then gave back every free entry,
one revoke each, in a file the counter could not see. v2 changes the port, and measures instead
of inferring.

- **0002, `pool_free_entry`**: an entry already given back is not given back again at the pool's
  end. Its slot is dropped (`ff_sublet_entry_end`), and the block's revoke at `pool_free` ends it
  with every other entry. A `free_entry_cb` still gets an alias, through a new handle that is
  dropped at once.
- **0001**: `av_buffer_pool_init(0, …)` no longer collides with the "not ported" sentinel. A
  comment that said the records are freed after the block revoke now says when they are.
- **`ffsublet.c`** reports its own file's revoke counter; `destroy-revokes` is gone.

Predictions, on the same compiler, emulator and configure as the A+B run:

- **A4/B4 — the pool's end, counted across FFmpeg's own uninit.** New counting fixtures 20
  (`AVRefStructPool`) and 21 (`AVBufferPool`) make 8 entries, return them, then end the pool. They
  read the pool layer's file counter and the heap's across each phase. The pool layer's revokes
  are **8 across the returns, which shows the counter can move, and 0 across the end**: marks
  `1400800` and `1500800`. v1's refstruct would have read 8 across the end.
- **Everything else as before.** M1-M5 MATCH with the flip control firing on both arms; fixtures
  11-17 as registered, N = 3; poolstock 20 and 21 print the heap's numbers and mark `0xE00F0`.
- **Reported, not predicted:** each run's heap figures (`FFAPP-HEAP`, including `peak-live`) on
  both arms, since the pools now take their storage from the heap and the heap's region is sized
  for it; and `ends` / `end-takes` per stage.
