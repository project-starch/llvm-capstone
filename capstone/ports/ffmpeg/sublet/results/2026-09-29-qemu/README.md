# The Sublet port of FFmpeg's own pools (capstone-qemu, 2026-09-29)

**Question.** The app port's Sublet heap revokes what `free` returns. FFmpeg's pools never call
`free` on a buffer they get back: they keep it on a LIFO freelist and reissue it. So a pointer kept
past `av_buffer_unref` or `av_refstruct_unref` still reads and writes the pooled memory. Can the
pools be ported so a return is a revoke, keeping their policy and FFmpeg's output unchanged?

**Answer: yes, on every pre-registered prediction, with a matched control arm for each fault.**
Predictions:

- `../../PREREGISTRATION.md`: `ce65f1da59ab`, before the first ported line, and its v2 addendum,
  `e258f12ff365`, before v2 was built;
- the control arm's predictions, in `app/host/safety-expect.txt`: `3873e2cf09a6`, before that arm
  was first built.

This folder holds three runs:

- **Phase A alone** (0001).
- **v1**, Phase A+B (`3873e2cf09a6`). A claim audit upheld its fault fixtures, its control and
  its inertness, and refuted its evidence for A4 (below).
- **v2** (`e258f12ff365`), which changes the port where the audit found it wrong and measures what
  v1 inferred.

## Verdicts (v2)

| | predicted | result |
|---|---|---|
| **A1** | fixtures 12, 13 FAULT temporal; 14 FAULT oob; 11 returns, len 64 | as predicted, N=3, in all three runs |
| **A2/B2** | M1–M5 reached, 30 frames, 0 mismatches, the flip control fires | as predicted: Phase A on poolsublet (the only arm it had), v1 and v2 on both arms |
| **A3** | 0001 mostly hierarchy (against H4) | hierarchy 52, metadata 6 added code lines (36/2 without `#if` guards): **H4 refuted for AVBufferPool** |
| **B1** | 15 FAULT temporal; 16 FAULT oob; 17 FAULT temporal, on the stale pointer's own authority | as predicted, N=3; 17 faults at the probe in `av_refstruct_unref`, before the header lookup |
| **B3** | 0002 metadata-dominated | metadata 84, hierarchy 46 (63/27 without guards): **H4 holds for refstruct** |
| **A4/B4** (v2) | a pool's end gives no entry back; the pool layer's revokes move 8 across 8 returns, 0 across the end | **as predicted, both pool types, N=3**: marks `1400800` and `1500800` |

## Each fault against its matched control

`poolstock` is built from the **same patched source** with `FF_SUBLET_POOLS=0`, so FFmpeg's pools
run upstream's path on the same Sublet heap. It also builds the pool fixtures, which no heap arm
does. Beside the macro, it lacks only what the macro calls:

- `ffsublet.o` is not linked;
- the stage driver does not print the pools' counts (`FFAPP_SUBLET_POOLS`, which only
  `ffapp_domain.c` reads).

Fixtures 1–17 compile identically in both arms (18 and 19 are Track B's, built later). Of the FFmpeg objects, only `buffer.o`,
`refstruct.o` and the three `version.o` differ: the last three by the embedded configure string.
Fixtures 20 and 21 differ by design, since only the port has a pool layer to count.

| fixture | poolstock (port off) | poolsublet (port on) |
|---|---|---|
| 12 read after the buffer's return | RETURN `c000a0`: its own byte, still readable | FAULT temporal at the touch |
| 13 read after the entry went to a new owner | RETURN `d0015b`: the new owner's byte, same address | FAULT temporal at the touch |
| 15 refstruct object read after its return | RETURN `f000a0` | FAULT temporal at the touch |
| 16 one byte below a refstruct object | RETURN (a byte of the in-band header area, inside the pointer's bounds) | FAULT oob: the object's bounds are its 64 bytes |
| 17 stale `av_refstruct_unref` after reissue | RETURN `1100177`: the live owner lost its reference and a third get aliased it | FAULT temporal, in the unref's probe |
| 14 one byte past a pool buffer | FAULT oob (a stock entry is its own heap object) | FAULT oob — not discriminating, and registered as such |

Fixture 17 is the one that matters most. Under stock pools a stale unref silently drops the *new*
owner's reference, and the next get hands the same object to a second owner. Under the port the
stale pointer faults before any header is read. The disassembly shows it: `lbu zero,0(a1)` in
`av_refstruct_unref`, the inlined probe, ahead of the index hash.

## A4: what the v1 evidence was, and what v2 measures

v1 reported `destroy-revokes`: the heap's revoke count, bracketed around
`__capstone_sublet_free_linear`. By that function's own code the count is always one revoke plus
one per buddy merge. `sublet.h` also counts per file, so the bracket could not see a give done by the
pool layer. It measured a code identity, not the pool.

It also hid a real difference. v1's refstruct `pool_free_entry` took and then gave back every free
entry at the pool's end, one revoke each. v2 gives nothing back there: the entry's slot is dropped,
and the block's revoke ends it.

Fixtures 20 and 21 measure this across FFmpeg's own `av_refstruct_pool_uninit` and
`av_buffer_pool_uninit`, reading the pool layer's own file counter and the heap's:

| per pool of 8 entries | pool layer, returns | pool layer, end | heap frees at the end |
|---|---|---|---|
| refstruct, port | 8 | **0** | 14 = 8 records + 2 blocks × (block, block record) + 2 (pool object and header) |
| refstruct, stock | — | — | 9 = 8 objects with their in-band headers + 1 |
| AVBufferPool, port | 8 | **0** | 13 = 8 records + 2 blocks × 2 + 1 |
| AVBufferPool, stock | — | — | 17 = 8 × (record, data) + 1 |

The 8 across the returns shows the counter can move. It reads the same quantity as `gives`: the pool
layer's only revoke is its give. v1's refstruct would have read 8 across the end. Nothing ran v1's
code in fixture 20, but the stage counters across the two builds agree: M5's gives fell from 354 to
330, by exactly v2's 24 ends, and M4's from 19 to 11, by its 8. Fixture 21 cannot tell v1 from v2,
because 0001's end path never gave in either.

So the claim, stated exactly: **at a pool's end the port revokes no entry's storage one by one.**
Each entry's record is still one heap free, as upstream's header or record is. The heap's figures
agree that every block revoke came back UNINIT and was written through: in every stage,
`init − merge = destroyed` (M5: 281 − 270 = 11). That a callback's alias dies with the block is by
the code, not measured: no fixture uses a `free_entry_cb`, and M5's three end-takes are followed
by no stale touch.

End-of-pool frees go DOWN for AVBufferPool (13 against 17: the entries' data needs no free) and UP
for refstruct (14 against 9: upstream kept each header inside its object). The decompositions in
the table are read from the code paths; the fixtures print revokes and merges, not frees.

## What changed in v2

- 0002: the pool's end drops an entry's slot instead of giving the entry back. A `free_entry_cb`
  still gets an alias, through a new handle that is dropped at once. M5's counts: 333 takes =
  330 gives + 3 such end-takes; 24 entries ended without a give.
- 0001: `av_buffer_pool_init(0, …)` no longer collides with the "not ported" sentinel.
- The pool takes no senior handle of its own. The design note in `docs/plans` considered one; the
  heap's handle on each block already covers everything carved from it.

## Costs, stated rather than hidden

- **Allocations.** A refstruct object is two allocations instead of one. Each pool takes blocks of
  4 entries, rounded by the heap to a power of two. M5's live heap objects peak at 187 against
  poolstock's 164.
- **Failed init.** An entry whose `init_cb` fails keeps its carved storage until the pool's end.
- **No thread safety.** The refstruct address index is one global table that plain objects use
  too, so a threaded build needs a **global** lock around it, not the pool's mutex; and refstruct's
  carve of a new entry runs after the pool's mutex is released, so it needs the lock as well
  (0001's carve runs under the mutex). This build is `--disable-pthreads`.
- **Bytes: not measured.** Both arms ran the same heap region. A pool's blocks are 4 entries
  rounded up to a power of two, and nothing here measured what that rounding costs in bytes:
  `peak-live` counts objects (M5: 187 against 164).

## Scope decides H4

`effort.txt` gives the count under two scopes, and with and without guard lines:

- **Counting only the patches**, as every port's `.classes` does, H4 is refuted for 0001 and holds
  for 0002.
- **Adding `ffsublet.c` and `ffsublet.h`** as hierarchy (the level below, where every Sublet call
  lives), metadata is 28% of the pair (v2's files: 110 and 26 code lines).

The wmem port's audit showed the same choice deciding its outcome. Neither scope is the answer on
its own, so both are reported.

## What this does not establish

1. **QEMU only.** capstone-qemu untags a revoked capability on load; the deployed silicon lets a
   stale access retire (**Q-11**). No board arm.
2. **Unchecked on silicon: the bounds of carved entries.** Entries are carved at entry-size offsets
   inside a block, not at naturally aligned addresses, and frame planes are tens of KiB.
   `sublet_heap.c`'s "WHY A BUDDY" note says carving like this can widen bounds on silicon.
   capstone-qemu represents them exactly. Not tested.
3. **One workload.** 30 frames of one mpeg4 stream; the fixtures are directed.
4. **N = 1 for each stage oracle**, and N = 3 per fixture.
5. **Compiler `3979abd8`**, which predates the C-50 fix. The C-50 gate finds 0 hits in every M5
   image, and patch 0003 stays applied.

## Files

- `result-lines.txt`: every verdict, counter and pool-end line, for Phase A, v1 and v2, both arms.
- `effort.txt`: `port-effort.py` over both patches, the guard-free counts, and the two-scope
  table.
- `SHA256SUMS`: every image and input each boot loaded, per run and arm.
