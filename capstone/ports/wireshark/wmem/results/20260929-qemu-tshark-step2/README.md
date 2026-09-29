# The wmem chunk port inside real tshark: step 2 (capstone-qemu, 2026-09-29)

**Question.** Step 1 ran the chunk port on synthetic traces and directed fixtures, with no packet
dissected. Does the same ported allocator run inside the real tshark app port, dissect real
captures unchanged, and close the wmem lifetimes the sublet arm leaves open?

**Answer: yes, on every pre-registered prediction, against a matched control.** Predictions:
`../../PREREGISTRATION-tshark-step2.md` and the `chunks` rows and `sublet 13` of
`app/host/safety-expect.txt`, pushed in `68ed711eca82` before the first chunks boot.

## The arm

`TSAPP_HEAP=chunks` is the sublet arm with one difference: tshark's `wmem_allocator_block.c` is the
chunk port's, the source the replay harness tests, plus patch 0007's block-size macro. Beneath it
are the port's own `chunks.c`, unchanged, and `app/src/tsapp-wmem-chunks.c`, which lends blocks
LINEAR from the Sublet heap. BLOCK_FAST, the packet scope, is the sublet arm's. The sublet arm,
rebuilt from the same tree, is the matched control. Of the objects both arms link, only
`wmem_allocator_block.o` and `tsapp-heap.o` differ; the latter only appends the chunk counts to the
exit line. One boot-level difference: the chunks arm's predicted faults ran one fixture per boot, the
control's three returning fixtures in one boot.

## Verdicts

| | predicted | result |
|---|---|---|
| **T1** build | gates pass, the control names `__capstone_region`; the ported file differs from the harness's only by the block-size macro | as predicted: `diff` shows exactly 0007's `#ifdef`; the chunk port is linked into the chunks arm only |
| **T2** fixtures | 1-10 as the sublet arm; 11 FAULT temporal, 12 FAULT oob, 13 FAULT temporal; the control returns 11-13 | **13 of 13 as predicted on chunks, 3 of 3 on the control** |
| **T3** dissection | the sublet arm's verdicts on dhcp, dns_port, http, arp, dns-ooo, ntp and four flips | **as predicted on both arms**: stdout and stderr MATCH on all but ntp, whose stdout DIFFERS by the known 24-line timezone gap; all four flip controls FIRE; M1-M5 REACHED and MATCH |
| **T4** counters | the chunk file's revokes = reset_revokes + close_revokes + retires, every run | holds in all 16 chunks runs, stages included (e.g. dhcp: 102 = 4 + 4 + 94). A consistency check by construction; the ablation arm shows it can fail (below) |
| **T5** node budget | first boot measures dhcp; later oracle boots hold floor(60000 / (1.5 × S_max)) runs | dhcp S = 14,117; S_max = 14,856 (dns_port), so 2 per oracle boot. No boot exhausted the pool (no `cap_rev_tree` assertion in any log; the grep finds one in an older, known case). The rule covers oracle boots only: the chunks stages boot ran 5 images and spent 46,042 nodes, above the rule's 40,000 margin and below the 65,536 pool |
| **T6** cost of not coalescing | reported, not predicted | every capture opened the same 4 BLOCK blocks. The heap's live-object peak is 5 above the control's on every capture and on M2-M5 (M1, which opens no block, is equal). Heap nodes are 13-29 above the control's on the captures, 9-25 on the stages; the chunk layer adds 1,374-1,507 nodes per full run |

**Why +5 is metadata and not blocks.** `wm_meta_alloc` has three callers and nothing frees their
allocations: one slab of 256 chunk headers at a time, the index once, and one slab of 256 block
headers at a time. So the metadata objects number 1 + 1 + ceil(H / 256), for H live headers. Every
capture's H lies between 513 and 768 (dhcp: 726 = 4 opens + 722 splits), so there are 5. Linear
blocks count in `peak_live` as malloc'd blocks do, since both come from the heap's shared carve, so
+5 leaves a block difference of 0.

**What T6 could not show.** These captures are small. Four allocators each needed only their first
1 MiB block on either arm, so "4 opens" had no way to reveal accumulated unreused free space. A
long capture is what would test that.

## The lifetimes this closes

| fixture | sublet (control) | chunks |
|---|---|---|
| 11 BLOCK reset, stale read | RETURN `b0015b`: the new scope's byte | FAULT temporal |
| 12 BLOCK neighbour write | RETURN `c000ee`: the neighbour overwritten | FAULT oob at the printed target |
| 13 BLOCK chunk free, block live | RETURN `d00020`: upstream's free-list link now covers the chunk's first byte | FAULT temporal |

Fixture 13 is the lifetime this port adds over every other arm of this app. **Its attribution, and
T4's positive control:** the ablation arm (`TSAPP_WMEM_ABLATE=1`, the chunk free's one give stubbed
out, registered in `312ada7cf012` before it was built) returns `d000a0` on fixture 13, the chunk's
own byte, because nothing revoked it. Its exit line reads `retires=1 revokes=0`, so the identity
fails there, as registered. Against the chunks arm it differs in `chunks.o` alone; the block
allocator object differs only by the length of its embedded source path. On the control, the
stale read returns 0x20, not the chunk's own 0xa0: upstream wrote its free-list link into the freed
chunk, which is the in-band metadata the port moves out.

## Found on the way (all fixed on this branch, none of them the chunk port's)

On dev, the tshark app port no longer built or judged runs:

1. `deps/env.sh` compiled 128-bit builtins that dev's shared soft-float list now also carries, so
   its control link failed on duplicate symbols.
2. musl-capstone's `mmap_shm_level0.c` omitted `__vm_wait`, one of `mmap.o`'s symbols, so any
   pthreads program linked musl's `mmap.o` too and failed on a duplicate `__mmap`.
3. `host/domain-stdout.py` required `LT-RESULT` to be the last line; dev's test host now prints its
   tables after it.

Each is fixed and tested both ways.

## What this does not establish

1. **QEMU only** (Q-11).
2. **Six captures**, each small. A long capture could accumulate unreused free space in a
   long-lived block allocator; these are too small to test that (see T6).
3. **N = 1** per fixture and capture (dhcp ran three times on the chunks arm, identically). Two boots were infrastructure and were retried: the chunks
   stages boot, whose guest stalled before setup finished (no section began), and one sublet boot
   that the runner ended with exit 75. Both attempts are kept in result-lines.
4. **Compiler `3979abd8`.**
