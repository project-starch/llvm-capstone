# A domain past the 4 MiB order ceiling, loaded from CMA (2026-09-25, capstone-qemu)

**Question.** The port has always been capped at a 4 MiB image, and the plan lists that as the
reason the `ffmpeg` CLI and libavfilter are out of scope. Since buildroot `2b8ad05` (pinned by #97)
a domain block beyond 4 MiB is served from CMA instead of failing. Does that actually work for this
port, and does building more into the image change what it decodes?

**Answer: yes, and no.** A **6,266,832-byte** image — h264 and libavfilter compiled in, on the
sublet arm — loads from an 8 MiB CMA-served block and reaches every stage M1–M5. The mpeg4 decode
is unchanged: 30 frames, **0 hash mismatches** against the native reference, with the flipped-input
control firing on 10 hashes in the same boot.

That control is what makes the MATCH mean something: the comparison is shown able to report a
difference in the same run in which it reports none.

## What ran

- **Image:** `ffapp_m5.dom`, sublet arm, configure adding `--enable-avfilter --enable-filter=join
  --enable-filter=aformat --enable-decoder=pcm_s16le --enable-decoder=h264 --enable-parser=h264
  --enable-demuxer=h264`. `FFAPP_ORDER_CEILING_MB=8`.
- **Guest:** the tshark port's private rootfs, whose module serves a block over 4 MiB from CMA
  (`MODULE-MD5 5696d0fe05f5fd5395a0935a0c7ff302`). **The shared buildroot checkout is still at
  `d04bd83`, before that change**, so the shared rootfs cannot load this image at all.
- **The kernel's own lines confirm both halves:** `Reserved 112 MiB` (the `cma=` this runner now
  passes) and `code size = 6266832, tot_size = 800000` for each of the six domains loaded.
- **Build gates at ceiling 8:** 20 images FIT, the C-50 disassembly scan found 0 hits in 1,018,765
  instructions, and the negative link control fired.

## The C-50 gate's positive control — patch 0003 is load-bearing, not droppable

Added 2026-09-28, prompted by a warning from the compiler lane that a C-50 gate pass is no longer
evidence now that `4c407f9456d4` has landed, because a fixed compiler does not emit the shape at
all.

**That warning does not apply to these builds, and checking why was the point.** They use compiler
`3979abd8e9a3`, which **predates** `4c407f9456d4` (`git merge-base --is-ancestor 4c407f9 3979abd8`
is false). The shape can still be emitted here.

A pass was nevertheless weaker than it looked, for a different reason: **0 hits is equally
consistent with patch 0003 removing the instance and with the shape never arising in this build.**
That had never been separated. So M5 was rebuilt with patch 0003 removed and nothing else changed:

```
HIT ff_mpv_alloc_pic_pool   955d8: addi a2, sp, 0x8  ->  955e8: sd zero, 0x0(a2)
1 hit(s) in 361737 instructions      (with 0003: 0 hits in 1,018,765)
```

**The gate fires, in the very function patch 0003 patches** (`libavcodec/mpegpicture.c`,
`ff_mpv_alloc_pic_pool`). So on this compiler the patch is load-bearing, the gate is proven able to
report the opposite, and the Phase 0 pass above is real evidence rather than an unproven check.

What this does **not** settle: whether `4c407f9` makes patch 0003 droppable. That needs a build on
a compiler at or past the fix, which is not built here. Note the trap in answering it — **on a
fixed compiler this control cannot fire**, so "the gate passes without 0003" would there be
uninformative, and droppability has to be argued from the emitted code or from a recorded pre-fix
disassembly. `c50-control.txt` holds this one.

## Sizes measured on the way (the reason this matters)

| | bytes | |
|---|---|---|
| `libavcodec.a` `h264*` members | 3,229,648 | 23 members — what forces the block size |
| `libavcodec.a` `mpeg*` members | 264,912 | 14 members |
| `libavfilter.a` (`join`, `aformat`, core) | 461,736 | carries `buffersrc`, `buffersink`, `framepool` |
| image, level0, no libavfilter | 7,164,624 | |
| image, sublet, with libavfilter | 6,266,832 | smaller: its heap is granted, not `.bss` |

So h264 is the expensive component and af_join is affordable, and the sublet arm is the one to size
a block from.

## What this does not establish

1. **Nothing decoded h264 or ran a filter.** They are compiled and linked in; the workload is still
   the 320x180 mpeg4 clip. This measures the image budget and proves the additions are inert, not
   that either component works in a domain.
2. **QEMU only**, and on this host's guest. No board arm (**Q-11**, **M6** unchanged).
3. **N = 1.** One boot, six sections.
4. **The default ceiling is still 4 MiB.** `FFAPP_ORDER_CEILING_MB` must be raised deliberately, and
   only together with `cma=` *and* a CMA-capable guest module. Raising it alone produces an image
   that fails to load, which reads as a stall rather than as a refusal.
5. **This boot under-reserved CMA and got away with it.** `REGION_MB` was 0 through a shell
   arithmetic error in the runner (`$(( 4 + ${FFAPP_POOL:+4} ))` expands to `4 + ` when `FFAPP_POOL`
   is unset), so the sublet arm's 4 MiB heap region per run was not counted: 112 MiB was reserved
   where 136 MiB was implied. It passed because there was slack. Fixed, with every case tested; it
   is recorded here because a reservation that is too small fails only sometimes, and then looks
   like a stall.

## Files

- `result-lines.txt` — the runner's verdicts, the guest's CMA and module lines, and the build gates.
- `SHA256SUMS` — the images, the loader, and the compiler/emulator/guest identities.
