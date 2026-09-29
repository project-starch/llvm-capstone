# The wmem corpus against the chunk port — predictions

Registered and pushed **before any domain image of this run was built**. The machine-readable
oracles are the `sublet-chunks` arm added to each `case.json` in the same commit.

## Why re-run

On 2026-09-21 the region-granular port caught **12 of 13** (`../20260921-qemu/`). Case 12 was the
recorded non-detection, the only case whose lifetime ends by an individual `wmem_free` into the
block allocator's recycler rather than by a pool reset.

The chunk port landed on dev in `a34caaedb1bc`. It gives every chunk of the block allocator a region
of its own, so a chunk free is a revoke. It is the port's default build (`WM_CHUNKS=ON`).

## Found before registering

**The corpus did not build on dev.** The chunk port added a `wm_probe` to the port
(`src/shared/backing.c`), and the corpus seam's driver defines its own `wm_probe`, the read probe
every oracle names. The link failed on the duplicate. The port's symbol is renamed
`wm_handback_probe` (it probes a pointer handed back to the allocator), in `backing.c`, `chunks.h`
and patch 0002's one call. The labelled fault site, `wm_widen_probe`, keeps its name. The chunk
port's own runs never built the corpus, which is how it went unseen.

**The runner judged a build against the wrong oracle.** `run-defects.py` took every protected
oracle from the `sublet` arm, which describes the region-granular port. It now reads the domain
build's `WM_CHUNKS`: ON is judged against `sublet-chunks`, and OFF, or a build from before the
option existed, against `sublet`. A build with no `CMakeCache.txt` is refused. All four states
were tested.

**The native, unprotected arm with the chunk port** (x86-64, `<program> 0 <case>`) completes all
13. So case 12's own check, that the next 16-byte allocation lands on the freed chunk's address,
holds under the chunk port: the case reaches its read.

## Builds

- **Chunk build:** current dev plus the fixes above, `WM_CHUNKS=ON` (the default),
  `-DWM_CORPUS_DIR=<this corpus>`, presets `capstone-domain` and `linux-guest`.
- **Control build:** the same tree with `WM_CHUNKS=OFF`, the region-granular hooks. Every hunk of
  patch 0002 is guarded, so the patch is applied and inert. It differs from the chunk build in that
  option alone.
- The compiler and emulator are those of the ports' runs (clang `3979abd8`). The guests boot a
  private, `e2fsck`-repaired copy of the rootfs, because the shared one carries ext4 errors.

## Predictions

| cells | build | mode | predicted |
|---|---|---|---|
| cases 0-12, N = 1 | chunk | spatial | **13 / 13 complete** |
| cases 0-11, N = 1 | chunk | sublet | **fault at the labelled read probe, cause 24**, as on 2026-09-21. Their pool is BLOCK_FAST, which the chunk port leaves unchanged |
| **case 12**, N = 3 | chunk | sublet | **fault at the labelled read probe, cause 24.** The free revokes the chunk's region. The reissue at the same address is a new region, so the registry's stale name no longer reaches it |
| case 12, N = 3 | control | sublet | **completes**: the recorded non-detection, one option apart |
| case 12, N = 1 | control | spatial | completes |
| all 26 chunk-build arms | chunk | `--negative-control` | **every arm FAILS**: the input record is refused before any case runs |

So the corpus goes from 12 / 13 to **13 / 13** under Sublet. The case-12 pair, chunk against
control, is the attribution: one build option apart, same program, same guest.

## What would refute it

- Case 12 completing on the chunk build, or faulting anywhere but the labelled read probe.
- Case 12 faulting on the control build: the attribution to the chunk port would then fail.
- Any of cases 0-11 changing outcome on either mode.
- The negative control letting any arm pass.
