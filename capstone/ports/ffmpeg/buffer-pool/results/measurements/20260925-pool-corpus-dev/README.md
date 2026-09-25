# The FFmpeg pool-defect corpus under Sublet, re-run on current dev (2026-09-25)

**Question.** Does the buffer-pool port catch the three temporal bugs of
`bug-corpora/ffmpeg/pool-repros/` (af_join, h264_refs, vidstab)? The corpus README says yes, in its
"Four systems" table. Until this folder no run of it was committed anywhere: the claim lived only
in commit messages (`f5b2a33c758f`, `cd6a2b5dc81a`) and prose.

**Answer, on capstone-qemu only.** In mode 2 (a Sublet lease on every pool get, revoked when the
buffer returns to the pool), each of the three probe cases faults at its labelled stale access with
cause 24. The same domain image in mode 0 (bounds only, no lease) completes all three. There the
pool reissues the same address, and the stale access reads the new owner's data, or for vidstab
overwrites it. Natively, the corpus's own fixtures reproduce all three defects without the upstream
fix and not with it.

| case | bug | mode 2 (Sublet) | mode 0 (control) |
|---|---|---|---|
| 36 | af_join: a reference never taken | fault, cause 24, at `ff2_probe_read` (`0x101a114f0`) | completes (`FF2 return=42044 status=0`) |
| 37 | h264_refs: a reset bounded by the count, not the array | fault, cause 24, at `ff2_probe_read` | completes |
| 38 | vidstab: a pointer parked in a library | fault, cause 24, at `ff2_probe_write` (`0x101a116a0`) | completes |

## What ran

- **The code under test** is FFmpeg 9.0.1's own `libavutil/buffer.c`, with the port's patch
  `patches/ffmpeg-9.0.1-0002-libavutil-pool-payload-lifetime-hooks.patch`:
  - `av_buffer_pool_get` calls `ff2_payload_issue`;
  - `pool_release_buffer` calls `ff2_payload_return`.
  In mode 2 those become `sublet_take`/`sublet_give` (`src/capstone-domain/payload-capabilities.c:56,86-92`,
  `src/allocators/sublet/pool-leases.c:9,13`). The built domain contains the `revoke`.
  It runs in the port's replay and probe domain, on the port's exact-size payload arena rather
  than `av_malloc`.
- **The bugs** are the port's hand-written reductions, cases 36–38 of
  `security-tests/shared/pool-lifetime-probes.c` (lines 285, 327, 358):
  - 36 and 37 contain the upstream defective statement (af_join's `if (j == i)` dedup bound; h264's
    reset loop bounded by the count), and assert that it fired;
  - 38 contains only the vidstab bug's shape, a pointer kept across the buffer's return.
  Sublet faults in all three because the lease died at the last return. What is specific to each
  bug is only how its stale alias arises.
- **The command** is the corpus README's:
  `security-tests/qemu/run.sh <out> --cases 36,37,38 --modes 0,2 --rounds 1`.
  It ran one boot per cell, under the shared QEMU lock; the first attempt timed out waiting for
  the lock and ran nothing.
- **Tree, toolchain and guest:**
  - tree: origin/dev `892bf63d11de`; nothing in the port or the corpus differed at dev's later tips;
  - compiler: `3979abd8e9a3`, dev's codegen (a private build, object-identical to the compiler
    lane's);
  - emulator: capstone-qemu `deb7d75756`, one opt-in switch behind dev's pin;
  - guest: buildroot `d04bd83b13cd`, before the CMA-for-large-domains module change, which a
    0.9 MB domain does not use.
- **One image for every cell:** all six cells ran the same domain image and loader (`SHA256SUMS`).
  The pairs differ only in the mode argument.

## What this does not establish

1. **The board.** Cause 24 here is capstone-qemu reloading a revoked capability untagged. The
   deployed silicon lets a stale data access retire (ISSUES Q-11), so no trap is expected there.
2. **FFmpeg's consumers.** No af_join, h264 or libvidstab code ran under Sublet, and neither did the
   corpus's own `case.c` files, which run natively only. The three bugs have not been run in the
   FFmpeg app port (`ports/ffmpeg/app`).
3. **Fix removal under Sublet.** No fixed arm runs in a domain. With the fix, cases 36 and 37 stop at
   their own assertion instead of completing.
4. **All pool temporal bugs.** The corpus was narrowed to the three this mechanism covers. The fourth
   pool-backed temporal specimen found by the triage (vp9, a refstruct pool) has no domain arm, and
   is a predicted miss.
5. **Repeatability.** One boot per cell today. The September 19–20 runs recorded the same outcomes in
   prose only.

## Files

- `result-lines.txt`: the runner's six verdicts, each boot's stage marker and fault or completion
  line, and the native fix-differential.
- `SHA256SUMS`: the domain image and the Linux-side loader every boot ran.

An adversarial audit re-read these artifacts, re-hashed the binaries, disassembled the domain and
re-ran the native binaries. It confirmed every row above. It also narrowed the claim to the limits
listed here.
