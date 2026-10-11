# The FFmpeg pool corpus against the Sublet port of FFmpeg's own pools — predictions

Registered and pushed **before any image of this run was built**. The per-cell predictions are
`../../runners/sublet-port-expect.txt`, and the `sublet-port` arm in each `case.json`, in the same
commit.

## Why a new arm

The corpus's protected Capstone arm is the buffer-pool port's probe cases 36-38. Those run against
that port's **substitute** allocator (`pool-allocator.c`), so FFmpeg's own `buffer.c` never ran
under protection there.

The Sublet port of FFmpeg's own pools (`ports/ffmpeg/sublet`, on dev since `584268d1179f`) is now
the real thing: FFmpeg's `AVBufferPool` takes its storage LINEAR from the Sublet heap, and a
buffer's return to its pool is a revoke.

This arm runs each case's `case.c` **unchanged** against that libavutil, in the FFmpeg app port's
domain. `src/capstone-domain/ffapp_corpus.c` stands in for `shared/driver.c`: same pool, same
calls, and the case prints its own verdict line.

The domain has no argv, so each case and variant is an image of its own: fixture
`40 + 2 * case + fixed` (`build-domain.sh`, `FFAPP_CORPUS_DIR`).

## Arms

- **poolsublet:** the Sublet port of the pools.
- **poolstock:** the same patched source with the port's macro at 0 (upstream's pools on the same
  heap), the matched control.
- Each case runs **with the defect** (upstream's code before its fix) and **with the fix**, the
  corpus's own `fixed` switch.
- Each boot runs the fix's image first and the defect's last: a fault ends the emulator.

## Predictions, N = 3 per cell

| case | poolstock, defect | poolstock, fix | poolsublet, defect | poolsublet, fix |
|---|---|---|---|---|
| 0 af_join | VERDICT DEFECT-REPRODUCED | VERDICT FIXED | **FAULT**, cause 24, at `case.c:85/86/89` | VERDICT FIXED |
| 1 h264_refs | VERDICT DEFECT-REPRODUCED | VERDICT FIXED | **FAULT**, cause 24, at `case.c:48/49/52` | VERDICT FIXED |
| 2 vidstab | VERDICT DEFECT-REPRODUCED | VERDICT FIXED | **FAULT**, cause 24, at `case.c:33` (the stale write) | VERDICT FIXED |

The two sides of each prediction mean:

- **Completing.** The case's own check holds: `capstone_main = 0`, and the verdict line is the one
  its native arm prints.
- **Faulting.**
  - It happens after the case's `case=<c> arm=buggy` line and before any verdict line.
  - Its pc must map, through the image's own line table, to a line that dereferences the stale
    pointer. The compiler may load it once for all of them.
  - Where the fault line reports `value_hi`, it must be non-zero: a revoked capability, not an
    integer.

The poolsublet fix column is what shows the port is not faulting on anything else: the same case,
its storage reused the same way, with the reference kept or the pointer dropped.

h264_refs is run here as its reduction. Its real code (FFmpeg's h264 decoder) was not run, because
that needs a malformed stream.

## What would refute it

- A poolsublet defect cell completing, or faulting at a line that does not dereference the stale
  pointer.
- Any fix cell faulting, on either arm.
- A poolstock defect cell not printing DEFECT-REPRODUCED. The case would then not have created its
  reuse, and the poolsublet fault could not be attributed to it.

## Correction, appended after the run (2026-09-29); no prediction changes, and the registered text is at `60491a5b4ad8`

"So FFmpeg's own `buffer.c` never ran under protection there" (above) is wrong.

- The buffer-pool port's probe cases 36-38 do run FFmpeg's `buffer.c`, with hooks that serve the
  pool's payloads from that port's own allocator, whose leases carry the revocation.
- What had not run under protection is FFmpeg's pools **ported**: storage from the Sublet heap, and
  the pool's own code revoking a buffer at its return.
- The predictions are unaffected.
- A second sentence above is inexact. "The poolsublet fix column ... its storage reused the same
  way" holds for cases 1 and 2. In case 0's fix the output keeps its reference, so the storage is
  never reissued (`reuse_same_address=0`).
