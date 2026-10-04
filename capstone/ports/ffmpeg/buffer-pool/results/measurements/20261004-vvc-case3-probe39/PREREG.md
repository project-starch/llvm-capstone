# PRE-REGISTRATION — probe case 39, the capability arm for pool-repros case 3 (2026-10-04)

**Registered before the domain build and before any boot.** Nothing here is a result.

## Why this arm exists

`bug-corpora/ffmpeg/pool-repros/03_5c66a3ab51_vvc_nonref_output_releases_tabs` is the only one of the
22 upstream cases across memcached/tshark/FFmpeg with **no measured capability arm**. Its `spatial` and
`sublet` arms are `"status": "not written"`, and its own note says why: *"spatial and sublet need a probe
case in `ports/ffmpeg/buffer-pool` (36-38 are taken by cases 0-2)"*. This registers **slot 39**.

This is **not** the `sublet-port` arm (fixtures 46/47). That one needs the FFmpeg *app* port, whose SDK
ABI gate refuses every toolchain on this host for want of the intcap extensions (PR #120 / C-72), and it
**remains a prediction**. The two must not be conflated: slot 39 is the *component* port's
`spatial`/`sublet` pair, which is how cases 0-2 were measured
(`results/measurements/20260925-pool-corpus-dev`).

## What the probe does

`security-tests/shared/pool-lifetime-probes.c`, `test == 39`. It takes two entries from a **simple**
`av_refstruct_pool_alloc(64, 0)` pool — the all-NULL-callback form VVC itself uses
(`vvc/dec.c:387,394`) — fills them with `0xA0` and `0xB0`, releases **both** in
`ff_vvc_unref_frame`'s order (`tab_dmvr_mvf` first, `rpl_tab` second), takes one entry back, writes `61`
into it, and reads through the **stale** pointer whose address matches.

Releasing *two* entries is what distinguishes this from tests 5/6/7, which release one. The free list is
LIFO (`pool_return_entry` pushes onto `pool->available_entries`, `refstruct_pool_get_ext` pops the head),
so the entry handed back is **`rpl_tab`**. Watching only `tab_dmvr_mvf` reads "no reuse" — which a correct
allocator and a nonexistent defect produce identically, and which cost a void native run on 2026-10-03.
So the order is asserted (CHECK 485) *and* `held` is chosen by comparison, keeping the stale access
correct either way.

## Predictions

Run as `security-tests/qemu/run.sh <out> --cases 39 --modes 0,2`, matching the modes the 2026-09-25
bundle used for cases 36-38.

| mode | meaning | prediction |
|---:|---|---|
| **0** | bounds only, no lease | **completed** — no fault; the stale read returns the new owner's `61`; the `Print = Scalar(0xff25000000000027)` setup stage appears |
| **2** | Sublet lease on every pool get, revoked on return | **fault** — exactly **one** capability fault, **cause 24** (`allowed` is `(24, 25)`), at the `ff2_probe_read` label, with the runner's `pc == expected_pc`, and **no** `FF2 return=42044` |

`0xff25000000000027` is `0xff25000000000000 | 39`, the marker `mark(39)` emits.

The runner computes `writing = case in (2, 4, 6, 38)`, so 39 is correctly treated as a **read**
(`site = 0`), and `fault = ... or (mode == 2 and case in (36, 37, 38, 39))`.

## What would falsify this, and what each failure would mean

- **mode 2 completes** → the lease is not taken on a simple refstruct pool get, or the revoke does not
  cover the entry. A real finding about the port, not about the defect.
- **mode 0 faults** → the probe is not a clean reduction; something other than the stale read is out of
  bounds.
- **exit code 485** → the pool did not hand back the last-released entry, i.e. the free list is not LIFO.
  The probe refuses rather than measuring, which is the intended behaviour; it would mean the LIFO
  reading of `refstruct.c` is wrong and the native case 3 bundle needs revisiting.
- **exit code 482** → the live control failed: the entry was not readable while held, so the probe path
  itself is broken and no conclusion may be drawn from mode 2.
- **more than one fault, or `pc != expected_pc`** → the fault is not the labelled access; the cell is
  void regardless of cause.

## Controls

- **Positive control in the same boot pair:** mode 0 must COMPLETE the identical binary. Same
  `domain_sha256` in both modes — the mode is a runtime argument, so the pair differs in exactly one
  thing. The 2026-09-25 bundle shows this works for 36-38.
- **Live access before the stale one:** CHECK 482 reads the entry through the same labelled probe while
  the frame still holds it, so a fault at mode 2 cannot be blamed on the probe instruction.
- **Negative-control note:** the runner raises `SystemExit(1)` on the first non-passing row, so a silent
  pass is not available.

Registered by the lane branch commit that carries this file, before the build.
