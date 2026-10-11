# FFmpeg at the 9.0.1 pin: we are NOT out of bugs, and what it takes to get them

**The short version.** FFmpeg's corpora hold 8 cases, 3 live at the pin. That is not a
ceiling and it is not a blocked road. In `n9.0.1..master`, upstream itself labels

| upstream's own `Fixes:` label | commits |
|---|---:|
| out of array access | 47 |
| out of array read | 29 |
| heap buffer overflow | 12 |
| out of array write | 10 |
| stack buffer overflow | 2 |
| stack buffer underflow | 1 |
| **spatial, total** | **101** |
| use after free | 3 |
| double free | 1 |
| **temporal, total** | **4** |

Those are upstream's classifications, not our keyword guesses, and they are the right two
classes for this study. The work left is per case — content-based liveness at the pin, then a
reduction — not a blocked dependency.

## Correcting this file's own earlier claim

An earlier version of this note said *"the binding constraint is the port's configuration, not
the search"*, and dismissed candidates because their code is not in the image this port builds
(5 decoders of 2,299; `HAVE_THREADS 0`; no hwaccels). **That reasoning is wrong for corpus
cases, and it is withdrawn.**

Every existing FFmpeg case is a **reduction** — `case.c` plus the shared driver, fidelity tier
*model-consumer, real allocator*. The heap-arms bundle says so of its own fixtures: *"they never
execute the upstream consumer"*. A reduction compiles its own consumer and calls the real
allocator, so **whether a decoder is enabled is not a precondition for making a case from a
defect in it.** The 101 spatial and 4 temporal labelled commits are all available this way.

What the configuration actually decides is the *stronger* fidelity tier — running the real
upstream consumer — which is a separate and more valuable goal, not a gate on this one.

## What a candidate does need

Three questions, in this order, because each is cheaper than the next:

1. **Is it live at the pin, by CONTENT?** `n9.0.1` is a release-branch tag that cherry-picks, so
   ancestry proves nothing. `79e10e5196` (*avcodec/dovi_rpudec: bound num_x/y_partitions*,
   "Fixes: out of array access", named finder) was the most promising subject in the whole
   population and the pin **already carries** both `VALIDATE` lines — not live. Checking content
   first avoids reducing a defect that is already fixed.
2. **Can the mechanism be expressed in a reduction that runs on our target?** This is where some
   candidates really do fall out, for reasons that are about the target and not about effort:
   - **big-endian only.** `b8b8d43935` (*avcodec/h274*, "Fixes: use after free") is real and live
     in the pin's source — `av_freep(ctx)` frees the context and the next line dereferences it
     through `c->buf` — but the line sits under `#if HAVE_BIGENDIAN`, and we are little-endian.
     No reduction can make it happen here.
   - **needs real threads.** `ead4378652` is entirely `ff_thread_await_progress` calls, and the
     domain has `HAVE_PTHREADS 0`.
   - **hwaccel private data is NOT such a reason.** `0661ef6bb3` and `d344929552` move an unref
     of `hwaccel_picture_private`; a reduction allocates that pointer itself, so the ordering
     defect is expressible without any hardware acceleration. Those two stay candidates.
3. **Is a trigger constructible?** Some are fuzzer states that resist hand-derivation.
   `e723ebf0e2` (*avutil/avsscanf*, "Fixes: stack-buffer-underflow") **is live** — the pin has the
   unmasked index at `avsscanf.c:442`, where `uint32_t x[128]` with `MASK 127` is written as
   `x[(z=(z+1 & MASK))-1]`, so `z == 127` indexes `x[-1]`. Reaching it needs `a` at 126 or 127
   and `z` at 127, since `LD_B1B_DIG` is 2. An instrumented build of the pin's own
   `avsscanf.c` — reporting at the site rather than trusting a sanitizer to see a four-byte
   stack write — records **0 hits** over 8 input shapes × lengths 1…2,600, and 0 even for `1.5`.
   The upstream report is a fuzzer finding and its input is not public. Recorded as a **triaged
   live defect without a trigger**, which is the state most of the mruby ledger is in. Note what
   it would test: the object is a **stack** array, so no allocator arm bounds it — that authority
   is the compiler's.

## Decoder widening: measured, cheap, and worth doing for the OTHER reason

Both probes cross-build with **zero** refused translation units, which was the entire variance
in an earlier estimate of mine that said 3-6 days:

| | baseline | +h264/hevc | +vp9/av1/vvc |
|---|---:|---:|---:|
| cross-build | — | rc=0 | rc=0 |
| `code_len` | ~2 MB | 5.9 MB | 7.9 MB (ceiling 256 MB) |
| translation units | 219 | 254 | 295 |

The decoders are verified present by symbol (`ff_h264_decoder`, `ff_hevc_decoder`,
`ff_vp9_decoder`, `ff_av1_decoder`, `ff_vvc_decoder`), not merely by `CONFIG_*`, and the safety
scan reports `0 hit(s) in 716502 instructions`. Image size is two orders of magnitude under the
ceiling and the build is one run, not a project.

So widening is cheap. It buys no *reduction* that we could not already write — but it is the
only way to reach the stronger tier where the **real upstream consumer** executes the defect,
which is worth more per case than a reduction is. That is the reason to do it.

## Where to start

The 101 spatial-labelled commits, newest first, checking content-liveness at the pin before
anything else. The four temporal ones are already triaged above: one big-endian-only, one
thread-only, two (`0661ef6bb3`, `d344929552`) still open and reducible.
