# FFmpeg at the 9.0.1 pin: triage by reachability first, and what it yields

**Question.** The FFmpeg corpora hold 8 cases, 3 of them live at the pin. How many more
spatial or temporal defects are there in the version we compile?

**Answer: one solid live spatial defect and one weak one — and the binding constraint is
the port's configuration, not the search.**

## The filter order was the problem

The earlier triage (`ffmpeg-live-defect-triage.md`) sieved 1,788 commits down to 15 on
**subject wording**, and its filter 1 named spatial wording as a *disqualifier*. Wording is
the wrong first filter in both directions: it drops defects whose subject is terse, and it
admits prose matches that are not defects at all. Three of the candidates it would keep are
ASan/MSan annotations, and one — `e5eb7581dd` — matched on "parsed against the **stale**
context" in a commit whose own message says the checks fired and the fix only stops an error
being logged per slice.

Filtering by **reachability** first is precise, costs nothing, and cuts far harder:

| | commits |
|---|---:|
| population `n9.0.1..master` | **1,912** |
| touch a translation unit the port actually compiles (219 TUs) | **111** |
| …with h264 + hevc added (254 TUs) | 149 |
| …with vp9, av1 and vvc as well (295 TUs) | 169 |

A 94% cut from configuration alone, and the survivors are few enough to **read**, which is
what produced the triage below. The 111 are mostly feature work; about six are candidate
memory defects.

## What the port's configuration decides

Read from the generated `config.h`/`config_components.h` of the built image, not assumed:

| | |
|---|---|
| `HAVE_THREADS`, `HAVE_PTHREADS` | **0** |
| hwaccels enabled | **0** |
| `HAVE_BIGENDIAN` | **0** |
| decoders enabled | **5 of 2,299** (h263, mpeg4 and three trivial ones) |

Four candidates die on these lines alone, which is why this check belongs **before** any
liveness test rather than after it:

- `ead4378652` h264_direct — the whole change is `ff_thread_await_progress` calls, and its
  comment says the two fields are "decoded by different threads".
- `0661ef6bb3`, `d344929552` — both move `av_refstruct_unref(&…hwaccel_picture_private)`;
  with no hwaccel that pointer is never allocated.
- `b8b8d43935` avcodec/h274 — **"Fixes: use after free"**, a real one: `av_freep(ctx)` frees
  the context and the next line dereferences it through `c->buf`. The pin has the faulty
  order, so it is live **in source**. The line sits under `#if HAVE_BIGENDIAN`, and
  `ff_h274_hash_freep` is not in the image at all (0 symbols). Live in source, absent from
  our binary.

## Liveness is decided by CONTENT, and it disqualified the best-looking candidate

`n9.0.1` is a release-branch tag that cherry-picks, so ancestry proves nothing.

- `79e10e5196` avcodec/dovi_rpudec, *bound num_x/y_partitions*, **"Fixes: out of array
  access"** with a named finder — the most promising subject in the whole population. The
  pin **already carries** both `VALIDATE` lines (dovi_rpudec.c:585-586). **Not live.**

## What survives

| commit | class | state |
|---|---|---|
| **`e723ebf0e2`** avutil/avsscanf | **spatial** | **LIVE.** `Fixes: stack-buffer-underflow` |
| `9fc8c785e2` avutil/encryption_info | spatial-ish | LIVE, but needs the caller to pass `num_key_ids > 0` with `key_id_size == 0` |
| `2c2f6e96e3` avutil/hdr_dynamic_metadata | — | live, but uninitialised output bytes: outside this study's spatial/temporal scope |
| `0a6f759027` avformat/utils | — | the author's own message says "no current caller could ever have been affected" |
| `9a688ce884` avformat/seek | — | the message calls it "hardening against API misuse; a caller passing such an index is the bug" |

### `e723ebf0e2`, the one real find, and why it has no trigger yet

`decfloat` keeps `uint32_t x[KMAX]` with `KMAX 128`, `MASK 127`, and indexes it as

    if ((a+i & MASK)==z) x[(z=(z+1 & MASK))-1] = 0;

When `z` is 127, `z+1 & MASK` is 0 and the index is **-1**: a four-byte write one element
below a stack array. The fix masks that index. The pin has the unmasked form
(`avsscanf.c:442`), and `avsscanf.c` is in the image, so **the defect is live in the version
we compile**.

**No trigger yet, and the reason is recorded rather than glossed.** `LD_B1B_DIG` is 2, so the
loop runs twice and the site is reached only when `z` equals `a` or `a+1`; the out-of-bounds
index additionally needs `z == 127`, hence `a` at 126 or 127. Reaching that means ~127 leading
zero groups dropped by the normalisation path plus carry propagation advancing `z`. An
instrumented build of the pin's `avsscanf.c` — reporting at the site itself rather than relying
on a sanitizer seeing the write — records **0 hits** across 8 input shapes × lengths 1…2,600,
and 0 hits even for `1.5` and `0.1`. The upstream report is a fuzzer finding
(`Fixes: YrUfC4ZuDBNi`, `ANT-2026-1B7JHPCG`) and its input is not public. Recorded as a
**triaged live defect without a trigger**, the same state most of the mruby ledger is in.

It is also worth noting what it would test: the object is a **stack** array, so neither the
system-allocator arms nor the nested-allocator arm bounds it. It belongs to whatever narrows
stack allocations, which is the compiler.

## Decoder widening: cheap, and it buys almost nothing

Both probes build with **zero** refused translation units, which was the entire variance in an
earlier 3-6 day estimate:

| | baseline | +h264/hevc | +vp9/av1/vvc |
|---|---:|---:|---:|
| cross-build | — | rc=0 | rc=0 |
| `code_len` | ~2 MB | 5.9 MB | 7.9 MB (ceiling 256 MB) |
| translation units | 219 | 254 | 295 |
| reachable spatial candidates | 5 | 8 | 10 |
| reachable temporal candidates | 3 | 7 | 10 |

The decoders are verified present in the image by symbol (`ff_h264_decoder`,
`ff_hevc_decoder`, `ff_vp9_decoder`, `ff_av1_decoder`, `ff_vvc_decoder`), not merely by
`CONFIG_*`. But every one of the seven candidates widening adds is hwaccel private data,
frame-threading state, or integer-overflow hardening — the same three classes the
configuration already rules out. **Widening is a half-day and yields no case**, so it should
be done for a reason other than this corpus.
