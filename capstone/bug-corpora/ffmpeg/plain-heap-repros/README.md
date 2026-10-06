# FFmpeg plain-heap spatial defects: the MALLOC BOUND itself, FFmpeg's not-nested row

The third corpus of its kind, after [`../../memcached/plain-heap-repros/`](../../memcached/plain-heap-repros/README.md)
and [`../../wireshark/plain-heap-repros/`](../../wireshark/plain-heap-repros/README.md), and a sibling
of [`../subobject-repros/`](../subobject-repros/README.md) and [`../pool-repros/`](../pool-repros/README.md)
rather than part of either. Those boundaries are *inside one allocation* and *storage an
`AVBufferPool` handed out*. Every case here leaves a buffer FFmpeg obtained **directly** from
`av_malloc_array` or `av_calloc`, with no inner layer.

**Under this inventory's axis — *who allocated the object* — all four rows are NOT NESTED.** Nothing
sub-allocated the crossed region.

**All four are fix-reversals**, read from the pinned source two-sided. Liveness is **recorded, never
required**; the convention is at
[`../../memcached/allocator-repros/README.md:132-135`](../../memcached/allocator-repros/README.md),
and requiring it is what kept FFmpeg's spatial count at four. Triage and the full candidate
disposition: [`docs/ref/ffmpeg-spatial-defect-triage.md`](../../../docs/ref/ffmpeg-spatial-defect-triage.md).

## The four rows

| shape | cases | upstream | the crossing | ASan |
|---|:--:|---|---|:--:|
| a scan inclusive of a bound the writer is exclusive of | 0 | `d133b4a231` | reads `tkernel[size]`, 4 bytes past `av_malloc_array(size, 4)` | **reports** |
| a shift loop guarding on `j` while reading `j + 1` | 1 | `bcbf3a5630` | reads one element past the format array; the value is **discarded** | **reports** |
| a one-shot mirror that reflects to a **negative** index | 2 | `56309e476a` | reads 4 bytes **BELOW** `av_calloc`'s base | **reports** |
| an allocation sized for one sub-buffer, carved into four | 3 | `495b402f27` | writes 33 bytes past `av_malloc_array(stride, 32)` | **reports** |

**ASan reports all four, measured two-sided** — the fixed arm silent and exiting 0 in every case. That
is the contrast this corpus exists to draw against `../subobject-repros/`, where ASan is blind to all
six of its shapes for the opposite reason: there the crossing stays inside one allocation, so there is
no redzone where it lands.

The runner keys its ASan check on **`heap-buffer-overflow`**, never on the string `AddressSanitizer`.
LeakSanitizer's summary line contains that string too, and on 2026-10-06 a probe in the sibling
sub-object corpus leaked and so read as a bounds detection on *every* arm — the exact inverse of the
truth. The sanitiser build is `-O0`, because at `-O1` a discarded out-of-bounds read (case 1's, by
upstream's own description) can be optimised away and the silence would be the compiler's, not ASan's.

### Two rows worth singling out

**Case 1 is the one where the crossed value is never used.** Upstream says so itself: *"Fortunately,
the excess element was never actually used, but it still triggers ASAN (and could in theory trigger a
segfault)."* The value read from `formats[nb_formats]` is written to an element the following
`nb_formats--` immediately puts out of range. So `damage` is **0 on both arms** and the *crossing
alone* is the finding — a test keyed to consequences cannot see this defect at all, while a bounds
check sees the read. That asymmetry is the point of the row.

**Case 2 is the only row in the whole inventory that crosses BELOW an allocation's base.** That is not
cosmetic. An upper-bound-only check passes it. And the mechanism that refuted
`memcached/plain-heap-repros/00`'s catch prediction — CheriBSD's `malloc` bounding to the allocator's
*usable size* rather than the request — cannot rescue it either, because no size class extends an
allocation *downward*. It is therefore the most robust catch prediction in the corpus.

## Why the platform allocator, and not the port's `av_malloc`

These cases call the platform's own `calloc`. The buffer-pool port's `av_malloc`
([`metadata-allocator.c:26`](../../../ports/ffmpeg/buffer-pool/src/shared/metadata-allocator.c)) is a
bump/freelist carve out of **one** arena: it returns a raw interior pointer and rounds every request up
to 64 bytes, and `__builtin_capstone_cap_shrink` appears only in
`src/capstone-domain/payload-capabilities.c:57`, on pool *payload* blocks. Routing these cases through
it would leave **no per-allocation bound to cross** *and* would absorb every small crossing in the
round-up — so the reading would be about the arena, not about the defect. The two sibling plain-heap
corpora make the same choice for the same reason.

## What is measured, and what is declared

**Measured:** `native-fix-differential` and `native-detect`, both two-sided, control arm first.
`runners/run-native.sh` exit 0. Result lines in `results/20261006-native-plain-heap/`.

**Declared predictions:** the Capstone, PoisonCap and CheriBSD arms. This corpus has no
capstone-domain runner, as is also true of the two sibling plain-heap corpora.

**The CheriBSD predictions are deliberately CONDITIONAL, and each names its own request size.** That
platform's `malloc` bounds a capability to the allocator's **usable size**, not to the request —
measured in-guest on this host on 2026-10-06: `calloc(1,1)` and `calloc(1,9)` both return length 16,
`calloc(1,17)` returns 32, `calloc(1,8192)` returns exactly 8192. `memcached/plain-heap-repros/00`
predicted a catch without that condition and was **refuted** by precisely this mechanism. So each arm
here records the bytes requested and the distance crossed, and says the reading must be taken in-guest
for *that* request size rather than carried over from the `calloc` table.

**N = 1 per cell.**
