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
| a buffer sized by one consumer and used by another | 4 | `b3c7ebc1ed` | writes 1 byte past a scratch row sized for the luma plane, copying a **chroma** row | **reports** |
| an allocation that omits the padding its readers require | 5 | `8553e6ef57` | reads up to 64 bytes past an exactly sized `av_buffer_alloc` | **reports** |
| a stream-supplied length consumed without checking the bytes left | 6 | `041d4f010e` | reads past the packet buffer by as much as the stream declares | **reports** |
| a copy sized by the source payload, not the destination | 7 | `8880a174d0` | writes the payload's length into a smaller caller buffer | **reports** |
| a writer given a constant capacity instead of the space left | 8 | `b2df2f4f22` | writes up to 128 bytes at a cursor with less than that remaining | **reports** |
| a transform whose output is a multiple of a declared width | 9 | `16b2049d4d` | writes `lowpass_width * 2` into a narrower plane allocation | **reports** |
| a stride computed before the directive that changes it | 10 |  |  |  |
| a clone sized by the character count of a terminated source | 11 |  |  |  |
| a buffer sized for its items but not the separator joining them | 12 |  |  |  |
| a single-reflection mirror that produces a negative index | 13 |  |  |  |
| an index used unmasked where its sibling paths mask it | 14 |  |  |  |
| a copy length and an allocation taken from unrelated fields | 15 |  |  |  |
| plane lengths from the geometry, read out of the packet | 16 |  |  |  |
| a skip that advances the output cursor instead of the input | 17 |  |  |  |
| a scroll whose last iteration sources the row after the last | 18 |  |  |  |
| a cursor initialised to the count rather than the last index | 19 |  |  |  |
| a buffer whose worst case is smaller than its writer's | 20 |  |  |  |
| a fixed-length read from a frame whose last partition is shorter | 21 |  |  |  |

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

**Also measured — `cheribsd-revocation`, on 2026-10-06** (`results/20261006-cheribsd/`), with
revocation **on** and both platform controls firing in the same boot. **3 of 4 CAUGHT, and the fault
is attributed to the labelled probe in every one of them**; case 1 is **not** caught, which refutes
its own pre-registered prediction. This is the first CheriBSD reading of any FFmpeg corpus.

| case | crossing | CheriBSD |
|---:|---|---|
| 0 | 4 B past a 16 B request | **CAUGHT** — `si_code` 1, `addr`=`pc`=`0x1020ba` = the resolved probe |
| 1 | 1 B past a 24 B request | **not caught** — the capability is 32 long, so offset 24 is inside |
| 2 | 4 B **below** the base | **CAUGHT** — `0x1020de` |
| 3 | 33 B past a 512 B request | **CAUGHT** — `0x102120` |

**Why case 1 is a clean refutation rather than a surprise.** CheriBSD's `malloc` bounds to the
allocator's **usable size**, not the request, and this corpus measured the lengths for *its own*
request sizes in the same boot instead of reusing the `calloc` table taken at 1/9/16/17/8192 for the
sibling corpora — because a size class is a step function:

| request | length | slack |
|---:|---:|---:|
| 16 | 16 | **0** |
| **24** | **32** | **8** |
| 512 | 512 | 0 |

The committed prediction for case 1 said in advance that 24 bytes "is not a size-class boundary, so
the usable size must be read in-guest before this is believed". It was read, and it was 32. This is
the **second** instance of the mechanism — `memcached/plain-heap-repros/00` was refuted by it at
request 9 → length 16 — so it is now measured at two size classes, and case 2 shows it cannot apply
below the base.

**Attribution is established here and is not in the sibling corpora.** They declare their probes
`static` in the header, so each translation unit gets a private copy and `supervise` cannot resolve
the symbol — their rows read *"attribution: not established"*. This corpus **declares** the probes in
`shared/corpus.h` and **defines them once** in `shared/driver.c`, so the fault address can be matched
against a symbol resolved from the ELF independently of the run. The native readings were re-run
after that change and came out byte-identical, so it is inert to everything except attribution.

**Still declared predictions:** the Capstone and PoisonCap arms. This corpus has no capstone-domain
runner, as is also true of the two sibling plain-heap corpora, and PoisonCap is unavailable on this
host.

**N = 1 per cell.**
