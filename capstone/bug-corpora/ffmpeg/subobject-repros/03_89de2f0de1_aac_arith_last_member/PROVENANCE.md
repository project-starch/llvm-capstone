# 89de2f0de1 — reads one element past `last[]` into the next member

## The defect

The AAC arithmetic-coding context reads `state->last[i + 1]` while `i` walks the whole
window. The window length is `2048 / 4 = 512`, which is also the array's length, so the last
iteration reads `last[512]` — one past the end, landing on the next member.

## Upstream defect

- **Fix:** `89de2f0de1`. It grows the array by one element rather than clamping the index: `last[512 /* 2048 / 4 */ + 1]`.
- **CVE:** none assigned.
- **Live at the `n9.0.1` pin: NO — the fix is already in at the pin.**

## The vulnerable code, quoted from upstream

The struct, `n9.0.1:libavcodec/aac/aacdec_ac.h:27-32` (post-fix at the pin; the `+ 1` is
the fix):

```c
typedef struct AACArithState {
    uint8_t last[512 /* 2048 / 4 */ + 1];
    int last_len;
    uint8_t cur[4];
    uint16_t state_pre;
} AACArithState;
```

The consumer, `libavcodec/aac/aacdec_ac.c:57-62`:

```c
uint32_t ff_aac_ac_get_context(AACArithState *state, uint32_t c, int i, int N)
{
    c = state->state_pre >> 8;
    c = c + (state->last[i + 1] << 8);
```

and the fix's own diff, which changes only the declaration:

```diff
-    uint8_t last[512 /* 2048 / 4 */];
+    uint8_t last[512 /* 2048 / 4 */ + 1];
```

**Liveness: a FIX-REVERSAL.** The fix is already in at the pin, read from the pinned
source rather than from ancestry — `git merge-base --is-ancestor` has called backported fixes live
before, so it is not used here. Two-sided, so the probe is known to fire:

- `n9.0.1:libavcodec/aac/aacdec_ac.h` contains the fixed `last[512 /* 2048 / 4 */ + 1]` — **1 occurrence**
- the same file contains the pre-fix `last[512 /* 2048 / 4 */]` — **0 occurrences**

The case reconstructs the **pre-fix consumer shape against the shipped allocator**, which is this
tree's convention and not a weaker kind of case: it is stated at
`../../memcached/allocator-repros/README.md:132-135`, and most cases in this tree are fix-reversals.
Liveness is **recorded, never required**; requiring it is what left this cell nearly empty, and that
inference was retracted on `dev`.

**Why this is a sub-object crossing and not an ordinary overflow.** Index 512 of a
`uint8_t[512]` sits at byte offset **512**, which is 4-aligned and is exactly where `last_len`
begins — so the one-byte read takes that member's **lowest byte** and shifts it into the
arithmetic-coding context. The whole struct is **one** allocation: `AACArithState ac` is a member of
`AACDecContext` (`libavcodec/aac/aacdec.h:156`), which is `avctx->priv_data`.

**Reachability note:** the USAC/xHE-AAC arithmetic-coded path of the AAC decoder.

## What is real here, and what is reduced

**Real:** the allocator. `av_refstruct_allocz` over the pinned `libavutil/refstruct.c`, where
`refstruct.c:109` is `av_malloc(size + REFCOUNT_OFFSET)` — header and payload in **one**
allocation. So the fact the case turns on, that the crossed region shares an allocation with its
neighbour, is a property of FFmpeg's allocator and this struct's layout, not of the driver.

The case **asserts** the layout rather than assuming it, so a compiler or platform that laid the
members out differently would fail the case rather than quietly measure something else.

**Reduced:** the struct is declared exactly as the header has it, at its real widths, and
`ff_aac_ac_get_context` is cut to the one term that forms the address, `state->last[i + 1] << 8`.
The rest of that function is arithmetic on values already read and cannot move the access.

## What the run establishes, and what it does not

**Establishes:** the defect is real and reproduces from the upstream fix differential, measured
two-sided with the control (fixed) arm run first.

**Does not establish** reachability upstream, anything about silicon, or the Capstone, PoisonCap,
CheriBSD and ASan readings. Those arms are **declared predictions** in `case.json` and are *not*
measured for this case. Measuring the Capstone arm needs a probe case in
`ports/ffmpeg/buffer-pool/security-tests` — the seam cases 0-2 use — and measuring ASan needs
`results/20261005-native-subobject/asan-probe.c` extended, with its positive control still firing.

**An arena caveat that bears on the Capstone and CheriBSD predictions.** This corpus's driver hands
the port's `av_malloc` **one** arena (`shared/driver.c` calls `ff2_memory_init`), and
`ports/ffmpeg/buffer-pool/src/shared/metadata-allocator.c:26` carves it by bumping a cursor,
returning a raw interior pointer and rounding every request up to 64 bytes;
`__builtin_capstone_cap_shrink` appears only in `src/capstone-domain/payload-capabilities.c:57`, on
pool *payload* blocks. So on this harness there is **no per-allocation bound on the struct at all**,
and a completion would be weaker evidence than "the bound is the whole allocation". The sub-object
claim does not rest on that — the crossing is interior by construction, which the case asserts on
the offsets — but the arm must not be read as a measured per-allocation bound.

**N = 1 per cell.**
