# Option (iii): what dropping exact bounds would cost the virtual runtime

**Bottom line.** mallocng needs one change, in one place, and it is cheap: **classify on
`repr_round(n)` rather than `n`, and size the big-path mapping from `repr_round(n)` too.** That
penalises 628 of 131,051 small sizes (0.48%) and none of the sizes an application requests on
purpose, and adds at most 16 pages to a large mapping. **No size-class table change, no slot
geometry change, no alignment change.** What is *not* cheap is the rest: one recorded verdict
flips, one documented sentence becomes false, every CSMINT descriptor in the image path must be
re-checked for representability, and seven runners plus a launcher refusal are wired to the
profile flag.

Everything below is computed, not reasoned: `capstone/tests/exact-bounds-iii/{sweep,shape}.py` read musl
1.2.5's own constants and `size_to_class` verbatim, and print their inputs. **There is no runtime
run in this estimate.**

## The rounding rule, and prior art

Confirmed against the primary source, `modcapstone` `capstone.c:274-290`:

    granule(L) = 1                      for L < 4096   (the encoder's exact branch)
               = 1 << (ilog2(L) - 9)    otherwise

The kernel rounds and then re-checks. Swept over 4096..2^22, **the second pass never fires** — one
pass always suffices — so the re-check is belt-and-braces, not a requirement.

**This is already implemented in an allocator in this repo.** `sh_narrow` in
`capstone/ports/musl-capstone/runtime/sublet_heap.c:206-224` does exactly this rounding
("Exactly n bytes when that is representable; above 4 KiB, n rounded up to the granule the
compressed encoding keeps"). That file was retired by `7901153f10fa`, but it is evidence the
rule works in an allocator here rather than only in the kernel.

## 1. mallocng under (iii)

### Where the bound can leave the slot

From `enframe` (`meta.h:196-226`), read rather than derived:

    stride = UNIT*size_classes[sc]          UNIT=16, IB=4
    slack  = (stride - IB - n) / UNIT       integer division
    p      = slotstart + UNIT*off           off in [0, slack]
    end    = slotstart + stride - IB

so the room from the user pointer to `end` is `stride - IB - UNIT*off`, which at the worst
offset (`off = slack`) is `n + ((stride - IB - n) mod UNIT)` — **at most 15 bytes of headroom**,
while the granule at 4 KiB is already 8 bytes and reaches 128 bytes by 64 KiB.

Measured consequence:

| | affected | max overshoot |
|---|---|---|
| size classes (at their `n_max`) | **20 of 48** — every class with stride >= 4672 (sc 28..47) | — |
| individual sizes, worst slot offset | **115,695 of 131,051** | 116 B |
| individual sizes, `off = 0` | 735 | 68 B |

The first affected class is sc=28 (292 units, stride 4672). Everything below 4 KiB is exact by
construction and cannot be affected.

### The fix, and what it costs

**(a) Classify on `repr_round(n)`, and hand `enframe` the rounded length.** Then `slack` is
computed from the rounded length and the offset cannot push the bound out — correct by
construction, not by a check. (The naive test "does the rounded bound fit at the worst offset
*after* promotion" is vacuous and can never fire; the non-vacuous condition is whether the chosen
class can hold `repr_round(n)` plus the in-band header, which fires 735 times without the fix and
0 times with it.)

* **628 of 131,051 sizes (0.48%)** promote to a larger class.
* **Every penalised size sits in the last 16 bytes below its class's capacity** — proven over the
  whole range, not sampled. The penalised set is exactly the sizes that only just fit their class.
* Penalty when it hits: 10-30% of the request; worst is `n=13073`, +3280 B (25.1%), which is
  promoted from the 13088-byte class to the 16368-byte one.
* **None of 17 common sizes is penalised** — every power of two from 16 to 131072, page multiples
  from 1 to 32 pages, and 1000/10000/100000 all come through at zero cost.
* 107 sizes near `MMAP_THRESHOLD` are pushed off the size-class table onto the mmap path by
  rounding; that path has to accept them.

**(b) Keep classification and pin `off = 0` for classes with stride >= 4096 — INSUFFICIENT.** It
still leaves **735 sizes** with the bound outside the slot, and it costs the offset cycling on 20
of 48 classes, which is mallocng's double-free trap. Strictly worse than (a) on both axes.

**(c) Derive the capability over `min(repr_round(n), room)`.** No allocator change, but the
application cannot touch the last bytes of its own allocation. Recorded only so the lead can see
it was weighed; it is a semantic break and I do not recommend it.

### The big path is the tighter constraint

`malloc.c:308-327`: `needed = n + IB + UNIT`, `maplen = ceil(needed/4096)`, and
`get_stride` (`meta.h:180`) returns `maplen*4096 - UNIT`. Headroom is again at most 15 bytes at
the worst offset, while the granule at 128 KiB is 128 bytes.

Sampled every 4096 B from `MMAP_THRESHOLD` to 64 MiB (16,353 points): **the rounded bound exceeds
the whole mapping at 13,312 of them (81%)**, with a maximum overshoot of 61,440 B. Sizing the
mapping from `repr_round(n)` fixes all of them and costs **1 to 16 extra pages**, distributed
`{1:3041, 2:2048, 3:1536, 4:1536, 5:1024, 6:1024, 7:1024, 8:1024, 9..16:512 each}`. Against a
minimum large allocation of 128 KiB that is a bounded, small relative cost.

### A documented sentence becomes false

`runtime/virtual/mallocng.md:96` says mappings are a "page-rounded requested length, without the
old power-of-two padding". Under (iii) that is false from **1025 pages = 4,198,400 bytes** upward:
of the first 16,383 page counts, **13,312 (81.3%) are not representable**. A page-rounded length
is not a representable length above ~4 MiB.

### One exposure point worth flagging

`get_nominal_size` (`meta.h:159-175`) keeps `reserved` footer bytes between `p+n` and `end`.
Rounding the exposed bound outward covers them. That does not reach a neighbour, so non-overlap
holds, but the capability then spans mallocng's own footer for that slot — the allocator's
metadata becomes writable through the user pointer. Under (a) the footer shrinks to the rounded
length's slack rather than disappearing, so this is a property to decide on deliberately, not a
side effect to discover later.

## 2. What else assumes exact bounds

Read on `origin/dev` (the QEMU paths at dev's submodule pointer `d35af47b1`, which is **not** the
working tree's `deb7d75756`).

**The hard gates — these refuse to run without the profile:**

* `runtime/virtual/exec.c:697-699` — `ioctl(device, CV_PROFILE, &profile)` and
  `!(profile & 0x100)` -> `die("exact-bounds CPU profile required")`. The launcher refuses
  outright when local mallocng is selected.
* `capstone-qemu` `tests/.../probe.S:67-71` — `csrr t0, 0x5c0; andi t0, t0, 0x100; beqz t0, fail`.
  CSR 0x5c0 bit 8 is the profile bit, the same one `exec.c` reads.

**CSMINT descriptors — the real audit item.** `probe.S:151-152` builds, under `EXACT_BOUNDS`,
`MINT_SLOT 352, DATA+1, DATA+4097, 6`: length 4096 with a base deliberately **one byte**
off alignment. For L=4096 the granule is 8, so that descriptor is not representable and the
default profile rejects it (`guarantees.md:74`: "The default profile continues to reject inexact
CSMINT descriptors"). Under (iii) **every CSMINT slot the launcher and image path construct must
be representable** — base aligned to the granule and length a multiple of it. That is the one item
here that is an audit of existing data rather than a flag flip.

**Context restore and spill/reload — simplified, not broken.** `probe.S:506-508`: "Non-representable
initial bounds must survive frame restore and an ordinary spill/reload, without widening either
edge." Under (iii) there are no non-representable bounds to preserve, so the guarantee becomes
vacuous rather than violated. Nothing to fix; the test arm loses its subject.

**The image-gp capability — already fine.** `runtime/virtual/README.md:191` already specifies
"a representable readable image capability". No change.

**ABI v5 headers — a version gate.** `runtime/virtual/vm-abi.h:4` ("It requires the explicit
exact-bounds QEMU profile"), `README.md:29`, `mallocng.md:116`, `guarantees.md:70-74`,
`isa.md:73,92,146`, `runtime.md:81`, `development.md:27,34,45,49`. All statements of the
requirement; they change together with the ABI version.

**The runners and the CLI — mechanical.** `host/capstone_vm/cli.py:319-323,443,504`;
`virtual/run.py:130`; `run-staged.py:20,37`; `run-fp.py:23,46`; `run-pthreads.py:23,46`;
`run-hosted.py:59`; `run-malloc.py:79`. Seven entry points, each passing or defaulting
`--exact-bounds`/`x-capstone-exact-bounds=true`.

**Round-trips that would widen authority:** `include/capstone/capability.h:266` already states the
contract — "representable; otherwise the encoder rounds them outward again" — so the round-trip is
already defined for inexact values. `include/sublet/sublet.h:26-28` likewise already reasons in
representable pieces ("always below 4 KiB, and a 2^(E+3)-aligned block above"). Neither needs a
change; both are evidence the codebase already carries the inexact model where it touches the
physical profile.

## 3. Effort, and which recorded results change

**Effort: 3-5 days**, dominated by the CSMINT audit and the re-run, not by mallocng.

* mallocng (a) plus the big-path mapping: **half a day.** Two expressions.
* CSMINT representability audit across the launcher and image path: **1-2 days.** This is the item
  that can surprise, because it is a property of every descriptor the runtime builds, and the
  existing test proves at least one deliberately non-representable descriptor is in use.
* ABI version bump, the seven runners, the launcher gate, and the doc sentences: **half a day.**
* Re-run and re-record the virtual suites: **1-2 days**, serialized on the shared QEMU lock.

**Recorded results that change — named, not "probably none":**

* **`results/local-mallocng/allocator.json`, verdict `bounds`: TRUE -> would FAIL.** This is the
  one spatial verdict in that file and it flips. `malloc-contract.c:177-181` allocates
  `malloc(200003)` — its own comment says "This request is not exactly representable in the
  compressed codec" — and then stores at `saved + 200003`, one byte past the end, expecting a
  fault. Under (iii): 200003 >= 2^17 so the granule is 256, `repr_round(200003) = 200192`, and the
  store at +200003 lands **inside** the bound. No fault, test fails. The test needs a new
  trigger: a store at or past `repr_round(n)`, not at `n`.
* **`capstone-qemu` `exact-bounds.sh`: 4/4 -> 2/4 as written.** The script is already two-sided —
  2 edge cases x 2 profiles, expecting PASS with the profile on and FAIL with it off. Under (iii)
  the `profile=false` arms become production behaviour, so the test does not merely fail, it
  changes meaning and has to be restructured. Its two edges name exactly what is lost: the
  unaligned-base CSMINT descriptor and the frame-restore/spill-reload preservation.
* **The four ports' verdicts do NOT change.** `results/local-mallocng/ports.json` records
  `cpython_exit`, `postgres_exit`, `ffmpeg-input_exit`, `tshark_exit`, plus output-equivalence
  checks (`cpython_json_gc`, `postgres_native_rows`, `ffmpeg_native_frames`,
  `ffmpeg_flipped_frames`, `ffmpeg_input_sensitivity`, `tshark_native_dns`,
  `tshark_missing_input`) and `resources_released`. These are **functional-equivalence** verdicts,
  and outward rounding is monotone with respect to them: a wider bound can only remove a fault,
  never introduce one. A correct program that passed still passes. The same argument clears the
  temporal verdicts in `allocator.json` (`stale`, `reuse-stale`, `reuse-free`, `remap-stale`,
  `inplace-stale`, `stale-free`) and `interior-free`, none of which depends on bound width.

## Caveats, stated rather than buried

* **No runtime run.** Static computation over musl's own constants plus code read on the ref.
* **No allocation-size distribution exists.** `runtime/virtual/results/` records only aggregates
  (`live_bytes`, `cached_bytes`, `node_bytes`); `heap.c:24`'s `bucket()` hashes a cursor, not a
  size. So the 0.48% figure is over the *size space*, not weighted by what any application
  actually requests. I did not synthesise a distribution. What replaces it is the structural
  result — the penalised set is exactly the sizes in the last 16 bytes of a class, and every
  common size tested comes through free.
* **The QEMU-side citations are at dev's submodule pointer**, not the working tree's.
* One of my own checks was vacuous on first writing and is called out above; the replacement is
  negative-tested both ways.

## Addendum: the bug-corpora virtual arms are the at-risk population, and the recorded data cannot settle them

The section above cleared `ports.json` and `allocator.json`. It did not account for a much larger
body of recorded virtual-profile results: **57 `verdicts.jsonl` files under
`capstone/bug-corpora/*/*/results/*virtual*/`, 825 judged rows, of which 430 are `CAUGHT`** — in
the arms `virtual-malloc` (447 rows), `virtual-nested-pools` (336) and `virtual-pg-pools` (42).

A `CAUGHT` verdict is a *detected fault*, so unlike a functional-equivalence check it can flip.
The flip condition is precise: the detecting access must lie inside the outward-rounding slack of
the object it overruns.

**The recorded data cannot decide this, and it is worth saying why rather than estimating.** A row
records `{cause, pc, symbol, offset}`. The `offset` field is **not** an overrun distance: it is the
faulting PC's offset into its function symbol — `verdicts.py:186-193` builds the `Fault` from
`symbols.lookup(pc, code)`, which returns `name, pc - addr` (`:182`). Nothing in a row carries the
allocation size or the distance past the object end. An earlier version of this analysis bounded
the risk from those offsets; that was measuring a code offset and the bound is withdrawn.

What can still be said without a run, and it is not nothing:

* **Rounding is zero below 4 KiB**, so only repros that overrun an object of 4 KiB or more are
  candidates at all.
* **The slack is at most `size/512`** (granule `2^(ilog2(L)-9)`), so a repro whose overrun distance
  exceeds a five-hundredth of its object's size cannot be swallowed. Heap-overflow repros
  typically overrun by one element, not by 0.2% of the buffer, so the expected number of flips is
  small — but that is a prior, not a measurement, and it is not the same claim as "none".
* **The flip is one-directional.** Outward rounding can only turn `CAUGHT` into a miss, never the
  reverse, so the 351 `MISSED` rows are unaffected and (iii) can only lower the corpus's detection
  rate, never raise it.

Settling it needs either the repro sources read case by case for the allocation size and overrun
distance, or a re-run of the virtual arms with the profile off. The re-run is the cheaper and more
honest of the two and is already inside the 1-2 days costed for re-running the suites — but it
should be counted as a *deliverable* of that re-run rather than assumed to come out clean.

## RETRACTION (2026-10-16, same day): "no alignment change" is wrong

**The claim withdrawn.** This document said option (iii) needs "no size-class table change, no slot
geometry change, no alignment change, no padding", and the decision commit `1cb732b55054` quotes
it as "mallocng changes in two expressions ... with no change to its size classes, layout or
alignment". **That is wrong.** Item 1 is not two expressions; it needs the group header and the
size-class strides to be granule-aligned, which is a layout and size-class change.

**Why it was wrong.** The estimate costed the LENGTH requirement and never checked the BASE.
Representability needs both: `modcapstone` `capstone.c:338` states it — "A region is representable
only if BOTH its length and its base are granule multiples" — enforces it at `:344`, and
`process.c:124` applies the same test to a DMA base.

**Confirmed in the encoder rather than in the kernel's description of it.**
`capstone/capstone-ariane/core/include/ariane_pkg.sv:786-845`. The granule rule this document used
is correct: the full scheme takes `leading_zeros = max(12, msb(len))`, `E = leading_zeros - 12`,
and the granule is `2^(E+3) = 2^(msb(L)-9)`. But the two edges are not treated alike:

    B[13:3] = {bounds.start >> E}[13:3];                     // base: TRUNCATED, no compensation
    T[11:3] = {bounds.base  >> E}[11:3];
    if (((bounds.base >> (E+3)) << (E+3)) != bounds.base)
        T[11:3] += 1;                                        // end: rounded UP

Only the end rounds up. A base that is not a `2^(E+3)` multiple is shifted and truncated, so the
capability **gains authority below the object** — up to 240 bytes for the largest class, past the
4-byte in-band header and into the previous slot. QEMU's physical profile refuses the same request
with 29 instead of widening it (`capstone/runtime/silicon/b6-sublet.c:19` records the divergence:
QEMU refuses an unrepresentable child, the RTL rounds outward). Either behaviour breaks the current
geometry — silent authority below the object, or `malloc` failing.

**Computed over mallocng's real geometry** (`capstone/tests/exact-bounds-iii/base.py`). The derive
is `heap.c` `publish:167-180`, `cap_derive(parent, key - base(parent), n)` — in our code, not
mallocng's — and `bind_slots` (`heap.c:64-69`) puts a slot at `group_base + 16 + i*stride`:

* **12 of the 20 classes with stride >= 4096 have an unrepresentable base for their first slot.**
  The `+ 16` group header plus strides that are not granule multiples does it, from sc=36 up. For
  sc=47 (stride 131056, granule 256), `(16 + i*stride) mod 256` is `[16, 0, 240, 224]` for i=0..3.
* **Rounding the length up and the base down fits the slot for only 8 of 20 classes.** For the
  other 12 the rounded range exceeds the stride — sc=47 needs 131328 against a 131056 stride.
* **Offset cycling is reduced, not preserved.** Only granule-aligned offsets remain usable, so
  mallocng's double-free entropy drops to 1-in-2 (granule 32) through 1-in-16 (granule 256) on
  those classes. This document said the cycling was untouched by (a); that applied only to the
  length.

**One part is still open, and it could cut the other way.** `compress_bounds` has a second,
*cursorless* branch, taken when `bounds.start == cursor`, which encodes the end relative to the
cursor in 21 bits with `E = max(20, msb(cursor XOR end)) - 20` — granule 1, i.e. **exact**, for
lengths below about 1 MiB. A freshly derived child has its cursor at its base, so it begins exactly
bounded. The reading this document now rests on is that the encoding reverts to the coarse full
scheme as soon as the program advances the cursor, which is what matters both for overrun detection
and for the base truncation above. If that reading is wrong, most mallocng objects are exactly
bounded under (iii) and the cost collapses. `rtl-oracle` has been asked to settle which scheme
applies to a CDERIVE child and what the first pointer increment does; the answer will be recorded
here either way rather than resolved in favour of the estimate.

**What this does not change.** The big-path mapping result, the `mallocng.md:96` page-rounding
result, the flipped `bounds` verdict, the `exact-bounds.sh` re-scoping, the ports' monotonicity
argument and the corpora addendum all concern lengths and are unaffected.

## Re-cost of item 1 WITH the layout change (2026-10-16)

The open question is settled against the collapse, and the retraction stands: there is one
scheme, so a derived child of 4 KiB or more is inexact from the moment it exists. Verified in the
submodule's own tree at `capstone-ariane d93a8d66b` — `compress_bounds` no longer takes a `cursor`
argument and its first line reads "capstone-sublet: one scheme for every capability (the
cursorless branch for start == cursor is gone)". The main checkout's `062f4453b`, which this
analysis first read, still has that branch; `4a4261bb4` removed it.

**The encoder is now measured, not modelled.** These numbers come from the project's own
validated oracle — `verif/tests/custom/capstone/bounds-model.py` at `d93a8d66b`, "a literal port of
the RTL's 27-bit bounds encoder and decoder ... bit-identical to capstone-qemu's cap_compress" —
loaded by revision rather than copied, so it cannot drift from the RTL
(`capstone/tests/exact-bounds-iii/load_model.py`, which errors rather than returning a zero if the
revision or the file is missing; negative-tested with a bad path). Its own self-check reproduces
the R-54 comment's figures exactly: 1,050 of 400,000 random regions decode short without the
renormalisation step, 0 with it. That is the positive control for everything below.

It also **corrects the granule rule upward** in some cases: the R-54 renormalisation
(`if ((tg - bg) >= 64'd1024) E = E + 1;`) can make the granule one step *coarser* than
`2^(msb(L)-9)`. The re-cost takes the granule from the oracle instead of assuming it.

### Today's geometry, measured

With `HDR = 16` and a page-aligned group base, taking each class's largest request at slot 0 and
offset 0 — the cheapest case there is:

**12 of the 20 classes with stride >= 4 KiB escape their slot even in that best case**, from sc=36
(stride 18704, granule 32) through sc=47 (stride 131056, granule 256). The base escapes downward by
16 bytes in every one of the twelve, and the end escapes upward by up to 112 bytes.

### The minimal geometry, and it is cheap

Containment needs `HDR`, the stride, and `UNIT*off` to be granule multiples; then the slot's base
and end are both granule-aligned and the decoded range cannot leave the slot. Concretely:

* **7 of 48 size-class entries change**, each by 1 to 4 units — and mostly to round numbers:
  `size_classes[36] 1169->1170`, `[38] 1637->1638`, `[39] 2047->2048`, `[41] 2730->2732`,
  `[43] 4095->4096`, `[45] 5460->5464`, `[47] 8191->8192`.
* **Slot-space cost: +0.022%** — 795,632 to 795,808 bytes summed across the 20 affected classes.
  Twelve of the twenty need no stride change at all.
* **Group header: 16 bytes to the class's granule**, so 32/64/128/256 bytes for the affected
  classes — **once per group, not per slot**, which is at most 0.183% of a single slot (sc=47) and
  less everywhere else.
* **Offset cycling: reduced on 12 classes, untouched on 8.** Usable offsets go from all to 1-in-2
  (granule 32), 1-in-4 (64), 1-in-8 (128) and 1-in-16 (256). The 8 classes with granule <= 16 keep
  every offset.

**Verified, not asserted.** Over the repaired geometry, **254,304 (class, request, offset)
combinations were swept — every request each class serves, at the lowest and highest surviving
offset — with 0 escapes.** An earlier version of this check looked only at the top 64 bytes of each
class, which is a narrowed view that would have shown exactly what it went looking for; the wide
sweep replaces it.

### `publish()` needs nothing beyond the geometry

The same 254,304-case sweep answers this: with a granule-aligned base and a granule-multiple
stride, a derive over the *requested* `n` is contained for every `n` the class serves. No length
round-up is needed in `heap.c publish`, and the 0.48% class-promotion cost from the original
estimate disappears with it — the geometry does that work instead.

### The table edit is safe, and the check that says so can fail

`size_to_class` finds the octave with a bit trick and refines with two comparisons against the
table, so the question is whether the edited table still satisfies the allocator's invariant.
Measured: monotonicity preserved, **0 of 131,051 requests mis-sized** by the new table, 128
requests change class, and the chosen class is the smallest fitting one for 18,718 of 18,722
sampled requests (one class of slack for the other 4; never too small). **Positive control:** with
`size_classes[43]` deliberately shrunk by 8 units the same check reports **112** mis-sized
requests, so the zero is a result rather than a dead check.

### Effort, revised

**4-6 days**, from 3-5. Only item 1 moves, from half a day to about one day: a 7-entry table edit,
a per-class granule-sized group header, a granule step in `enframe`'s offset cycling, the big-path
mapping sizing, and the `mallocng.md:96` sentence. Items 2-5 are unchanged — the CSMINT audit at
1-2 days is still the largest, and the re-record now carries the 430-row corpora question as a
deliverable rather than an assumption.

**What this means for the choice.** The retraction was real — item 1 is a geometry change, not two
expressions — but the geometry change costs +0.022% of slot space, a per-group header, and some
double-free entropy on 12 classes, against (ii)'s side store, 496 MiB of map, 1.9x capability
traffic and one to two bitstream cycles. On these numbers (iii) remains much the cheaper option;
the decision does not turn on the correction.
