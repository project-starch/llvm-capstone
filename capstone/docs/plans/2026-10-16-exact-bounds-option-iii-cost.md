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

## Correction to the containment test: the slot's last 4 bytes are the NEXT slot's header

**My containment test was wrong, and the peer lane caught it.** It asked whether the decoded bound
stays inside the slot. It must stay inside the slot's *usable* region: `enframe`
(`meta.h:196-201`) places slot `idx` at `storage + stride*idx` and ends its usable region at
`p + stride - IB`, so the final 4 bytes of each slot are `p'[-4..-1]` for the next slot
`p' = storage + stride*(idx+1)` — the check byte, index and offset bits that `enframe` itself
reads. A rounded end reaching them gives an object authority over its **neighbour's allocator
header**, which exact bounds never permitted. The correct limit is `slot + stride - IB`.

**Measured on the repaired geometry** (`capstone/tests/exact-bounds-iii/nextheader.py`), at the
lowest *and* highest surviving offset — the peer lane ran offset 0 only:

| | escapes the slot | reaches the next header |
|---|---|---|
| original table, 16-byte header (positive control) | 90,016 | 35,160 |
| repaired geometry, no length fix | **0** | **94,088** of 252,736 |
| repaired geometry **with** the length fix | **0** | **0** |

**The exposure is much wider than the near-capacity sizes.** At offset 0 only the sizes with
`n > stride - granule` reach the header, which is where the figure of 912 comes from. But offset
cycling exists precisely to move the object within its slot, and at the highest surviving offset
`p + n` sits within one `UNIT` of the usable end for *every* request — so about 37% of all
(request, offset) combinations in the affected classes reach the neighbour's header, not a few
hundred sizes. The fix is the same either way; the scope is not.

**The fix is the first estimate's rule, restored:** classify on `repr_round(n)` against
`stride - IB`. **912 sizes move up one class** — which matches the peer lane's count exactly. It
works at every offset because the slack is then computed from the rounded length and both the base
and the length are granule multiples, so the end is exact and at or below `stride - IB`.

**The per-class breakdown is not `granule - IB`.** That rule gives 1,136 and does not match its own
total; the granule has to be taken per *request* rather than per class, because within one class
the requests straddle a renormalisation boundary. Measured per class, summing to 912:

    sc 28-31  (g=8)     4 each        sc 39  (g=64)    28   <- not 60
    sc 32-35  (g=16)   12 each        sc 43  (g=128)   60   <- not 124
    sc 36-38  (g=32)   28 each        sc 47  (g=256)  124   <- not 252
    sc 40-42  (g=64)   60 each
    sc 44-46  (g=128) 124 each

The three exceptions are exactly the renormalised classes, whose repaired strides are the clean
powers of two 32768, 65536 and 131072.

**The slot's own footer stays reachable, and that is accepted.** `get_nominal_size`
(`meta.h:159-175`) keeps `reserved` bytes between `p + n` and `end`, and the rounded bound spans
them. No neighbour's data and no neighbour's header is reachable, and mallocng validates that
footer at `free`, so the exposure is to the object's own allocator metadata only. This is recorded
as a deliberate acceptance rather than an oversight.

## Correction: the promotion count is 1,024, not 912 — classification cannot see the base

**912 was computed at one particular base.** `size_to_class` runs at `malloc(n)` entry, before the
object has an address, so the granule it must assume is the worst case over every base the repaired
geometry can produce — not the granule at some chosen base. The three figures, all measured:

| granule evaluated at | promotions |
|---|---|
| base `G0 + class granule` (what the earlier sweep used) | 912 |
| base `G0 + UNIT` | 960 |
| **worst case over granule-aligned bases (what classification must use)** | **1,024** |

The spread is the R-54 renormalisation: for 72 request sizes the trip condition depends on the
base's alignment, so the same length encodes with granule 16 at one base and 32 at another. A
classification rule that picks the smaller one under-promotes and leaves the neighbour's header
reachable for those sizes.

**So the implementable rule is `repr_round_classify(n)` using the base-free worst-case granule, and
1,024 of 131,051 sizes (0.78%) move up one class.** That is the number to cost and the number the
change implements; 912 and 960 are artifacts of a fixed base and should not be quoted.

**Slot coverage widened.** The earlier sweep checked slot 1 only. Slot 0's base alignment depends
on the header pad and the last slot abuts the group's tail, so both are now included: slots 0, 1
and 31 (the largest index a 32-bit `avail_mask` allows), **0 slot escapes and 0 next-header hits**
with the fix in place.

## Implementation note: offset `storage`, not `mem`

The cheap-looking repair — place `g->mem` inside its mapping so `storage` lands granule-aligned —
**is wrong**, and would have failed at runtime rather than in a check. `g->mem` is the mapping base:
`realloc.c:34` passes it to `mremap(g->mem, g->maplen*4096UL, ...)` and `free.c:21-25,148` unmaps
`mi.len = g->maplen*4096UL` from it. An offset `mem` is not page-aligned, so `mremap`/`munmap`
return EINVAL, or unmap from the wrong base.

The repair is a per-class **storage** offset instead, applied where slots are addressed:
`meta.h:200` (`enframe`) and `:223`, `free.c:108`, `realloc.c:15,38,39`,
`aligned_alloc.c:31,32,40`, `donate.c:30` — with the group's allocated size grown to match at
`malloc.c:254`, `malloc.c:326`, `realloc.c:32` and `donate.c`'s `(size_classes[sc]+1)*UNIT`.

**`donate.c` is in scope**, which is not obvious: `donate.c:17` is
`for (int sc=47; sc>0 && b>a; sc-=4)`, so donated memory hosts sc 47, 43, 39, 35 and 31 — the
largest classes, and five of the twenty affected ones.

## Per-port derive sites: mruby's page lifetime is REFUSED, and it was on the "small children only" list

Under (iii) QEMU's compact profile refuses an inexact `CDERIVE` child with cause 29, so every port
that derives a child of 4 KiB or more must place it representably. Three ports were expected to
derive only small children. **One of them does not**, and the two-level structure is the reason:
each of these allocators derives a *large parent lifetime* as well as small per-object children.

**mruby — REFUSED, measured.** `patches/4.0.0-rc2/0008-gc-slots-as-sublet-lifetimes.patch:120`:

    page->life = capstone_cap_derive(page,
        cap_get_cursor(page->objects) - cap_get_base(page), sizeof(page->objects));

with `#define MRB_HEAP_PAGE_SIZE 1024` and `RVALUE objects[MRB_HEAP_PAGE_SIZE]` in the same patch,
so the child is `1024 * sizeof(RVALUE)` — at least 16 KiB — at an offset of
`offsetof(mrb_heap_page, objects)`.

Compiled with the capability-ABI toolchain rather than worked out on paper
(`capstone/tests/exact-bounds-iii/mrbpage.c`, `clang ... -target capstone64-unknown-elf`):
`sizeof(void *) = 16` and **`offsetof(objects) = 16464`**. Checked against every plausible
`sizeof(RVALUE)` from 16 to 128 bytes: **unrepresentable in all nine cases** — 16464 is 16 mod 32,
16 mod 64 and 80 mod 128, while the granule for those lengths is 32, 64, 128 or 256. The verdict
does not depend on `sizeof(RVALUE)`: any length of 16 KiB or more gives a granule of at least 32,
and the offset is 16 mod 32.

The cause is the patch's own addition. `struct RBasic *issued[MRB_HEAP_PAGE_SIZE]` is 1024
capability pointers — 16,384 bytes — inserted before `objects`, which moves it from roughly offset
80 to 16,464. **The Sublet patch introduced the misalignment it now has to fix.**

The repair is padding plus a compile-time assertion: `1024 * sizeof(RVALUE)` is known at compile
time, so the granule is too, and `offsetof(objects)` can be padded up to a multiple of it with a
`_Static_assert` that fails the build if it ever drifts. Cost: at most 240 bytes on a page of 32 KiB
or more, under 1%.

**CPython pymalloc — representable, no change.** `cpython-3.13.7-0014...patch:168` is
`capstone_cap_derive(address, 0, ARENA_SIZE)`. The project's own pymalloc model records
`#define ARENA_SIZE 1048576UL` (`ports/cpython/pymalloc/src/allocators/sublet/block-lifetimes.c:18`),
so 1 MiB, granule 2048, **offset 0**, and the arena comes from `_PyObject_Arena.alloc`, which is
mmap-backed and therefore page-aligned — so the base is 2048-aligned and 1 MiB is a 2048-multiple.
The per-block children (`...:136`) are pymalloc blocks, at most 512 bytes, granule 1, so their
non-zero offsets do not matter. The upstream `ARENA_SIZE` is still to be quoted from CPython 3.13.7
itself rather than from our model.

**Perl — safe by 16 bytes, and worth flagging as fragile.**
`perl/sublet/patches/5.36.3/0001-sv-heads-as-sublet-lifetimes.patch:191` derives `PERL_ARENA_SIZE`
bytes. Perl's documented default is 4080, which is **below** the 4096 threshold and therefore
exactly representable with granule 1 at any offset. That is a 16-byte margin: raising
`PERL_ARENA_SIZE` to 4096 or more puts this site in scope immediately. The value is upstream and not
yet quoted here. The per-SV children (`...:132`, `sizeof(SV)`) are small and safe.

**A simplification worth stating, because it bounds the whole audit:** a child shorter than 4096
bytes has granule 1, so it is exactly representable at *any* base and *any* offset. Only derive
sites that can produce a child of 4 KiB or more need auditing at all — which is why the per-object
derives in all of these allocators are safe by construction and only the parent-lifetime derives
matter.

## RETRACTION of the 1,024 figure: it is 912, and the renormalisation cannot fire at all

**Withdrawn: "the promotion count is 1,024, not 912".** It is 912 — the figure the peer lane gave
originally. My correction was an over-correction, and the cross-check below caught it.

**What I got right and what I got wrong.** Right: `size_to_class` runs before the object has an
address, so it cannot evaluate a base-dependent granule. Wrong: I concluded it must therefore
assume the worst case over *all* bases. The repaired geometry places every slot on a granule-aligned
base, so the misaligned case the worst-case granule accounts for **cannot occur**.

**And the renormalisation cannot fire on any range the predicate accepts.** The condition is
`(tg - bg) >= 1024` with `tg = roundup(end, g)/g` and `bg = floor(start/g)`. If the base and the
length are both granule multiples then `tg - bg` is exactly `len/g`, and since `len < 2^(msb(len)+1)`
with `g = 2^(msb(len)-9)`, that is `len/g < 1024`. So it is never met. The renormalisation fires only
for ranges that are *already* unrepresentable — which is precisely what we are eliminating.

A pleasing consequence: with the repaired strides the three classes that used to renormalise
(32752, 65520, 131056 → 32768, 65536, 131072) no longer do, so **no class in the repaired table
renormalises at all**. The granules become the plain 32, 64 and 128 rather than 64, 128 and 256.

**Re-measured with the predicate as shipped:** 7 of 48 size-class entries change (unchanged),
**912 of 131,051 sizes (0.70%) promote one class**, and containment is re-verified over slots 0, 1
and 31 at both the lowest and the highest surviving offset — 6,036 checks, **0 slot escapes and 0
next-header hits**.

### The predicate, and how it is held to the encoder

`capstone/runtime/include/capstone/representable.h` is the one place this arithmetic now lives:
`capstone_repr_granule`, `capstone_repr_round`, `capstone_repr_ok(base, len)` and
`capstone_repr_base_down`. Every (iii) item needs it — the allocator geometry, the CSMINT audit and
the ports' derive sites — so it is a header rather than four copies.

It is checked against the encoder rather than against reasoning:
`capstone/tests/exact-bounds-iii/repr-cross-check.py` compiles the **real header** into a host
driver and compares its answers to the RTL oracle over 24,168 cases — every length below 5,000 at
four bases, 4,000 random large ranges, and the renormalisation band explicitly. **0 disagreements**,
and the header accepts 16,902 of the 24,168, so it is not trivially accepting. The same header is
separately asserted to compile for `capstone64`, because "it builds for the target" and "it computes
the right thing" are two claims and one run cannot cover both.

**The check is two-sided.** A mutant predicate that ignores the base — the exact mistake the first
cost estimate made — disagrees with the encoder 46 times and fails the check. Without that control
the 0 above would be unproven.

**An earlier version of the header was wrong in the safe direction and the check still caught it:**
it doubled the granule inside a supposed renormalisation band and so *rejected* exactly
representable ranges, `len = 16368` at an aligned base among them. A predicate that is too strict
costs memory rather than safety, which is exactly the kind of error that survives a one-sided test.

**Also checked and NOT a defect:** the kernel's own `capstone_repr_granule` (`capstone.c:274`) omits
the renormalisation step. For the LENGTH that is benign — rounding up with the finer granule inside
the affected band lands on the next power of two, which is a multiple of the coarser granule too,
measured over the first 2,000 affected lengths with 0 decoding short. I checked before reporting it
as a bug; it isn't one.

## The ports' derive sites: the fix is a rounded length per site, and the comparison does NOT reopen

Offsets and lengths extracted from the patches themselves
(`capstone/tests/exact-bounds-iii/derives.py`, which balances parentheses rather than grepping
line by line, because the calls span lines). Twenty derive sites across the five ports the
comparison turns on:

| port / patch | offset 0 | non-zero offset | the length |
|---|---|---|---|
| FFmpeg `0003` pools | 2 | 1 | `FFMAX(pool->size,1)`, `REFCOUNT_OFFSET + pool->size`, `size` |
| memcached `0006` slabs | 2 | 1 | `cache->bufsize`, `size` |
| wmem `0001` | 3 | 1 | `size` |
| PostgreSQL `0003` contexts | 2 | 2 | `size` |
| SQLite memsys5 | 2 | 4 | `mem5.aReq[i]`, `nByte`, `db->lookaside.sz`, `(u64)sz*cnt` |

**Offset-0 sites need only the LENGTH rounded, because the base comes free.** The parent is an
allocation mallocng has already placed on a granule-aligned base, and the child's granule is at
most the parent's (its length is no larger), so an aligned base for the parent is aligned for the
child. One `capstone_repr_round()` at the site is the whole change.

**With one condition that is easy to miss:** `CDERIVE` checks containment against the parent's
*exact* bounds *before* compression — `capstone_dyn_unit.anvil:250` refuses when
`rs2.cursor > rs1.end - rs1.cursor` — so the parent has to hold the **rounded** length. The site
must therefore *allocate* `repr_round(size)`, not merely derive over it.

**That costs under 0.2%.** The overhead is `(granule-1)/size`, bounded by `1/512`; measured maximum
over 4 KiB to 4 MiB is **0.1953%**, and it is 0 for every power of two and every page multiple —
`1048576` costs nothing, `1048577` costs 2,047 bytes.

**Non-zero-offset sites additionally need a granule-aligned offset.** Three shapes appear, each
with a local fix:

* a slot index times a power-of-two stride (`i * mem5.szAtom`): aligned whenever the stride is a
  power of two at least the granule — a startup assertion, since memsys5's atom size is a runtime
  configuration;
* a pointer difference inside a parent (`p - base(parent)`): needs the sub-object placed on a
  granule boundary, i.e. the containing allocator's own geometry, which is the mruby-style repair;
* `cursor(p) - base(p)` on a fresh allocation: zero in practice but not by construction, so it
  wants an assertion rather than an assumption.

**Nothing is in the shape that cannot be fixed site-locally** — a caller-controlled length with no
ceiling *and* an offset the port does not choose. None of the twenty sites is in that shape on the
patch side.

### Effort, and the comparison

**Port changes: 2-3 days** — twenty sites across eight ports, each a `capstone_repr_round()` on an
allocation plus an assertion on an offset, against a shared header that is already written and
cross-checked. Plus mruby's struct padding. The ports' re-qualification is the dominant remaining
cost and overlaps item 4's re-run rather than adding to it.

**Total for (iii): 6-9 days**, from 4-6.

**The comparison with (ii) does not reopen, and the margin is not close.** (iii) now costs
6-9 days of runtime and port work, +0.022% of mallocng slot space, under 0.2% on the affected port
allocations, a per-group header, and reduced double-free entropy on twelve size classes. (ii) costs
the hardware side store with its L1 array and the shared-core widening, 496 MiB of memory map with
the CMA and monitor conditions that follow, 1.9x capability traffic, and one to two bitstream
cycles — each of which is roughly ninety minutes of synthesis plus a reflash, and a reflash needs
the lead. Days of software against silicon area, memory bandwidth and bitstream cycles is not a
close call.

The per-port **maxima** still to be quoted decide *which* sites are in scope. They do not change
the cost shape: a site in scope costs a rounded allocation, and a site out of scope costs nothing.

## Two base alignments my "the base comes free" claim did not cover — both CONFIRMED

That claim holds only for parents **mallocng placed**. Two cases fall outside it.

### 1. The big path above 4 MiB: a page is not a granule

Confirmed with the oracle, two-sided. The granule exceeds a page exactly from 4 MiB up
(`granule(L) > 4096` iff `msb(L) > 21`), and `mmap` guarantees only a page:

| length | granule | page-aligned base, length already rounded |
|---|---|---|
| 2 MiB | 4096 | exact |
| 4 MiB − 4 KiB | 4096 | exact |
| 4 MiB | 8192 | **inexact** |
| 8 MiB | 16384 | **inexact** |
| 16 MiB | 32768 | **inexact** |
| 64 MiB | 131072 | **inexact** |
| 88 MiB | 131072 | **inexact** |

Those are the peer lane's figures reproduced independently, including the two exact cases they
named. The check yields both outcomes, so it is not a one-sided pass. **My earlier "size the
mapping from `repr_round(n)`" covered the length only** and left this open.

**Fix: over-map one granule and trim the base up to it.** Verified: 6 of 6 exact after trimming,
including 100 MiB + 7. Cost is one granule — **0.195% at 4 MiB, 0.1953% at 64 MiB**, the same
1/512 bound as everything else here. `realloc.c`'s `mremap` path needs the same treatment, because
a moved mapping can land page-aligned but not granule-aligned.

This matters in practice rather than in principle: the virtual workloads allocate 8, 16 and 64 MiB
in the pymalloc replay, and all three are in the inexact column.

### 2. Parents mallocng did not place — one site, and it is the static arena

Every offset-0 derive's parent traced to its origin:

| site | parent | placed by |
|---|---|---|
| FFmpeg `buf->sublet_life`, `ref->sublet_life` | `buf->data` from the pool allocator | mallocng |
| wmem `s->life` | a wmem block from `g_malloc` | mallocng |
| memcached `ret` | the slab/cache allocation | mallocng |
| PostgreSQL `b->life` | "the malloc'd object `alloc`" (the patch's own comment) | mallocng, **at a non-zero offset** |
| **SQLite memsys5 `mem5.pHeap`** | `zByte`, **one static array** | **the linker** |

`arms.json`'s `virtual-sqlite-memsys5` says it outright: "memsys5 stock over **one static array**
… every SQLite allocation is an offset inside that arena". So its base alignment is the linker's,
and the arena needs an explicit `__attribute__((aligned(g)))` for its own size's granule — a
compile-time constant, since the arena size is a build parameter.

**So exactly one offset-0 site needs an explicit alignment**; the rest inherit it from mallocng
once the geometry and the big-path trim are in. PostgreSQL's is the separate non-zero-offset case
already listed.

## The mallocng policy guard forbids the cheap geometry, and the alternative costs 100x more

`capstone/ports/musl-capstone/check-mallocng-policy.py` is a release gate on mallocng's decision
code: it hashes the verified musl 1.2.5 archive and requires token-identical text for

* `UNIT`, `IB`, `MMAP_THRESHOLD`;
* **`size_classes`**, `small_cnt_tab`, `med_cnt_tab`;
* **`size_to_class`**, `size_overflows`, `step_seq`, `record_seq`, `account_bounce`,
  `decay_bounces`, `is_bouncing`, `try_avail`, `alloc_slot`, `okay_to_free`, `nontrivial_free`;
* `alloc_group`'s group size/count/mmap decisions, and `malloc.c` from `sc = size_to_class(n);` to
  `success:`.

**So the 7-entry stride edit this document has been costing is not permitted**, and neither is
changing `size_to_class`.

**What IS permitted, checked function by function rather than assumed.** None of the slot-base
arithmetic sites falls inside a locked function: `free.c:108` is in `free()`, not the locked
`nontrivial_free` (which ends at line 100); `realloc.c:15` precedes the locked moving-realloc
region; `aligned_alloc.c` and `malloc_usable_size.c` are not locked at all; and `enframe` is not in
`meta.h`'s locked set. The `n >= MMAP_THRESHOLD` block precedes `sc = size_to_class(n);` and is
therefore editable. And the request can be rounded in **our** `heap.c` `malloc()` wrapper, which is
outside musl entirely — so no classification change is needed inside the locked region.

**But the stride cannot be made a granule multiple, and that is what the geometry needed.** A slot
base is `storage + pad + i*stride`; for every `i` to be granule-aligned the stride must be a
granule multiple. With the table frozen, **13 of the 20 large classes qualify and 7 do not** —
sc 36, 38, 39, 41, 43, 45 and 47, whose strides are `16 x odd` and so guarantee only 16-byte
alignment, while their own capacities need granules of 32 to 128.

The compliant alternative is to round each request up into a class that both holds
`repr_round(n)` **and** has a granule-multiple stride. Measured over every request:

| | the forbidden design (7 stride edits) | the policy-compliant design |
|---|---|---|
| requests promoted | **912** (0.70%) | **45,420** (34.66%) |
| pushed onto the mmap path | 107 | **26,347** |
| total slot bytes | **+0.022%** | **+2.13%** |
| worst single allocation | +25.1% at n=13073 | **+71.7% at n=21793** |

**That is roughly a hundredfold difference in allocator memory**, and it comes entirely from seven
table values changing by one to four units each.

**This is a question for whoever owns that guard, not a line to weaken.** The guard exists so that
mallocng's allocation *decisions* stay those of verified upstream, and seven stride values are
exactly what it is built to catch. What can be said for relaxing it narrowly: `size_to_class` would
not change at all (the rounding lives in our wrapper), the structure of every decision is
unchanged, the edits move each stride by at most 64 bytes, and the edited table was measured to
preserve the allocator's own invariant — monotone, 0 of 131,051 requests mis-sized, with a positive
control that reports 112 when the table is deliberately broken. What can be said against: the
qualification argument is "token-identical to a verified archive", and a narrower lock is a weaker
claim than the one the port currently makes.

**Not blocked meanwhile.** The big-path granule trim, the `mremap` path, the per-class storage
offset, memsys5's static-arena alignment and item 2's checker are all outside the locked regions
and proceed either way.

## The launcher needs NO alignment change — `rounded()` returns powers of two

Scope reduction, and a correction to my own premise. I wrote a change to `reserve()` on the
assumption that `image_bytes` and `stack_bytes` are page-rounded and therefore not powers of two,
which would have made its `& ~(bytes - 1)` a meaningless mask. **That premise was false, and I
inferred it from the function's name instead of reading it.** `exec.c:123-129`:

    static size_t rounded(size_t n)
    {
        size_t size = 4096;
        if (!n || n > CV_MAX_REGION_BYTES) return 0;
        while (size < n) size <<= 1;
        return size;
    }

It **doubles from 4096**, so every result is a power of two of at least 4096. Verified by running
it over 200,000 inputs plus the large cases: **0 results that are not a power of two and a granule
multiple**. The two constant sizes are powers of two as well — `CAPSTONE_DELEGATE_META_BYTES`
16384, `CAPSTONE_LAUNCH_BYTES` 65536.

A power-of-two length is always a granule multiple (the granule is `bytes/512` at most), and
`reserve()` aligns the base to `bytes`, which for a power of two implies granule alignment. **So
every arena the launcher grants is representable by construction**, and the `reserve()` change was
reverted.

**What was kept is the check at the chokepoint.** `grant()` is the one function every launcher
arena passes through, and it now refuses an unrepresentable one rather than minting a capability
silently widened below its base. The comment states plainly that **it cannot fire today** — the
construction above guarantees it — and why it is there anyway: that construction is an invariant
of three separate pieces of code, none of which says so, and a future size that bypasses
`rounded()` would otherwise hand the domain a widened capability.

**A check that cannot fire needs a test that can.**
`capstone/tests/exact-bounds-iii/grant-repr-test.c` asserts the launcher's real shapes pass — every
`rounded()` size at a `bytes`-aligned base, plus both constants — and that four guarded shapes are
refused: a non-power-of-two length at a page-aligned base, a length that is not a granule multiple,
a base one byte off (`exact-bounds.sh`'s `edge=1` descriptor), and a 16-byte-aligned static arena
(memsys5's case). Negative-tested: against a predicate that ignores the base, the test fails with
exactly those three base-dependent cases.

## Port audit completed: three corrections to my own cost, and the port estimate rises

The per-port maxima came back from a delegated search. Its numbers are recorded below as **its**
findings with its quoted evidence; the two that contradict what I had already reported I verified
myself first, and one of them is a retraction.

### RETRACTED: CPython's pymalloc arena is NOT free

I wrote that the 1 MiB arena derive needs no change because "the arena comes from
`_PyObject_Arena.alloc`, which is mmap-backed and therefore page-aligned". **That is wrong, and
the port's own build refutes it.** `capstone/ports/cpython/app/prepare-cpython-capstone.sh:192`
sets `ac_cv_func_mmap=no`, and `:188-189` says why — "With HAVE_MMAP obmalloc maps its arenas
directly (`_PyMem_ArenaAlloc`, 3 calls at startup measured natively), and every one would fail;
without it obmalloc takes them from malloc" — with a hard abort at `:281-282` if configure finds
mmap anyway.

So `_PyMem_ArenaAlloc` returns `malloc(size)`, and a 1 MiB arena's base comes from **mallocng's
large-allocation path**. Its alignment is therefore not free: it is exactly what the big-path
granule trim above has to provide. CPython's arena moves from "no change needed" to "covered by
item 1's big-path fix, and only by it".

### The audit is wider than the CDERIVE sites: 7 files use SPLIT/SHRINK instead

My table counted 20 `capstone_cap_derive` sites across five ports. Seven further files set bounds
with `sublet_split` / `sublet_carve` + `SHRINK` and never call CDERIVE at all —
`sublet-3530300.patch`, both nginx patches, `ngx_subpool.c`, both APR lease files, whisper's
`regions.c`, and FFmpeg's `payload-capabilities.c`. The same representability rule applies, and
`capstone/runtime/include/sublet/sublet.h` says so in its own `sublet_split` comment: bounds
"become exact whenever the piece is representable: always below 4 KiB, and a 2^(E+3)-aligned block
above".

### QUALIFIED: "nothing is in the unfixable shape" — fixable, but not by one call

I wrote that no site has "a caller-controlled length with no ceiling AND an offset the port does
not choose". The shape I should have described is narrower than I made it sound: several sites
*do* combine a request-driven length with an offset that is only 16- or 64-byte aligned, which is
more than a `capstone_repr_round()` on the length can fix. They remain fixable **locally**, because
the allocator chooses its own chunk positions — but the fix is granule-aligned *placement*, not a
rounded length, and that is a different amount of work per site.

Reported by the search, with its own quoted bounds (not re-verified by me):

* **wmem `wmem_sublet_issue`** — length is the caller's request and the jumbo path allocates "a
  new block of exactly the right size" with no clamp it could find; offset is the chunk position,
  aligned to `2 * sizeof(size_t)`. Its worst site.
* **SQLite `pcache1Issue`** — `szPage` **defaults to exactly 4096** and is read from the database
  file header (bytes 16-17) up to 65536; the bulk-pool offset is `k * szAlloc` where
  `szAlloc = szPage + szExtra + ROUND8(sizeof(PgHdr1))`, not a power of two.
* **PostgreSQL `MemoryChunkIssue`** — ≥4096 with **no configuration change** (an 8224-byte small
  chunk at the default `allocChunkLimit` of 8192), up to `MaxAllocSize`/`MaxAllocHugeSize` on the
  large path, at offset `ALLOC_BLOCKHDRSZ` = 112, 16-aligned only.
* **whisper `wg_region_create`** — up to `WG_PAYLOAD_BYTES` 384 MiB with a 16-byte-aligned base and
  16-byte-multiple length, and no granule rounding anywhere on the path.
* **FFmpeg `ff2_payload_carve`** — up to 64 MiB, 64-byte granular, and the representable-alignment
  correction in `pool-allocator.c:121-129` is `#ifdef FFPOOL_CHERI` only — **the CHERI arm already
  rounds to a representable length and the Capstone arm does not.** `ff2_payload_issue_pointer`
  also shrinks to `b->requested` exactly, with no rounding.
* **memcached `do_slabs_alloc`** — up to 1 MiB by configuration, 512 KiB default, at offset
  `k * p->size` where the class sizes are 8-byte-aligned ×1.25 products.

Independent corroboration of the mruby result: the search computed
`offsetof(mrb_heap_page, objects) = 16464` from the struct layout, matching the **16464 I measured
with the capability-ABI compiler**, and `sizeof(RVALUE) = 80`, giving a length of 81920 — inside
the 16-to-128-byte range my sweep had already shown unrepresentable at every value.

Perl's 4080 is confirmed, though it lives in `sv_inline.h` rather than `perl.h`, and it is
configurable through the `#ifndef` guard, so a build that raises it enters scope.

**UNRESOLVED and recorded as such:** APR's `SMALL_NODE_SIZE` and `APR_MEMNODE_T_SIZE` (upstream
apr-1.7.4 / apr-util-1.6.3 are not on this machine), the 3530300 memsys5 pool size (a host-granted
runtime region), and whether a 1 MiB `malloc` is page-aligned today — which is now a question about
mallocng's big path rather than about CPython.

### Revised effort

**Port changes 4-6 days, from 2-3.** The rise is the offset-alignment work: a length round is one
call, but granule-aligned chunk *placement* touches each allocator's own geometry, and there are
seven more SPLIT/SHRINK files than the CDERIVE table showed. **Total for (iii): 8-12 days**, from
6-9.

**The comparison still does not reopen.** Eight to twelve days of software against the side store,
496 MiB of map, 1.9x capability traffic and one to two bitstream cycles is the same call as before,
with more days on one side and nothing new on the other.

## Item 1 implemented in part, with what remains measured rather than estimated

**Landed on the lane branch.** The 912 figure is corrected first: it counted 17 requests at or
above `MMAP_THRESHOLD` (sc 47's repaired capacity runs to 131068 while the threshold is 131052),
which never reach a slot. Verified by reproducing my own loop: 912 − 17 = 895 against the edited
table, and against **upstream behaviour** the honest figure is **735 changing class — 628 in-class
plus 107 onto the mmap path**. 628 was my first estimate, before I started correcting in the wrong
direction. The peer lane's independent audit matches on all of it.

**The guard, narrowed and pinned.** `check-mallocng-policy.py` now accepts a `size_classes` table
that differs from upstream in **exactly** entries 36, 38, 39, 41, 43, 45 and 47, at **exactly**
1170, 1638, 2048, 2732, 4096, 5464 and 8192. It is a pin, not a relaxation: an eighth edit, a wrong
value, a missing entry, a lowered entry, a swapped pair and a wrong table length all still fail.
Everything else the guard holds — `size_to_class`, `try_avail`, `alloc_slot`, `okay_to_free`,
`nontrivial_free`, the group size/count/mmap decisions, the slot/class selection, the moving-realloc
sequence, `UNIT`, `IB`, `MMAP_THRESHOLD` — is untouched.

**The guard is proven two-sided against the real tree, not a synthetic table:** on a tree with
0001-0009 applied it reports `FAIL size_classes` and exits 1; with the seven entries raised it
reports `PASS size_classes` and exits 0. `test-mallocng-policy-narrowing.py` adds the unit-level
control: one accepted table, seven refused near misses.

**`0010-mallocng-representable-bounds.patch`** carries the seven entries and the `enframe` change
that steps each object's base **down** to a granule-aligned offset — down and never up, so it can
never exceed `slack` and overrun the slot, and a no-op for any granule at or below `UNIT`, which is
every length below 8 KiB, so the double-free offset cycling is untouched there. Applies cleanly to
a 0001-0009 tree and reverse-applies to a patched one, so `prepare-musl-capstone.sh` skips it when
it is already in.

**`heap.c`** does the two things that must live outside musl: `malloc()` rounds the request with
`capstone_repr_round` and passes **the same value** to mallocng and to `publish()`, so `set_size`
records exactly what the capability covers and neither `malloc_usable_size` nor an in-place
`realloc` can report unreachable bytes; and `__capstone_malloc_map` now asks `cap_vm_acquire` for
**granule** alignment rather than 4096 and rounds its length to the granule. That last one needed
no over-map and trim after all — `cap_vm_acquire`'s third parameter is already an alignment.

### What remains, with the number that says why

Simulated against the RTL bounds oracle over 6,060 (class, request, slot, offset-cycle) cases,
modelling the code exactly as now written:

| group header | cases failing |
|---|---|
| 16 bytes, as today | **1,497 of 6,060** |
| granule-sized | **0 of 6,060** |

So **the per-class granule-sized group header is required** — the `enframe` step-down cannot rescue
`off = 0`, where the user pointer is `storage + stride*idx` and `storage` is `mem + 16`, i.e. 16 mod
the granule. The first failure the simulation reports is sc=36, n=16480, slot 0, offset 0: base
`0x400000010` decoding as `[0x400000000, 0x400004080)`, truncated 16 bytes below the object. That
change touches the eleven `storage + stride*idx` sites — `meta.h` `enframe` and `:223`, `free.c:108`,
`realloc.c:15,38,39`, `aligned_alloc.c:31,32,40`, `donate.c:30` — none of which is inside a locked
function, plus the group sizing at `malloc.c:254`, `:326`, `realloc.c:32` and `donate.c`'s
`(size_classes[sc]+1)*UNIT`. **`donate.c` is in scope**: its loop is
`for (int sc=47; sc>0 && b>a; sc-=4)`, so donated memory hosts sc 47, 43, 39, 35 and 31.

**And one gap with no local fix.** `__capstone_vm_remap(unsigned long, unsigned long, unsigned long)`
takes an address, an old length and a new one — **no alignment parameter**, unlike
`cap_vm_acquire`. So a remapped mapping can land page-aligned but not granule-aligned, and
`__capstone_malloc_group_remap` cannot ask for better. The cheap correct answer is to verify the
remapped base and return `MAP_FAILED` when it is misaligned: `realloc.c` already falls through to
its `malloc` + copy path on a failed `mremap`, so the behaviour degrades to a copy rather than to a
wrong bound. Extending the VM service ABI would avoid the copy and is the larger change.

## The group header: where the pad goes is settled, how to pay for it is not

Tracing it through the patched tree made the change **smaller than the eleven sites I listed**, and
then hit one constraint that needs a decision.

**Smaller, because 0009 already routes every consumer through our slot capability.** `free.c:119`,
`realloc.c:15`, `aligned_alloc.c:31` and `malloc_usable_size.c:10` each compute
`start = storage + stride*idx` and then, under `#ifdef CAPSTONE_MUSL_MALLOC`, immediately replace it
with `__capstone_malloc_slot_start(&g->cap_slots[idx])`. And `get_meta` takes its capability branch —
`__capstone_malloc_find(p)` returning `s->group` — so the stored slot offset, the
`base = p - UNIT*offset - UNIT` reconstruction and the `size_classes[sizeclass]*index` asserts are
all **compiled out** in this build. (Those asserts would otherwise have fired: `glue.h:42` defines
`assert` as `a_crash()`, so they are always live.)

So the pad has exactly **two** homes: `bind_slots` in our `heap.c`, which is where a slot's
capability bounds are actually set, and `enframe`'s `p = storage + stride*idx`, which must agree
with it. The remaining `storage` use, `enframe`'s stored offset at `:287`, is only a cycling seed in
this build (`enframe` reads it back masked `& 255`), so a pad perturbs the cycle and nothing else.

**The constraint: the group must grow by the pad, and that sizing is locked.** Slots would occupy
`UNIT + pad + cnt*stride`, so the last slot's end overruns a group allocated for
`UNIT + cnt*stride`. The sizing lives in `alloc_group`, whose prefix up to the first
`#ifdef CAPSTONE_MUSL_MALLOC` the guard holds token-identical ("group size/count/mmap decisions").
The options, with what each costs:

1. **Over-map in our own `__capstone_malloc_group_map`** — add `pad` to the bytes we ask for, leaving
   musl's `needed` and `maplen` untouched. Needs checking whether the release path uses musl's
   `maplen` (`free.c`'s `mi.len = g->maplen*4096UL`) or our VM record; if musl's, up to one granule
   per group is never released — 0.1% of a 128 KiB group, but a genuine leak that would need saying
   out loud.
2. **Grow `struct group`'s `pad[]` to the maximum granule** (256 bytes) for every group. Simple and
   leak-free, but it taxes the small classes, which are the numerous ones, by 240 bytes a group for
   a problem only the large classes have.
3. **Widen the guard's `alloc_group` exception** so the sizing can take the pad per class. Smallest
   runtime cost, largest process cost — it is a second narrowing of a gate that has just been
   narrowed once, and it touches the decision code rather than a table of constants.
4. **Shorten the last slot by the pad** and teach `enframe` its reduced capacity. No extra memory and
   no guard change, but it makes one slot's capacity differ from its class's, which is exactly the
   invariant `size_to_class` is built on.

I am not picking between these. (1) trades a small leak for no process cost, (3) trades process cost
for no runtime cost, and the choice depends on how much the narrowed guard is meant to hold — which
is the question the lead already answered once. **Everything else in item 1 is implemented**; the
measured consequence of leaving the header alone is the 1,497-of-6,060 figure above, so this is the
last correctness gap in the allocator.

## The group header costs NOTHING, and the allocator geometry is complete

**The fork above is moot.** None of the four options is needed: the pad fits inside the page
rounding `alloc_group` already does. `malloc.c` computes `needed = size*cnt + UNIT` and then
`needed += -needed & (pagesize-1)`, so there is slack after the last slot, and measured over **every
large class and every reachable `cnt`** — the `med_cnt_tab` value, each halving the two loops can
reach, 1, and the `cnt==1 -> 2` rule — that slack is **always at least the pad**. No over-map, no
second guard narrowing, no leak, no reduced capacity.

The margin is exact in places, which is why it is asserted rather than trusted: class 36 at `cnt` 7
has **16 bytes of slack for a 16-byte pad**, and class 44 at `cnt` 7 has 112 for 112.
`cap_hdr_fits(size, cnt, pagesize)` is checked in `alloc_group` — placed *after* the first
`#ifdef CAPSTONE_MUSL_MALLOC` so the guard's locked prefix is untouched, confirmed by the guard
still reporting 22 PASS and no FAIL.

**The geometry is now complete and verified.** Re-simulated against the RTL bounds oracle with all
three pieces as implemented — the seven strides, the granule-sized header in both `enframe` and
`bind_slots`, and the step-down — over the same 6,060 (class, request, slot, offset-cycle) cases:

| | failing |
|---|---|
| step-down alone, 16-byte header | 1,497 of 6,060 |
| **all three, as implemented** | **0 of 6,060** |

Every case is exactly representable *and* contained within its slot, short of the neighbouring
slot's in-band header.

**What is left in item 1 is only the remap fallback**, which the runtime lane has approved with two
conditions: a counter for "remap landed unaligned, fell back to copy", reported from the suite runs
and the Tier 2/3 workloads, and a negative test that forces an unaligned remap and checks both the
`MAP_FAILED` return and that `realloc` yields a fresh, correctly bounded block with the old
contents. `mremap` returns page-aligned addresses, so at a 128 KiB granule most huge reallocs will
take the copy — which is exactly why the count has to be measured rather than assumed, and why an
aligned-remap ABI change is the follow-up if the workloads hit it often.

## Item 1 complete: the remap fallback, its counter, and its two-sided test

`__capstone_vm_remap` takes an address and two lengths with **no alignment parameter**, unlike
`cap_vm_acquire`, so a granule-aligned remap cannot be asked for. `__capstone_malloc_group_remap`
now checks the returned base and refuses an unusable one:
`__capstone_malloc_remap_usable(base, bytes)` is `capstone_repr_ok(base, repr_round(bytes))`, and on
a refusal it increments a counter, clears the grant and returns `MAP_FAILED`. `realloc.c` falls
through to its malloc-plus-copy path on a failed `mremap`, so **the cost is a copy and never a
widened bound**.

**The count is measured, not inferred**, because `mremap` returns page-aligned addresses and the
granule is 8 KiB at 6 MiB — so the copy is the *expected* path and its frequency is the input to
whether an aligned-remap ABI change (reserve an aligned range, then `MREMAP_FIXED` into it, in the
module) is worth making. `__capstone_malloc_remap_unaligned` is read by `malloc-contract.c`'s new
`remap-align` mode, which grows 6 MiB to 12 MiB, verifies **every page of the old contents
survives** and that the grown object is writable to its last byte, and prints
`MALLOC_REMAP_UNALIGNED <n>`. `run-malloc.py` runs the mode, checks its exit, records the number in
the result JSON as `remap_unaligned_fallbacks`, and — importantly — asserts the line was printed at
all (`remap_align_reported`), so a missing count fails the suite instead of recording nothing.

**Two-sided, on the host.** `capstone/tests/exact-bounds-iii/remap-usable-test.c` exercises the
decision: a page-aligned base is **accepted** at 1 and 2 MiB, where the granule is at most a page,
and **refused** at 4, 8, 16 and 64 MiB, and accepted again once the base is granule-aligned. Six
refusals and four acceptances, so the fallback is demonstrably not dead code.

### Item 1 as landed

| piece | where | verified by |
|---|---|---|
| 7 stride entries | `0010` patch, guard-pinned | guard `FAIL`→`PASS` on the real tree |
| granule-sized group header | `enframe` + `bind_slots` | 1,497→0 of 6,060 against the oracle |
| granule-aligned object placement | `enframe` step-down | the same simulation |
| the header fits free | `cap_hdr_fits` assertion | slack ≥ pad for every class and `cnt` |
| request rounding | `heap.c malloc()` | one value to mallocng and to `publish` |
| big-path alignment | `cap_vm_acquire` alignment argument | 4/8/16/64/88 MiB measured |
| remap fallback | `__capstone_malloc_group_remap` | 10-case host test, 6 refusals |
| guard narrowing | `check-mallocng-policy.py` | 1 accepted table, 7 refused near misses |

Next: the ports' derive sites, then the two-sided suite run with the profile off as production and
on as control.

## NOT COVERED — to carry verbatim into the landing commit

Stated so that no claim in the landing commit overstates what the tests show.

* **Forcing an unaligned remap is not tested end to end.** `remap-usable-test.c` proves the
  *decision* two-sided (six refusals, four acceptances), and `malloc-contract.c`'s `remap-align`
  mode proves the *contents survive whichever path was taken* and reports the fallback count. What
  neither does is make the service return an unaligned mapping on demand — that needs a stub, and
  it is a follow-up. So the claim is "the fallback's decision is correct and its use is counted",
  **not** "the fallback has been exercised deliberately".
* **No runtime run of any of this yet.** Every number in this document is static: compiler output,
  code read on a ref, and simulation against the RTL bounds model. The two-sided QEMU suite run is
  item 4 and has not happened.
* **The ports' maxima for FFmpeg, memcached, wmem, PostgreSQL, SQLite, nginx, APR and whisper are a
  delegated search's findings** with its quoted evidence, re-verified by me only where they
  contradicted something I had already reported. APR's `SMALL_NODE_SIZE` and `APR_MEMNODE_T_SIZE`
  and the 3530300 memsys5 pool size are UNRESOLVED.
* **`mrbpage.c`'s `sizeof(RVALUE)` is a stand-in**, not mruby's real union. The offset result holds
  for every value from 16 to 128 bytes, so the conclusion does not depend on it, but the figure 80
  comes from a layout computation and not from a build with debug info.
* **`enframe`'s offset cycling is reduced, not preserved**, on the twelve classes with a granule
  above `UNIT`: 1 offset in granule/`UNIT`. The double-free trap is weaker there in proportion.

## Ports, first site done: memsys5's static arena

The only offset-0 parent not placed by mallocng. `arms.json`'s `virtual-sqlite-memsys5` runs
"memsys5 stock over one static array", and memsys5 derives `mem5.pHeap` and `mem5.zPool` over the
**whole** arena, so the arena's own base must be a multiple of its bounds granule. All three
declarations were `__attribute__((aligned(16)))` — enough while QEMU kept exact shadow bounds, not
enough under the compact profile.

`CAPSTONE_REPR_GRANULE_POW2(n)` gives the granule as a compile-time constant usable in
`aligned(...)`, which cannot call a function: for a power of two, `msb(n) = log2(n)` so the granule
is `n/512`. Checked against the real function at 4096, 8192, 256 KiB, 1 MiB, 4 MiB and 64 MiB —
**agrees at every one**. `CAPSTONE_REPR_ASSERT_POW2` pairs with it so a non-power-of-two size fails
the build rather than silently under-aligning the arena; negative-tested with `-DSQLITE_HEAP_SIZE=3000000U`,
which fails with the assertion's message.

Measured, not assumed: at 4096, 256 KiB and 1 MiB the arena's `__alignof__` is 16, 512 and 2048 and
its address is a multiple of each.

Remaining port work: the eleven offset-0 sites (round the allocation with `capstone_repr_round` and
derive over the rounded length — the parent's alignment now comes free from item 1), and the nine
non-zero-offset sites, which need granule-aligned chunk placement in each allocator's own geometry.

## A sibling patch would have broken the SDK build: the changes belong in 0009

`0010-mallocng-representable-bounds.patch` is **withdrawn** and its content folded into
`0009-virtual-mallocng-capabilities.patch`.

**Why, found by running the gate rather than by reading it.** `build-sdk.sh:9-14` establishes that
the musl tree is current by requiring **every** patch to reverse-apply *independently*:

    for patchfile in .../musl-patches/*.patch; do
        if ! patch -d "$musl" -p1 -R --dry-run -s -f < "$patchfile" >/dev/null 2>&1; then
            echo "musl source is not current: run prepare-musl-capstone.sh (missing ...)" >&2
            exit 2

0010 edited lines that 0009 *adds* — the `enframe` slot-base expression and the line beside
`__capstone_malloc_group_map`. With 0010 on top, **0009 no longer reverses**, so the check reports
`0009 ... MISSING` and the SDK build refuses outright. Confirmed both ways: on a 0001-0009 tree 0009
reverses cleanly, and with 0010 applied it does not.

So the chain's structure is a constraint, not a convention: **two patches must not touch the same
region.** Representable bounds is part of the same concern as mallocng's capability representation,
so it belongs in 0009, whose header now records it.

Re-verified on a tree built from the new chain: all nine patches reverse-apply independently
(**currency gate PASS**) and `check-mallocng-policy.py` reports **22 PASS, no FAIL**.

## Critical path, as of 2026-10-17

1. **SDK build** from the prepared musl — `build-sdk.sh` then the application SDK. Not yet run.
2. **The two-sided suite run**, profile off as production and on as control: `run-malloc.py`,
   `run.py`, `run-fp.py`, `run-pthreads.py`, `run-hosted.py`. **This is the long pole** — it is
   serialized on the shared QEMU lock, and `run-malloc.py`'s own default timeout is 1800 s. It also
   produces the first `remap_unaligned_fallbacks` number, which decides whether the aligned-remap
   ABI change is worth making.
3. **Land the SDK** once mallocng and the launcher pass, before the ports are finished, and tell the
   board lane so Tier 3's rebuild can start — today's images require the exact-bounds bit and are
   refused without it.
4. **The ports**, one at a time in this session: the eleven offset-0 sites, then the nine needing
   granule-aligned placement. memsys5's static arena is done.

**Not parallelised with subagents, deliberately.** CLAUDE.md keeps "compiler/codegen and
capability-ABI changes" in the main session, and changing how an allocator derives a capability is
exactly that. The read-only half — bounding each site's maximum length — was delegated and is done.

## The SDK build: the 10-10 toolchain cannot do it, and musl cannot reach the runtime headers

Two findings from the first SDK attempt, both from the build's own controls rather than from reading.

### The 2026-10-10 toolchain is a dead end for the SDK

It is the only build on this machine with `capstone-image-gp` (`bd372e25f5fb`), so it looked like the
fast path. It **ICEs on 101 files** — every one under `src/math/` or `src/complex/` —

    clang: .../LegalizeDAG.cpp:1352: ... Assertion `(Res.getValueType() == Node->getValueType(0)
    || Node->getValueType(0) == MVT::Glue) && "Type mismatch for custom legalized operation"' failed.

triggered by the virtual profile's `+f`/`+d`/`lp64` flags, not by the sources: `src/math/trunc.c`
compiles fine on all three available builds with the base flags. The survey expects **1361** files
to compile for the virtual profile and got **1254**. The arithmetic closes exactly:
**1361 − 1254 = 107 = the 101 ICEs + the 6 mallocng files** that were failing for my own reason
below. So with my reason fixed the shortfall is precisely the ICEs, and **no SDK can be built with
that compiler**. The dev-current build is the only path; it is running.

### musl cannot include the runtime's headers, so the granule is deliberately duplicated

My `meta.h` change included `<capstone/representable.h>`. The survey compiles musl with
`-nostdinc` and **only musl's own include directories** (`survey-musl-capstone.py:91-99`), so that
header is unreachable, and widening the flag set would let musl reach arbitrary runtime headers.

The established bridge is the arch overlay: `ports/musl-capstone/arch-capstone64/malloc_capstone.h`,
a self-contained declaration header that `prepare-musl-capstone.sh` copies into
`arch/capstone64/`. `enframe` needs the granule on **every allocation**, so a cross-TU call would
cost one per `malloc`; the arithmetic is therefore defined **inline there**, duplicating
`representable.h`.

**Two copies can drift, so a test makes that safe rather than a comment claiming it is.**
`capstone/tests/exact-bounds-iii/granule-agree-test.c` compiles both headers and compares them over
**1,048,772 lengths** — every value to 1 MiB, plus every power of two to 2^40 with its neighbours
and granule boundaries: **0 disagreements**. Negative-tested by perturbing one copy's exponent by
one, which fails at 4096 (`granule 8 vs 16`).

### A control worth naming, because it did its job

`survey-musl-capstone.py:71` and `:253`: `src/malloc/mallocng/malloc.c` is `CONTROL_MUST_FAIL` in
the non-virtual profile and must **PASS** in the virtual one — a two-sided control on the harness
itself. It is what caught my breakage, with the right words: *"The harness is not measuring what it
claims. If this is an intended improvement, retire the control deliberately."* My failure was two
things, neither of them the code: my private musl tree never had the `arch-capstone64/` overlay
applied (so `malloc_capstone.h` was absent), and then the unreachable runtime header above.

## vp2's 16-bit node index: a module cap and a parameterised contract count

The semantics owner's constraint: the first virtual bitstream with a lifetime table keeps the
RTL's 16-bit node index, so a **flat** table holds at most 65,534 usable ids (0 and 1 reserved).
The 30-bit widening arrives with the paged table.

**The module refuses growth past the ceiling.** `node_max_records` is a new `module_param`,
defaulting to **0 = no cap beyond the namespace size**, so a build that does not set it behaves
exactly as before. Set to 65,536 under vp2, no id above 16 bits is ever allocated. It is checked in
both places that can grow the table — `ensure_nodes` beside the existing
`required > CAP_REV_TABLE_NAMESPACE_SIZE` test, and the `CV_NODES` ioctl beside its own — so the
refusal travels the ordinary ENOMEM/resource path rather than truncating an id.

**The two contract counts are parameters.** `node-contract.c`'s `NODES` and
`malloc-contract.c`'s `COUNT` were both a literal 70,000, which exceeds the cap by construction.
They now read `CAPSTONE_NODE_COUNT` / `CAPSTONE_MALLOC_COUNT`, **default 70,000 unchanged**, and
each prints the count it used (`NODE_COUNT_USED`, `MALLOC_COUNT_USED`).

**The runners record it, and one of them needed it more than it looked.**
`run-nodes.py:77` had the count hardcoded *inside a check* —
`log.count('NODE_LIVE:0:70000') == 2` — so a different count would have made that check compare
nothing while still reporting a verdict. It is now built from `--node-count`, and `run-malloc.py`'s
check name `70000_simultaneously_live_objects` is renamed off the literal. Both record the count in
`result.json`, and both assert the domain printed it, so a missing count **fails** rather than
reading as a zero.

**The substitution is asserted, because without it the parameter would have been silently inert.**
Both runners write the count into their gate script by replacing a placeholder; the first version
of this change added the placeholder and never substituted it, which would have exported
`CAPSTONE_NODE_COUNT=__NODE_COUNT__`, failed `strtoul`, and fallen back to 70,000 — working
perfectly at the default and ignoring every other value. There is now an
`assert '__NODE_COUNT__' not in gate_text`, the hash recorded as `gate_sha256` is taken from the
substituted text so the record describes the script that ran, and both substitutions are verified
to produce `70000` and `60000`.

**Headroom, from the owner's figures:** the runtime gates peak at 6,372 live nodes and the ports'
high-water mark is 13,054, both far below 65,534.

**Not compile-tested here:** the module needs kernel headers and `cap_rev_table_abi.h`, which comes
from a submodule a worktree does not have. The three C files that can be checked on the host are
clean; the module's build will check the rest.

## The ABI v6 SDK exists and builds v6 domains — 2026-10-17

**Toolchain.** `/tmp/capstone/llvm-dev-current-20261017/bin`, clang `85b1daddf230`. That is my lane
branch, but the compiler is **dev's**: `git diff origin/dev..HEAD -- llvm/ clang/ lld/` is 0 files,
and of the 26 commits on dev since my base, **0** touch the compiler. A standalone `bin` + `lib`
copy (867 MB) made out of the session scratchpad so it survives the session, with `ldd` confirming
**zero** libraries resolve back to the original build directory — the RUNPATH is `$ORIGIN/../lib`
with an absolute fallback, so a partial copy would silently keep using it.

**C-80's ICE is gone, measured two-sided.** Of the six files that ICE on the 10-10 build under the
full virtual flag set: **6 compile here, 0 compile there.** The test discriminates the compilers
rather than merely passing.

**The musl survey is clean: 1,361 compiled, 0 failed, 100.0%** — against an expected 1,361. The
earlier 1,254 was 101 C-80 ICEs plus the 6 mallocng files my own tree-prep and include had broken.
The policy guard passes, including `size_classes` with exactly the seven authorised entries verified
by value in the canonically prepared tree.

**Verified end to end, not inferred:** a domain built with the SDK's own `capstone-cc` carries
`.capstone_virtual = 0x364d56564e4f5043` — `CPONVVM6`, **ABI v6**. v5 was `0x354d…`.

**Three configuration traps, each of which cost a build, recorded so nobody repeats them:**

1. **`BUILD_SHARED_LIBS=ON` is required.** Capstone is a RISCV fork and both targets export the same
   non-static symbols (`getPredicatedOpcode` at `CapstoneInstrInfo.cpp:1799` and
   `RISCVInstrInfo.cpp:1640`, `isRegImmLoadOrStore`, the `isTune*Fusion` family). With static
   archives, linking `bin/lld` pulls both in and dies with pages of `multiple definition`. Not a dev
   regression — those symbols date from the original rename, and all three existing builds link
   `lld` fine. Copy the flags from an existing `CMakeCache.txt`.
2. **`-DLLVM_USE_LINKER=lld` fails at configure**: the host compiler cannot use lld.
3. **`CAPSTONE_LLVM_BUILD_DIR`, not `CAPSTONE_LLVM_BIN`,** is what `capstone-domain.cmake` derives
   `CMAKE_C_COMPILER`, `CMAKE_AR` and `CMAKE_LINKER` from. Setting only the latter leaves the
   compiler unset, cmake dies at `${CMAKE_C_COMPILER} -print-resource-dir` with "No such file or
   directory", and then reports a **misleading** "CMake was unable to find a build program
   corresponding to Ninja" — ninja was on `PATH` the whole time.

**NOT YET DONE, and the board lane has been told so explicitly:** the two-sided suite run. The SDK
exists and produces v6 domains; it has **not** been through `run-malloc.py`, `run.py`, `run-fp.py`,
`run-pthreads.py` or `run-hosted.py` with the profile off as production and on as control. Landing
waits on that, as does any claim that ABI v6 works rather than merely builds.

## Cross-checks from the board lane's kit, and what they settled

**The module compiles.** The change I had flagged as never-compiled builds cleanly for both kernels
— the QEMU guest's 6.1.0 (`97334f0fadd8ffd4`) and the board's 6.4.14 rv64g-no-RVC
(`dc0a0de0ea408a7b`), no warnings in either log. The flagged risk did not materialise.

**The SDK reproduces byte for byte.** All 89 built files — objects, the `.a` libraries, headers and
`capstone-cc` — plus `libc-capstone-virtual.a` are byte-identical between an independent kit build
and mine. The 13 files that differ are CMake/ninja bookkeeping and `sdk.json`, and every difference
is an absolute path.

**So the tree hash I published was the wrong instrument**, and the right one is now on record: a
whole-tree sha256 over a *build directory* cannot match across two paths, because the build records
its own location. `d0cddbb4…` is only useful for detecting a change to *my* copy. Comparing the
**built artifacts** is what establishes reproducibility, which is what the kit did.

**QEMU: two binaries at the same revision, and the difference is a configure option.** Both are
`d35af47b163c`. Mine is 63,722,024 B (`562bdded…`) from
`--target-list=riscv64-softmmu --disable-docs --disable-werror --enable-slirp`; the board lane's is
41,925,792 B (`40264f5de09ef153`) from `--enable-slirp --enable-debug --target-list=riscv64-softmmu`.
The material difference is **`--enable-debug`**, which theirs has and mine does not — QEMU
assertions on, optimisation off. Mine matches the recipe the project's READMEs document.

**Named binary: the board lane's.** Reasons, in order: it is already in use for the pilot and the
board's predictions, so one binary spans both lanes without a re-run; and **assertions on is the
safer side for results that feed a prediction** — C-80 is precisely a case where a release build
"passed the mismatch on silently" while an assertions build stopped. Mine stays as the cross-check,
and both hashes are recorded so a future disagreement can be attributed.

**Also merged:** `origin/dev` into the lane branch (29 commits), which brings `e012dd105f00` — the
fix for the app-port SQLite breakage my `423d13f5b81f` caused. Without it `run.py`'s
`sqlite_write`/`sqlite_read` steps have no `sqlite3.dom`. Merged rather than rebased, on a tree
other lanes share.


## Two launcher defects, and a RETRACTED attribution for the failures that found them

The v6 malloc contract fails. Two real defects were found in the launcher while chasing it, both
fixed; **neither of them explains the failures**, and the attribution that said they did is
withdrawn here. The failures are in the ALLOCATION path and are still open.

### Defect 1: a heap of 4 MiB or more could never grow

`acquire_mapping` had an exact-size mapping path keyed on `alignment == 4096`, and it is the only
path that sets `bytes = visible`; everything else takes `reserve(rounded(visible))`, which pads to a
power of two. `remap_mapping` matches a heap on `m->visible == old && m->bytes == old`. Option
(iii) made `__capstone_malloc_map` request the bounds granule, which exceeds a page at and above
4 MiB, so every heap that large silently left the exact-size path, got padded, and from then on
`remap_mapping` refused it unconditionally.

The condition conflated two things: *the heap needs a mapping of exactly its visible size* (its
growth contract) with *4096 is the alignment `mmap` gives for free*. It is now keyed on the heap,
and the requested alignment is honoured by `reserve_aligned()` — over-map by `alignment - 4096`,
align up, trim both ends — which at 4096 has zero slack and reduces exactly to the previous
`mmap`.

### Defect 2: the domain-side representability refusal was a use-after-move

`remap_mapping` grew a heap with `mremap(MREMAP_MAYMOVE)`, whose result is only page-aligned, so a
grown heap's base can be unrepresentable; in-place growth alone trips it, since 2 MiB -> 4 MiB keeps
the base and moves the granule from 4096 to 8192.

`__capstone_malloc_group_remap` was written to absorb that, and **cannot**: it runs after
`__capstone_vm_remap` has already moved the mapping and re-registered it through
`CV_REMAP_BEGIN`/`END`. Returning `MAP_FAILED` there sends `realloc.c` down a malloc-plus-copy path
that reads from an address which no longer maps. So the commit message of `5326e1026189` and this
document both claimed "the cost is a copy and never a widened bound", and that is **withdrawn**: a
nonzero `__capstone_malloc_remap_unaligned` was a correctness bug, not a cost.

The launcher now reserves a granule-aligned destination and uses `MREMAP_MAYMOVE|MREMAP_FIXED`, with
everything that can fail done before the source is touched — `mremap` fails atomically, so an error
leaves the old mapping intact and only the reservation to release. The counter and its contract mode
stay, as an assertion: nonzero now means a launcher bug.

**The generalisable shape, which is why this is written down rather than just fixed:** a guard that
runs after the state it guards has already changed is not a guard. The design said "refuse and fall
back"; nobody asked *where does the refusal run relative to the move?* That question is cheap and it
is the one that settles whether a fallback exists at all.

### RETRACTED: that these defects cause the observed contract failures

They do not, and my own negative control is what refuted it. A pre-fix launcher
(`539fa80816c490b3`) and the fixed one (`54581b7c0930a21e`) were run against the same domain and
images: **all 22 checks identical.** The condition also provably cannot bite below 4 MiB, because
`__capstone_malloc_map` clamps the requested alignment to 4096 and `granule(n)` only exceeds that at
n >= 4 MiB — while the first failure is a 5001-byte `posix_memalign`.

**The question that was available in advance and not asked: what can this control not tell apart?**
The remap-family checks cannot distinguish anything, because their *setup allocation* fails and the
remap code never runs. One reading of the contract source would have shown that before the arm was
spent.

A second misreading, corrected: the `exit:139` lines are the **expected** denial faults
(`domain fault cause=25`, then SIGSEGV), i.e. those checks PASS. They were reported as a collapse.

### Where the failures actually are

Every real failure is an explicit `exit:1` with a reason, and all of them are allocations:

| marker | the call |
|---|---|
| `MALLOC_FAIL:aligned allocation` | `posix_memalign(&p, 4096, 5001)` returns nonzero |
| `MALLOC_FAIL:population array` | an array allocation returns NULL |
| `MALLOC_FAIL:remap-align setup` | the 6 MiB allocation, before any remap |
| `MALLOC_FAIL:remap stale setup` | same shape |
| `MALLOC_FAIL:precise allocation` | `malloc(200003)` returns NULL |

The board lane's independent Tier 3 readings are the same family: NULL at `malloc(6144)`,
`calloc(1, 34816)` and `calloc(1, 35840)`, each the **first** allocation in a fresh process, while
`malloc(3072)` and `calloc(1, 12288)` succeed as first allocations in theirs. So it is deterministic
and history-independent, which makes it reproducible without the board and without a boot.

Not an arena exhaustion: `CAPSTONE_VM_STATS` reads `arenas=9` at the failing basic test and
`arenas=12` at the churn test, which passes.

Two live suspects, both of which I wrongly ruled out when first asked:
- **the request rounding in `malloc()`** (`capstone_repr_round` on the requested size);
- **the granule-sized group header** in `bind_slots`/`enframe`. This one only grows above a stride
  of 16388, which fits 34816 and 35840 but **not** 6144 — so there are plausibly two distinct
  causes in that table, and treating it as one is how the next wrong attribution happens.

`aligned_alloc` is the sharpest lead for the 5001 case: it over-allocates and then **re-bases the
object inside its slot**, adjusting the offset field, and option (iii) requires the re-based
base/length pair to be representable. That re-basing is exactly what the `0009` patch touches.

### Reporting defects fixed in `run-malloc.py`, found while reading its output

Three, all of the same shape — a constant or an input echoed where a reading belongs:
- `virtual_vm_abi` was a hard-coded `5`, so **every v6 run reported "ABI 5"**. It now reads the
  magic off the image and errors if it cannot identify it.
- `malloc_count` echoed the **requested** count; the collapsed arm published `malloc_count: 70000`
  beside `malloc_count_reported: false`. Now reported only when the domain echoes it back.
- My own three-arm wrapper recorded `rc=0` for arms whose status was `FAIL`, because `rc=$?` was
  inlined into an `echo` containing `$(date -Is)` and the command substitution resets `$?`. The
  project rule is "branch on a gate's status, never echo it"; this is that, in the harness, one turn
  after fixing two reporting defects in the same file.

### `stub.S` reproducibility, and how nearly it was dismissed

`capstone-vexec` was not bit-reproducible: with no `.file` directive the assembler records the
driver's randomly named temporary. Fixed with `.file "stub.S"`.

Worth recording **how** it was confirmed. A byte-scan of the shipped binary for `ccXXXXXX.s` and for
`/tmp/` paths found **zero** of each, which reads as "does not reproduce" and would have discarded a
correct report. The build-twice test settles it: without the directive, two builds of identical
source give `73ce7b1bc0411029` and `b1b9c8a5670e137a`; with it, `54581b7c0930a21e` twice. A probe
keyed to one spelling of a symptom is the narrowed view the house rules warn about, and the
unfiltered artifact here is the second build, not a better grep.


## Cause A found, fixed, and PRE-REGISTERED before the confirming run

### What it was

The granule-sized group header had nowhere to live on the two **single-slot** shapes.
`alloc_group`'s `cnt<=7` branch maps exactly `pageround(n+IB+UNIT)` and the `>= MMAP_THRESHOLD`
path maps `n+IB+UNIT`; each then binds ONE slot covering the whole mapping after the header. So the
slot ended `granule - UNIT` bytes past the mapping, `bind_slots`' shrink could not be derived, and
`malloc` returned NULL.

Measured over 300,000 request sizes, by path actually taken:

| path | sizes | overrun | first |
|---|---|---|---|
| multi-slot `group` | 10,908 | **0** | — |
| `group`, cnt==1 branch | 120,143 | **114,687 (95.5%)** | n = 16,365 |
| `>= MMAP_THRESHOLD` | 168,949 | **168,949 (100%)** | n = 131,052 |

**`cap_hdr_fits` was blind to every one of them.** It is evaluated on the CLASS stride and on
`needed = size*cnt + UNIT`, and neither is what those paths map — so it passed on all 114,687
overrunning group cases. An assert that cannot fire for the shape that needs it is the
"gate whose condition the mandated control always satisfies" pattern, in the allocator.

This accounts for every failure in the suite and in the board lane's Tier 3 readings **except two**,
and the two exceptions are both `cnt == 3` multi-slot allocations — `malloc(6144)` and the
`malloc(9081)` behind `posix_memalign(4096, 5001)`. Those are **Cause B, still open**; the model
says their geometry fits, so it is a different mechanism and must not be folded in.

### The fix

`cap_single_slot_bytes` reserves the header for both shapes, as a fixed point in two variables: the
header depends on the stride, the stride on the mapping size. It is also a fixed point of the
**bounds** rounding, because `__capstone_malloc_map` applies `capstone_repr_round` and a merely
page-aligned size can be granted a larger mapping than `maplen*4096` records — 6,303,744 becomes
6,307,840 at a granule of 8192 — which would leave `get_stride` and `bind_slots` disagreeing about
where the slot ends. `get_stride` subtracts the granule header rather than `UNIT` for these groups,
so enframe's slack, free's bounds check, realloc, `malloc_usable_size` and `aligned_alloc` all see
the stride that was actually bound.

Verified in the model and again in C over 300,003 sizes: converges in **at most 3 rounds**, 0
overruns, 0 capacity failures, 0 sizes where the granted mapping differs from `maplen*4096`. Only
4.8% of sizes need a larger mapping at all; the worst growth is 32 pages, at 64 MiB.

### The gate had to be extended, and that is flagged deliberately

`check-mallocng-policy.py` FAILED on "group size/count/mmap decisions", correctly: it freezes
`alloc_group`'s decision region and the mapping SIZE is computed there. The size genuinely has to
change, so the region cannot stay byte-identical.

It is extended the way the seven size-class strides already are — `ALLOWED_GROUP_SIZE_EDIT`, pinned
as exact text on BOTH sides — and **not** relaxed to "the region may differ". The path selection is
untouched: the branch condition is still evaluated on upstream's own `req`, and only `needed`
changes. `test-group-size-narrowing.py` proves the allowance is narrow, 8 of 8: the authorised edit
passes, a revert to upstream passes (policy-identical — so a PASS there does **not** mean the fix is
present, which the test says in its own output), and six near misses are all refused, including
reserving for `req` instead of the original request, a second edit elsewhere in the region, a
relaxed mmap-worthiness condition, the stock build losing upstream's sizing, the non-convergence
check dropped, and the branch condition itself made Capstone-specific.

Flagging it rather than burying it: this changes what the gate certifies. It still certifies
upstream's allocation POLICY; it no longer certifies that the mapping size is upstream's.

### A consequence of option (iii) that is not a bug and should not be discovered later

`malloc()` rounds the request up to the granule and publishes a capability over the ROUNDED size,
so an allocation can legitimately address up to **granule - 1 bytes past what the program asked
for**: 189 bytes at `malloc(200003)`, whose granule is 256. A spatial overflow inside that slack is
therefore NOT caught. The contract's own `bounds` mode is written to expect a fault at exactly
`+200003` and so can no longer pass as written.

This is the cost of option (iii) over a wider capability, it is measurable, and it belongs in the
discussion of what exact bounds buy. Recording it here; the paper is not this lane's to edit.

### PRE-REGISTRATION — written before the run, so it cannot be fitted afterwards

Inputs: domain `01482e9dee66d862` (fixed), control domain `2c1daf0f0833a4fb` (pre-Cause-A),
launcher `54581b7c0930a21e`, SDK runtime `24ebdd1f8b936e1f`, lane commit cited in the landing.

| marker / check | pre-fix | predicted | why |
|---|---|---|---|
| `MALLOC_FAIL:precise allocation` — `malloc(200003)` | fails | **clears** | large path, Cause A |
| `MALLOC_FAIL:population array` — `malloc(70000*16)` | fails | **clears** | 1,120,000 B, large path |
| `MALLOC_FAIL:remap-align setup` — 6 MiB | fails | **clears** | large path |
| `MALLOC_FAIL:remap stale setup` — `malloc(200000)` | fails | **clears** | large path |
| `MALLOC_FAIL:aligned allocation` — `malloc(9081)` | fails | **STILL FAILS** | cnt==3, Cause B |
| `bounds` | false | **still false, and NOT at "precise allocation"** | request rounded to 200192, so `+200003` is in bounds and cannot fault |
| `remap_unaligned_fallbacks` | null | **0** | the launcher now lands growth granule-aligned |
| `MALLOC_WHY` line | absent | **present at the aligned-allocation failure** | the new probe names which step refused |
| control arm | — | **must still show the four Cause-A failures** | otherwise the attribution is void a second time |

If the control arm does NOT reproduce them, this attribution goes the way of the last one.


## The pre-registration FAILED: Cause A is not the mechanism either

The run is in, against the pre-registration above, and it refutes it. Inputs as registered:
domain `01482e9dee66d862`, control `2c1daf0f0833a4fb`, launcher `54581b7c0930a21e`.

**Every one of the four markers predicted to clear still fails**, and the control arm is
**identical to the fixed arm in all 22 checks** — the same null result as the launcher control
before it. So the single-slot geometry was not what returned NULL, and Cause A joins the launcher
mapping defects as a real but LATENT bug. Writing the prediction down is what made this
unambiguous; without it the four surviving failures would have been easy to read as progress.

**What the probe says, and it is the first direct evidence in this investigation:**

    MALLOC_WHY bind_refused=0 ng_null=1 ...

`bind_refused = 0`. The slot shrink was **never refused**, in any arm. That is the step the whole
geometry argument turned on, so the argument is dead regardless of whether the arithmetic was
right — and the arithmetic was right, which is the instructive part. A correct model of the wrong
step predicts nothing.

**What replaces it: ONE threshold explains all 20 observations.** The request fails if and only if
the group mapping mallocng would need exceeds **16384 bytes**, i.e. four pages:

| call | needed | rounded() | predicted | observed |
|---|---|---|---|---|
| `malloc(3072)` | 16384 | 16384 | ok | ok |
| `calloc(1, 12288)` | 16384 | 16384 | ok | ok |
| `malloc(6144)` | 20480 | 32768 | FAIL | FAIL |
| `posix_memalign(4096, 5001)` -> `malloc(9081)` | 28672 | 32768 | FAIL | FAIL |
| `calloc(1, 34816)` / `(1, 35840)` | 36864 | 65536 | FAIL | FAIL |
| `malloc(200003)`, `malloc(200000)` | 200704 | 262144 | FAIL | FAIL |
| population array, `malloc(70000*16)` | 1122304 | 2097152 | FAIL | FAIL |
| 6 MiB | 6295552 | 8388608 | FAIL | FAIL |

plus the ten small successes from 64 to 1664 bytes. **20 of 20**, and the discriminating pair is the
inversion: `malloc(6144)` fails while `calloc(1, 12288)` succeeds, because 6144 lands in a group
needing 20480 bytes and 12288 lands in one needing exactly 16384.

This also collapses the Cause A / Cause B split. There was never a second mechanism: both
"Cause B" cases — 6144 and 9081 — are on the same side of this one threshold. Splitting them was an
artefact of modelling the geometry rather than the limit.

**Where the limit is NOT.** `CV_MAX_REGION_BYTES` and `CAP_VM_MAX_BYTES` are both 256 MiB
(`wire.h:60`, `vm.h:17`), and `CV_MAX_ARENAS` is 512 against a measured `arenas=9`, so neither is
it. The only 16384 in the runtime headers is `CAPSTONE_DELEGATE_META_BYTES`
(`include/capstone/delegate.h:50`), which is suggestive and unproven.

**The honest state of the localisation.** The launcher's acquire appears in the trace as
`CAPSTONE_VM_SERVICE nr=0xC0DE0001`, and there are exactly three such calls in the failing process,
all `result=0`, none at the failure. Read literally that says mallocng returned NULL without ever
requesting a mapping. It is NOT yet conclusive, for two reasons I would rather state than paper
over: a `CAPSTONE_VM_BOUNDARY kind=2` does appear immediately before the failure and its result is
not printed, and my own probe conflates two different NULLs — `ensure_ids(256)` failing and
`__capstone_mallocng_malloc` returning NULL both increment `ng_null`. `ensure_ids` is called with a
constant and cannot be size-dependent, so it is unlikely, but "unlikely" is not the standard this
investigation has already failed twice.

**Next, and it is cheap:** split that counter, and record the acquire's return value in
`__capstone_malloc_map`. One rebuild, one run, and the answer is either "the acquire was refused,
with this errno" or "mallocng never asked". Everything above is then either confirmed or replaced.

### What this cost, and the rule that would have stopped it

Three attributions in one session, two of them published to other lanes before a control had
spoken:

1. the launcher's `alignment == 4096` keying — refuted by its own control;
2. "growth-timing, not size classes" — refuted by the same control;
3. the single-slot group geometry — refuted by this pre-registered run.

All three were derived from correct readings of real code, and all three named a step that was
never the failing one. The project rule that covers this is already written — *ask what the
instrument cannot distinguish, before the claim goes out* — and the specific omission each time was
the same: no measurement said which step refused. The probe added this round is the first one that
does, and it answered in a single run. **The lesson is to instrument the choke point before
modelling the mechanism**, because a model of the wrong step is indistinguishable from a model of
the right one until something measures.


## The mechanism: the module enforces a predicate far stricter than representability

Found by the board lane reading `capstone_vm.c` on this branch, verified here in source and by an
arm already on disk. Two gates, both keyed on `c->exact_bounds`, which option (iii) deliberately
runs with **OFF**:

- **`add()`** — `(!c->exact_bounds && (!is_power_of_2(r->bytes) || (r->address & (r->bytes - 1))))`
  returns `-EINVAL`. With exact bounds off, every arena must be a **naturally aligned power of
  two**.
- **`remap_begin()`** — `if (!c->exact_bounds || ...) return -EINVAL`. With exact bounds off, **no
  heap growth is permitted at all.**
- `c->exact_bounds` comes from CSR `0x5c0` bit `0x100` (`capstone_vm.c` ~1107).

mallocng's group mappings are page multiples and usually **not** powers of two — 20480, 28672,
36864, 200704, 1126400, 6295552 — so `CV_ADD` refuses them and `malloc` returns NULL, printing
nothing. That is the whole failure set, and `remap_begin` is `heap growth/zero` directly.

### The confirming run, pre-registered, 7 of 7

Domain `de4da05131662957`, exact bounds OFF, with the choke point instrumented:

    map_calls=3  map_acquire=2  map_early=0
    last_bytes=1126400  last_align=4096  last_errno=22 (EINVAL)
    bind_refused=0  ng_null=2  ids_failed=0

All four well-formed readings show errno 22, and 1126400 is not a power of two. `map_calls`
advancing also corrects my earlier reading that mallocng never requested a mapping — it did; the
absence of a `CAPSTONE_VM_SERVICE nr=0xC0DE0001` line at the failure was not the evidence I took it
for.

**Two-sided, from an arm already paid for:** the same suite with exact bounds **ON** shows **zero**
occurrences of all four `MALLOC_FAIL` markers, against the off arm failing every one. That is what
makes the `exact_bounds` keying established rather than inferred, and it is also why dev's v5 run
passed — both gates are skipped when the profile is on.

### Why my own "needed > 16384" was withdrawn

It and `is_power_of_2(needed)` both score 20/20 on the same 20 observations and are
**indistinguishable on that data**, because every size that succeeded is BOTH a power of two AND at
most 16384. The source decides, and it says `is_power_of_2` literally. The empirical separator, if
one is ever wanted, is `malloc(9341)`: its group needs 32768, a power of two above 16384.

This is the third time this session a fit over real data named the wrong mechanism. The pattern is
now explicit enough to state as a rule: **a predicate inferred from a sample is a hypothesis about
the sample, not about the code — go and read the condition.**

### How bad the current predicate is, measured

Over 400,000 request sizes, for the mapping each one actually produces:

| predicate | admits | refuses |
|---|---|---|
| `is_power_of_2(bytes)` (what the module enforces today) | **28,220 (7.1%)** | 371,780 |
| `capstone_repr_ok(base, bytes)` (actual representability) | **400,000 (100%)** | 0 |

So the module is refusing 93% of the allocator's mappings to enforce an invariant the hardware does
not require. Power-of-two-plus-natural-alignment is a small subset of representability, not a
restatement of it.

### The proposed fix, and why Cause A's fix is a prerequisite rather than dead weight

- **`add()`**: replace the exact-bounds-off branch with `capstone_repr_ok(r->address, r->bytes)`.
  Strictly more permissive and still sound — it is the encoder's actual condition. The module keeps
  its own page checks, the `CV_MAX_REGION_BYTES` and total-bytes limits, and the overlap test;
  `capstone_repr_ok` returns granule 1 below 4 KiB and so adds no page rule of its own.
- **`remap_begin()`**: replace the blanket `!c->exact_bounds` refusal with the same predicate on the
  post-growth base and length, so growth is allowed exactly when the result is representable.

**One predicate, shared, not duplicated.** `include/capstone/representable.h` has no `#include` at
all and uses only `__builtin_clzl`, `_Static_assert` and `unsigned long`, every one valid in kernel
C, and `build-adapter.sh:10` already copies `cap_rev_table_abi.h` out of the QEMU tree into the
module build dir. Copying `representable.h` the same way gives the module the same predicate that
`repr-cross-check.py` validates against the encoder oracle — so there is no module-side copy that
could drift from the silicon encoder, which is the failure mode that would hand out a widened
capability on the board with no QEMU signal. Copying it from the tree being built also stops a kit
pairing a module and an SDK from different revisions.

**And this is where Cause A's fix stops being latent.** The relaxed predicate needs the mapping
LENGTH to be a granule multiple. `granule > 4096` only happens at 4 MiB and above, which is exactly
the single-slot path — and the Cause A fix is what made that path's size granule-stable. Multi-slot
groups are capped at `65536*UNIT` = 1 MiB by mallocng's own `size*cnt >= 65536*UNIT` guard, so
their granule is at most 2048 and a page multiple always satisfies it. Measured: 0 of 400,000
mapping lengths fail the granule test. The two changes are a pair.

**Not proposed:** rounding every arena up to a power of two. It would also pass `add()` as it
stands and would waste up to half of every mapping.

### Still open before this is written

The module reaches silicon — the board runs option (iii) with exact bounds off, so both gates refuse
the heap there too and the same `capstone_vm.c` builds both modules. So this is not a QEMU-only fix
and it is not this lane's to land unannounced.

And one gap is under audit rather than settled: `EINVAL` is **also** what the launcher's
`acquire_mapping` returns for its own validation failures, so the errno alone does not prove the
refusal came from `add()` rather than from the launcher before it. `grant()`'s representability
check returns `ERANGE`, which separates one path but not all three. That audit is the gate on
recording this as the root cause.


## The audit of 502ca09f0e87: one limb SUPPORTED, two REFUTED

`claim-auditor` was run over the previous section before it should have been pushed — the commit
502ca09f0e87 says "that audit gates recording this as the root cause" and then recorded it anyway,
which is the defect, not the audit's finding. The corrections below supersede it.

### SUPPORTED, and more strongly than I had it

**`add()`'s power-of-two gate is what refuses mallocng's group mappings, and the launcher is
EXCLUDED, not merely unlikely.** The gap I asked to be attacked hardest did not survive the attack:

- For all **four** measured `(visible, alignment)` pairs, **no** `acquire_mapping` EINVAL disjunct
  fires (`exec.c:243-253`), evaluated numerically rather than by hand.
- `grant()`'s representability check returns **ERANGE**, not EINVAL (`exec.c:197`), and computes
  `repr_ok = True` for all four anyway.
- `madvise`'s EINVAL is explicitly tolerated (`exec.c:131`); an mmap failure gives ENOMEM.
- The `CV_ADD` ioctl branch has no EINVAL pre-check — only `copy_from_user` (EFAULT) and
  `find_thread` (ESRCH) (`capstone_vm.c:902-907`).
- Inside `add()`, every disjunct before line 466 is false for all four, so
  **`!is_power_of_2(r->bytes)` is the first true disjunct in each case**.

**An internal control that was already in my own data and I did not use:**
`map_calls - map_acquire - map_early` is 1 or 2, so in the *same process* as each refusal at least
one `cap_vm_acquire` **succeeded**. That proves the launcher, module and thread state all work
there, and the only thing differing on the failing calls is the size — which closes the two
state-dependent EINVAL residuals (`capstone_vm.c:472`, and the overlap test at `:479-481`) that no
amount of input arithmetic could reach.

**One clause the claim was missing, and it matters for where the fix belongs.** The non-heap path
grants a naturally aligned power of two — `rounded()` doubles to one (`exec.c:123-129`) and
`reserve()` natural-aligns to it (`exec.c:149`) — and so *passes* line 466. The gate bites only
because `exec.c:266` deliberately sets `bytes = visible` for the v6 local-malloc heap. The module
predicate is the refusal; that line is why the heap meets it.

### RETRACTED: "`remap_begin()` refuses ALL heap growth when exact bounds are off"

The source text is real (`capstone_vm.c:562`). **Nothing shows it executed, and opening the gate
does not restore growth**, which is evidence against the causal half:

- `CAPSTONE_VM_REMAP_OK` is **0 in all three arms, including exact-bounds-ON**. The instrument is
  proven live — it is gated by the same `static int trace` as `CAPSTONE_VM_BOUNDARY` and
  `CAPSTONE_VM_SERVICE`, which fire hundreds of times in every log — so that zero is meaningful:
  **no heap growth completed anywhere, gate open or shut.** The gate has no positive control.
- `real_linux_mremap_growth_and_shrink` is False in **off, ctl AND on**. If `!c->exact_bounds` were
  what refuses growth, it should pass with the profile on. It does not.
- In the OFF arm the only growth test never reaches a remap at all: `remap_align`
  (`malloc-contract.c:183-207`) dies on its *setup* `malloc(6u<<20)`, an `add()` refusal, before the
  `realloc`.
- **"`remap_begin` is run.py's `heap growth/zero` directly" is unsupported.** run.py's only growth
  token is `run.py:207`, a `VIRTUAL_HEAP_THREADS_OK ...` line count; nothing ties its value to
  `remap_begin`.

### RETRACTED: the two-sided control, as I stated it

I wrote that the exact-bounds-ON arm "shows zero occurrences of all four `MALLOC_FAIL` markers".
That is true of those four strings and **false as a control**:

- the ON arm's own verdict is **FAIL**, with 5 of 8 failures surviving;
- **only 3 of 22 checks change** when the profile is flipped (`simultaneously_live_objects`,
  `remap_align_contents_survive`, `remap_align_reported`);
- two of the four markers were **replaced by different failures, not by passes** — `aligned
  allocation` becomes a `Cap mem access OOB` and `MALLOC_BASIC_EXIT:139`, and `precise allocation`
  becomes **`MALLOC_FAIL:denial survived`**, i.e. a *safety* failure where an access that should
  have been denied was not;
- flipping the profile is not a single-variable change: `c->exact_bounds` comes from CSR `0x5c0`
  bit `0x100`, so the **hardware encoder changes too**, and the ON log carries a fault class absent
  from OFF (`[CAPSTONE] SHRINK illegal operand value`).

Grepping four marker strings was a narrowed view that excluded the ON arm's own fifth `MALLOC_FAIL`
and its segfault. This is the project's own "a positive finding from a narrowed view needs the same
suspicion as a clean zero", and the unfiltered artifact was the result JSON I had already parsed.

### Factual corrections to the previous section

- **Five markers, not four.** I omitted `MALLOC_FAIL:remap stale setup`.
- **All FIVE readings show errno 22**, not four. The fifth was interleaved with serial output, not
  lost — its tail reassembles to `last_bytes=28672 last_align=4096 last_errno=22`, and 28672 is also
  not a power of two. Reassemble an interleaved line rather than discarding it.
- **The mapping list mixed measured with modelled and contained a wrong number.** Measured: 28672,
  200704, 1126400, **6307840**. `6295552` was wrong by one page; `20480` and `36864` were modelled
  and never measured. The 1122304-vs-1126400 slip I had already caught is the reason not to print
  modelled sizes as readings.
- **"93% stricter" is withdrawn from the headline.** The 7.1%/100% table cites no script, unlike
  every other table in this doc, and the 100% row is **circular as stated**: `capstone_repr_ok`
  takes a base, and the base is granule-aligned by construction because `heap.c:137-138` asks for
  granule alignment and `reserve_aligned` honours it. It is a property of the current launcher's
  alignment, not an independent finding — and it does not carry to `remap_end`, which checks only
  page alignment (`capstone_vm.c:597`).
- **RETRACTED: "that is why dev's v5 run passed".** `exec.c:376-378` refuses a v5 image outright
  (`ENOEXEC` at `:385`), so this launcher cannot run one and that sentence cannot be the
  explanation. I never examined that run.
- **The `ctl` arm belongs in the record** and was unmentioned: domain `2c1daf0f` with
  `--no-exact-bounds`, identical to `off` in all 22 checks, which is the evidence that the Cause A
  fix is latent.
- `last_bytes` records the **last** refusal in a process, not necessarily the one that produced the
  reported failure. Harmless here, since both refusals in each process fall the same side of the
  predicate, but the wording should not imply otherwise.

### Still source-only, and the two cheap runs that finish it

The *mechanism* is established. The **predicate** is not: `is_power_of_2(needed)` and my withdrawn
`needed > 16384` are still only separated by reading the source, because every size measured so far
is on the same side of both. Two arms settle it in one run, and the ON arm can never do it, because
turning the profile on disables the very gate whose behaviour is in question:

- **`malloc(9341)`** — its group needs 32768, a power of two **above** 16384. Power-of-two predicts
  it SUCCEEDS; the threshold predicts it is refused.
- a request whose group needs **12288** — not a power of two, **below** 16384. Power-of-two predicts
  it is REFUSED; the threshold predicts it succeeds.
- and **record `last_bytes` on the SUCCESSFUL acquire too**, not only in the failure branch
  (`heap.c:143-146`). That is the one empirical hole: nothing currently shows a non-power-of-two
  mapping being *admitted* while the gate is on.


## The predicate is settled by MEASUREMENT: it is `is_power_of_2`, not a size threshold

Domain `a20249b9d909b5ec`, exact bounds OFF (the profile must stay off — turning it on disables the
gate under test). Two requests chosen so the two surviving predicates make **opposite** calls, and
neither is on the single-slot path, so the Cause A fix cannot confound them:

| request | group needs | `is_power_of_2` predicts | `needed > 16384` predicts | MEASURED |
|---|---|---|---|---|
| `malloc(9341)` | 32768 — a power of two, **above** 16384 | ALLOCATED | NULL | **ALLOCATED** |
| `malloc(2029)` | 12288 — **not** a power of two, below 16384 | NULL | ALLOCATED | **NULL** |

    MALLOC_PRED pow2_above=ALLOCATED
    MALLOC_PRED nonpow2_below=NULL
    MALLOC_PRED admitted=2 last_ok_bytes=32768 last_ok_align=4096
    MALLOC_PREDICATE_EXIT:0

**2 of 2 for `is_power_of_2`; 0 of 2 for the threshold.** My `needed > 16384` is now refuted by
measurement as well as by source, and `capstone_vm.c:466` is confirmed as the operative condition
rather than merely the plausible one.

`last_ok_bytes=32768` also closes the empirical hole the audit identified: nothing until now showed
a mapping **above** the old threshold being *admitted* while the gate was armed, because the
exact-bounds-ON arm structurally cannot show it. Recording the success path, not only the refusal,
is what produced it — one array write.

The arm is a **reading, not a verdict**: `predicate_arms_reported` asserts only that both lines
appeared. Scoring it pass/fail would reintroduce exactly the error it was built to remove.

## Revision: what is now established, and what is NOT

**Established, with the launcher excluded and the predicate measured:**
`capstone_vm.c:466`'s `!is_power_of_2(r->bytes)` is the first true disjunct for every measured
refusal — errno 22 at 28672, 200704, 1126400 and 6307840 — and the heap meets that gate only
because `exec.c:266` sets `bytes = visible` for the v6 local-malloc heap, where every other path
grants a naturally aligned power of two and passes.

**NOT established, and deliberately out of the proposed fix:** why heap growth never completes.
`CAPSTONE_VM_REMAP_OK` is 0 in **every** arm including exact-bounds-ON, with the instrument proven
live, and `real_linux_mremap_growth_and_shrink` is False in all three arms. So `remap_begin`'s
`!c->exact_bounds` refusal has no positive control and is not shown to fire. **Relaxing it is not
justified by anything measured**, and there is a second, unidentified reason growth does not
complete. That is its own investigation, and the launcher growth fix in `09e1896f6bb3` remains
latent until it is done.

**So the fix narrows to one gate:** `add()`'s exact-bounds-off branch uses
`capstone_repr_ok(r->address, r->bytes)` instead of power-of-two-plus-natural-alignment, keeping the
module's page checks, the region and total limits and the overlap test. `remap_begin` is left alone.

`representable.h` is shared rather than duplicated — it has no `#include` and uses only
`__builtin_clzl`, `_Static_assert` and `unsigned long`, all valid in kernel C — copied into the
module build dir by `build-adapter.sh` beside `cap_rev_table_abi.h`, from the same tree being built,
so a kit cannot pair a module and an SDK across revisions and there is no second predicate that
could drift from the silicon encoder.

**Prediction for the add()-only change, to be pre-registered before it runs:**

| | predicted |
|---|---|
| the five `MALLOC_FAIL` allocation markers | **clear** |
| the board lane's NULLs at 6144, 34816, 35840 | **clear** |
| `malloc(2029)` (group 12288, non-pow2) | **ALLOCATED**, where it is NULL today |
| `malloc(9341)` (group 32768, pow2) | still ALLOCATED |
| `heap growth/zero`, `real_linux_mremap_growth_and_shrink` | **still FAIL** — not addressed |
| `bounds` | still FAIL: the request rounding puts `+200003` inside the object |

The sixth row is the point of stating the prediction: a fix that cleared the allocation failures and
left growth broken would otherwise read as a partial success, when it is exactly what the evidence
predicts.

**This is not landed.** The same `capstone_vm.c` builds the board module and the board runs this
profile with exact bounds off, so relaxing a capability-bounds check there changes what the silicon
enforces, not only what QEMU does.


## SETTLED: both disjuncts of `add():466` are operative, and one makes the failure NONDETERMINISTIC

Two independent instruments, built by two lanes without coordination, agree.

### Launcher-side, printing add()'s own disjuncts on the real values

`grant()` now prints on a `CV_ADD` refusal. The domain cannot do this — `cap_vm_acquire` clears its
slot on failure, so the base never reaches it, which is why a power-of-two size refused with EINVAL
was indistinguishable from a misaligned one:

    CAPSTONE_VM_ADD_REFUSED base=0x3f8aeb4000 bytes=49152 errno=22 pow2=0 naturally_aligned=1
    CAPSTONE_VM_ADD_REFUSED base=0x3f8af01000 bytes=28672 errno=22 pow2=0 naturally_aligned=1
    CAPSTONE_VM_ADD_REFUSED base=0x3f9483c000 bytes=49152 errno=22 pow2=0 naturally_aligned=0

**Every recorded refusal has `pow2=0`,** and two were refused while satisfying alignment — so the
SIZE half refuses non-power-of-two mappings whatever the address.

*Caveat on my own print:* `naturally_aligned` is computed as `address & (bytes-1)`, which is
meaningless when `bytes` is not a power of two (the mask is not a mask). That is precisely why
`add()` tests `is_power_of_2` first. So the field carries information only on the pow2 rows, and I
am not reading the `naturally_aligned=1` on a 49152 row as anything.

### The same arm gave OPPOSITE results on consecutive runs

| run | `nonpow2_below` | admitted | refused |
|---|---|---|---|
| `pred2` | **NULL** | last 32768 | last **8192** |
| `pred4` | **ALLOCATED** | last **8192** | last 49152 |

Same domain source, same images, same launcher behaviour — and an 8192 mapping is refused in one
run and admitted in the next. 8192 **is** a power of two, so the size half cannot explain it: the
only remaining disjunct is `address & (bytes - 1)`, and the launcher requests only
`max(granule, 4096)` alignment, so an 8 KiB mapping lands 8 KiB-aligned about half the time.

### Independent confirmation from the board lane, with no new instrument

Four boots of the Tier 3 pilot, unchanged kit, images and binaries:

- `malloc(6144)` — group 8192 — **admitted in 1 of 4 boots**, NULL in 3. Same binary, same input,
  both outcomes. 1 in 4 is within chance of the ~1 in 2 an 8 KiB-aligned landing gives.
- `calloc(1, 34816)` and `calloc(1, 35840)` — non-power-of-two groups — **NULL in all 4 boots**,
  8 refusals.

One case varies with the boot, one never varies. That is exactly what the conjunction predicts, and
it was measured without reference to my launcher print.

### What this means, beyond the fix

**A class of allocation failures here is NONDETERMINISTIC, keyed on where `mmap` happens to land.**
Any power-of-two group is refused or admitted depending on the boot. That is why the earlier
attempts to fit a size rule kept almost working: the data was a mixture of a deterministic
mechanism (non-power-of-two sizes, always refused) and a stochastic one (power-of-two sizes,
refused about half the time). Both the "needed > 16384" fit and the "predicate is
`is_power_of_2`, 2 of 2" reading were averaging over two mechanisms, which is why each matched its
sample and neither survived a purpose-built arm.

It also means **N=1 was never enough here**, and the project rule about N=1 on a system with known
nondeterminism applied the whole time without anyone knowing the nondeterminism existed.

### The fix, now fully justified on both halves

`capstone_repr_ok(r->address, r->bytes)` loosens **both** disjuncts to the encoder's actual
condition:

- **length**: a granule multiple, instead of a power of two;
- **base**: granule-aligned, instead of naturally aligned to the size.

And the second half is satisfiable as the launcher already stands: `heap.c` requests
`max(granule, 4096)`, so at these sizes `repr_ok` accepts a 4096-aligned 8192 mapping — which
removes the nondeterminism rather than merely widening the accepted set. `remap_begin` stays out:
it is still not shown to fire.

### Prediction for the add()-only change

| | predicted |
|---|---|
| the five allocation markers | clear |
| non-power-of-two groups (34816, 35840, 200003, population, 6 MiB) | clear, **deterministically** |
| power-of-two groups (`malloc(6144)` → 8192) | clear **on every boot**, not 1 in 4 |
| `CAPSTONE_VM_ADD_REFUSED` | **absent** |
| `heap growth/zero`, `real_linux_mremap_growth_and_shrink` | still FAIL — not addressed |
| `bounds` | still FAIL — the request rounding puts `+200003` inside the object |

The boot-to-boot row is the sharpest test: a fix that cleared the deterministic failures and left
`malloc(6144)` varying between boots would mean the alignment half is still biting.


## Both disjuncts now have PRINTED rows, and the fourth cell is empty

The board lane pointed out a real gap in the record above: **none of the three printed
`ADD_REFUSED` rows had a power-of-two size**, so the alignment half rested on the same arm giving
opposite answers across two runs plus a 1-in-4 board result — behavioural inference, not a measured
disjunct.

That row did not need a lucky boot. A power-of-two mapping is refused whenever `mmap` does not land
it on its own size, so forcing many group mappings in one process makes it near-certain: each class
holds only `cnt` slots, so keeping every object live forces a fresh mapping every `cnt`
allocations. The `align-probe` mode does that across a spread of classes and is deliberately
**model-free** — it does not assume which group a request lands in, since assuming that is what made
the earlier `malloc(2029)` arm measure the wrong thing.

    MALLOC_ALIGN_PROBE attempted=600 ok=187 failed=413

420 refusal rows, classified by the two disjuncts `add():466` actually evaluates:

| `pow2` | `naturally_aligned` | rows | what it establishes |
|---|---|---|---|
| 0 | 1 | **109** | refused *while satisfying alignment* — the SIZE half, directly |
| 1 | 0 | **87** | a power of two refused with a misaligned base — the ALIGNMENT half, directly |
| 0 | 0 | 224 | both conditions fail |
| 1 | 1 | **0** | nothing satisfying BOTH was ever refused |

Examples of the row that was missing, with the arithmetic spelled out:

    base=0x3f9c1dd000 bytes=8192  errno=22  base mod bytes = 4096
    base=0x3f9c1d2000 bytes=16384 errno=22  base mod bytes = 8192
    base=0x3f9c1a1000 bytes=16384 errno=22  base mod bytes = 4096

**The empty fourth cell is the confirming negative.** The conjunction predicts that a mapping which
is a power of two AND naturally aligned can never be refused by line 466; across 420 refusals, none
was. Had that cell been populated, a third disjunct would be in play and the account would be
incomplete.

So neither half is inference any more, and the earlier behavioural evidence — the pred2/pred4
inversion and the board lane's 1-of-4 — is now corroboration rather than the basis.

**Severity, measured rather than estimated:** 413 of 600 allocations fail in this profile, 69%.

### What still rests on inference, stated plainly

- That the module is enforcing an invariant the hardware does not require. `capstone_repr_ok` is
  validated against the encoder oracle by `repr-cross-check.py`, so the *predicate* is grounded, but
  no run here has exercised a granule-aligned, non-power-of-two arena through the proposed relaxed
  `add()` — that is what the fix's own run will show.
- Why heap growth never completes. `remap_begin` is still not shown to fire, so it stays out of the
  fix and remains open.


## The refusal diagnostic was itself trace-gated, so its absence proved nothing

Found by the board lane's positive control, and it applied to **this suite too**, not only to the
corpus harness. `CAPSTONE_VM_ADD_REFUSED` was printed under `if (trace)`, and an audit of this
suite's own gate script found **four of its invocations run without `CAPSTONE_VM_TRACE`** —
`population`, `churn`, `remap-align`, and the denial-case loop. Those are exactly the stages where
refusals occur, so the "ADD_REFUSED absent" row of the fix prediction would have been an unproven
zero over most of the run.

**Turning the trace on globally is not the fix.** `churn` alone performs 70,000 allocations and
would bury the serial in boundary lines; that is presumably why the flag is selective in the first
place. The print is now **unconditional**: a refusal is rare, so it cannot flood anything, and
gating a rare error-path diagnostic behind a verbosity flag is precisely what made its absence
unreadable.

**Positive-controlled, two-sided on the same images.** Stage regions that contributed **zero**
refusal rows before now contribute them — the denial loop 9, `basic` 2, `predicate` 2,
`population` 1, `hot-control` 1 — and the total rose from 420 to 432. (The per-stage attribution is
keyed to the preceding `EXIT` marker, so a label names the stage before the one that emitted the
row; the robust claim is that previously-silent regions now report, not the exact split.)

So the prediction's `ADD_REFUSED absent` row is only now a claim that can be falsified, on this
suite and on the kit. This is the "a gate that has never blocked anything is not a passing gate,
it is an unproven one" rule applied to a diagnostic rather than a gate — and it was caught by
someone else's control, not by mine.


## Why heap growth never completes: my own Defect-2 fix refuses it

Instrumenting `remap_mapping`'s refusal paths — unconditionally, for the same reason as the
`CV_ADD` one — answered this in two runs, and the answer is a regression I introduced in
`09e1896f6bb3`.

**The exact-bounds-OFF arm cannot answer it at all:** `remap-align` exits 1 on its *setup*
`malloc(6 MiB)`, refused by the module gate, so the `realloc` never runs and **zero** REMAP lines
appear. Growth is unreachable there, behind the `add()` defect. The ON arm is the only profile
where the setup succeeds, and there it says:

    CAPSTONE_VM_REMAP_REFUSED stage=args base=0x3f8fd4f000 old=200704 bytes=536875008 page_ok=1
    CAPSTONE_VM_REMAP_REFUSED stage=granule bytes=12587008 align=16384

- The first is **CORRECT and expected**: `malloc-contract.c:72` deliberately requires
  `!realloc(p, 512UL<<20) && errno == ENOMEM` ("failed VM growth preserves original"), and
  512 MiB + 4096 exceeds `CV_MAX_REGION_BYTES`. Not a defect.
- The second is **mine**. `realloc.c:38` computes `needed = (n + base + UNIT + IB + 4095) & -4096`
  — page-rounded and **never granule-rounded**. For the 12 MiB growth that is 12 MiB + 4096 =
  12587008, whose granule is 16384 and `12587008 % 16384 = 4096`. My `if (bytes % align) return
  -ERANGE` therefore refuses it, `realloc` falls through to malloc-plus-copy, and the growth never
  reaches the launcher.

**So the Defect-2 fix traded a use-after-move for a silent refusal of every large growth.** The
refusal is the safer of the two — it cannot corrupt — but it disables the functionality, and it is
why `CAPSTONE_VM_REMAP_OK` reads 0 in every arm of every profile.

**And it makes that commit's own claim false.** `09e1896f6bb3` says "a nonzero
`__capstone_malloc_remap_unaligned` now means a launcher bug, not a measurement". The counter in
fact reads 0 because my check fires *before* anything can set it. That is a clean zero from a check
that cannot fire — the class this project names as its most expensive — in a sentence I wrote while
fixing an instance of it.

**The fix mirrors the single-slot sizing, and must be domain-side for the same reason.** `needed`
has to be granule-rounded where `maplen` is computed, i.e. in `realloc.c`, so mallocng's accounting
equals what the launcher maps; rounding it up inside `remap_mapping` would leave `maplen*4096`
disagreeing with the mapping, which is the trap `cap_single_slot_bytes` was made bounds-stable to
avoid. It also touches two regions the policy gate pins (`moving realloc sequence`,
`realloc g->sizeclass>=48`), so it needs the same pinned-allowance treatment and narrowing test as
the `alloc_group` edit. Not written yet.

### Still open after this

`real_linux_mremap_growth_and_shrink` (the `realloc(p, 400000)` growth and `200000` shrink at
`malloc-contract.c:76-77`) produces **neither** a refusal nor a `REMAP_OK`, so `remap_mapping` is
never called for it. Its `needed` of ~405504 has granule 512 and `405504 % 512 == 0`, so my check
would pass it — meaning `realloc` itself decided not to ask, most likely via
`g->maplen*4096UL == needed ? g->mem : ...` at `realloc.c:39` reusing the mapping outright. Why
that check fails is not yet measured, and I am not going to model it.


## Heap growth fixed, and the granule header turns out to forbid in-place growth

Four iterations, each one measured rather than predicted-and-hoped. The suite went from
**17 true / 6 false to 21 true / 2 false**, and both remaining failures are tests encoding the
pre-option-(iii) behaviour rather than runtime defects.

### The three defects, in the order the instrument found them

1. **`needed` was page-rounded, never granule-rounded** (`realloc.c:38`), so my own
   `remap_mapping` granule check refused every large growth and `realloc` fell silently through to
   malloc-plus-copy. This is why `CAPSTONE_VM_REMAP_OK` read 0 in every arm of every profile.
   Granule-rounding it produced the **first completed growth ever observed** here:
   `CAPSTONE_VM_REMAP_OK old=200704 bytes=401408`.

2. **`__capstone_malloc_group_remap` still passed `bytes - 16`** while its sibling
   `group_attach` had been fixed to pass `bytes - group_header(bytes)`. On a 401408-byte mapping
   that overruns the slot by 496 bytes. I had fixed one call site and missed the other, and it only
   became visible once growth began completing at all.

3. **Granule-rounding the page-rounded size still left no room for the header.** At 12 MiB it gave
   a usable span of 12582908 against a request of 12582912 — short by exactly `IB` — so
   `set_size`'s `reserved = end - p - n` went negative and `end[-reserved] = 0` stored one byte
   past the object. Measured, not inferred: `Cap mem access OOB … addr = 0x3fbccf8000,
   bounds = (0x3fbc0f8000, 0x3fbccf8000)`, i.e. `addr == bounds end`. `cap_single_slot_bytes`
   reserves `IB` and the header and is already verified over the whole range, so the fix was to use
   it here too rather than write a second sizing rule.

### The finding that matters beyond this fix

**A granule-sized group header makes in-place growth across a granule boundary unsound**, and this
is a property of option (iii)'s design, not a coding slip. The header's size follows the mapping's
size, `mremap` leaves the bytes at their original offsets, so growing 6 MiB to 12 MiB moves the
header from 8192 to 16384, shifts the slot base by 8192, and every byte of the payload is then read
8192 bytes from where it actually is. It presented as `MALLOC_FAIL:remap-align contents` **with the
growth itself reporting success**, which is worse than any refusal.

So `realloc` now refuses an in-place remap whose header would shift, and falls through to its
malloc-plus-copy path, which relocates the payload correctly. The cost is a copy on growths that
cross a granule boundary. The alternative — `memmove` the payload inside the mapping — is the same
copy plus a window in which the slot bounds and the data disagree, so it is strictly worse.

**The prediction for that change said the headline number would go DOWN, and it did:**
`CAPSTONE_VM_REMAP_OK` 1 → 0, because both growths in the suite cross a granule boundary
(256 → 512 at 200704 → 401408, and 8192 → 16384 at 6 → 12 MiB). Writing that down first is what
kept a correct fix from reading as a regression.

| prediction | measured |
|---|---|
| `REMAP_OK` 1 → 0 | **0** |
| remap-align exits 0, not 139 | **0** |
| `remap_align_contents_survive` | **True** |
| OOB faults | **0** |

### The two surviving failures are tests, not defects — and are NOT mine to quietly edit

- `real_linux_mremap_growth_and_shrink` is `log.count('CAPSTONE_VM_REMAP_OK') >= 2`
  (`run-malloc.py:153`). It requires the **launcher remap to complete twice**, which a correct
  implementation now prevents whenever the header would shift. The check encodes the old, unsound
  behaviour as its pass condition — the "a regression may be a test encoding the defect" shape. It
  wants re-scoping to accept either a completed remap or a correct copy, and that is proposed here
  rather than done, because silently making a failing check pass is the thing the house rules
  forbid.
- `bounds` expects a fault at exactly `+200003` with cause 28. The request rounding puts that
  address inside the object, so no fault can occur. Same shape, already recorded above as a
  consequence of option (iii).

### Gate status

`check-mallocng-policy.py` passes **22 of 22 with no new allowance**: the changes lie outside both
regions it pins here — it pins the `g->sizeclass>=48` *condition* text and the moving-realloc region
beginning at `new = malloc(n);`. Verified by running it, not by reading it. The
`test-group-size-narrowing.py` suite still behaves 8 of 8, and the regenerated `0009` patch
reverse-applies, which is `build-sdk.sh`'s currency check.


## DECISION REQUIRED: relax `add()`, or avoid the module change

Written as a decision rather than a description, because it had been sitting in message history as
"with the lead" without ever being put as an answerable question, and the lead decides from the
plan. Everything below is measured; the choice is not a technical unknown, it is a question about
what the module should enforce.

**The situation in two lines.** `capstone_vm.c:466` refuses any arena that is not a naturally
aligned power of two whenever exact bounds are off, which is the profile option (iii) runs in.
mallocng's mappings are page multiples, so **413 of 600 allocations fail**, and power-of-two ones
fail or succeed depending on where `mmap` lands — so part of the failure is nondeterministic.

### Option A — relax `add()` to the encoder's real condition (recommended)

Replace the power-of-two-and-natural-alignment test with `capstone_repr_ok(r->address, r->bytes)`,
sharing `representable.h` into the module build. Keeps the module's page checks, region and total
limits, and overlap test.

- Admits **400,000 of 400,000** of mallocng's real mapping shapes, against 28,220 (7.1%) today.
- Removes the nondeterminism: `heap.c` already requests `max(granule, 4096)`, so the base condition
  holds by construction rather than by luck.
- The predicate is the one `repr-cross-check.py` validates against the encoder oracle, and it is
  *shared*, not copied, so it cannot drift from what the hardware does.
- **Cost:** it changes what the module enforces, and the same `capstone_vm.c` builds the board
  module, so it changes what the silicon enforces. It is a loosening — a widened arena would be
  admitted where it is refused today — which is why it is not a lane's call.

### Option B — leave the module alone, round every arena to a naturally aligned power of two

No module change, so nothing on silicon changes.

- **Costs 46% more mapped memory on average**, measured over 400,000 sizes (80.9 GB becomes
  118.2 GB in total), worst case 1.97x on a single mapping (266240 becomes 524288).
- And `mmap` does not provide natural alignment, so the runtime needs over-map-and-trim at up to
  twice the size again for the reservation, transiently.
- It does remove the nondeterminism, by satisfying both halves by construction.

### Option C — defer

Option (iii) stays unusable in the exact-bounds-off profile; the allocation failures stand. The
growth fixes already landed are unaffected either way.

### What I recommend, and why it is not mine to take

**Option A.** B pays 46% of the heap to preserve an invariant the hardware does not require, and
the module's current test is not a restatement of representability but a much smaller subset of it.

It is not mine to take because it loosens a capability-bounds check on silicon. That is the one
class of change where being right about the predicate is not sufficient — the question is whether
the module should be the place that enforces a stricter invariant than the encoder, and that is a
design call.

**What would make it safe to take, if the answer is A:** the board lane has the prediction
registered row by row and rebuilds both modules from the revision, so a wrong relaxation shows up
as a failed row rather than as silence.


## DECIDED: Option A, after auditing the predicate against QEMU's own encoder

The lead delegated the choice with a steer — *"maybe like in current qemu model"* — and asked for an
audit first. The audit changes how the steer reads, so it is recorded before the change.

### The audit: what QEMU's model actually does

**Its representability rule is identical to ours.** Read from `cap_compress.c:40-57`:
`lz = msb(len)` floored at 12, `E = lz - 12`, `B = (base >> E) & ~7`, `T` likewise with `+8` when
the end has bits below `1 << (E+3)`. So the quantum on **both** edges is `8 << E` = 2^(msb(len)-9),
with an exact branch below 4 KiB — which is exactly `capstone_repr_granule`. Compared over 300,004
lengths: **0 disagreements**, and the positive control fires (a granule off by one power of two
disagrees on 295,904 of 299,999). This is a *new* check: `repr-cross-check.py` pins the header to
the **RTL** model, and nothing had pinned it to QEMU.

**But QEMU has no refusal path.** `B = (base >> E) & ~7` truncates the base **down** and `T += 8`
rounds the end **up**, so an unrepresentable range is silently **widened** — the capability gains
authority below the object. That is the hole exact bounds exists to close.

**So the steer resolves cleanly, and not the way it first sounds:** follow QEMU's model for *what
is representable*, and not for *what to do when a range is not*. Copying its behaviour would
reintroduce silent widening.

*(An earlier attempt at this audit built a compress/uncompress round-trip oracle and reported 4051
then 6788 disagreements. Both were artefacts of my transcription — first missing that `B` is packed
into 14 bits and `T` into 12, then a wrong `msb`/`carry_out` in `cap_uncompress`, which I had
transcribed from line 100 without its preceding lines. The encode path is short and unambiguous and
is all the decision needs, so the round-trip model was discarded rather than debugged further. Its
numbers are not evidence of anything.)*

### The change

`capstone_vm.c` `add()`, exact-bounds-off branch:
`!is_power_of_2(r->bytes) || (r->address & (r->bytes - 1))` becomes
`!capstone_repr_ok(r->address, r->bytes)`. `representable.h` is **shared** into the module build by
`build-adapter.sh`, beside `cap_rev_table_abi.h` and from the tree being built, so the module, the
launcher's `grant()` and the allocator agree on one oracle-checked rule and a kit cannot pair a
module and an SDK across revisions. Verified that the header compiles `-ffreestanding -nostdinc
-Werror`, which is how the module will see it. `is_power_of_2` now has 0 uses in the module.

**The refusal stays**, which was the condition that mattered: an unrepresentable arena is still
refused, and the launcher prints `CAPSTONE_VM_ADD_REFUSED` with the base, size, errno and both
disjuncts on that path unconditionally.

### Why this is a loosening of a bounds check, and what bounds it

The alignment half is **not** dropped — it becomes granule alignment instead of natural alignment to
the size, and it still binds wherever the granule exceeds a page, i.e. at and above 4 MiB. Below
that the granule is small (16 for an 8192-byte mapping), so every page-aligned base satisfies it —
**which is precisely why the change removes the boot-to-boot nondeterminism** rather than merely
widening the accepted set.

`module-arena-test.c` states that as cases rather than as a claim: the four shapes
`grant-repr-test.c` guards stay refused, the launcher's own arena shapes still pass, nine real
mallocng mapping sizes are newly admitted *and* asserted to have been refused by the old rule, and
the two halves are separated by a case each. Three negative controls, all firing: dropping the
alignment half gives 4 failures, dropping the length half 1, reverting to the old power-of-two rule
11.

Two defects in my own test found while controlling it, both recorded in it: an assertion that 8192
at `0x40001000` is refused (wrong — `granule(8192)` is 16, so every page-aligned base satisfies it),
and a missing case for an *aligned* base with a non-granule-multiple length, without which a
base-only predicate passed the whole test. A third was in the control harness: one arm failed to
compile and, because I had redirected stderr and not checked the status, ran a **stale binary** and
reported a pass.

### What is NOT done here, and why it is not an assumption

**The module is not built or run.** It needs a prepared kernel tree, and there is none on this host:
the module's own `.capstone_vm.o.cmd` records only paths relative to the make's `-C` directory, the
session transcript has no literal `KERNEL_BUILD`, and no `arch/riscv/include/generated` exists under
any of the scratch, offload or buildroot build directories that were searched. Tried and recorded, not
assumed. The board lane builds both modules and has the prediction registered row by row, so the
verification happens there.

**The landing commit must state** that the board module's enforcement changes from "naturally
aligned power of two" to "representable", because the same `capstone_vm.c` builds both profiles.


## SCORED: the allocation rows all held, the suite-total row did not

Run against the board lane's kit built from `89ee98be99e5` — their launcher and, for the first time,
their compile of the **module** change — with this lane's verified domain `aa9f98535ff98f07`, exact
bounds **OFF**, which is the profile the gate applies in.

| registered row | result |
|---|---|
| the five allocation markers clear | **held** — only `denial survived` remains, which is the predicted `bounds` mode |
| `align-probe` `failed=0 of 600` | **held** — `attempted=600 ok=600 failed=0`, against 413 failed before |
| `CAPSTONE_VM_ADD_REFUSED` absent | **held** — 0, and the print is unconditional, so this is falsifiable |
| `CAPSTONE_VM_REMAP_OK` 0 | **held** — 0 |
| both predicate arms ALLOCATED | **held** — `pow2_above=ALLOCATED nonpow2_below=ALLOCATED` |
| **suite 21 true / 2 false** | **FAILED — 18 true / 5 false** |

So the module change does what it was predicted to do on every allocation measure, and my
*aggregate* prediction was wrong. Both halves matter.

### The three extra failures are one event, and it is a defect the allocation failure was masking

`basic_inplace_mremap_failure`, `nonlinear_and_linear_tag_transfer` and `remap-stale` all fail
because `basic` now dies: `MALLOC_BASIC_EXIT:139`, from

    domain fault cause=29 pc=0x3fb2d1133c address=0x0 last=0xc0de0001

Before the change `basic` exited early on `MALLOC_FAIL:aligned allocation` — a clean NULL from a
refused arena. Now the allocation succeeds, execution runs further, and a fault appears that was
previously unreachable. `MALLOC_TAG_COPY_OK` prints once instead of twice for the same reason.
This is the "a fix that fails further on may have exposed an older defect" shape, and the right
reading is that the three rows are one new finding rather than three regressions.

### Ruling out the dangerous reading, which is the one that mattered

A relaxed bounds predicate could in principle admit an arena whose bounds are **wrong**, and that
would show as a bounds violation. It does not:

- **0 `Cap mem access OOB` lines** in the entire arm;
- **0 `SHRINK illegal operand` lines**;
- **cause 28, which `run-malloc.py:134` uses as the bounds class, does not appear at all.** The
  causes present are one 2, five 25 (the expected use-after-free denials) and two 29.

So the evidence that would indicate a widened capability is absent, and the predicate is not
implicated in the new fault. That is the check worth doing before calling a loosening safe, and it
is a negative result with a known positive form — cause 28 does fire in this suite, for the
`bounds` mode, which is how we know the class is observable.

### The new thread, stated as a lead and not a conclusion

`cause=29` with `address=0x0` immediately after a `kind=2` boundary matches the shape the RTL lane
described for a refused `CSRUNV`: a terminal event `{kind 2, cause 29, pc = the entry PCC's cursor,
addr 0}`, where QEMU raises a synchronous 29. That is suggestive and unverified — `last=0xc0de0001`
says the last service was the acquire, not an entry — so the next step is to instrument it rather
than to reason further, which is the pattern that has worked every time in this investigation and
failed every time it was skipped.


## The relaxation is validated, AND it brings a cause-29 fault in `capstone_launch_unpack`

### Validated, by the board lane's full set on `89ee98be99e5`

`run.py` **PASS, 41/41** — `normal`, `parallel_one` and `parallel_two` clear `heap growth/zero`
(the computed row), and `heap-threads`, `sqlite_write`, `sqlite_read` and `mruby` all exit 0 (the
inferred rows). `CAPSTONE_VM_ADD_REFUSED` 0, against **35** in the `b61` before. Tier 3 bundle 5
**13/13** cell for cell; cases 00 and 10 **CAUGHT in 4 of 4 boots**, against 3 of 6 pre-fix, which
is the nondeterminism row and it held; bundles 13 and 14 unchanged. The board module compiles —
`07ba2c4f7425310f`, 0 RVC, vermagic 6.4.14, with `representable.h` `7c41a409acb7a897` copied beside
it.

**And the fault profile is identical before and after:** `13 ×2, 15, 24, 25 ×3, 28 ×2`, every one an
expected denial, with QEMU's six capability-violation classes unchanged. In particular **cause 28
fires twice in `run.py` both before and after**, so the bounds class is observably live and the
relaxation did not move it. That is the strongest available evidence that the predicate is not
handing out widened capabilities.

### And a fault that appears only with the change

This lane's `ok` mode now dies at `MALLOC_BASIC_EXIT:139` from

    domain fault cause=29 pc=... address=0x0 last=0xc0de0001

Located: **both cause-29 faults are inside `capstone_launch_unpack`**, not in the contract's
allocation code. The instruction is `lw a3, 0x14(a4)` at ELF `0x1133c`, where `a4` comes from
`shrink a4, a0, a0+0x58` two instructions earlier — so the derived capability is invalid and the
load faults at address 0. Nothing prints beforehand, not even `MALLOC_BASIC_OK`.

**Single-variable control, same domain and images, bounds OFF, differing only in the module:**

| module | cause-29 | `basic` | `ADD_REFUSED` |
|---|---|---|---|
| old, power-of-two | **0** | exit 1 on `MALLOC_FAIL:aligned allocation` | 445 |
| new, representability | **2**, in `capstone_launch_unpack` | exit 139 | 0 |

### What the control cannot settle, stated rather than resolved in my favour

It **cannot separate** "the change created this fault" from "the fault was always in that path and
the refused allocation stopped it being reached". `last=0xc0de0001` says the fault follows an
acquire service, and with the old module that acquire was refused, so the unpack never ran. **Both
readings predict exactly what was measured.** That is the same under-determination that cost three
attributions earlier in this investigation, so it is recorded as open.

**A framing I gave the board lane and have retracted:** that "a step which now gets further and then
faults is the masked-defect shape". I sent that before locating the fault, and `capstone_launch_unpack`
is startup code — so "further" is not supportable. The revised discriminator is the fault CLASS, not
the progress: a 28 or an OOB line implicates the predicate, a 29 at address 0 is this thread.

### The lead worth taking next

The fault does not appear in any of `run.py`'s steps or Tier 3's, so the faulting path is something
this contract's `ok` mode exercises and those do not — the board lane's pointer at the allocation
sizes diverging between `contract.c` and `malloc-contract.c` is the right place to look. The
instruction sequence narrows to an 88-byte window and reads two words at `+0x10` and `+0x14` which
it compares as negatives of one another, so there is a descriptor-layout question underneath it.
Instrument it; do not reason about it.


## The cause-29 fault: located precisely, cause still open, two hypotheses refuted

A grant-logging launcher (every ADMITTED grant printed, not only refusals) paired with each module
in turn — **same launcher, same domain, same images, the module the only variable**. Arms identified
by the module hash in `result.json`, not by directory name, after a `sed` accumulation wrote the
control into `g-oldmod-newmod` and the first diff attempt read a path that did not exist.

### What the grant diff shows, with addresses normalised

The first diff compared raw strings and "found" a difference at event 0 that was only the per-boot
ASLR base. Normalised to the address-independent fields plus `base % granule`:

| # | new module | old module |
|---|---|---|
| 0–7 | **identical** — 524288, 1048576, 16384, 262144, 65536 (the launcher's own arenas) and three linear heap mappings, all power-of-two and naturally aligned | identical |
| 8 | **OK 49152**, pow2=0, `base%granule=0` | **REFUSED**, errno 22, pow2=0 |
| 11 | **OK 28672**, pow2=0 | **REFUSED**, errno 22, pow2=0 |
| 12 | **FAULT cause=29 addr=0x0** | — (already failed out) |

So the loosening admits **exactly** the intended shapes and nothing else, and the launcher's own
arenas are untouched. 28672 is the `posix_memalign(4096, 5001)` group, i.e. the allocation that
used to print `MALLOC_FAIL:aligned allocation`.

### The fault's location, established by byte match rather than by symbol

The symbolizer reported `.Lpcrel_hi12 launch.c:0:0` — a **local label** and no line, because the
runtime archive carries no debug info, so that attribution was not usable. Instead: the 48-byte
window at the faulting vaddr was extracted from the final image and searched for in every runtime
and libc object. It appears in **`launch.c.obj` and nowhere else**. `launch.c` defines only
`append`, `capstone_launch_pack`, `string_at` and `capstone_launch_unpack`, and only the last is
reachable at runtime.

The instruction is `lw a3, 0x14(a4)` after `shrink a4, a0, a0+0x58`.
`sizeof(struct capstone_launch_header)` is exactly **0x58 = 88**, and `+0x10`/`+0x14` are its
`argc` and `envc`. So the fault is reading the launch header through a `buffer` capability that is
invalid, and `address=0x0` is consistent with that.

This also resolves why nothing prints: the grants at events 5–11 are **startup** heap mappings, and
`capstone_launch_unpack` runs after them, so `basic()` never begins.

### Two hypotheses refuted, one of them mine

- **"The launch block is refused or granted differently."** No: event 4 (65536, reg=12, perms=4) is
  **identical** in both arms.
- **"Registering more arenas invalidates an earlier capability."** No, and this was refuted by data
  already in hand rather than by a new run: the new module reaches arena **slot 168 in `churn` with
  0 cause-29 faults**, while the fault happens at max slot **10**. Arena count is not the mechanism,
  and neither is "admitting a non-power-of-two arena" in itself, since `churn` admits many without
  faulting.

### What remains true, and what the next instrument is

Still solid: the fault is **not** a bounds failure — zero `Cap mem access OOB`, zero `SHRINK
illegal`, and cause 28 absent here while firing twice in the board lane's `run.py` both before and
after. The predicate is not handing out widened capabilities.

Still **open**, and not resolved in the flattering direction: whether the change created this fault
or made an existing one reachable. Nothing measured so far separates those.

**Next instrument, named rather than reasoned about:** print the launch-block capability's tag and
bounds in `application.c` immediately before `capstone_launch_unpack` is called, in both arms. That
is the one reading that says whether the capability is already invalid when it is handed over or is
invalidated inside the unpack. It needs an SDK rebuild, which is a known-good path here.


## RETRACTED: the fault is not in `launch.c`. My addresses were wrong by 0x10000

### The error

I translated the fault pc as `ELF vaddr = pc - code_base`, which was never verified against an
anchor — the `entry=` field in the fault line is the launcher's trampoline and lies outside the
image's `PT_LOAD`. The board lane's own convention, stated in their report, is
**`VA = (pc - code_base) + 0x10000`**, and it cross-checks: they place `cap_malloc_validate` at
VA `0x20d78`, and in this image its symbol is at `0x20dc8`.

So everything I concluded from vaddr `0x1133c` is withdrawn, including the byte-match "confirmation"
that the fault was in `launch.c.obj`. **That match was real but at the wrong address.** A byte match
confirms where bytes came from; it cannot confirm that the fault happened there, and I presented it
as if it did.

### What the fault actually is, at the corrected VA `0x2133c`

    21324: lcc  t0, a2, 0x3        # base
    21328: lcc  t1, a2, 0x4        # end
    2132c: sub  t1, t1, t0
    21330: bltu t1, a3             # the function's own window test -- PASSES
    21338: scc  t2, a2, t0
    2133c: <unknown, funct7 0x51>  <-- faults

That is instruction-for-instruction `capstone_cap_derive` from
`runtime/include/capstone/capability.h:42-48`, inlined into **`aligned_alloc`** at +0x19c
(`aligned_alloc` is at `0x211a0`, `posix_memalign` at `0x213b4`).

**So this is the board lane's signature (2), not a separate fault.** Their description —
"cause 29 inside `capstone_cap_derive` (+0x9c), at its CDERIVE (funct7 0x51), after the function's
own window test passed" — matches mine exactly. There is **one** signature here, not two, and it is
reached through `posix_memalign(4096, 5001)`, whose 28672-byte arena the `add()` relaxation newly
admits.

### The cause numbers, from the source rather than from inference

`cpu_bits.h:693-699`: `0x18` `UNEXP_OP_TYPE`, `0x19` `INVALID_CAP`, `0x1a` `UNEXP_CAP_TYPE`,
`0x1b` `INSUF_CAP_PERMS`, `0x1c` `CAP_OOB`, `0x1d` `ILLEGAL_OP_VAL`, `0x1e` `INSUF_RESOURCES`.

So **24 = `UNEXP_OP_TYPE`**, which confirms the board lane's reading of their signature (1) from the
cause name alone: a non-capability where a capability was required, i.e. `free()` reached with an
integer pointer. And **29 = `ILLEGAL_OP_VAL`**; **28 = `CAP_OOB`** is the bounds class, which is why
its absence was the right thing to check.

### An inconsistency I have NOT resolved, stated rather than smoothed over

`RISCV_EXCP_ILLEGAL_OP_VAL` is raised at exactly three places in this QEMU pin — `op_helper.c:1087`
(SHRINK, with a debug print), `:1264` (`csinit`, no print) and `:1561` (STC through an uninitialised
capability with a non-zero imm, with a print). **None of them is a CDERIVE.** Further, funct7 `0x51`
on opcode `0x5b` has **no entry** in this pin's `insn32.decode` (0 matches), there is no `cderive`
in the decode table or in `op_helper.c`, and the disassembler prints `<unknown>`.

Those facts do not yet fit together: an undefined instruction would normally raise cause 2, and the
arm does contain exactly one cause-2 fault. So either the pc is still off by one instruction, or
funct7 0x51 reaches a raise site I have not found. **That is the next thing to settle, and it is
also the most likely explanation of the board lane's signature (2) in a corpus that my change did
not touch** — which would make the mechanism independent of the relaxation, with the relaxation only
making the path reachable.


## RESOLVED: the cause-29 fault was my own rounding, applied to `malloc` and not its siblings

The board lane flagged this as an "unrelated observation" and said it was not the cause of their
FFmpeg signature. They were right about theirs and it is the complete cause of **mine**.

`malloc` publishes `capstone_repr_round(n)`. `realloc` (`heap.c:359`) and `aligned_alloc` (`:366`)
published the **raw `n`**. And `posix_memalign(4096, 5001)` reaches `aligned_alloc(4096, 5001)`,
where `granule(5001) = 8` and `5001 % 8 = 1` — so the child derived over the raw size is **not
representable**, and the CDERIVE inside `capstone_cap_derive` raises `ILLEGAL_OP_VAL`, cause 29.

**So the open question is answered, and not in the flattering direction.** The relaxation did not
create the fault and the predicate was never implicated: the relaxation made reachable a defect in
**the rounding change this lane had already landed**, which had only ever been applied to `malloc`.
Before, the 28672 arena was refused, so `aligned_alloc` returned NULL and never reached `publish`.

### The fix, and why it rounds the REQUEST and not just the published size

Publishing a rounded size over an unrounded block would hand out a capability past the allocation —
the opposite defect. So `realloc` and `aligned_alloc` now ask mallocng for `want` and publish
`want`, exactly as `malloc` does, and all three `publish()` call sites pass a representable size.

### Pre-registered, and every row held

| row | before | after |
|---|---|---|
| cause-29 faults | 2 | **0** |
| `basic` exit | 139 | **0** |
| `MALLOC_TAG_COPY_OK` (needs 2) | 1 | **2** |
| `CAPSTONE_VM_ADD_REFUSED` | 0 | **0** |
| align-probe | — | **600 ok, 0 failed** |
| suite aggregate | 18 true / 5 false | **21 true / 2 false** |

21/2 is the aggregate predicted before the `add()` run and missed; the two remaining failures are
`bounds` and `real_linux_mremap_growth_and_shrink`, both already recorded as checks that encode
pre-option-(iii) behaviour rather than defects.

### What this says about the method

Three of this investigation's attributions failed because a control under-determined. This one was
settled by someone else reading my code and noticing an inconsistency I had introduced and not
looked for — which no amount of instrumenting the fault would have found, because the fault was a
faithful consequence of a wrong input. **Worth keeping: when a derive is refused, check what size
was asked for before instrumenting where it was refused.**

The board lane's other two signatures are separate and remain theirs: the FFmpeg pool tag loss
(cause 24, a capability stored to memory returning untagged, independent of `aligned_alloc` since
their subobject bundle defines it away) and the memcached allocator-repros CDERIVE refusal. The
latter may share this mechanism — an unrepresentable size reaching a derive — but through their own
port's allocator rather than through `heap.c`, and that is unverified.


## Two corrections from reading the WRONG TREE, and they strengthen the decision

The board lane caught both. I grepped the `capstone-qemu` **checkout**, which is at
`deb7d757565d`, while the QEMU actually run is the pin `d35af47b163c`. This project's own rule
covers it exactly — *a lagging checkout cannot prove absence; verify a citation in its own tree with
`git show <ref>:<path>`* — and I did not follow it.

### RETRACTED: "funct7 0x51 is not a defined instruction"

It is. In the pin, `insn32.decode:984` reads
`csderive 1010001 ..... ..... 001 ..... 1011011`. The checkout I grepped has **0** occurrences of
`csderive`; the pin has it. So the "inconsistency I have not resolved" in the previous section
dissolves: the faulting word is `csderive`, there is a `helper_csderive`, and nothing was undefined.

### RETRACTED and REFINED: "QEMU has no refusal path, it silently widens"

`helper_csderive` in the pin does this, with the comment *"Never silently widen a new object's
authority in the compact profile"*:

    cap_compress(&child, &lo, &hi);
    cap_uncompress(lo, hi, &back);
    if (!riscv_cpu_cfg(env)->capstone_exact_bounds &&
        (back.bounds.base != child.bounds.base || back.bounds.end != child.bounds.end))
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_OP_VAL, ra);

So **QEMU refuses a non-representable CHILD at derive time**, as a compress/uncompress round trip,
exactly when exact bounds are off. What widens silently is `cap_compress` used as an encoder; what
refuses is the derive instruction. My audit sentence conflated the two.

**This strengthens Option A rather than weakening it.** The argument I gave was "follow QEMU for
what is representable, not for what it does when a range is not, because it widens". The better
argument is simpler: **`csderive` already refuses exactly what is not representable, so a module
arena test of `capstone_repr_ok` makes the module agree with the instruction** instead of enforcing
a stricter, unrelated rule. The decision is unchanged; its justification is now the stronger one.

It also means my abandoned round-trip oracle was the right *shape* — QEMU itself round-trips — and
only my transcription of it was wrong. Worth recording so the next reader does not conclude the
approach was misguided.

### And it closes the mechanism, consistently

My cause-29 fault was `csderive` refusing a child of size 5001, whose granule is 8 and which is
therefore not representable — the round trip above returns different bounds and the instruction
raises. The board lane's memcached Sublet signature is the same refusal, confirmed by them with a
single-variable run: the identical binaries, SDK and module with only
`x-capstone-exact-bounds=true` added read CAUGHT 8 + 1 control-failed, every cell equal to the
record, against 8 mismatches without it. That is vp2's Q-15 binding inside a virtual context.

**The open question for that arm is the one this lane's own fault answered: what SIZE the carve
passes to the derive.** If it derives over a raw request, rounding at the carve is the fix, and
`heap.c`'s `malloc`/`realloc`/`aligned_alloc` are the worked example.


## The FFmpeg pool tag loss: narrowing "the SDK" to five arms, and two of my own hypotheses refuted

Not this lane's defect, but option (iii)'s mallocng is one of the candidates, so the narrowing is
recorded here. The board lane established the compiler is exonerated for the case: the same program
built with the kit's clang (85b1daddf230) and the 10-10 record's clang (bd372e25f5fb) against a
fixed v6 SDK gives **byte-identical images**, and both fault cause 24 at
`refstruct_pool_uninit+0x78`. So the variable is what the SDK brings.

### Two hypotheses of mine, both refuted before being sent

**`memcpy`'s mis-phased byte fallback is NOT a defect.** `string_bounds_safe.c`'s `memcpy` uses the
tag-preserving granule loop only when source and destination share a 16-byte phase, and otherwise
copies every byte — which does drop tags. But the file's own header already states the rule: *"A
tagged value that is not 16-aligned in both source and target cannot be preserved by any copy, which
is also CHERI's rule, and the bytes of it are still copied correctly, only untagged."* It is the
documented semantics, not a bug. Checking prior art before writing it up is what stopped that.

**The missing `-fno-builtin` is NOT the differentiator.** C-78's entry records that this exact SDK
flag difference once produced an untagged capability — *"reads the copied capability untagged in the
SDK without -fno-builtin, and passes in the SDK with it"* — and the v6 SDK does not pass it. But
**neither does the 10-10 SDK**: `virt-stack/out/sdk/sdk.json`'s codegen is
`-mllvm -capstone-gp-free -mllvm -capstone-image-gp` and nothing else. So the flag is constant
across the two records and cannot explain a difference between them. A good hypothesis with a
recorded precedent, killed by looking at the other side.

### What "the SDK" actually is: five single-variable arms

Comparing the two `sdk.json` codegen lines entry by entry, the v6 SDK **adds** exactly three things
and removes nothing:

| added | what it is |
|---|---|
| `-Xclang -target-feature -Xclang +f` and `+d` | hard float |
| `-Xclang -target-abi -Xclang lp64` | the ABI, stated explicitly |
| `-Xclang -target-feature -Xclang +movc-keeps-integer-source` | `movc` semantics |

So the remaining variable is five arms, each testable against the same compiler and sources:
the three flag groups above, `libapplication-runtime.a`, and `libc-capstone-virtual.a` (musl plus
patches 0001–0009, i.e. option (iii)'s mallocng).

**`+movc-keeps-integer-source` is the one I would try first** for a tag-loss symptom, because it is
the only added flag that changes how a value moves between registers, and the symptom is a value
that keeps its address and loses its tag. That is a reason to order the arms, not a prediction.


## The v6 SDK loses a capability tag, and `-fno-builtin` is the cause — reduced, two-sided

Reached by reaching for an existing instrument instead of a rebuild. The board lane's runs 2, 3 and
4 had excluded the module change, the compiler change and `+movc-keeps-integer-source`, leaving "the
runtime/libc contents or the pool sources", and the archive-swap arms cannot link. So rather than
rebuild archive sources, the **hosted-application probe** —
`runtime/virtual/hosted-contract.c`, the instrument that found C-78 — was run against the v6 SDK.
It tests tag survival across fourteen copy shapes with controls that MUST lose the tag.

**One check fails, and only one:**

    virtual:memcpy_constant_size_unknown_alignment   FALSE

Everything else passes, including `memcpy_24_capability_head`, `memcpy_padded_struct`,
`struct_assignment_64_pointers`, `realloc_moving` and `memmove_overlapping` — and both tag-loss
controls behave (`control_byte_copy_drops_tag` and `misaligned_copy_has_no_tag` are true), so the
instrument can report the opposite.

**The shape.** `copy_pair` is `noinline` and does `memcpy(d, s, sizeof(struct pair))` through `void
*`, with the probe's own comment: *"The compiler sees only void pointers: no alignment to rely
on."* Both operands are 16-aligned statics in fact, so musl's run-time-checking `memcpy` would
preserve the tag — but without `-fno-builtin` the compiler is free to inline the copy, cannot prove
alignment, takes the scalar path, and the tag is gone. Its known-alignment neighbour,
`memcpy(&pair_dst, &pair_src, sizeof pair_dst)` on the same types, passes.

**Two-sided and single-variable:** rebuilding only the probe with `-fno-builtin` added, nothing else
changed, flips that check to TRUE and leaves **no virtual failures at all**. The two images differ
(`683a6b6f350a7764` vs `d6de4abcd44a568d`), so this is not a silent rebuild of the same code.

### This corrects my own earlier refutation, which was right about the wrong question

Earlier in this document the `-fno-builtin` hypothesis was recorded as refuted, on the grounds that
the 10-10 SDK does not pass it either. **That refutation answered "does the flag DIFFER between the
two records?" — it does not — and I then over-generalised it into "the flag is irrelevant".** The
flag demonstrably causes a tag loss in the v6 SDK. Both statements can hold: the flag is a real
defect in both SDKs and therefore cannot by itself explain a difference between them.

### What is NOT established

That this explains the FFmpeg pool fault. The 10-10 record **passed** that case and, if its SDK also
lacks the flag, it should have had the same exposure — so either the pool's copy is a different
shape, or something else in v6 makes it reachable. **The decisive test is one rebuild and belongs
with the case:** add `-fno-builtin` to the SDK's codegen and re-run subobject case 00. If it clears,
the chain is closed; if not, the pool fault is a separate shape and this is a second, independent
defect found on the way.

### The fix, and why it is not applied unilaterally

`-fno-builtin` belongs in the virtual SDK's codegen, next to the flags `capstone-cc` already
validates. It is one entry. But it changes every port's images, and the board lane has runs in
flight against the current SDK, so it is proposed rather than dropped in mid-experiment — and it
should go in as a *validated* arm, with this probe as its gate, since the probe is exactly the test
that would have caught the omission.


## The `-fno-builtin` OBSERVATION stands; my MECHANISM for it does not

The two-sided result is unchanged and reproducible: with the v6 SDK as shipped,
`hosted-contract.c`'s `memcpy_constant_size_unknown_alignment` reads FALSE; adding `-fno-builtin`
and changing nothing else makes it TRUE and leaves no virtual failures, with different images.

**The explanation I attached to it is withdrawn.** I wrote that "without `-fno-builtin` the compiler
inlines the copy, cannot prove alignment, takes the scalar path, and the tag is gone". A minimal
reduction of exactly that shape — a `noinline` function doing `memcpy(d, s, sizeof(struct pair))`
through `void *`, `sizeof` 32 — shows the opposite in **all four** arms (kit clang 85b1daddf230 and
the 10-10 clang bd372e25f5fb, each with and without the flag): `copy_pair` **tail-calls `memcpy`**.

    auipc a2, %pcrel_hi(memcpy) ; addi a3, ... ; li a2, 32 ; mv t1, a3 ; ... ; jr t1

So for that shape the compiler already makes the conservative, tag-preserving choice, and the flag
changes nothing about it. Whatever `-fno-builtin` is doing in the probe, it is **not** preventing an
inline expansion of that call.

**And the reading that first said otherwise was a detector artifact of mine.** My script counted
`ldc`/`stc`/`ld`/`sd` in the function body and looked for `call.*memcpy|tail.*memcpy`. It found no
call — because a tail call through `auipc` + `jr t1` matches neither pattern — and it counted the
**prologue spills of `s0` and `ra`** as the copy, so it reported "inlined as CAPABILITY (tag kept)"
for a function that does not copy anything itself. Dumping the whole body is what caught it, and is
what should have been done before reading counts.

**So two things are now open rather than one:**

1. The board lane's question — if the 10-10 SDK lacked the flag too, why did its run pass? My
   reduction was meant to answer it by showing the old compiler lowered the copy differently. It
   shows the two compilers lowering it **identically**, so the question stands untouched.
2. **Why the flag changes the probe's result at all**, given that the shape it names is a call in
   both builds. The next step is to diff the probe's own generated code with and without the flag
   around that check, rather than to reason from a reduction that does not reproduce. The two probe
   images are kept for exactly that: `683a6b6f350a7764` and `d6de4abcd44a568d`.

The arm is still worth running: the observation is what it is, and if case 00 clears on arm 6 that
is a fact about case 00 whether or not the mechanism is yet understood.
## RETRACTED: there is no measured compiler regression, and my compiler labels were swapped

**Everything I wrote in this document's previous section about a lowering regression between
`bd372e25f5fb` and `85b1daddf230` is withdrawn.** It rested on two disassemblies I attributed to
two compilers. Re-measured with the revision printed in the same command as the numbers, the body I
labelled "the 10-10 compiler, tag kept" is **the kit compiler's own output**:

    85b1daddf230, SDK v6 codegen flags, -O2, NO -fno-builtin -- copy_pair, 21 instructions:
      auipc a0, %pcrel_hi(.L_MergedGlobals) ; li a1, 1616 ; scc a0, gp, a0 ; delin a0
      mv a2, a0 ; add a1, a2, a1 ; shrink a0, a2, a1
      ldc a1, 32(a0) ; ldc a2, 16(a0) ; stc a1, 96(a0) ; stc a2, 80(a0)

Capability loads and stores: **the tag is kept here.** And the four arms settle it completely --
the two compilers emit **byte-identical** bodies:

| arm | insns | copy | body sha |
|---|---|---|---|
| `bd372e25f5fb` plain | 21 | `ldc`x2 + `stc`x2 inline | `0f832598dde9` |
| `85b1daddf230` plain | 21 | `ldc`x2 + `stc`x2 inline | `0f832598dde9` |
| `bd372e25f5fb` `-fno-builtin` | 23 | tail-call `memcpy` (`jr t1`) | `9871b6a90cc3` |
| `85b1daddf230` `-fno-builtin` | 23 | tail-call `memcpy` (`jr t1`) | `9871b6a90cc3` |

**There is no compiler difference in this function at all**, so there is no regression to claim and
nothing was "lost" between the revisions. The `32x (lbu + sb)` body is reproduced by **neither**
compiler and I no longer know where it came from. The 16-byte size delta between the two plain
objects lies elsewhere in the file, not here.

**And both flag settings keep the tag at `-O2`** -- plain inlines the capability copy, `-fno-builtin`
tail-calls musl. So at this optimisation level the check cannot read FALSE either way, which is the
real content of the next paragraph.

One more thing the compiler axis was never going to show: `sdk-arm-6/sdk.json` names
`cc = /tmp/capstone/vp2-prep/v6-fix/llvm-build/bin/clang`, which is a **symlink** resolving to
`/tmp/capstone/llvm-dev-current-20261017/bin/clang-22`, i.e. `85b1daddf230`. The gate and my arms
used one compiler throughout. A configured path is not a compiler.

### What survives, because it never depended on the disassembly

* **Arm 6's gate.** `memcpy_constant_size_unknown_alignment` reads FALSE without `-fno-builtin` and
  TRUE with it, with two distinct probe image hashes (`683a6b6f350a7764` vs `d6de4abcd44a568d`).
  That is an image-level two-sided reading.
* **The board lane's run 6.** Arm 6 restores every cell in all three FFmpeg pool bundles --
  subobject 9/9, pool stock 4/4, pool Sublet 4/4 CAUGHT.

The flag's *effect* is measured. Its *mechanism* is not, and I should have kept those apart.

### The finding that is actually worth keeping, about the instrument

**My `-O2` reproduction does not reproduce the gate**, and that is the informative part. At `-O2`
*neither* flag setting loses the tag in `copy_pair`, so the image whose check reads FALSE cannot
have been built the way I just built it. The `-O2` was my choice, not the build's. The build flags are the missing fact; `run-hosted.py` does
not compile the probe at all, it consumes a prebuilt `hosted.dom`.

And the reason the two can differ at all is a property of the check itself:

    /* The compiler sees only void pointers: no alignment to rely on. */
    __attribute__((noinline)) static void copy_pair(void *d, const void *s)
    { memcpy(d, s, sizeof(struct pair)); }
    ...
    copy_pair(&pair_opaque, &pair_src);          /* the only call site */

`copy_pair` is `static` with **one** call site and both operands are file-scope statics. `noinline`
stops inlining; it does **not** stop interprocedural constant propagation. At `-O2` the compiler
substitutes `&pair_opaque` and `&pair_src` into the body -- which is why the body references
`.L_MergedGlobals` and never reads `a0`/`a1` -- and then it *knows* both operands are 16-aligned
statics. **So the check named `memcpy_constant_size_unknown_alignment` does not necessarily test
unknown alignment.** Whether it does is an optimisation-level property, and the comment asserting
otherwise is defeated by a pass the comment does not mention. That is a real defect in the probe,
independent of the flag question, and it is the kind that makes a check read differently for reasons
that have nothing to do with the thing under test.

### Process

Four instrument failures ran before any number here was trustworthy, each silent:
`llvm-objdump` absent at the path I used with **stderr folded into the body file**, so "No such
file" rendered as `0` in all four arms -- the "no data as a zero" shape, in the same week I
sharpened this repo's rule about it; `--disassemble-symbols` stopping after the prologue; an address
slice bounded by the next `t` symbol, which is a `.Lpcrel_hi` label 20 bytes in, not the next
function; and an assembly matcher requiring the label line to equal `copy_pair:` when LLVM emits
`copy_pair:  // @copy_pair`. The reading only became real once two independent views agreed.

The single cheap check that would have caught the whole thing at the start: **print the compiler
revision next to each body**. The rule is already in my notes, I did not apply it, and a swapped
pair of labels is exactly what it exists to prevent.
