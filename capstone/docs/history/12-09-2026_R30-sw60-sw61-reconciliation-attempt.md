# R-30 after the flash: what sw60 and sw61 can and cannot settle, one refuted hypothesis, and the instrument gap that blocks the rest

**2026-09-12, RTL lane.** The board lane handed this over as an unreconciled conflict rather than a
re-diagnosis, which was the right call. This note is the attempt, and it ends with a **negative
result plus a one-line instrument fix**, not an answer.

## What the two readings actually say

| boot | reading |
|---|---|
| sw60 | the monitor's reclaim of a region the host asked for as **1,419,584 bytes** falls **1,728 bytes** short of `end` (`RCSH:000006C0`, 108 granules of 16) and takes its designed `while(1)` |
| sw61 | the Sublet port on the same silicon reports **`init=5334`**, with every counter bit-identical to QEMU |

**R-30's registry headline does not survive either reading.** It says INIT is *unreachable* and that
the shortfall is *exactly one byte*. INIT ran 5,334 times in one boot, and the observed shortfall is
1,728 bytes. Both particulars are wrong as stated.

**But the two readings are not in conflict with the R-30 FIX, and that distinction matters.** R-30's
one-byte claim was about INIT's *precondition* — a filled UNINIT leaves the cursor **at** `end`, and
INIT required `cursor > end` strictly, so it was unreachable by one position. The fix made INIT legal
at `cursor == end`. sw61 is that fix working, 5,334 times. sw60 is a **fill that did not reach
`end`**, which is a different failure at a different step: with the cursor short of `end`, INIT is
correctly refused. **So sw60 is not evidence against the R-30 fix. It is evidence of a second,
previously unnamed effect that only shows on a large region.**

## The hypothesis that fit the arithmetic exactly, and is REFUTED

1,728 is exactly the distance from 1,419,584 up to the next **2 KiB** boundary (1,421,312):

    round up to  2048  -> 1421312, gap 1728   EXACT
    round up to  4096  -> 1421312, gap 1728   EXACT (same value at this size)
    round up to  1024  -> 1420288, gap  704
    round up to   512  -> 1419776, gap  192

That is a striking fit, and it suggested **capability bounds compression**: a compressed format's
representable `end` is rounded up for a large object, the fill reaches the true end, and the readback
reports the rounded one. It would also have explained sw61 for free, since small blocks are exactly
representable and would never round.

**It is wrong.** The RTL uses a `fat_cap_t` with **full 64-bit bounds** — `start : logic[64]` and
`end : logic[64]` (`core/anvil_build/capstone_unit.anvilh:321` and the `fat_cap_metadata_t`
constructors at `:396-442`). There is no bounds compression in the hardware capability to round
anything. The arithmetic fit is a coincidence until something else explains it.

**Two further reasons not to have shipped it even if the RTL had cooperated**, both worth keeping:

* **One datapoint cannot separate 2 KiB from 4 KiB.** They give the same gap at this size. A single
  reading that fires and still under-determines is the documented failure shape, not a finding.
* It would have been a curve fit to **one** number, with the region size taken from what the host
  *asked for* rather than from what the capability *reports*.

## Why the loop arithmetic cannot produce 1,728 on its own

The fill is `C_RECLAIM` (`sbi_capstone.c:252-256`), and it is short enough to reason about completely:

    n = (cap_end(cap) - cap_base(cap)) >> 4        // C, before the asm
    mv %1,%3 ; 1: beq %1,x0,2f ; stc(x0,%2,0) ; addi %1,%1,-1 ; j 1b
    2: lcc(%0,%2,2) ; lcc(%1,%2,4) ; sub %1,%1,%0  // shortfall = end - cursor

Every store advances the cursor by one 16-byte granule, so after `n` stores the cursor sits at
`base + 16*((end-base)>>4)`. The **only** shortfall this arithmetic can produce is
`(end - base) mod 16`, i.e. **at most 15 bytes**. The region is granule-aligned anyway
(1,419,584 mod 16 = 0), so on the reported size the predicted shortfall is **0**.

1,728 is 108 granules. So either 108 stores did not advance the cursor, or `n` was computed 108 too
small — which needs `cap_end` at bound time to differ from `lcc …,4` at check time, and both are
selector 4 on the same capability. **Nothing in the source accounts for it**, which is the honest
state.

## THE INSTRUMENT GAP, and it is one line

**The reading cannot be pushed further because the macro does not report the two values that would
settle it.** `C_DO_RECLAIM` (`sbi_capstone.c:275-280`) reports `RCSH` (the shortfall) and
`CAPSTONE_TAG_BASE` (the base) and nothing else. So from sw60 nobody can compute:

* the capability's **true size**, `end - base` — the 1,419,584 above is what the host **requested**,
  not what the capability reports, and the whole question is whether those differ;
* where the cursor **actually stopped**.

**Fix: report `cap_cursor` and `cap_end` alongside the shortfall on the RCSH path.** It is inside a
path that is already halting with `while(1)`, so it cannot perturb any measurement, and it converts
the next occurrence from "1,728 short of something" into a closed arithmetic statement. This is the
project's own rule — name in advance the observation that proves the condition — applied to firmware.

## Pre-registered predictions, written BEFORE the next run

With cursor and end reported, one boot separates the live accounts:

| reading | what it means |
|---|---|
| `end - base` = 1,419,584 and cursor = base + 1,417,856 | 108 stores genuinely did not advance the cursor. A fill/STC question, and the serious one. |
| `end - base` = 1,421,312 | the capability is **larger than the request** — the region allocator rounded, and `n` was right while the request figure was the misleading number. A monitor/allocator question, not an ISA one. |
| `end - base` not a multiple of 16 | the truncation case after all, and then the shortfall must be < 16, so 1,728 would refute this in the same reading |

**And the cheap discriminating pair, independent of the above:** run the same reclaim on a region
whose size is an exact multiple of 4,096, and on one that is not. If the shortfall tracks the
round-up distance it is an allocator/representation effect; if it stays 1,728 or scales with region
size it is not.

## What must NOT be written down yet

* Not "R-30 is fixed" — sw61 shows INIT reachable, which is the fix working, but sw60 shows a large
  fill still failing and that is unexplained.
* Not "R-30 is not fixed" — sw60's failure is at the fill, one step before INIT, and the R-30 change
  was to INIT's precondition.
* Not the compression story above. It is recorded here **because it was refuted**, so the next reader
  does not spend the same hour rediscovering the same attractive fit.

The sim test passing at this revision an hour before the board contradicted it is worth keeping in
view: `r30-fill-init.S` fabricates its UNINIT with the Custom3 debug ops on a small buffer, which
R-30's own entry already calls a test working around the defect rather than reporting it. A directed
test on a small fabricated capability cannot speak to a 1.4 MB region reached through the real
share/revoke path.

---

## ADDENDUM, same day: my discriminating pair is dead, the "end moved" account is REFUTED, and one live hazard

Three results after the board lane built the `RCEN`/`RCCU` instrument. Two of them remove options
rather than add them, which is the useful direction.

### 1. My pre-registered pair cannot separate anything — the board lane is right

The bound is `(end - base) mod 16`, **at most 15 bytes, regardless of what `end - base` is**, because
`n` is computed from that same quantity and scales with it. So the "allocator rounded the region up
and the request figure was the misleading number" branch **cannot produce 1,728 either**: a rounded
region gives a proportionally larger `n` and the cursor still lands on `end`. My exact-multiple-of-4096
versus non-multiple pair therefore predicts the same `<= 15` on both arms and would have spent a boot
to learn nothing. Withdrawn.

### 2. "END MOVED DURING THE FILL" is REFUTED from the RTL

The board lane's third account — `n` computed before the loop, `end` grown by 1,728 during it, every
store advancing correctly — is excluded at the flashed revision. `STC`'s UNINIT path is
(`capstone_dyn_unit.anvil`, `1bfff7776`):

```
else if(rs1_v.metadata.cap_type==cap_type_t::CAP_TYPE_UNINIT){
    let new_cursor = rs1_v.cursor + 64'd16;
    let new_rs1 = call create_capability(rs1_v.metadata,new_cursor);
```

**The metadata — which carries `end` — is passed through unchanged and only the cursor advances, by
exactly 16.** No path in `STC` writes `end`, the capability lives in a register nothing else in the
loop touches, and there is no concurrent agent. So `end` cannot move during the fill.

Also read while there, and it confirms the loop is correctly bounded rather than lucky: `STC`'s check
is `rs1_up > (end - 16)` → OUT_OF_BOUNDS, so the last legal store sits at `cursor == end - 16` and its
advance lands the cursor exactly on `end`. A fill of `(end-base)/16` stores is exactly right, and one
more store would fault rather than silently stop.

### 3. So `RCEN`/`RCCU` will narrow LESS than hoped, and the honest version of the prediction is shorter

With "end moved" excluded, both surviving readings collapse into the **same** account and differ only
in the denominator:

| RCEN | RCCU | what it means |
|---:|---:|---|
| 1,419,584 | 1,417,856 | 88,724 stores attempted, 108 did not advance the cursor |
| 1,421,312 | 1,419,584 | 88,832 attempted, 108 did not advance the cursor |
| anything | `RCEN - RCCU != RCSH` | the reclaim path is mis-instrumented |

**The instrument is still worth having** — the self-check is the good part, and an inconsistency
surfacing as an inconsistency rather than a plausible number is exactly right. But it will name the
denominator, not the mechanism. **The question it leaves is which 108 stores, and why.**

**A discriminator that does work, because it does not lean on the mod-16 arithmetic:** run the same
reclaim on **two different large region sizes**. A constant 1,728 is a fixed tail effect; a shortfall
that scales with region size is a proportional store-failure rate. Those are different bugs and one
pair of boots separates them.

### 4. LIVE HAZARD in the fill macro, checked in the generated artifact rather than assumed

`C_RECLAIM`'s outputs are declared `"=r"`, **not** `"=&r"`. There is no early-clobber, so the compiler
is entitled to allocate an output register overlapping an input — and this template violates the
assumption that licenses that, because it writes `%1` in its **first** instruction (`mv %1, %3`) and
reads `%2` throughout, then writes `%0` while `%2` is still live. An overlap of `%0`/`%2` would have
`lcc(%0, %2, 2)` destroy the capability before `lcc(%1, %2, 4)` reads `end` off it.

**Measured, not reasoned:** in the shipped generated assembly all five sites allocate
`dest=t0, scratch=a1, cap=a2, n=a3` — **disjoint**, so it did not bite and **is not the cause of the
1,728**.

**But the change just made adds two more reports inside that same halting branch**, which raises
register pressure at exactly the site with no early-clobber. **Re-read the generated
`sbi_capstone_dom.c.S` for all five instances after the rebuild and confirm `%0`/`%1` still do not
overlap `%2`.** Cheap: grep the fill-loop line and read the four register names. Adding `&` to both
output constraints would remove the hazard permanently and is a one-character change per operand,
but it is the board lane's file.

### 5. The INPUT side was the worse hazard, and the instrument is sound for a reason nothing declares

Found by the board lane while checking (4), and it is the more dangerous half of the same template.
`C_RECLAIM` declares the capability `"r"(cap)` — **input-only** — while the hardware **advances its
cursor**. By the constraints alone every read of `cap` after the template is stale. That is not a
cosmetic complaint: it would have made `RCCU` report **0** and `RCEN` the pre-fill end, the
self-check `RCEN - RCCU == RCSH` would have **fired**, and it would have pointed squarely at the
instrument-fault branch. **The instrument would have blamed itself, plausibly, and been wrong.**

**It is saved, and by nothing in the constraints.** capstone-c's linear discipline writes a
capability back to its home slot after every use. Confirmed here in the generated assembly rather
than argued — around the fill site the sequence is:

    stc(t0, sp, 64)      ; capability spilled to its home slot
    ldc(a2, sp, 64)      ; reloaded into the template's operand
    <the fill template>  ; hardware advances a2's cursor
    stc(a2, sp, 64)      ; POST-FILL capability written BACK to the same slot

so any later `cap_cursor`/`cap_end` reloads the advanced capability, not a stale one.

**Record the dependency, because it is invisible at the C level.** `RCCU`'s correctness rests on that
write-back discipline and not on the asm constraints. Anyone changing how capstone-c spills
capabilities can silently invalidate this instrument, and the failure mode is the self-check firing
and accusing the wrong branch.

The `&` early-clobber was added to both outputs anyway; the allocation comes back identical at all
five sites, so it costs nothing and the output-side hazard stops resting on luck.

**Two decode traps worth inheriting**, both of which produced a clean-looking absence: the tag
constants are `lui`/`addiw` immediates and never stored literals, so searching a binary for their
bytes finds **none** of them — including one that provably fired on the board; and the capability
load/store forms are I-type and S-type, so the fields that look like `funct7` and `rs2` are immediate
bits, and comparing them as opcodes makes four of five sites look like they have no write-back.

---

## 6. RETRACTION: my refutation was wrong, the compression account is RIGHT, and here is the mechanism

**I refuted the compression hypothesis on a narrowed view and the refutation does not stand.** I read
`fat_cap_t`'s full 64-bit `start`/`end` and concluded there was nothing to round. Capabilities are
**fat inside the pipeline and COMPRESSED ON WRITEBACK**; I never looked at the writeback layer. The
board lane found `compress_bounds` (`core/include/ariane_pkg.sv:782`) and retracted their endorsement
of my refutation at the same time. Same failure shape as the others today: a clean-looking answer read
out of half the file.

**My STC reading was correct and INCOMPLETE.** Cursor advances by 16, metadata passes through
unchanged — true of the pipeline, and it simply does not reach the layer where compression happens.
The two arguments were never in contact, which is why both of us thought the question closed.

### The mechanism, every link quoted

1. **STC advances the cursor, metadata untouched** (`capstone_dyn_unit.anvil`, UNINIT path):
   `let new_cursor = rs1_v.cursor + 64'd16; let new_rs1 = call create_capability(rs1_v.metadata,new_cursor);`
2. **That capability becomes the result's `cap_rs1`** (`capstone_unit.anvilh`, `create_result_pack`):
   the third argument is assigned `cap_rs1 = rs1`, and STC passes `new_rs1` there.
3. **The DYN unit's rs1 IS re-compressed on writeback** — this is the site the board lane could not
   find, and it is on the dyn path rather than a separate store path, because **STC is a DYN op**
   (`decoder.sv:1309`, `fu = CAPSTONE_DYN`). `ex_stage.sv:1227`:
   `rs1_metadata : {(capstone_dyn_res.cap_rs1.metadata.cap_type != NOT_CAP), compress_cap(capstone_dyn_res.cap_rs1)},`
4. **`compress_cap` calls `compress_bounds` with the cursor** (`ariane_pkg.sv:869`).
5. **`compress_bounds` rounds the top up** when the cursor is off base (`:827-828`):
   `if(((bounds.base >> (E+3))<<(E+3)) != bounds.base) T[11:3] += 1;` with `E = leading_zeros - 12`.

### Why the timeline produces exactly 1,728, and why every store succeeded

The branch at `:783` is the hinge: **`if (bounds.start == cursor)`** selects a *cursorless* scheme
whose top is `t = (bounds.base >> E) & 21'h1FFFFF` — **no round-up term at all**.

    cursor AT base   -> cursorless scheme, exact top.  cap_end reads the TRUE end.
                        n = (end-base)>>4 = 88,724.   [RCPR confirmed cursor==base here]
    first stc        -> cursor moves off base
    every writeback  -> rounding scheme. len 1,419,584, top bit 20, E = 8,
                        granule 2^(E+3) = 2048, top rounds 1,419,584 -> 1,421,312
    after the fill   -> lcc(...,4) decodes the ROUNDED end; cursor is on the TRUE end
                        shortfall = 1,421,312 - 1,419,584 = 1,728 = 0x6C0

**So all 88,724 stores succeeded and the cursor is exactly where it should be.** "108 stores did not
advance the cursor" — the only account I had left standing — is refuted. The shortfall is a readback
artefact of bounds re-encoding, not a fill failure.

`E == 0 && len[12] == 0` (`:818-821`) is exact, so the effect **cannot** appear on small regions. That
is why 5,334 Sublet inits succeeded and nothing before sw60 ever showed it.

### What this changes about R-30

The R-30 fix is fine and sw61 is it working. sw60 is **not an R-30 failure at all**, and not a fill
failure either: it is a **separate defect in bounds re-encoding** that makes `end` read high once the
cursor leaves base. It deserves its own registry entry rather than more R-30 text.

**Still not proven, and worth saying:** the chain above is read from source. sw62's aligned arms are
the empirical test, and they are the right one — a granule-aligned region needs no round-up, so
compression predicts a CLEAN reclaim exactly where any store-failure model predicts a shortfall.

## 7. Working the "what else reads a bound after moving a cursor" list — the exposure is ONE window

The board lane listed six monitor sites to check and asked for the cursor state at each rather than
treating the list as a finding. Worked, and the answer is better than feared.

**INIT RESETS THE CURSOR TO BASE, which is what closes most of the list.** `capstone_flu_unit.anvil`,
INIT's success path:

    let val = rs2.cursor;
    let rd_temp = call modify_cap_type(rs1,cap_type_t::CAP_TYPE_LINEAR);
    let new_cursor = val + rd_temp.metadata.start;

The monitor's `csinit` passes **`x0`** as rs2 (`.insn r 0x5b, 0x1, 0x9, %0, %2, x0`), so `val = 0` and
`new_cursor = start`. **A completed reclaim therefore hands back a capability whose cursor is at base**,
which re-selects the exact cursorless encoding. Nothing downstream of a successful reclaim inherits a
rounded `end`.

**The caller-visible region-field query is NOT exposed in practice.** `CAPSTONE_REGION_FIELD_END` and
`..._LEN` read `regions[]` or a CPMP entry. Those capabilities are freshly created or post-INIT, so
their cursor is at base. The worry was right to raise and does not survive the check.

**The share path's `ALEN` trace is safe by explicit ordering**, not by luck — its own comment says the
trace *"deliberately runs FIRST"*, before `C_DO_RECLAIM`, and `RCPR` verifies `cursor == base` on the
very next line.

**So the only window in the monitor where a bound is read with the cursor off base is inside
`C_RECLAIM`'s own postcondition** — between the first store and the INIT. That is exactly where the
defect surfaced, and nowhere else reaches it.

**Two caveats, because this is a property of the current call sites and not a guarantee:**

* It holds for the code as written. **Any new code that reads a bound off a mid-fill capability hits
  this**, and nothing in the type system or the API says so.
* `compress_bounds` truncates the **bottom** as well as rounding the top (`B[13:3] = {bounds.start >> E}[13:3]`
  with no correction term), so `cap_base` can read **LOW** by up to granule−1 on the same encoding.
  It did not bite on sw60 because that region's base is 2048-aligned, but a non-granule-aligned base
  would move `cap_base` too, and `..._FIELD_LEN` subtracts the two.

### A firmware-side fix that does not wait on the RTL

The postcondition does not need to read `end` back at all. It already knows the answer before the
fill: `n` stores of 16 from a cursor verified at base must land the cursor at `base + 16n`. So

    shortfall = (base + 16*n_initial) - cursor_readback

compares two quantities neither of which is a re-encoded `end`, and `cap_base`/`cap_cursor` are read
either side of the same window. That makes the check immune to the re-encoding regardless of whether
the RTL defect is ever fixed, and it keeps the postcondition doing its real job — catching a fill that
genuinely fell short. It is the board lane's file and their call.

## 8. sw62 confirms it, and R-33 is WORSE than "end reads high": the re-encoding WIDENS AUTHORITY

sw62 settles the mechanism — arm 4 read the pre-registered compression figure (448) against 432 for a
proportional rate, both granule-aligned arenas reclaimed clean, and `RCCU` showed the cursor reached
the **true** end with every store advancing. `ALEN` and `RCEN` disagreeing **within one arm** —
354,880 before the cursor moved, 355,328 after — is the defect in a single boot with no cross-boot
comparison. Both of the surviving "stores did not advance" accounts, mine and the board lane's, are dead.

**The board lane's addition to the postcondition fix is correct**, and I verified the direction rather
than taking it. The flashed INIT raises on `cursor < end`:

    if(rs1.cursor < rs1.metadata.end){
        call raise_exception(data.trans_id,ex_code::ILLEGAL_OPERAND_VALUE)

So a postcondition that stops reading `end` lets a large unaligned region reach `csinit`, whose cursor
sits at the true end while `end` decodes higher — an `ILLEGAL_OPERAND_VALUE` trap inside the monitor,
which is the uninformative failure the postcondition exists to prevent. Two checks with separate tags,
as they propose, not one substitution.

### The consequence neither of us has stated, and it is the serious one

**The rounded end is not merely REPORTED high — the exact end is LOST, and the capability's AUTHORITY
grows with it.** The pipeline reconstructs its fat bounds by *decompressing the register*
(`ex_stage.sv:784-785`, `cap_rs1 : decompress_cap_tagged(...)`), so after the first store re-encodes
the bounds every later decode yields the **rounded** end. STC's bound check then uses that end:

    let rs1_end = rs1.metadata.end - 64'd16;   // dyn unit, STC
    ... ((rs1_up>rs1_end)||(rs1_up<rs1.metadata.start))  -> OUT_OF_BOUNDS

For sw62 arm 4 that authorises stores up to `base + 355,312`, whose last byte lands at
`base + 355,328` — **448 bytes past the 354,880 the allocator actually reserved.** The hardware will
permit those writes. They land in whatever follows the allocation.

**So R-33's headline should not be "`end` reads high after the cursor moves". It should be "moving a
cursor WIDENS a capability's bounds beyond its allocation, by up to granule−1 bytes".** The reclaim
shortfall is a symptom; over-permissive authority is the defect, and it is a capability-soundness
issue rather than an instrumentation nuisance.

### Which makes the CAUSAL fix an allocator one, and it is cheaper than it sounds

Compressed-bounds formats are lossy by construction and the standard contract is that allocators hand
out **representable** objects. Nothing here enforces that: the monitor's region allocator passes the
requested size straight through, so a size that is not a multiple of its own encoding granule is
silently widened. sw62 is exactly that experiment — **both granule-aligned arenas were clean and only
the unaligned one failed.**

**Round region sizes up to the representability granule at creation** (granule `2^(E+3)` with
`E = floor(log2(len)) - 12`, exact when `E == 0 && len[12] == 0`), and the reclaim works, the
region-field API reads true, and no authority is widened. That fixes the cause; the two-check
postcondition remains worth having as the thing that catches it if the contract is ever broken again.

### Two limits on all of the above, stated because they bound R-33

* **Only STC-advanced cursors on the DYN path are demonstrated.** `CINCOFFSET` moves a cursor too and
  its `cap_rs1` is compressed on the FLU writeback (`ex_stage.sv:1215`), which is structurally the
  same path — so the same widening plausibly applies to ordinary LINEAR capabilities the moment their
  cursor leaves base. **Untested, and it should not be asserted until it is**, but if it holds the
  scope is far wider than the reclaim.
* **The bottom truncation is unobserved.** `B[13:3] = {bounds.start >> E}[13:3]` has no correction
  term, so `base` can decode LOW on the same encoding. Every sw62 region was base-aligned, so nothing
  there could have shown it.
