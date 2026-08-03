# The silicon blocker — everything known

**Living document.** Update it whenever a claim is added, refuted, or measured. Every entry
must say how it is known: MEASURED (board), SOURCE (quoted file:line), or INFERRED.
Last updated: 2026-08-03.

---

## 2026-08-04 — Defect 2: SOURCE pointer refuted; records 1+ are getting ZEROS, not stale data

`sv` gives all three globals the SAME initial value (7). If the blob source pointer never
advanced, every record would copy record 0's bytes -- identical by construction -- and all three
would be correct (777).

    sv = 700   ->  sv0 = 0, sv1 = 0, sv2 = 7

Still only the first-processed record lands. **The source pointer is NOT the defect**, and the
"every record copies record 0's data" model is refuted: later records receive **zeros**, not
duplicated data.

Combined with the cap table being proven correct and distinct (`pk = 117`), that leaves a narrow
set:

* records 1+ take the **zero-init path**. The descriptor's `blob_off == -1` means ".bss, fill
  with zeros" (`ld t5, 16(t0)` reads it per record). If `t5` is clobbered inside the loop, or the
  field is misread for later records, they are memset to zero instead of copied -- which is
  EXACTLY "zeros, not stale data".
* or the copy is **skipped** for records 1+ because a loop counter (`a5`/`a6`) is not
  re-established per record, leaving nothing to copy.

Both predict zeros; both are loop-carried state; neither involves the carve, the slot pointer,
or the source. **`t5` is the prime suspect** because it is read once per record and its `-1`
sentinel selects the zero-fill branch.

**Next probe:** publish the per-record `blob_off` the loop actually read (same return-a-number
technique as `pk`, which is what finally made progress here). If record 1 sees `-1` where the
descriptor holds a real offset, the bug is `t5` clobbering and the fix is a register reallocation
in the loop.

Elimination table for defect 2 so far — all MEASURED, each a one-variable probe:
    cap-table slot pointer   fresh-from-gp derivation changed nothing      NOT IT
    split not consuming sp   RTL writes narrowed rs1 unconditionally       NOT IT
    the carve / slots        pk=117: valid, distinct, NONLIN               NOT IT
    blob source pointer      sv=700 with identical values                  NOT IT
    zero-init path / counter                                               <- REMAINING

## 2026-08-04 — DEFECT 2 LOCALISED: the cap table is CORRECT; only ONE record's COPY lands

Two direct observations, replacing inference with measurement.

**1. The cap table is fine.** `pk` reads slots 0 and 1 with `ldc gp[i]` and inspects them with
`lcc` (so the diagnostic does not depend on global init):

    pk = 117  ->  type0 = type1 = 1 (NONLIN), flags = 7:
                  slot0.start != 0, slot1.start != 0, and the two are DISTINCT

So the record loop carves per-record correctly and populates every slot with a valid, distinct
storage capability. **The carve, the slot pointer and the descriptor walk are all exonerated.**
Defect 2 is purely in the CONTENT copied into those (correct) carves.

**2. Exactly one record's copy lands — and it is NOT record 0.** `vv` returns the actual bytes of
three distinct globals as `b0 + 10*b1 + 100*b2` (correct = 321):

    unfixed glue          vv = 0     all three zero (defect 1: size-2 goes to the byte tail)
    with copy-length fix  vf = 300   b2 = 3 CORRECT, b1 = 0, b0 = 0

Only the LAST-DECLARED global has its data. Since the glue walks descriptors in the reverse of
declaration order, that is the **first-processed** record. So the shape is: *the first copy the
loop performs succeeds; every later copy silently does nothing.*

This corrects the earlier reading of `mf2`/`mf3` = 1: I assumed "record 0 works". It is the
first-PROCESSED record that works, which is a different statement and points at loop-carried
state in the COPY, not at ordering of the globals.

**What remains for defect 2:** the copy's per-record source/destination state — the blob view
`s1`, the destination pointer `t6`, or the byte counters `a5`/`a6` — is not re-established for
the second and later records. The carve and cap-table are proven good, so nothing outside the
copy needs re-examining.

**Cheapest next probe:** publish the copy's source and destination ADDRESSES for the first two
records (same `lcc`/return-a-number technique as `pk`), rather than inferring them from the
bytes. If the destination is identical across records the destination pointer is stale; if the
source is identical the blob view is; if both advance correctly the failure is in the loop's
counter reset.

Glue reverted to its committed state.

## 2026-08-04 — Also REFUTED: the "NONLIN sp is not consumed by split" theory

The interp glue does `delin(sp)` early (`start-gp-captable-interp.S`, C-4b), while the generated
glue delins only on the copy path. That suggested: once `sp` is NONLIN, `split(t2, sp, t1)`
returns `t2` without narrowing `sp`, so every record carves the SAME region and only one global
ends up correct — which matches the 1/1, 1/2, 1/3 shape exactly.

**The RTL says otherwise** (`capstone_dyn_unit.anvil`, `func SPLIT`, :139-145):

    let rs1_orig = rs1;
    let rs1 = call modify_cap_end(rs1, val);         /* rs1 keeps [start, val) */
    let rd  = call modify_cap_start(rs1_orig, val);  /* rd  gets [val, end)    */
    let result = call create_result_pack(..., rs1, rd);   /* BOTH written back */

The write-back of the narrowed `rs1` is unconditional; the LINEAR/NONLIN test at `:120` is a
type GUARD, not a gate on the update. So `split` narrows `sp` whether it is LINEAR or NONLIN,
and this theory is refuted without spending a boot.

**Standing state of defect 2** — "only record 0 initialises, at every global count":
* NOT the cap-table slot pointer (fresh-from-`gp` derivation changed nothing) — MEASURED.
* NOT `split` failing to consume a NONLIN `sp` — REFUTED from the RTL above.
* Remaining candidates: the blob SOURCE pointer's per-record advance, or `t1`'s walk-down
  arithmetic, or the descriptor read itself returning stale/zero fields for records 1+.
  The cheapest discriminator is a probe that RETURNS the per-record values the loop computed
  (blob_off, size, the carve base) for records 0 and 1, rather than inferring them from whether
  the global ended up correct — the loop already has a diagnostic-publish mechanism
  (`INTERP_PEEK_OFF`, STAGE 11) built for exactly this.

Glue is at its committed state; both experimental fixes reverted.

## 2026-08-04 — Defect 2 is NOT the cap-table slot pointer (chained-cincoffsetimm hypothesis REFUTED)

Hypothesis: `cincoffsetimm(s2, s2, 16)` at the end of the record loop is the CHAINED form
(`rd == rs1`), so `s2` becomes unusable after its first derivation and only record 0's slot is
written. This project had already built stages 128/129 to test chained-vs-independent derivation
and never measured them, so it looked promising.

Tested by deriving the slot FRESH from `gp` every record instead of chaining
(`slli a4,s3,4; cincoffset(a4,gp,a4); stc(t2,a4,0)`), on top of the copy-length fix:

    r14sl   4   OK      control
    g1      1/1 OK
    g2      1/2 WRONG   <- unchanged
    g3      1/3 WRONG   <- unchanged

**No change.** Exactly one global still initialises at every count, so the cap-table slot pointer
is NOT the defect. Chained `cincoffsetimm` is exonerated for this path.

**Where defect 2 must therefore live:** the other per-record state — the STORAGE carve
(`split(t2, sp, t1)` with `t1` walking down) or the blob source pointer. The carve is the more
suspicious of the two: the GENERATED glue performs the same `split` per global and works at 192
globals, but it is UNROLLED straight-line code, whereas this is a loop reusing `sp`/`t1` across
iterations. If `sp` or `t1` is degraded by the first `split`, every later carve is wrong while
record 0's is fine — which is exactly the observed shape.

**Next probe, and it is cheap:** make the loop derive its carve base fresh each record (or
unroll two records by hand) and see whether record 1 lands. If it does, the defect is the reuse
of `sp`/`t1` across `split`s, and the generated glue works only because it never reuses them.

Both fixes are REVERTED in-tree; the glue is at its committed state. Backup of the two-fix
version: `/tmp/capstone/interp-glue.bak` is the ORIGINAL, so re-apply from the quoted edits.

## 2026-08-04 — ROOT CAUSE: TWO defects in the interp glue's globals copy

Probes with DISTINCT per-global values and an individual check per global, so the return value
counts correctly-initialised globals instead of summing them (which hid which one failed).

    unfixed glue          mg1 = 0/1    mg2 = 0/2    mg3 = 0/3
    copy length -> 8      mf1 = 1/1 OK mf2 = 1/2    mf3 = 1/3

**MINIMAL REPRO is now ONE global**: a single `static char m[2] = {1,0};` under the interp glue
reads back 0 on silicon (`mg1`), QEMU correct, in-boot control passing. No struct, no loop, no
second global.

**Defect 1 — the byte-wise tail.** For a size-2 global the copy's `blt a5,8,21f` branches
straight to the byte tail, so the tail is the ONLY path that runs. Rounding the copy length up
to 8 (`addi a5,t3,7; andi a5,a5,-8`) makes the single-global case CORRECT (`mg1` 0 -> `mf1` 1).
Safe by construction: the carve is already `align_up(size,16)`.

**Defect 2 — the per-iteration advance.** With the length fixed, **exactly ONE global is correct
at every count** — 1 of 1, 1 of 2, 1 of 3. So globals after the first are still not initialised,
independently of size handling. Something the loop carries between descriptor records (the
destination pointer `t6`, the blob source, or the descriptor pointer `t0`) does not advance
correctly, so only record 0 lands.

**Defect 2 is the one that matters for SQLite**, and it is the best R-16 candidate yet: an image
with 181 globals of which only the first initialises is catastrophically broken before
`domain_main`, which is exactly where R-16 stalls. Note it is INDEPENDENT of size — it would
break an all-8-multiple image too.

**Status of the fix:** the length rounding is REVERTED in-tree (backup at
`/tmp/capstone/interp-glue.bak`, edit quoted above). It is a genuine partial fix but landing it
alone would leave the far more damaging defect 2 in place while making the symptom look better.

**Next:**
1. Read the loop's inter-record state (`t0`, `t6`, blob source) and find why record 1 is not
   written. This is a source question, no board time.
2. Fix both, then re-run `mg1`/`mg2`/`mg3` — success is 1/1, 2/2, 3/3.
3. Then SQLite stage 10, which is the R-16 test.

## 2026-08-04 — The copy loop IS involved: rounding its length is a PARTIAL fix (0 -> 1 of 2)

Located the blob->storage copy in `BUILD_GP_CAPTABLE_INTERP`: it sets the remaining-byte
counter from the RAW descriptor size (`mv a5, t3`), takes an 8-byte fast path
(`li a6,8; blt a5,a6,21f; ld/sd; addi a5,a5,-8`) and falls into a **byte-wise tail**
(`addi a5,a5,-1`) for the remainder. Non-8-multiple globals are exactly the inputs that reach
that tail.

Tested a one-line change — round the copy length up to 8 so the tail never runs. Safe by
construction: the carve is already `align_up(size,16)`, so writing up to 7 bytes past `size`
stays inside the global's own carve.

    r14sl   4   OK      control
    fxe8    2   OK      fixed glue, char[8]  (unchanged, as expected)
    fxe2    1   WRONG   fixed glue, char[2]  -- was 0, now 1 of 2 globals correct
    sze2    0   WRONG   unfixed glue, char[2]

**So the byte-wise tail is PART of the defect but not all of it.** One global of the two now
initialises; the other still does not. The fix direction is confirmed and the remaining error is
narrower than before, but this is not a working fix.

**The change has been REVERTED** — `start-gp-captable-interp.S` is back at its committed state.
A half-fix in glue shared by every interp build would silently change behaviour for SQLite and
every future rung while still being wrong; the backup of the attempt is at
`/tmp/capstone/interp-glue.bak` and the exact edit is described above, so it is one command to
re-apply when someone can finish it.

**Next, in order:**
1. Work out why exactly ONE of two identical `char[2]` globals is fixed by the rounding. The
   asymmetry is the strongest clue available — two identical globals, same size, same path, and
   they behave differently. Suspect the per-global `blob_off` arithmetic or the destination
   pointer advance, not the length.
2. Only then re-apply the rounding as part of a complete fix.
3. The SQLite test (pad globals to 8-byte multiples, run stage 10) is still worth doing
   independently — it does not depend on the glue fix and answers whether R-16 is this bug.

## 2026-08-04 — Narrowing the interp size fault: the CARVE is not the bug

Source read following the `sze8` PASS / `sze2` FAIL pair. What is now excluded inside the glue:

* **The per-global carve DOES round up.** `start-gp-captable-interp.S` (record loop):
  `addi t4, t3, 15` / `andi t4, t4, -16`, i.e. `stor = align_up(size, 16)`, matching the
  generated glue's `align_up(size, 16)` (`gen-gp-captable-glue.py:181`). So a `char[2]` global
  gets a correctly-sized, 16-aligned carve; the storage capability is not undersized.
* **The granule-alignment block is DISABLED.** The `li a4, 8 /* granule */` code sits under
  `#if defined(INTERP_GRANULE_ALIGN)`, and its own comment records it as off and
  non-discriminating ("the residual misalignment is LARGER in a domain that passes (304 B) than
  in the one that wedges (240 B)"). It is not in play.

So the fault is NOT the carve size and NOT that disabled granule path. It remains somewhere in
how the initialised CONTENT reaches the carved storage for a non-8-multiple global — the monitor's
blob copy is 8-byte granular by construction (`sbi_capstone.c:786-791`), and the descriptor
carries the raw `size` (24 B records: `u64 size ; u64 align ; i64 blob_off`, `blob_off == -1`
meaning zero-init). **I did not locate the blob->storage copy loop itself, so the final mechanism
is NOT established — do not write one into ISSUES.md on the strength of this section.**

**What IS established and is the usable result:**

    sze8   2   OK      interp, two char[8]   -- 8-byte multiple
    sze2   0   WRONG   interp, two char[2]   -- NOT an 8-byte multiple

one variable, in-boot control passing, QEMU correct for both. That is a genuine, minimal,
silicon-only reproducer and it is enough to act on.

**Two next steps, both cheap and independent:**
1. **Find the copy.** Grep the record loop for the load/store pair that moves `blob_off`
   content into the carved storage and check its length handling against a size of 2. If it
   iterates in 8-byte units bounded by the raw size, `floor(2/8) == 0` iterations explains
   everything -- but that is a hypothesis until the loop is read.
2. **Test the R-16 link directly, no source needed:** pad SQLite's globals to 8-byte multiples
   and see whether a stage-10 image enters. If it does, R-16 is this bug and the whole SQLite
   blocker falls out of it.

## 2026-08-03 — ROOT CAUSE: the interp glue mishandles globals whose size is NOT an 8-byte multiple

**One-variable pair. Same global COUNT (2), same code, only the size differs:**

    r14sl   4   OK      control (generated glue)
    wbi     4   OK      interp, globals 64 B and 8 B
    sze8    2   OK      interp, two char[8]   -- 8-byte multiple
    sze2    0   WRONG   interp, two char[2]   -- NOT an 8-byte multiple

QEMU computes the right answer for both. On silicon, the non-8-multiple build returns 0 --
every global reads as zero.

**This retires the "count" story entirely.** The bisection that looked like a low count
threshold (`ri2`/`ri4`/`ri8`/`ri16`/`ri32`/.../`ri192` all failing, `wbi` passing) was an
artefact of my probe sources: every `rc*` rung used `char[2]` globals, and `wbare`/`wbi` used a
64-byte array plus 8-byte pointers. Count never mattered; SIZE did. Correspondingly the earlier
"interp + many carves is a conjunction" reading is withdrawn -- `ri192` failed for the same size
reason as `ri2`.

**It is consistent with the glue's own documented constraint.** `gen-gp-captable-glue.py` rejects
non-copy-eligible globals with *"needs the large-RO copy path (file-scope symbol, 8-mult size)"*
and computes `size % 8`; the monitor's blob copy is explicitly in **8-byte units**
(`sbi_capstone.c:786-791`, "8-BYTE UNITS now, not 16 ... emits a scalar ld/sd"). A global whose
size is not a multiple of 8 therefore has a tail the 8-byte-granular path cannot express. The
generated glue emits straight-line per-global code and does not hit this; the interp glue's
descriptor-driven loop does.

**Why this is very likely R-16's cause.** SQLite is full of globals with non-8-multiple sizes
(char arrays, small structs, string tables) and it uses the interp glue. Every SQLite image has
been running a glue path that silently mis-initialises exactly those globals on silicon.

**What is measured vs inferred:**
* MEASURED: interp glue + non-8-multiple global sizes => all globals read 0 on silicon, correct
  under QEMU, with an in-boot control passing. One-variable pair (`sze8` vs `sze2`).
* INFERRED, not yet shown: that this is the same defect as R-16's ENTRY STALL. `sze2` returns 0;
  R-16 hangs. Same glue, same class of input, different symptom.

**Next:** (1) find the exact tail handling in `start-gp-captable-interp.S`'s copy loop and in
`gen-gp-captable-glue.py`'s descriptor emission -- the fix is likely rounding the per-global
copy length up to 8 or handling the remainder; (2) rebuild an SQLite stage-10 image with all
globals padded to 8-byte multiples and see whether it enters; that is the direct test of the
R-16 link and it needs no new mechanism.

## 2026-08-03 — SHARPENED: the interp glue fails on silicon at a LOW global count (3..32)

Bisection under `DOMAIN_GLUE=interp`, one boot, control first, every image QEMU-validated:

    r14sl    4    OK      control (generated glue)
    wbi      4    OK      interp,   2 globals
    ri32     0    WRONG   interp,  32 globals
    ri64     0    WRONG   interp,  64
    ri96     0    WRONG   interp,  96
    ri128    0    WRONG   interp, 128
    ri192    0    WRONG   interp, 192

**It is not a scale effect.** Everything from 32 up fails identically (all globals read zero),
and the same sources on the GENERATED glue pass at every one of those counts (`rc32`..`rc192`,
measured earlier). So the boundary is somewhere in **3..32 globals**, not near SQLite's 181 and
not near the 127/2047 B 12-bit limit I expected to find.

Caveat on the pair: `wbi` has 2 CAPABILITY-BEARING globals while `ri32` has 32 plain `char[2]`
globals, so count is not the only difference between the passing and failing arms. The clean
next step is to bisect with the SAME kind of global: build `ri4`, `ri8`, `ri16` from the `rc`
sources under interp and find the first failing count.

**Why this matters for R-16.** SQLite uses the interp glue with 181 globals, i.e. far past this
boundary — so every SQLite image has been running on a glue configuration that fails on silicon
at a tenth of that population. This is now the most likely explanation for R-16 that has ever
been on the table, and it is reproducible in an 11 KB-source rung.

**Still not established:** `ri*` RETURN 0 while R-16 HANGS. Same glue, same over-threshold
population, different symptom. Treat "the interp glue is broken on silicon above a small global
count" as measured, and "therefore R-16" as a hypothesis until a rung actually hangs.

**Next, cheapest first:** (1) bisect 4/8/16 under interp with the `rc` sources for the exact
threshold; (2) read `start-gp-captable-interp.S:117-243` for a counter, offset or register reuse
that breaks past a handful of descriptors; (3) diff the interp glue's emitted loop against the
generated glue's straight-line code for the same rung — the generated one works at 192.

## 2026-08-03 — R-16 LEAD: interp glue AND many carves is a silicon-only failure (conjunction)

The last untested combination, and it is the one SQLite actually uses.

    r14sl   4     OK      control
    wbi     4     OK      interp glue + 2 globals + SQLite geometry
    ri192   0     WRONG   interp glue + 192 carves + SQLite geometry   (QEMU returns 192)

Valid differential: QEMU computes 192 from the identical binary, silicon returns 0 — every one
of the 192 globals reads as zero. Contrast with the two halves, each already measured:

    interp glue alone      wbi     2 globals,  interp     -> PASS
    many carves alone      rc192   192 carves, generated  -> PASS
    BOTH                   ri192   192 carves, interp     -> FAIL

**So it is a conjunction, and it matches SQLite's configuration** (interp glue, 181 carves).
This is the first rung-scale silicon failure that reproduces SQLite's own build shape, and the
first real progress on R-16 since the elimination table.

Note what it revises: "carve count is ruled out" was established with `rc192` on the GENERATED
glue, and stands only for that glue. Under interp, 192 carves fails. The elimination table
entries should be read as glue-qualified from here on.

**Mechanism candidate, not yet established.** The interp glue is descriptor-driven — a FIXED
loop over `.capstone_gp_initdesc` records — where the generated glue emits straight-line
per-global code. All 192 reading zero points at that loop failing to populate the cap table on
silicon while QEMU's looser model tolerates it. Next probes, cheapest first:

1. **Bisect the carve count under interp**: build ri16 / ri64 / ri128 and find where it breaks.
   That gives a threshold, and a threshold is what makes this diagnosable.
2. If a threshold exists, compare it against the 12-bit immediate boundary (127 globals /
   2047 B table) that the generated glue used to `die()` on — the interp loop may have an
   analogous limit it silently exceeds rather than diagnosing.
3. Read the interp glue's descriptor loop (`start-gp-captable-interp.S:117-243`) for an
   offset/counter that could overflow or wrap at scale.

**Whether this IS R-16 is not yet established:** `ri192` RETURNS 0, whereas R-16 HANGS at entry.
Same configuration, different symptom. It may be the same defect with the SQLite images failing
harder, or a second fault in the same glue. Do not conflate them without measuring.

## 2026-08-03 — RESOLVED: the GENERATED glue never initialises capability-bearing globals

One-variable pair, same source, same boot, in-boot control:

    r14sl   4     OK      control
    wbi     4     OK      DOMAIN_GLUE=interp     -- cap-init RUNS
    wbare   -62   WRONG   DOMAIN_GLUE=generated  -- cap-init is dead code

**So `__capstone_cap_init` WORKS on silicon, and there is no hardware fault here.** Capability-
bearing initialised globals are correct on the board when the glue actually calls the
initialiser. Everything I wrote about a silicon cap-init defect is retracted; the cause was
entirely on our side.

**The real defect, and it is ours:** `start-gp-captable-generic.S` contains **zero** references
to `cap_init` (`start-gp-captable-interp.S` has 15), while `build-ladder-domain.sh:22` makes
`generated` the **default**. So any domain built through the ladder path with a capability-
bearing initialised global silently gets an UNTAGGED global — the raw 8-byte address word from
the monitor's template copy — and no capability tag. QEMU is permissive about that; silicon is
not. The failure is silent at build time and looks exactly like a hardware bug at run time.

Fix options, in order of preference:
1. have the generated glue call `__capstone_cap_init` when the image defines one (the symbol's
   presence is already discoverable — it is what `.capstone_cap_init` records); or
2. make the build FAIL LOUDLY when `DOMAIN_GLUE=generated` is combined with a non-empty
   `.capstone_cap_init`, so the mis-pairing cannot be built at all.

(2) is a few lines in `build-ladder-domain.sh` and would have saved this entire detour.

**Scope of what this does and does not overturn.** It retracts only the cap-init thread
(`wcap`, `wbare`, `wdrf`, `wc64`, `wc160`). It does NOT touch: the R-16 elimination table
(image size, carve count, their conjunction, dom_data geometry, blob size, loader — those rungs
have no capability-bearing globals, so the generated glue was adequate for them), or R-14
(`k800`/`k1200`, `zoff`, `h2adj`/`h2far`), which remains a confirmed silicon fault with a
packaged reproducer.

**R-16 itself is still open**, and the glue axis is now properly testable: `wbi` proves the
interp glue works end-to-end at SQLite's geometry on an 11 KB rung, so a rung/SQLite comparison
across `DOMAIN_GLUE` is finally a valid one-variable pair.

## 2026-08-03 — RETRACTED: `wbare`/`wcap`/`wdrf` never ran cap-init at all (WRONG GLUE)

**The probes did not test what I said they tested.**

    build-ladder-domain.sh:22   DOMAIN_GLUE=${DOMAIN_GLUE:-generated}    <- I never set it
    start-gp-captable-generic.S  0 references to cap_init  -> NEVER CALLS __capstone_cap_init
    start-gp-captable-interp.S  15 references

Every rung I built (`wcap`, `wbare`, `wdrf`, `wc64`, `wc160`) used the **generated** glue, which
does not call `__capstone_cap_init`. The function is emitted into the binary — I disassembled it
and reasoned about its `stc` immediates — but **it is dead code in those images**. So:

* **"`__capstone_cap_init`'s capability stores do not take effect on silicon" is RETRACTED.**
  The stores never executed. Nothing about cap-init was measured.
* The `-62` / `2` results have a much simpler explanation: with cap-init never run, a
  capability-bearing initialised global receives only the raw 8-byte address word from the
  monitor's template copy (plain `ld`/`sd`, `sbi_capstone.c:786-791`), and **never gets a
  capability tag**. An untagged word is not a usable capability on silicon; QEMU is permissive
  about it. That is a configuration error in my probe, not a hardware defect.
* It also explains `wc64`/`wc160` returning 0 **under QEMU too** — untagged globals, not a
  miscompile, and not the print artifact either.

**What this does NOT retract:** the elimination table stands (image size, carve count, their
conjunction, dom_data geometry, blob size, and the loader were each ruled out by one-variable
pairs whose rungs did not depend on cap-init), and R-14 stands (`k800`/`k1200`, `zoff`,
`h2adj`/`h2far` — all measured, none involving cap-init).

**The corrected experiment, and it is now the obvious one:** rebuild `wbare` with
`DOMAIN_GLUE=interp` so cap-init actually runs, and compare against the `generated` build in the
same boot with an `r14sl` control:

    interp returns 4, generated returns -62  -> cap-init is REQUIRED and works; my probes were
                                                simply mis-built, and there is no silicon fault here
    interp also returns -62                  -> NOW there is a real cap-init fault, measured for
                                                the first time, with a three-line repro

This also finally makes the glue axis testable for R-16, because the earlier
`DOMAIN_GLUE=interp` attempt failed for a linker-script reason since corrected (use the DEFAULT
`link-gpfree.ld` with `DOMAIN_WINDOW=`, never the `-sq`/`-2m`/`-32k` scripts).

**Method note worth keeping:** the QEMU differential caught four bad probes today, but it could
not catch this one — a probe where the code under test never executes passes QEMU *and* looks
plausible on silicon. The check that would have caught it is the one already written in the
board-run skill: *verify the artifact does what the source says*, extended from "is the
construct present" to **"is it reached"**.

## 2026-08-03 — RTL STUDY: cap-init is NOT R-14's mechanism (closed from the disassembly)

An RTL read of `capstone-ariane` flagged one shared-mechanism candidate as most promising and
left it open pending disassembly: does `__capstone_cap_init` emit R-14's failing shape — an
`stc` with a NON-ZERO immediate off a register-form `cincoffset` base? **It does not:**

    ldc  a1, 0x0(gp)
    ldc  a0, 0x20(gp)
    stc  a0, 0x0(a1)            <- immediate 0x0, base from ldc gp[i]
    cincoffsetimm a0, a0, 0x10
    ldc  a1, 0x10(gp)
    stc  a0, 0x0(a1)            <- immediate 0x0

Both stores use **imm=0** with a cap-table-derived base. R-14 fails specifically on non-zero
immediates off a register-form `cincoffset` base (`k1200` fails / `h2adj` passes), and `zoff`
already showed that forcing imm=0 does NOT rescue R-14. **So the cap-init fault and R-14 are
different shapes** — the "same defect at different scales" idea is refuted from both ends, and
should stop being repeated.

**What the RTL read established (quoted, worth keeping):**

* `STC` checks only the DESTINATION (`rs1`) — type, permission, bounds, revocation validity
  (`capstone_dyn_unit.anvil:356-430`). An invalid destination FAULTS; it does not silently
  miscompute. So "the destination slot was invalid" does not explain a silent wrong value.
* Cursor and compressed metadata are written in the SAME commit-queue entry / same physical
  request (`store_buffer.sv:171-176`), so the "cursor lands but metadata doesn't" hypothesis is
  **REFUTED** — there is no split-write path.
* The monitor's globals blob copy is one-shot inside `create_domain`, strictly before first
  entry (`sbi_capstone.c:589-836,792-796`), using plain `ld`/`sd` by design so it never touches
  `compress_cap`. So blob-copy-clobbers-cap-init is not an ordering inversion at monitor level.
* **A real, structural, QEMU-invisible divergence exists:** the RTL compresses bounds losslessly
  only in special cases (`compress_bounds`/`compress_cap`, `ariane_pkg.sv:753-835`), while
  QEMU's memory-side tracking (`cm_map`, `op_helper.c`) keeps EXACT fat bounds and never
  compresses. A whole class of precision-loss bugs therefore cannot reproduce under QEMU by
  construction. Not proven to be `wbare`'s cause — `wbare`'s bare-symbol initialisers hit the
  compressor's lossless `cursorless` branch — but it is the standing reason QEMU agreement is
  weak evidence about silicon.

**Still UNRESOLVED and needing more than a source read:** whether the glue re-runs cap-init on
re-entry; a race in the `STC` exception-vs-commit handshake (`capstone_store_syncer`,
`dyn_unit.anvil:680-751`); and a cache-level forwarding hazard between the blob copy's plain
`sd` and a later `stc` to the same line. The first is a source question; the last two need
simulation, not reading.

**The cheapest open experiment is unchanged and still unrun:** build the same rung with
`DOMAIN_GLUE=generated` (whose glue never calls cap-init) and compare the address word against
the `interp` build — it decides whether the zeros are written BY cap-init or by the template
copy, in one boot.

## 2026-08-03 — RETRACTED: there is NO `base + offset` compiler bug (print artifact)

I reported a compiler root cause -- "cap-init drops `base + offset` initialisers, this is ours,
not the board's" -- on the strength of `-capstone-cap-init-print` showing `value=` blank for
those leaves. **It is a diagnostic-print artifact.** Verified in the emitted binary:

    __capstone_cap_init:
      ldc  a1, 0x0(gp)          # slot cap for wcap_ptr
      ldc  a0, 0x20(gp)         # cap for wcap_data
      stc  a0, 0x0(a1)          # wcap_ptr  = wcap_data
      cincoffsetimm a0, a0, 0x10   # <-- THE +16 IS APPLIED
      ldc  a1, 0x10(gp)
      stc  a0, 0x0(a1)          # wcap_ptr2 = wcap_data + 16

The print is `errs() << It.Value->getName()` (`CapstoneCapGlobalInit.cpp:222`), and for a
`base + offset` initialiser `It.Value` is an **unnamed `ConstantExpr` GEP**, so `getName()`
returns `""`. The pass's own comment at `:112-115` says the full interior pointer is what gets
stored. Nothing is dropped.

**Consequences:** (a) there is ONE fault here, not two -- the "compiler bug + silicon bug" split
is withdrawn; (b) `wc64`/`wc160` were NOT "miscompiled by this bug", so why they return 0 under
QEMU is open again; (c) the only real defect in the pass is the print itself -- it should show
the operand, not `getName()`.

**Reframing that matters more than the retraction.** `wbare`'s null test compiles to an 8-byte
INTEGER load of the address word (`ldc a0,0(gp)` then `ld a0,0(a0)`), and the glue's carve loop
copies the `.data` template -- `.quad wbare_data`, a nonzero link-time address -- into that same
storage (`start-gp-captable-interp.S:534-560`). So a slot reading back ZERO does not mean "the
store never happened": something actively wrote zeros over a nonzero template.

Leading explanation, single-fault: **`ldc a0, 0x20(gp)` returned a null/zero capability and the
`stc` faithfully wrote 16 zero bytes.** The same `a0` feeds both slots, which explains both
leaves and the clean return. That puts the defect on the LOAD FROM THE CAP TABLE, the same
family as the isolated `r14lp` result, and `__capstone_cap_init` does three `ldc gp[i]` where the
passing control `r14sl` does two.

**Cheapest discriminator, one boot, no new mechanism:** build the same rung with
`DOMAIN_GLUE=generated`, whose glue never calls cap-init at all (`start-gp-captable-generic.S`
has no `RUN_CAP_INIT`), and compare the address word against the `interp` build. Nonzero with
`generated` and zero with `interp` => the zeros are written BY cap-init. Zero in both => the
template copy is at fault and cap-init is exonerated. Run both alongside an `r14sl` control.

**Also worth knowing: capstone-c is not a reference for this problem.** It has no static
initialisers at all -- `visit_declaration` never reads `init_declarator.initializer`
(`capstone-c/src/lang.rs:346-368`), and a `char *p = data;` at file scope makes the reference
compiler panic. Its globals are runtime-carved and uninitialised. Where it does store a
capability it emits exactly our shape (`ldc gp[i]`; `stc`) with no extra `delin`/`scc`/`split`,
and its `AddressOf` uses `ldc` + `cincoffsetimm` byte-for-byte as we do. Its own samples never
exercise the global path.

## 2026-08-03 — CORRECTION: the cap-init capability is WRONG-VALUED, not NULL

`wdrf` dereferences a cap-init'd global UNGUARDED (no null test), the way SQLite uses such
globals:

    static char  wdrf_data[64] = { 'A', 0 };
    static char *wdrf_p = wdrf_data;
    return wdrf_p[0];                       /* expect 65 = 'A' */

    QEMU  : 65   correct
    board : 2    WRONG -- and note: NO fault, NO hang, control returned 4 in the same boot

**So "the stores do not take effect / come back NULL" is too strong — retracted.** The
capability is present and dereferenceable; it points at the WRONG PLACE. Reads yield 2 here and
0 in `wbare` where 'A'=65 is expected, so the value differs between builds — garbage, not a
consistent constant, and not a trap.

Re-reading `wbare` (-62 => n==0) with this: `n = (p1 ? p1[0] : 0) + (p2 ? 1 : 0)`. n==0 is
satisfied by p1 non-null with `p1[0] == 0` AND p2 null — consistent with wrong-valued
capabilities rather than uniformly null ones.

**Revised statement of the silicon fault:** `__capstone_cap_init`'s capability stores land, but
the capability subsequently read back from the global has an incorrect cursor/bounds, silently
and without trapping. QEMU produces the correct capability from the identical binary.

**This is also NOT yet linked to the R-16 hang.** `wdrf` was built to convert the fault into a
hang if they were the same defect; it returned a wrong value instead. So the NULL/wrong-value
fault and the entry stall remain two separate observations, and the "same defect at different
scales" idea is now weaker, not stronger.

Two subagents are running on this: one reading `capstone-c` for how the reference implementation
initialises capability-bearing globals, one reading the `capstone-ariane` RTL for a path where a
capability store lands with cursor but wrong/garbage metadata and no exception.

## 2026-08-03 — CONFIRMED: `__capstone_cap_init`'s capability stores DO NOT TAKE EFFECT ON SILICON

The decider. `wbare` uses BARE-SYMBOL initialisers only, so the compiler bug below cannot apply;
`-capstone-cap-init-print` confirms both leaves resolve:

    leaf 0 holder=wbare_p1 value=wbare_data holder_size=16
    leaf 1 holder=wbare_p2 value=wbare_data holder_size=16

    QEMU  : __CAPSTONE_LADDER_WBARE_PASSED__ (retval = 4)   correct
    board : retval = 4294967234 ( = -62 )                   BOTH pointers NULL
    control r14sl in the same boot: 4  OK

**So there are two independent faults, and this is the second one, now proven:**

    1. COMPILER  base+offset cap-global initialisers lose their value (empty `value=` in the
                 print output). Reproduces under QEMU. Ours, in CapstoneCapGlobalInit.cpp.
    2. SILICON   even fully-resolved capability stores in __capstone_cap_init do not take
                 effect on hardware, while QEMU executes the same binary correctly.

**THE MINIMAL REPRO — three lines, ~11 KB domain, no SQLite, no struct, no loop:**

```c
static char  wbare_data[64] = { 'A', 0 };
static char *wbare_p1 = wbare_data;
static char *wbare_p2 = wbare_data;
/* n = (p1 ? p1[0] : 0) + (p2 ? 1 : 0);  return n - 62;   QEMU 4, silicon -62 */
```

`__capstone_cap_init` is the routine the glue calls **before `domain_main`**, and it is the code
path R-16 stalls in. Every other axis was eliminated by a one-variable pair with an in-boot
control: image size, carve count, their conjunction, dom_data geometry, blob size, and the
loader (a stalling image stalls under `lpc` too).

**Still to establish:** whether this NULL-store fault and the R-16 *hang* are the same defect at
different scales. `wbare` returns rather than hanging, so that link is inferred, not measured.
The next probe is a rung that USES a cap-init'd global the way SQLite does (dereference through
it rather than null-testing it), which should convert the NULL into the fault the SQLite images
take.

## 2026-08-03 — ROOT CAUSE of the NULL capabilities: cap-init drops `base + offset` initialisers

`-mllvm -capstone-cap-init-print` (the flag the pass provides for exactly this) names it:

    wcap : leaf 0  holder=wcap_ptr   path=  value=wcap_data  holder_size=16   <- resolved
    wcap : leaf 1  holder=wcap_ptr2  path=  value=           holder_size=16   <- EMPTY
    wc64 : leaf 0  holder=wc64_p0    path=  value=wc64_data                   <- resolved
    wc64 : leaf 1..63                path=  value=                            <- ALL EMPTY

Store counts are correct (2 for `wcap`, 64 for `wc64`), so nothing is truncated -- the clamp is
disabled and the stores exist. **What is missing is the VALUE.** Only an initialiser that is a
bare symbol (`= wcap_data`) resolves; every `base + offset` form
(`wcap_data + 16`, `wc64_data + i%48`) emits a store with an empty value, which lands as a NULL
capability. `CapstoneCapGlobalInit.cpp` is a compiler pass, so **this is ours, not the board's.**

**It explains the invalid scale probes.** `wc64`/`wc160` used `data + i%48` for every pointer,
so all 64/160 were empty-valued and NULL under QEMU AND silicon -- which is precisely why they
returned 0 in both. They were not "broken builds" in some vague sense: they were miscompiled by
this bug, and the QEMU differential correctly refused to call that a silicon result.

**What it does NOT yet explain.** `wcap`'s leaf 0 IS resolved (`value=wcap_data`), yet silicon
returned -62, i.e. `n == 0`, meaning `wcap_ptr` was ALSO null on hardware while QEMU read it
correctly (retval 4). So there are two distinct failures in play:

    1. COMPILER: `base + offset` cap-global initialisers lose their value  -- proven above,
       reproduces under QEMU, ours to fix in CapstoneCapGlobalInit.cpp.
    2. SILICON:  even a correctly-resolved leaf 0 reads back NULL on the board but not under
       QEMU -- this is the silicon-only part of `wcap`, and it is still unexplained.

**Next, in order:** (a) fix (1) -- find where the pass resolves the initialiser operand and why
a GEP/offset constant expression yields no value; (b) rebuild `wcap` with a bare-symbol-only
variant (`p1 = data; p2 = data;`) so BOTH leaves resolve, and re-run on the board: if it still
returns NULL, (2) is confirmed as an independent silicon fault with a two-line repro; if it now
returns 4, then (1) was the whole story and the "silicon-only" reading of `wcap` was an artefact
of a miscompiled second leaf.

**(b) is one compile plus one boot, and it decides whether R-16 is a compiler bug or a hardware
bug.** Do it before anything else.

## 2026-08-03 — SOURCE READ: the 8-byte `.capstone_cap_init` clue was a RED HERRING

Read `llvm/lib/Target/Capstone/CapstoneCapGlobalInit.cpp` before spending more board time.

* **`size=000008` regardless of global count is BY DESIGN.** The section holds a single
  PC-relative table entry pointing at a synthesized `void __capstone_cap_init(void)`
  (`:196-205`), which the domain glue calls before `domain_main` (`:23`). It is one function
  pointer, NOT per-global records. My reading of "it does not scale, so globals are
  under-registered" was wrong — retracted.
* **The diagnostic clamp is NOT active.** `capstone-cap-init-limit` is `cl::init(0)`
  (`:67`), i.e. disabled, so truncation does not explain `wc64`/`wc160` returning 0.
* The per-global stores are emitted straight-line into that function, one store per capability
  leaf, and are marked **volatile so they are never elided** (`:19-26`).

**So why `wc64`/`wc160` return 0 under QEMU is still unknown**, and those probes remain invalid.
The standing minimal repro is unchanged: **`wcap`**, 2 capability-bearing initialised globals,
QEMU 4 vs silicon -62.

**Next diagnostic, and it is built for exactly this:** `capstone-cap-init-print` (`:72`, also
`cl::init(false)`) names WHICH global each emitted store corresponds to, and
`capstone-cap-init-limit=N` bisects to the offending index — the header at `:54-62` says the
limit exists so a bisect can name the first bad store, and records that a previous bisect
localised a failure to "the first entry, a 16-byte capability accessed at +32". Run
`-mllvm -capstone-cap-init-print` on `wcap` and on `wc64` first: if `wc64` emits far fewer
stores than 64, the probe is malformed at the IR level and the board result was never
meaningful; if it emits 64, the fault is downstream and `wc64` becomes a valid scale probe.

## 2026-08-03 — SCALE PROBES wc64/wc160 ARE INVALID (broken under QEMU too)

Scaling `wcap` to 64 and 160 capability-bearing initialised globals to see whether "returns
NULL" becomes "hangs":

    r14sl    4            OK      control
    wcap     -62          WRONG   VALID -- QEMU returns 4 from the same binary
    wc64     0 (exp 64)   WRONG   INVALID -- see below
    wc160    0 (exp 160)  WRONG   INVALID -- QEMU also returns 0

`wc160` returns **0 under QEMU** (`Called dom (1-th time) retval = 0`), so the scale probes are
broken builds, not silicon results — the fourth invalid probe this investigation, and the fourth
caught by the QEMU differential before it was reported as a board finding.

**A clue in the failure:** `.capstone_cap_init` is `size=000008` for `wcap` (2 cap-bearing
globals), `wc64` (64) and `wc160` (160) alike — it does NOT scale with the number of
capability-bearing globals. Either the section holds a count/pointer rather than per-global
records, or only one global is being registered and the rest are never initialised at all, which
would explain 0 under both QEMU and silicon. Read `CapstoneCapGlobalInit.cpp` and the glue's
consumer before building another scale probe.

**So the standing minimal repro remains `wcap`** — 2 capability-bearing initialised globals,
QEMU 4 / silicon -62 — and the "does it become a hang at scale" question is UNANSWERED, not
answered negatively.

## 2026-08-03 — MINIMAL REPRO AT RUNG SCALE: `__capstone_cap_init` produces NULL capabilities

**Ten lines of C, ~11 KB domain, QEMU-correct, silicon-wrong.**

    static char  wcap_data[64] = { 'A', 0 };
    static char *wcap_ptr  = wcap_data;        /* capability-bearing INITIALISED globals */
    static char *wcap_ptr2 = wcap_data + 16;
    ... n = (wcap_ptr ? wcap_ptr[0] : 0) + (wcap_ptr2 ? 1 : 0);  return n - 62;

    QEMU  : __CAPSTONE_LADDER_WCAP_PASSED__ (retval = 4)    correct
    board : RESULT wcap retval=4294967234  ( = -62 )        n == 0

`-62` means **both** pointers tested false: the capability-bearing globals came back NULL on
silicon. The same binary initialises them correctly under QEMU.

**Why this was never hit before — the presence-vs-execution trap.** `.capstone_cap_init` is
**size 0 in every ladder rung ever run** (`r14sl`, `wbhi`: `size=000000`) and non-empty in every
SQLite image (`f10`: `size=000008`). The routine builds capability-bearing globals AT DOMAIN
ENTRY — exactly where R-16 stalls — and no rung had ever executed a single byte of it. Five
rounds of minimisation scaled size, carves and geometry while leaving this code dead.
`wcap` is the first rung with `cap_init size=000008`, matching SQLite.

**Everything else is ruled out**, each by a one-variable pair with an in-boot control:

    image size            rz1m   1087 KB, 1 carve              PASS
    carve count           rc192  192 carves / 3072 B table     PASS   (> SQLite's 181/2896)
    both together         rzc1m  192 carves + 1.1 MB           PASS
    dom_data geometry     wsq    order 9, globals_off=0x150000 PASS   (SQLite's exact layout)
    blob size             wbhi   blob 90320                    PASS   (> the 84336 that stalls)
    the loader            strim  stalls under lpc AND sqlite_host      (loader exonerated)
    ---
    cap-bearing globals   wcap   returns -62 instead of 4      FAIL   <-- silicon only

**Status: this is a NULL-capability miscompute, not yet the entry stall itself.** `wcap`
entered and returned (SHA5->SHA6->RESULT); R-16 hangs. The honest claim is that the code path
R-16 lives in is demonstrably broken on silicon at rung scale, in a repro that costs one boot.
Whether more cap-bearing globals turn "returns NULL" into "hangs" is the next experiment: scale
`wcap` to tens/hundreds of capability-bearing initialised globals, which is what SQLite has.

Also settled in passing: the `SHA5` operand is a domain-sequence index, not a failure code —
`wsq` showed `SHA5:00000001 -> SHA6:00000001` and PASSED. Do not read it as diagnostic.

## 2026-08-03 — R-16: THE LOADER IS EXONERATED, the IMAGE carries the stall (MEASURED)

The zero-rebuild experiment the audit proposed, run in one boot with the control first. A
known-stalling SQLite image (`strim`, `SQLITE_STATIC_BUILTINS=1`) was loaded by the LADDER
controller `lpc` instead of `sqlite_host.user`:

    r14sl (control) via lpc :  SHA5:00000000 -> SHA6:00000000 -> RESULT retval=4   ENTERED
    strim           via lpc :  SHA5:00000001 -> no SHA6                            ENTRY STALL

**It stalls under BOTH loaders.** By the pre-agreed decision rule that means the loader is NOT
the variable, and the image is. This eliminates in one boot every loader-side difference that
had never been varied:

* `lpc` builds an anonymous, pre-`memset`, per-segment-`memcpy`'d buffer
  (`ladder_perf_ctl.c:243-254`); `libcapstone.c:135-137` passes a pointer into a `MAP_SHARED`
  file mapping the kernel then `copy_from_user`s ~1.4 MB from (`module/capstone.c:98`).
* `lpc` creates ONE region and ONE share; `sqlite_host.c:115-140` creates two and shares twice.

None of that matters to R-16.

**Two consequences.**

1. **The ladder is now a valid vehicle for R-16.** Its loader reproduces the stall on a real
   stalling image, so a rung built with SQLite's geometry is a legitimate comparison — and
   `DOMAIN_WINDOW=0x150000` on `build-ladder-domain.sh` produces exactly that at ~11 KB, with no
   new linker script (see the corrections above). That is the minimal-repro path, now unblocked.
2. **The `SHA5` argument differs**: `00000000` for the entering control vs `00000001` for the
   staller. Worth reading `sbi_capstone.c`'s SHA5 emit to learn what that operand is — it may
   name the failing step directly, and it is free to check offline.

Method note: this cost ONE boot and no rebuild, because both binaries and both controllers were
already in the initramfs. It eliminated a whole half of the search space that five rounds of
building could not.

## 2026-08-03 — AUDIT CORRECTIONS to the R-16 rounds (three of my claims were wrong)

Independently verified against the sources; each was load-bearing.

1. **"`ladder_perf_ctl` does not pack the globals offset, so the monitor falls back to 0x1000"
   — REFUTED.** `capstone/tests/rtl-smoke/ladder_perf_ctl.c:282,311` derives it from the section
   address and packs it: `a.entry_offset = c.entry_offset | (c.globals_off << 32)`. The rungs
   prove it works: `rz1m` returned 7 with `globals_off=0x108020`.

2. **My custom linker scripts CAUSED the 0xB10B I blamed on the controller.**
   `build-ladder-domain.sh:42-45` already accepts `DOMAIN_WINDOW=<any>` and seds it into the
   correct script — the same mechanism `build-sqlite-silicon.sh:299` uses. My hand-made
   `link-gpfree-sq.ld` / `-2m.ld` (and the stock `-32k.ld`) do **not** place
   `.capstone_gp_initdesc`, so lld orphan-places it next to `.text`, `globals_off` reads as
   ~0x3f0, and the monitor's guard (`sbi_capstone.c:744-754`) fires arithmetically.
   `DOMAIN_WINDOW=0x150000` gives an 11 KB rung with SQLite's geometry today, no new script.
   **Treat `-sq`/`-2m`/`-32k` as traps** until they place initdesc and put `.bss` in the
   `PT_NULL` nobits phdr the way `link-gpfree.ld:53-58,88-97` does.

3. **"Every passing rung is order-5 / 128 KB" — FALSE.** Only the tiny control is.
   `rz1m` = pages 281 and `rzc1m` = pages 283, both **order 9 / 2 MB** — already the class I
   called unique to the stalling images. Verified with the project's own budget script.

The audit also reached, independently, the same conclusion as the section below: **`f10` — the
known-good in-boot control — has pages-for-pages the stalling geometry (371 pages, order 9,
`globals_off=0x150000`, blob 75120)**, so no static geometry attribute can be the discriminator.
It further notes `sqlite_silicon` (full SQLite, 179 carves) is recorded ENTERING reliably at
`SILICON-BLOCKER.md:1536,1551`, i.e. the premise "all SQLite images stall" was refuted 1200
lines earlier in this very document and I relaunched from it anyway.

**Also flagged, and correct:** every "ruled out" rung verdict is N=1, on a failure this document
itself records as not strictly per-image. Those axes are *not* safely ruled out; they need a
stall-RATE (3-5 runs per image in one boot) before the wording goes anywhere durable.

### Best next experiment (from the audit, and better than mine): swap the LOADER, not the image

Zero rebuilds. Run an already-stalling `.dom` through the ladder controller `lpc` in the same
boot as `sqlite_host.user`. Both binaries are already in the initramfs. The two loaders differ
in ways never varied: `lpc` builds an anonymous pre-`memset`, per-segment-`memcpy`'d buffer
(`ladder_perf_ctl.c:243-254`), while `libcapstone.c:135-137` passes a pointer into a
`MAP_SHARED` file mapping that the kernel `copy_from_user`s 1.4 MB from; `lpc` creates one
region and one share, `sqlite_host.c:115-140` creates two and shares twice.

    enters under lpc, stalls under sqlite_host -> the IMAGE is exonerated; the loader is the variable
    stalls under both                          -> the image is implicated, and the ladder becomes a valid vehicle

## 2026-08-03 — R-16 REFRAMED: it tracks SQLITE_STATIC_BUILTINS, and the two issues are COUPLED

**A false premise ran through this whole investigation, including my own write-ups above:**
"small ladder rungs enter, SQLite-derived images stall". **`f10` is an SQLite-derived image** --
built by `build-sqlite-silicon.sh`, 1624096 B, 181 carves, allocation order 9 -- and it has
ENTERED reliably as the in-boot control dozens of times today. So image size, carve count and
dom_data class were never able to be the axis, which is consistent with rounds 1-5 ruling every
one of them out, and I should have noticed that the control itself refuted the premise.

**What actually separates them is one build flag.** Every image built with
`SQLITE_STATIC_BUILTINS=1` has entry-stalled -- `st10`, `sb10`, `swa`, `swa8`, `swa9`, `strim`:
**6/6** -- while `f10`, the same builder without the flag, enters. That is a one-variable pair
with six replicates on the failing side.

**The geometric difference is the BLOB** (`domdata-budget.py`), and it is the only one:

    image   blob     cap table        storage   allocation      verdict
    f10     75120    2896 (181 gl)    354320    order 9 / 2 MB  ENTERS
    swa     84336    2896 (181 gl)    354320    order 9 / 2 MB  STALLS
    strim   82592    2784 (174 gl)    352736    order 9 / 2 MB  STALLS

Carve count, storage and allocation class are identical or near-identical; only the blob moves,
by ~9 KB. **The blob is the initialised-globals template that is COPIED AT DOMAIN ENTRY, before
`domain_main` runs** -- exactly where R-16 stalls. It also explains why five rounds of ladder
probes never reproduced it: `.bss` is uninitialised and `.rodata` padding is never copied, so
none of them grew the blob at all.

### The practical consequence: R-14's workaround TRIGGERS R-16

`SQLITE_STATIC_BUILTINS=1` restores `aBuiltinFunc` to a compile-time-initialised static, which
is precisely what adds ~9 KB of initialised globals to the blob. So the R-14 workaround is what
pushes the image across the R-16 threshold. **That is why it could never be validated on
SQLite** -- and it is not a coincidence to be worked around by redrawing: every draw carries the
same extra blob.

Consequence for the default flipped earlier today: `SQLITE_STATIC_BUILTINS` now defaults to 1,
which fixes R-14 and guarantees R-16. Either the blob threshold gets understood, or the R-14
workaround needs a form that does NOT add initialised globals (e.g. a static that is
zero-initialised and filled at run time, keeping it in `.bss` rather than the blob). **That
last idea is cheap and is the obvious next experiment.**

### Status of the blob axis on the ladder path

Attempted and NOT yet working: growing the blob with a large initialised array. `static` fails
to link (`undefined symbol`) because the generator's large-RO copy path emits `lla <sym>` and
needs external linkage; made file-scope it links, but with the custom linker script the reported
`globals_off` collapses to 0x410 and the blob reads as ~1.4 MB, i.e. the measurement is not
meaningful yet. Unresolved plumbing, not a board result.

## 2026-08-03 — R-16 ROUNDS 3-5: three more axes ruled out; the last two are TOOL-BLOCKED

**Enabling fix landed first.** `gen-gp-captable-glue.py` used to `die()` above 127 globals
("cap-table (%d B) exceeds a 12-bit immediate; needs li+sub (TODO)"), which is exactly why the
rungs (<=96 carves) and SQLite (181) could never be compared on one glue. Two 12-bit limits were
involved and both are now handled: the table reservation (`addi` -> `li`+`sub`) and the
per-entry store offset (`stc t2, i*16(gp)` -> `li`/`cincoffset` register form past entry 127).
The <=2047 path is byte-identical, verified: `rc96` hashes the same before and after.

**Measured, each with an in-boot control, QEMU-validated first:**

    rc128   128 carves, 2048 B table                    -> 128   OK
    rc192   192 carves, 3072 B table  (> SQLite's 181/2896)  -> 192   OK
    rzc1m   192 carves AND ~1.1 MB image, together      -> 192   OK

So **carve count is ruled out even ABOVE SQLite's value**, and the conjunction of size and
carves — the shape that turned out to be R-14's answer — is ruled out too. Together with
rounds 1-2: image size (to 1087 KB), carve count (to 192), cap-table size (to 3072 B) and
their conjunction all fail to reproduce R-16.

**What actually distinguishes the stalling images, per `domdata-budget.py`:**

    every passing rung   allocation pages=18   order=5   tot_size=131072    globals_off small
    every SQLite image   allocation pages=371  order=9   tot_size=2097152   globals_off=0x150000

That geometry — not image size, not carve count — is the remaining difference. Two attempts to
mirror it on the ladder path, and BOTH are blocked by the harness rather than by the board:

* **`.bss` growth** (raise storage into the order-9 class): fails with
  `capstone_error 0xB10B` (blob does not fit) at 64 KB and above with the 32 KB globals window.
* **Large globals offset** (`link-gpfree-sq.ld`, 0x150000, mirroring SQLite exactly): still
  `0xB10B`, even trimmed to 361 pages, i.e. BELOW SQLite's 371. Cause is the controller:
  `ladder_perf_ctl` does not pack a large globals offset into `entry_offset`, so the monitor
  falls back to 0x1000 — the failure mode `run-sqlite-silicon.sh` already documents for a stale
  `sqlite_host.user`.

**Both failures were caught by the QEMU differential before reaching the board**, as was the
`DOMAIN_GLUE=interp` breakage in round 2. Three invalid probes, zero false silicon findings —
the QEMU gate is doing the work the retractions used to.

### Where R-16 stands, and the one path left

Ruled out: image size, carve count, cap-table size, their conjunction, and (for R-14) every
compiler-side mechanism. Untested and tool-blocked: the **interp glue** and the **dom_data
geometry / globals-offset class**. Nothing reproduces R-16 outside SQLite-derived images.

Minimise-from-below has reached the limit of the ladder harness. The remaining approach is
**minimise-from-above**: keep `build-sqlite-silicon.sh` (which sizes the globals offset, packs
`entry_offset` via `sqlite_host.user`, and uses the interp glue — all the machinery the ladder
path lacks) and shrink the PROGRAM instead. Note this is already partly informative: `:0`
returns before any SQLite code runs and still stalls, so the program is NOT the variable —
which means the target is to shrink the IMAGE while keeping the SQLite build path, e.g. via
`SQLITE_TRIM=1` / `SQLITE_OMIT_*` until the geometry crosses back into the passing class.
Alternatively, teach `ladder_perf_ctl` to pack the globals offset, which makes the whole
geometry axis testable at 11 KB rung cost.

## 2026-08-03 — R-16 ROUND 2: the glue axis is NOT YET TESTABLE (the pairing tool is broken)

The most promising remaining axis was the entry glue: ladder rungs use
`start-gp-captable-generic.S`, SQLite uses `start-gp-captable-interp.S`, and R-16 is an ENTRY
stall. `build-ladder-domain.sh` exposes `DOMAIN_GLUE=interp`, which looked like a ready-made
one-variable pair.

Board result, ascending with `r14sl` (generic) as the in-boot control:

    r14sl      4   OK      generic glue, control
    gi_r14sl   0   WRONG   SAME SOURCE, interp glue      (oracle 4)
    gi_rz1m    0   WRONG   SAME SOURCE, interp glue      (oracle 7)

**This is NOT a silicon finding — the interp build is simply broken for these rungs.** QEMU
differential on the identical pair:

    generic glue   QEMU = 4   board = 4     correct
    interp  glue   QEMU = 0   board = 0     wrong in BOTH

Returning exactly 0 is the "every global reads as zero" signature, and it reproduces under
emulation, so `DOMAIN_GLUE=interp` does not initialise the ladder's generated cap-table. The
descriptor-driven glue evidently needs the descriptor emission that the SQLite path provides and
this path does not.

**So the glue axis remains UNTESTED**, and the tool that was supposed to test it needs fixing
first. Recorded because the board result on its own looks exactly like a silicon miscompute, and
reporting it as one would have been the fourth false mechanism of this investigation. One QEMU
run cost less than a retraction.

Ways forward on this axis, cheapest first:
1. Fix `DOMAIN_GLUE=interp` in `build-ladder-domain.sh` so it emits/consumes the descriptors the
   interp glue expects — then the one-variable pair exists and costs one boot.
2. Go the other way: build a MINIMAL domain through `build-sqlite-silicon.sh` (which already
   uses the interp glue) with the amalgamation reduced to nothing, so the SQLite-side variable
   is the program rather than the glue.
3. Leave the glue and vary carves ABOVE 127, which needs the `li+sub` TODO in
   `gen-gp-captable-glue.py` fixed regardless.

**R-16 status after two rounds:** carve count (to 96) and image size (to 1087 KB) are ruled out;
the glue is untested; nothing yet reproduces R-16 outside the SQLite-derived images.

## 2026-08-03 — R-16 MINIMISATION ROUND 1: carve count and image size are BOTH RULED OUT

First application of the minimise-first strategy to R-16 rather than R-14. The unexploited
clue: every ~10 KB ladder rung has ENTERED reliably, every 1.6 MB SQLite-derived image has
ENTRY-STALLED. Two ladders separate the obvious axes, each holding the other fixed, each run
ascending with `r14sl` as the in-boot control.

    AXIS 1 -- carve count (image ~40-44 KB throughout)
      rc32   carves=32   image=40336    -> 32   OK
      rc64   carves=64   image=42224    -> 64   OK
      rc96   carves=96   image=44112    -> 96   OK

    AXIS 2 -- image size (carves = 1 throughout)
      rz64k  carves=1    image=104264   -> 7    OK
      rz256k carves=1    image=300872   -> 7    OK
      rz1m   carves=1    image=1087304  -> 7    OK

**Both axes pass end to end.** Carve count up to 96 and image size up to **1087 KB** -- two
thirds of the 1633 KB SQLite image -- neither triggers an entry stall. So R-16 is NOT simply
"the image got big" and NOT simply "too many cap-table entries", which were the two natural
first guesses and are now excluded.

**Ceiling worth knowing:** the carve axis cannot go past ~127 on this glue --
`gen-gp-captable-glue.py` aborts with *"cap-table (2048 B) exceeds a 12-bit immediate; needs
li+sub (TODO)"*. SQLite reaches 181 only because it uses a DIFFERENT glue
(`start-gp-captable-interp.S`). That is now the most interesting remaining difference.

**What still differs between the passing `rz1m` and the stalling SQLite images** — the next
axes to test, in order of promise:

1. **The entry glue itself.** Ladder rungs use `start-gp-captable-generic.S`; SQLite uses
   `start-gp-captable-interp.S`. Entirely different entry code, and R-16 is an ENTRY stall.
   This is the strongest remaining candidate and has never been varied.
2. **Carves ABOVE 127**, which needs the interp glue or the `li+sub` TODO fixed.
3. **The combination** (large image AND many carves) -- each is individually harmless, and this
   investigation has already produced one genuine conjunction (R-14's).
4. `dom_data` geometry, e.g. `SQLITE_HEAP_SIZE=256 KB`, which the rungs do not have.

Cost so far: 2 boots, ~12 minutes, and it eliminated the two leading hypotheses.

## 2026-08-03 — R-16 SEPARATED FROM BOARD HEALTH FOR THE FIRST TIME (MEASURED)

Every previous R-16 reading was ambiguous: an image that did not enter looked exactly like a
board, firmware or boot failure, which is how a whole session was spent blaming the board. The
fix is a **per-boot health control** — run a KNOWN-ENTERING image first, in the same boot, from
the same firmware:

    SQLITE_STAGE_DOMS="/test-domains/f10.dom:0,/test-domains/<image>.dom:0,..."

MEASURED (2026-08-03, firmware md5 `8686cad424cb`, bitstream `working-caplifive-captype-fixed`):

    draw d140   f10ctl=0 (RETURNED)   d140:0 = STALL
    draw d141   f10ctl=0 (RETURNED)   d141:0 = STALL

So in the *same boot*, on the *same firmware*, a known-good image enters and the image under
test does not. **R-16 is a property of the image, not of the board or the firmware.** That was
previously assumed; it is now measured, and it makes every later stall verdict attributable.

### The whole 140-146 family stalls, and it is not a lottery

`min.dom` stalled 2/2 under a correctly-calibrated watchdog (`ENTRY_STALL_S=260`; the earlier
45 s default aborted runs mid-upload — see HOW-TO-LAUNCH-ON-FPGA.md). `d140` and `d141` then
stalled with the control returning. All are built by `build-sqlite-silicon.sh` and differ ONLY
in `CAPSTONE_SQLITE_STAGE`, which selects which `#if` block compiles; `f10.dom` is the same
builder with stage 10.

I proposed from this that the 140-146 block's **presence in the image** blocks entry — a
layout effect rather than an execution one, since `:0` returns before any ladder code runs
(`run_sqlite_staged()` opens `if (stage <= 0) return 0;`).

**RETRACTED the same session, by the narrowed images below.** `n144` CONTAINS the block and
its `:0` returned cleanly; `n140` contains a strictly smaller version of it and stalled. So
presence of the block does not determine entry, and **R-16 remains unexplained** — carve count,
`.text` size, merged-string bytes, dom_data geometry and now "carries the ladder block" all
fail to separate entering from stalling images. Recorded because the inference ran one step
past the evidence: two near-replicate draws (differing in one constant) were treated as
evidence for a general structural claim.

### `CAPSTONE_LADDER_ONLY` — the minimisation now in flight

`sqlite_capstone_domain.c` gained `CAPSTONE_LADDER_ONLY=<n>`, compiling exactly one ladder arm,
to ask which ingredient carries the property:

    n146   four plain scalars + arr[4]    no struct, no 64-entry array
    n140   one struct VARIABLE            struct, no array
    n144   struct kv5 a[64]               the 2 KB stack array

An arm excluded by the knob returns the sentinel **99**, deliberately not 0, so "not compiled
in" can never be misread as stage `:0`'s legitimate answer.

### MEASURED result — R-14 reproduced on silicon with BOTH controls passing

    n146   f10ctl=WEDGE                    VOID -- control failed, no verdict
    n146   f10ctl=0  | :0=STALL            re-run: R-16 on this image too; arm not measured
    n140   f10ctl=0  | :0=STALL            R-16 on this image; arm never measured
    n144   f10ctl=0  | :0=0 | :144=WEDGE   <-- the result
    n144   f10ctl=0  | :0=0 | :144=WEDGE   CONFIRMED 2/2, deterministic

`n144` is the first clean silicon measurement of the R-14 construct. All three verdicts come
from ONE boot, and the two controls both pass: the board/firmware are healthy (`f10ctl=0`) and
this image enters (`:0=0`, so no R-16). Only then does the construct wedge. Every earlier R-14
"wedge" was confounded with a possible entry stall; this one is not.

The construct that wedges (`CAPSTONE_LADDER_ONLY=144`, nothing else compiled from the block):

```c
struct kv5 { const char *z; const char *y; };
struct kv5 a[64];
a[0].z="ltrim"; a[0].y="aaa0";  a[1].z="rtrim"; a[1].y="aaa1";
a[2].z="trim";  a[2].y="aaa2";  a[3].z="max";   a[3].y="aaa3";
for (i = 0; i < 4; i++) {
  unsigned nz=0, ny=0; const char *z=a[i].z, *y=a[i].y;
  while (z && z[nz]) nz++;
  while (y && y[ny]) ny++;
  if (z && y && nz > 0 && ny > 0) ok++;
}
return ok;                        /* expect 4; the domain never returns */
```

Eight capability stores into a stack array of structs, then a read-back loop. This is the same
shape as `sqlite3RegisterBuiltinFunctions`'s local `FuncDef capstoneBuiltinFunc[]`, which is the
control-validated SQLite blocker — so the minimal repro and the real blocker now match.

**Open, in flight:** `n146` (four plain scalars, no struct, no array) is the discriminator —
if it RETURNS 4 the struct/array is implicated; if it WEDGES the construct is not the axis at
all and the fault lies in something all these images share. Both `n146` boots so far failed to
enter (one void control, one genuine stall), so the arm is still unmeasured. Four redraws
(`q141/q142/q143/q145`, arm 146 pinned via `CAPSTONE_LADDER_ONLY=146`, differing only in the
compiled default stage) are staged to find one that enters.

Note what this costs: R-16 does not merely add retries, it **silently biases which constructs
can be measured at all**. `n144` was measurable and `n140`/`n146` were not, for reasons
unrelated to what they test. Any claim of the form "arm X wedges and arm Y does not" must state
which arms were actually reachable.

### Redraw outcome — arm 146 is STILL UNMEASURED after six attempts

    n146   :0=STALL                      (x2, one boot void on the control)
    q141   f10ctl=0 | :0=STALL
    q142   f10ctl=0 | :0=STALL
    q143   f10ctl=0 | :0=STALL
    q145   f10ctl=0 | :0=0 | :146=STALL  <-- image ENTERED for :0, then hung on :146

`q145` is the informative one and it needs reading from the transcript, not from the runner's
label. Its `:146` block ends:

    SQ: A/dom-ok  B/mkregion1  C/mkregion2  D/mapped  E/share1  SHA5:00000002   [end]

no `SHA6`, no `F/share2`, no `G/enter`. So it hung **inside the FIRST capability-share call**,
earlier than domain entry — the arm's code never executed and **arm 146 has no verdict**. The
classifier prints `146=STALL`, which is right in kind but reads like a statement about the arm;
it is not.

Two consequences worth keeping:

* **R-16 is not strictly per-image.** The SAME `q145` binary, in the SAME boot, entered for
  `:0` and then hung at `:146`. The per-image-determinism model (used above to justify
  redrawing rather than re-running) is therefore incomplete: it holds across boots for a given
  image, but a later invocation within one boot can still hang. Redrawing remains the right
  strategy — six images, one entry — but "deterministic" overstates it.
* **`SHA5` last does not mean "entry stall" by itself.** A domain that enters and wedges
  immediately would also leave `SHA5` last. Distinguish on `SQ: G/enter`: present => the domain
  ran (a real wedge, as in `n144:144`); absent => it never entered (as here).

**Where this leaves the minimisation.** The array arm (`n144`) is the only one measured, and it
wedges 2/2. The two arms that would isolate the ingredient — `n140` (struct, no array) and
`n146` (scalars, no struct) — have never entered, across 8 images and 9 boots. Isolating the
ingredient therefore needs a way around R-16, not more redraws of the same shape.

### What the wedge ACTUALLY is (MEASURED, in-session debug mux, reproducible 2/2)

The runner reads the debug mux while the core is still wedged, with the trap latch cleared
before each domain, so this is attributable to `n144:144`. Identical in both wedge runs:

    sw=255  TRAP LOG {seen,mcause[6:0]}   0x9c   seen=1, mcause=28
    sw=224  {excommit,...,flush,privM}    0x9f   privM=1   <- core is in M-MODE
    sw=225  {tbe,wstore,wload,wrev,...}   0x80   wrev=0, memwait=0
    sw=249/250  rev_node_head             602    overflow=0
    sw=251-254  rev_node_serving_idx      0

`ex_code` (`capstone_unit.anvilh:289`) numbers the capability faults 24..29 —
UNEXPECTED_OPERAND 24, INVALID_CAPABILITY 25, UNEXPECTED_CAP_TYPE 26,
INSUFFICIENT_PERMISSION 27, **OUT_OF_BOUNDS 28**, ILLEGAL_OPERAND_VALUE 29. So:

**The domain takes an OUT_OF_BOUNDS capability fault, traps to M-mode, and the M-mode side
wedges.** It is not a silent hang in the store path — a real exception is raised and taken.

REFUTED by this reading, all three previously plausible:

* **R-12 / revocation-node hang** — the leading prior theory. `wrev=0`, `serving_idx=0`,
  head=602 of 1023 with `overflow=0`. Not blocked on a node query, pool not exhausted, rev
  unit not walking. The `stc` path's unbounded `get_node_query_validity`
  (`capstone_dyn_unit.anvil:399`) is real but is NOT what fires here.
* **`stc` consumes its source capability** (the spill-then-reuse story: the compiler spills the
  merged-string blob cap `a1` with `stc a1,-0x440(a2)` and keeps deriving from `a1`).
  `capstone_dyn_unit.anvil:428` returns `rs2_v` unchanged on the normal path; only an UNINIT
  destination writes `cnull` back (`:408-416`).
* **Plain stack exhaustion.** `run_sqlite_staged` allocates a ~26 KB frame
  (`cincoffsetimm sp,sp,-0x7f0` then `cincoffset sp,sp,a1` with a1=-0x5EA0), while
  `domdata-budget.py` reports **211824 bytes of stack** in `dom_data`. Fits with 8x margin.

**Open — which capability is out of bounds.** Not yet identified. The two candidates worth
separating, both visible in the arm's disassembly around `0x3563c`:

1. the **destination** `a2` (`cincoffset a2,s0,<-0x4000>` then `cincoffsetimm a2,a2,0x5a0`),
   i.e. the stack-array pointer, versus
2. the **stored values**, each derived from the merged-string blob base `a1` by
   `cincoffsetimm a4,a1,0x6da` / `cincoffset a4,a1,<reg>` — if `a1` carries per-string rather
   than per-blob bounds, offsetting it to a *different* literal is out of bounds by
   construction, which would make `-capstone-merge-string-constants=true` the trigger.

Cheapest discriminator: read **mepc/mtval** at the wedge (names the faulting instruction
outright), or build arms writing a[0] only vs a[3] only — if the fault tracks the offset it is
(1), if it tracks which literal is stored it is (2).

### 2026-08-03 later — THE FAULT IS NONDETERMINISTIC; three "axes" refuted

All measured on images whose `:0` control returned in the same boot, with `f10:0` passing too.

    n144  :141  = 1  (CORRECT)   x3 boots      1 entry, 2 stores
    n144  :142  = WEDGE                        2 entries, 4 stores
    n144  :143  = WEDGE                        4 entries, 8 stores, ONE literal ("dup")
    n144  :145  = WEDGE                        4 entries, 8 stores, 5 literals
    co    :141  = WEDGE          attempt 1     SAME SOURCE ARM as n144:141
    co    :141  = 0 (exp 1)      attempt 3     SAME SOURCE ARM, third distinct outcome
    co    :147  = 1  (CORRECT)                 2 stores at HIGH offsets 0x60,0x70
    co    :148  = 2  (CORRECT)                 3 stores at low offsets
    rt    :149  = 80,80,80                     8/8 cursors round-trip, 0 nulls

**Refuted, in order:**

* **"Distinct string literals are the trigger."** `:143` stores the SAME literal eight times —
  one derived capability, no distinctness at all — and wedges. This kills the merged-blob
  derivation story that the disassembly suggested (all values derived from one blob base by
  `cincoffsetimm`).
* **"Offset/bounds of the destination slot."** `:147` puts both stores at the HIGH offsets
  (0x60/0x70) and returns correctly.
* **"Store count, boundary between 2 and 4."** `:148` does 3 stores and returns correctly; and
  `:141` (2 stores) both WEDGED and returned 0 in a later image. **The boundary was an artefact
  of one image** — recorded here because it was briefly stated as a result.

**The finding that replaces them: the same source arm produces three different outcomes.**
`:141` returned 1 (n144, 3 boots), wedged (co attempt 1), and returned 0 (co attempt 3). Frame
size is byte-identical between those images (~26 KB prologue in both, verified by
disassembly), so it is not frame geometry.

**Current reading (INFERRED, not established).** A capability stored into the stack array is
not reliably usable when read back: sometimes correct, sometimes null (arm returns 0), and
sometimes right-address/wrong-bounds, whose dereference is exactly the measured
`mcause=28 OUT_OF_BOUNDS`. Arm `:149` does NOT refute this: `q == p` compares only the 64-bit
cursor, so a capability can lose tag or bounds and still compare equal — 80 proves the ADDRESS
survived, not the capability.

**Next probe, STILL UNMEASURED after 5 images / 9 boots:** arm `:150` dereferences the literal
repeatedly WITHOUT any stack round-trip. If `:150` returns 8 while `:141` faults or returns 0,
the damage is in the round-trip rather than in the literal's capability. Every image carrying
it has failed before reaching it:

    ct    :0=STALL x3 (+1 void control)
    c141  :0=STALL     c143  :0=STALL     c145  :0=STALL
    c142  :0=0, then :150=STALL   x2      <- enters, but stalls ON the 150 invocation

Do NOT read "images carrying arm 150 are cursed" into this yet — that is the same shape as the
"block presence blocks entry" claim already retracted above, on a sample of five near-replicate
draws. What is fair to say: `:150` is the one probe that would separate round-trip damage from
literal damage, and R-16 has blocked it on every attempt so far.

Worth noting for whoever picks this up: `c142` failing ON the `:150` invocation while its `:0`
returns twice means the stall is not purely a property of the image — it tracks which
invocation runs, consistent with the `q145` observation above that the same binary entered for
`:0` and hung at `:146` in one boot.

### String merging is NOT the trigger — settled, and the earlier "confounded" note was wrong

I first built `-capstone-merge-string-constants=false` through `build-sqlite-silicon.sh`,
measured **1026 carves** against the ~1020 rev-node pool, and recorded the test as confounded.
**That was a statement about SQLITE, not about the repro** — the build compiles the whole
amalgamation, so the 1026 carves are SQLite's globals. The minimal repro contributes ~10.

Settled without spending a boot:

* `CapstoneMergeStrConstants.cpp:80` declares the flag `cl::init(false)` — **merging is OFF by
  default**, and `build-ladder-domain.sh` never sets it (`build-sqlite-silicon.sh` enables it
  explicitly, with a comment saying it is opt-in precisely so ladder rungs keep their geometry).
* Therefore the standalone `r14b_app.c`, **board-measured returning 4 where 16 is correct**,
  was built with merging **OFF**. The fault reproduces without merged string constants.
* Independently, `:143` stores the SAME literal eight times — one derived capability — and
  still wedges.

Two independent directions, same answer: **merged string constants are not necessary for the
fault.** The disassembly's "everything derives from one blob base by `cincoffsetimm`" is a true
description of the merged build and a false explanation of the bug.

Both arms are built and differ properly, for whoever runs the direct comparison:

    r14b.dom        merging OFF   10 globals, table 160 B, ldc-gp=10, 10896 B
    r14b_merge.dom  merging ON     1 global,  table  16 B, ldc-gp=2,  10272 B

### USE THE STANDALONE REPRO — it is 150x smaller and should dodge R-16

`r14b.dom` is **10896 bytes with 10 carves**. The SQLite-derived ladder images are **1624128
bytes with 181 carves**. Every measurement blocked tonight was blocked by R-16 on the big
images; the small one has a completely different profile, and being ~11 KB many variants fit in
one firmware, so a batch costs one boot instead of one boot per draw.

It is also the better instrument on its own terms: `r14b` **returns 4 where 16 is correct**
rather than wedging, so every run yields a number (the project's "make every run RETURN" rule).
And `r14b_app.c` already records the shape of the failure:

    the four STRAIGHT-LINE entries pass, the twelve LOOP-ASSIGNED ones fail

which points at capability stores through a **computed (loop-variable) address** rather than at
immediate-offset stores — a different axis from everything tested tonight, and not yet examined.

Runner: `run_ladder_perf_fpga.py`, rungs overridable via `LADDER_RUNGS`, `LADDER_ONE_BOOT=1`
to run them all in one boot; `DOMAIN_EXTRA_CFLAGS` toggles the merge flag per build. **This
runner transfers each `.dom` over UART (gzip+base64, per-chunk sha), so it needs NO firmware
rebuild and no initramfs staging** — the entire rebuild/restage/reflash loop that dominated
2026-08-03 disappears. Cost is UART time: 16 chars per emit, so an ~11 KB domain is minutes.

### The STRAIGHT-LINE vs LOOP discriminator (built 2026-08-03, running)

The axis none of the `:14x` arms tested. `r14b_app.c` records that its four straight-line
entries pass and its twelve loop-assigned ones fail, i.e. the suspect is a capability store
through a **computed (loop-variable) address**, not an immediate-offset store.

Two rungs isolate exactly that, holding everything else constant — the SAME two literals in
every field, the SAME number of capability stores, the SAME array slots, the SAME read-back
loop. Confirmed matched at build time: **both are 2 globals, table 32 B, `ldc-gp=2`**.

    r14sl   a[0..3].z/.y assigned STRAIGHT-LINE (immediate offsets)   oracle 4
    r14lp   the same four entries assigned IN A LOOP (computed addr)  oracle 4
    r14b    the known-failing reference, same batch                   oracle 16

Reading: `r14sl` passing while `r14lp` fails names the addressing form as the trigger, and
would be the first mechanism-level answer for R-14. Both failing points away from the store
form entirely. Both passing means 4 loop-assigned entries are not enough and the count matters
after all — in which case raise `r14lp` to 16 entries, since `r14b` fails on its 12
loop-assigned ones.

Files: `silicon-ladder/r14{sl,lp,b}_kernel.h`, `_fpga_app.c`, `_host.c`.

#### MEASURED 2026-08-03 — the loop arm fails, the straight-line arm passes

    rung     retval  oracle  cycles  instret  correct
    r14sl    4       4       4752    1092     YES     straight-line, immediate offsets
    r14lp    None    4       None    None     NO      SAME 4 entries, assigned in a LOOP
    r14b     None    16      None    None     NO      <- NO INFORMATION, see below

`r14sl` ran FIRST in the boot and passed, so it is also that boot's health control: the board,
firmware and transfer path were all good when `r14lp` failed. The two differ only in how the
store address is formed.

**CONFIRMED 2/2, with a pass-fail-pass bracket.** Second run, `LADDER_RUNGS="r14sl r14lp r14sl"`:

    r14sl    4       4       4771   1092   YES
    r14lp    None    4       None   None   NO
    r14sl    4       4       4771   1092   YES    <- ran AFTER the failure, still correct

The trailing `r14sl` passing after `r14lp` failed, in the SAME boot, is the strongest control
available here: the board stayed healthy across the failure, and `r14sl` is bit-for-bit
deterministic (4771 cycles / 1092 instret in both runs).

It also corrects a caveat written after run 1: `r14lp` does NOT take the core with it, so
`r14b`'s failure in run 1 was informative rather than collateral. (`r14b`'s historical
behaviour was RETURNS 4 where 16 is correct, under the older non-perf harness; under this
harness it produces no END marker. Different domain_main, so do not treat the two as the same
measurement.)

This is the first result that names a MECHANISM rather than excluding one: a capability store
to a stack struct array through a **computed (loop-variable) address** fails where the
identical store through an **immediate offset** succeeds. It is consistent with everything
else measured: `:147`/`:148` (straight-line) passed; `:142`/`:143`/`:145` all contain
loop-driven read-back over entries; and `r14b`'s own note that its straight-line entries pass
while its loop-assigned ones fail.

#### 2026-08-03 — SQLite + workaround: BLOCKED BY R-16, not by R-14

The workaround validated on the minimal repro (`w1stat`) was carried to SQLite:
`SQLITE_STATIC_BUILTINS=1`, stage-10 domain, **181 carves** (inside the ~1020 rev-node budget),
1633312 B. Four distinct draws built and tested, each in its own boot with `f10:0` first:

    swa    :0  ENTRY STALL (R-16)      no SQ: G/enter -- the domain never ran
    swa8   :0  ENTRY STALL (R-16)      no SQ: G/enter
    swa9   :0  ENTRY STALL (R-16)      no SQ: G/enter
    swa11      VOID -- the f10 CONTROL itself wedged, so that boot carries no verdict

**The workaround was never exercised.** `:0` returns before any SQLite code runs, so an entry
stall says nothing about whether the static-builtins change fixes stage 10. This is the same
wall that stopped `st10`/`sb10` earlier in the investigation, and it now stands at 5 static-
builtins images that never entered.

Deliberately NOT concluding "static-builtins images are cursed" — that is the same shape as the
"block presence blocks entry" claim already retracted above, and the sample is again a set of
near-replicate draws differing in one constant. What IS supportable:

* **R-16, not R-14, is now the binding constraint on getting SQLite onto silicon.** R-14 has a
  validated workaround at minimal-repro scale; R-16 has no workaround, no mechanism, and blocks
  measurement of everything at SQLite scale.
* The `f10` control wedging in the `swa11` boot is a fresh reminder that a single control pass
  is weak evidence — the control itself fails roughly 1 in 5.

**Strategic consequence.** Further R-14 refinement is cheap (11 KB rungs, ~5 min/boot) but no
longer on the critical path for SQLite. The critical path is R-16: it is per-image, unexplained,
survives every structural attribute checked, and it is what makes large images unmeasurable.

#### 2026-08-03 — WORKAROUND: move the big local OUT of the frame (outlining does NOT work)

Two candidates, each the only unknown in its own boot, after a known-good control:

    w1stat   4      OK      the big local array becomes a file-scope STATIC
    w3out    0      WRONG   stores outlined into a noinline helper -> SILENT MISCOMPUTE
    k1200    None   FAIL    reference: this firmware still reproduces the bug

**`w1stat` WORKS.** Moving the large local out of the frame removes the failure even with the
1200 B pad still present. This is exactly what `SQLITE_STATIC_BUILTINS=1` already does for
`sqlite3RegisterBuiltinFunctions`, so the SQLite workaround is on the shelf and now has a
minimal-repro validation behind it.

**`w3out` IS A TRAP — do not use outlining.** Keeping the array local but moving the stores into
a `noinline` helper (whose own frame is tiny and whose base is a plain argument, verified: 0 lui
sites) returns **0 where 4 is correct**. It converts a hang into a SILENT WRONG ANSWER, which is
strictly worse: a wedge is at least loud. Recorded because it is the natural thing to try next
and it looks correct at the source level.

Note this also further constrains the fault: `w3out`'s helper stores through an argument-derived
base in a small frame and is still wrong, so "the storing function must have a big frame" is not
the whole story either — the CALLER's frame is big, and the array being addressed lives in it.

#### 2026-08-03 — VERDICT: R-14 is an RTL DEFECT, not a codegen bug

Three independent lines now agree, and the compiler-side explanations have all failed.

**1. The capability at the failing address is WELL FORMED.** `bnd2` dumps every `lcc` field at
`&a[3].y` -- the address the failing store targets -- and encodes a verdict
(`+1` cursor>=start, `+2` cursor+16<=end, `+4` start 16-aligned, `+100*type`).

    bnd2 = 107   ->  type = 1 (cap_type 2 = NONLIN, valid for stores)
                     v = 7 = 1+2+4  ->  ALL THREE CHECKS PASS

So: cursor inside the bounds, the 16-byte store fits before `end`, base 16-aligned, correct
capability type. `bnds` separately measured **1312 bytes of headroom** against a 16-byte store.
There is no architectural reason to fault this access. Neither probe performs a capability
store, so neither can wedge, and both always return.

**2. The same binary is CORRECT under QEMU.** `run-ladder-qemu.sh k1200` ->
`__CAPSTONE_LADDER_K1200_PASSED__ (retval = 4)`. Board-free, and the reference implementation
of the ISA computes the right answer from the identical source.

**3. Every compiler-side mechanism proposed this session has been REFUTED on the board:**
merged string constants (`:143`, and `r14b` fails with merging off), repeated `ldc` from one
cap-table slot (`clp16` does 16 and passes), count of `ldc`-from-gp (`cdif8`), capability stores
as such (`cst8`), `ldc`+store in one loop (`cgs8`), frame size alone (`r14sl`/`r14hl`/`cgpad`),
loops (`e3rd`/`e4wr`), and finally the non-zero `stc` immediate -- `zoff` forces every store to
`imm=0`, verified in the disassembly, and **still fails**.

**Conclusion: the hardware faults an architecturally legal capability store.** The compiler is
not handing it a malformed or undersized capability. This is the finding to take to the board
owner.

**PACKAGED AS `capstone/tests/fpga-repros/R14-frame-pad/`** -- frozen `.dom` images + the `lpc`
controller (~41 KB total, pinned by `images/SHA256SUMS`), sources, run instructions, the
refuted-hypotheses table, and an explicit "what is NOT established" section. That directory is
the thing to hand over; it supersedes `R14-strline-struct/`.

**The hand-off artifact is built and self-contained** -- no SQLite, no compiler needed to
reproduce, ~11 KB each, baked into the image:

    k800   PASSES   struct a[32] + 800 B dead volatile pad
    k1200  FAILS    IDENTICAL SOURCE, 1200 B dead pad
    bnds   1322     headroom end-cursor = 1312 B at the failing address
    bnd2   107      type NONLIN, cursor in bounds, 16 B fits, start aligned
    (QEMU: k1200 returns the correct 4)

**Gaps that remain, and they are small:**
* **Permissions were not read** (`lcc` field 5). A missing write permission would be an
  architectural reason to fault, and would move this back toward the compiler/glue. Cheapest
  possible next probe -- one extra `lcc_f(perm, p, 5)` in `bnd2`.
* `bnds`/`bnd2` measure a capability the compiler materialised for a `volatile` access, not
  provably the same register the faulting `stc` uses. Reading `lcc` of the base immediately
  before the faulting store would close this.
* No `mcause`/`mepc` for `k1200`/`zoff` themselves; every `mcause=28 OUT_OF_BOUNDS` still comes
  from the SQLite-derived `:144`. The baked driver needs the debug-mux read that
  `run_sqlite_stages_fpga.py` already performs on wedge.

#### 2026-08-03 — COMPILER vs RTL: the evidence points at the RTL

Two probes aimed at the question that decides who fixes this.

    rung    retval        meaning
    r14sl   4    OK       boot control
    bnds    1322          headroom (end - cursor) = 1312 B at the failing address
    zoff    NO RESULT     the predicted COMPILER FIX -- all stores forced to imm=0 -- STILL FAILS
    k1200   NO RESULT     reference: this firmware still reproduces the failure

**1. The failing access is IN BOUNDS, so the compiler is not handing the hardware an
undersized capability.** `bnds` reads the capability at `&a[3].y` -- the address the failing
store targets -- with `lcc` (field map `capstone_dyn_unit.anvil:182-188`: 2=cursor, 4=end) and
returns `end-cursor+10`. Measured **1312 bytes of headroom**, against a 16-byte capability
store. `bnds` performs no capability store itself, so it cannot wedge and always returns.

If the bounds legitimately cover the access, an `stc` there is a LEGAL operation and faulting
it is an RTL defect. **This is the first direct evidence that R-14 is a hardware bug rather
than a codegen bug.**

**2. The predicted compiler fix does NOT work — mechanism retracted again.** `zoff` is the
`k1200` shape with every capability store forced through an explicitly materialised address, so
the emitted stores are `stc 0x0(a0..a3)` with NO non-zero immediate anywhere (verified in the
disassembly before running). It still fails. So *"an `stc` with a non-zero immediate off a
lui-derived base"* -- the characterisation recorded one section below -- is **REFUTED**: the
immediate is not the mechanism, and no codegen constraint on immediates would fix this.

**Caveats, stated because this is the third mechanism to fall in one session:**
* `bnds` measured a pointer the compiler materialised for a `volatile` access; that is
  *a* capability to the failing address, not provably the *same* register the failing `stc`
  uses. Confirm by reading `lcc` fields of the base immediately before the faulting store.
* `zoff` uses `volatile` pointers, which changes more than the immediate; the honest reading is
  "forcing imm=0 does not rescue it", not "imm=0 is irrelevant in isolation".
* No `mcause`/`mepc` has been read for `k1200`/`zoff` themselves; every `mcause=28
  OUT_OF_BOUNDS` reading still comes from the SQLite-derived `:144`.

**What this makes the next step.** The compiler-side story has now failed three times
(merged-string blob, repeated `ldc`, non-zero immediate), while the in-bounds measurement
points at the RTL. Priority is to read `mcause`/`mepc` on one of THESE rungs -- the baked driver
needs the debug-mux read that `run_sqlite_stages_fpga.py` already does on wedge -- and then hand
the board owner a self-contained repro: `k800` (passes) vs `k1200` (fails), identical source
apart from a dead pad, plus the `bnds` headroom measurement.

#### 2026-08-03 FINAL — `stc` with a NON-ZERO immediate off a lui-derived base

Sharpest characterisation so far, and the first stated as an INSTRUCTION PATTERN.

    rung    verdict         what it is
    k800    PASS            struct a[32] + 800 B pad -> frame 3776, lui-addressing sites = 0
    k1200   FAIL            IDENTICAL source, 1200 B pad -> frame 4576, lui sites = 13
    h2adj   PASS  (524)     flat p[16], two stores/iteration, big frame, BOTH at offset 0x0
    h2far   WRONG (272)     flat p[16], two stores/iteration to p[i] and p[i+8]

`k800`/`k1200` differ ONLY in the size of a dead `volatile char pad[]`. The predicate that
flips with them is compiler-observable and is NOT raw frame size: it is whether the frame needs
`lui`+register-form `cincoffset` to be addressed. (`f2nop` at frame 2176 has zero lui sites and
passes; the switch happens between pad 800 -> 3776 B / 0 sites and pad 1200 -> 4576 B / 13
sites.)

The store forms settle what "the conjunction" actually was:

    k1200 (FAILS):   stc 0x0(a3)   and   stc 0x10(a1)     <- NON-ZERO immediate
    h2adj (PASSES):  stc 0x0(a1)   and   stc 0x0(a3)      <- both ZERO immediate

So it is not "two capabilities per iteration", not the struct, and not the loop. **It is a
capability store whose immediate offset is NON-ZERO and whose base register was produced by a
register-form `cincoffset` (the lui frame-addressing idiom).** With the field offset folded
into the `stc` immediate the access is wrong; with the address fully materialised into the base
(`imm=0`) the identical work succeeds.

`h2far` is the same fault in a milder form: it RETURNED 272 where 524 is correct, so the
failure mode spans wrong-value and hang, consistent with the earlier
`mcause=28 OUT_OF_BOUNDS` and with the arms that returned 0. A derived base whose bounds are
wrong faults at +0x10 while +0x0 is still inside.

**Retracted from the section below:** "the trigger is a big frame AND a loop storing TWO
capabilities per iteration to a computed struct element". `h2adj` does exactly that (big frame,
two capability stores per iteration, computed element) and PASSES -- because both its stores
use immediate 0. The conjunction was a description of `r14lp`'s codegen, not the mechanism.

**Predicted compiler fix, not yet tested:** when the base of a capability store is
register-derived, do not fold the field offset into the `stc` immediate -- materialise
`base+off` and emit `stc rs, 0(base)`. Cheapest confirmation: a probe that writes `a[i].y`
through an explicitly computed `&a[i].y`, in the `k1200` shape. If that passes, the fix is a
codegen constraint and R-14 is closed on the compiler side.

**Also still open:** read `mcause`/`mepc` for one of THESE rungs. Every `mcause=28` reading so
far came from the SQLite-derived `:144`, not from `k1200`/`r14lp`; the baked driver does not
yet do the debug-mux read that `run_sqlite_stages_fpga.py` does on wedge.

#### 2026-08-03 LATER — BAKED-IN RUNGS, and the characterisation that survives

**Method change that made this possible.** The rungs are now BAKED INTO THE FIRMWARE
(`overlay/test-domains/` + `lpc` controller) and driven from the shell by
`fpga_driver/run_baked_rungs_fpga.py`, instead of shipped over UART by
`run_ladder_perf_fpga.py`. UART transfer was 16 chars per HTTPS round trip -- minutes per
domain. Baked: **10 rungs in ONE boot in ~5 minutes.** The domains are ~10 KB, so they cost
nothing on the JTAG upload that happens anyway.

**READ THIS BEFORE USING THE DRIVER: it does NOT reboot between rungs.** A wedged rung takes
the core, so every rung after the first failure is COLLATERAL, not a result. This bit us
immediately: in the bisection sweep `e3rd` failed at position 4, and `e4wr` (5) and `r14lp` (6)
were recorded as failures. Re-tested one-unknown-per-boot, **`e4wr` PASSES**. Rule: put at most
ONE unknown per boot, last, after a known-good control.

**Valid results only** (each read no further than its run's first failure):

    arm      caps stored per loop iteration   frame(prologue)  lui sites  verdict
    e1sml    2  (.z and .y), a[8]                    608           0       PASS
    e2one    1  (.z only),   a[64]                  1088           0       PASS
    f2nop    2, a[32]                               2176           0       PASS
    cgs8     1, p[8]                                small          0       PASS
    cgpad    1, p[8] + 2200 B padding               4960          16       PASS
    r14sl    stores UNROLLED, a[64]                 4256          16       PASS
    e4wr     stores UNROLLED, a[64]                 4256          16       PASS
    r14hl    2, but literal loaded from STACK       4288          23       PASS
    clp1..16 16 x ldc gp[0] in a loop, no stores    small          0       PASS
    cdif8    8 ldc from 8 DISTINCT slots            small          0       PASS
    cst8     8 capability stores, straight-line     small          0       PASS
    ---
    e3rd     2 per iteration, read unrolled         4224          27       FAIL
    f1pad    2 per iteration, a[32] + 1400 B pad    4960          16       FAIL
    r14lp    2 per iteration                        4224          19       FAIL

**What is REFUTED by this table** (each had been proposed, some by me as a "root cause"):

* *repeated `ldc` from one cap-table slot* — `clp16` does 16 and passes.
* *count of `ldc`-from-gp* — `cdif8` does 8 from 8 distinct slots and passes.
* *capability stores as such* — `cst8` does 8 and passes.
* *`ldc`-from-gp + capability store in one loop* — `cgs8` does exactly that and passes.
* *frame size alone* — `r14sl`, `e4wr`, `r14hl`, `cgpad` all have big lui-addressed frames and
  pass. Frame size is NECESSARY (every small-frame arm passes) but NOT sufficient.
* *loops* — `e3rd` unrolls the read loop and still fails; `e4wr` unrolls the store loop and
  passes. So it is the STORE side, not looping in general.
* *the array's size or the struct* — `f1pad` has only a[32] (1024 B, identical to the passing
  `f2nop`) and fails purely because dead padding enlarges the frame.

**What survives — the trigger is a CONJUNCTION**, and each half is individually harmless:

    a big (lui-addressed) frame  AND  a loop storing TWO capabilities per iteration
    to a computed struct element (offsets 0x0 and 0x10 off a recomputed base)

`cgpad` isolates one half (big frame, ONE cap per iteration -> passes); `e1sml` isolates the
other (two caps per iteration, small frame -> passes). Only together do they fail. In `r14lp`'s
disassembly each iteration recomputes the element base and stores twice:
`cincoffset a3,a0,a2; ldc a2,0x0(gp); stc a2,0x0(a3)` then
`cincoffset a1,a0,a1; ldc a0,0x10(gp); stc a0,0x10(a1)`.

Consistent with the SQLite blocker: `sqlite3RegisterBuiltinFunctions` builds a large local
`FuncDef` array (big frame) with multiple capability fields written per entry.

**Next probe (untested):** whether "two caps per iteration" specifically means two stores to a
RECOMPUTED base, or simply two stores per iteration. Flat `p[2*i] = g0; p[2*i+1] = g1;` with
padding for a big frame separates them, and needs no struct at all.

#### ROOT CAUSE ISOLATED 2026-08-03 — repeated `ldc` from the SAME cap-table slot

Third arm, `r14hl`: the loop of `r14lp` with the cap-table loads HOISTED out of it. Same loop,
same register-form `cincoffset` computed addresses, same eight capability stores.

    rung     retval  oracle  cycles  instret  correct
    r14sl    4       4       4759    1092     YES     straight-line, one ldc per literal
    r14hl    4       4       5251    1246     YES     SAME loop + computed addrs, ldc HOISTED
    r14lp    None    4       None    None     NO      SAME loop, ldc from gp INSIDE the loop
    r14sl    4       4       4759    1092     YES     closing bracket

Totals across the session: **`r14hl` 5/5 pass, `r14lp` 6/6 fail, `r14sl` 6/6 pass.**

**CORRECTIONS to how this was first written up (found by an adversarial audit, then verified
against the UART capture):**

* **There was never a "same boot" bracket, and no interleaving.** The driver forces a cold boot
  after any rung that produces no END marker (`run_ladder_perf_fpga.py:443-449` resets
  `booted_once`), so *nothing can ever run in the same boot after a failing rung*. Every
  `BGr14lp` in the capture is followed immediately by an OpenSBI banner. The "pass-fail-pass in
  one boot, strongest control available" statement was false; so was "alternating them inside a
  single boot removes ordering effects".
* **Cycles are NOT bit-identical**; only `instret` is. Measured spread: `r14sl` 4752/4754/4759/
  4768/4770/4771 (1092 instret), `r14hl` 5250/5251/5252/5266 (1246 instret). Quote instret for
  determinism and give cycles as a <=0.4% band.

#### The R-3 same-VA confound — TESTED AND EXCLUDED

All three rungs link at **`entry=0x10000`** (`readelf -h`; `DOMAIN_BASE_VA` was never set), and
the sweeps used `LADDER_ONE_BOOT=1`. The runner's own precondition
(`run_ladder_perf_fpga.py:357-363`) says one-boot is *"Only valid when the rungs are linked at
DISTINCT entry VAs … because R-3 hangs a second domain reused at the SAME VA within one boot"*,
and warns the failure mode is *"a silent hang that looks like a rung result"* — exactly
`r14lp`'s symptom. Reconstructed from banners, `r14lp` had never run first-in-boot.

Re-run with `LADDER_ONE_BOOT` unset, i.e. a power cycle per rung so EVERY rung is the first
domain of a clean boot:

    r14lp    None    4    NO      <- first domain of its own fresh boot
    r14hl    4       4    YES
    r14sl    4       4    YES

Capture confirms it: boot at line 8968 -> `BGr14lp` at 9956 with no domain in between.
**`r14lp` fails as the first domain of a clean boot, so R-3/position-in-boot is excluded** and
the arm, not the position, is the discriminator.

**Therefore:**

* the **computed (loop-variable) address is innocent** — `r14hl` uses it and passes, which
  retracts the mechanism named one step earlier ("a store through a computed address fails");
* **repeated `ldc` from a STACK slot is innocent** — at `-O0` `r14hl`'s loop reloads the hoisted
  literal from a stack slot every iteration (`ldc a0, 0x0(a0)`, verified in the disassembly)
  and still passes;
* **re-loading the SAME gp cap-table slot is the trigger** — the one thing only `r14lp` does
  (`ldc a2, 0x0(gp)` inside the loop, executed 4x instead of once).

**MECHANISM RETRACTED — the linear-clearing explanation is REFUTED.** I proposed
`capstone-ariane/CLAUDE.md`'s *"after an LDC that loads a linear capability, the source memory
location is cleared to prevent aliasing"*: first `ldc gp[i]` clears the slot, later ones read a
cleared entry. **The precondition does not hold.** `gen-gp-captable-glue.py:192-193,263` emits
`split(t2, sp, t1)` then **`delin(t2)`** before `stc(t2, gp, i*16)`, and the built binary
confirms it — `delin t2` immediately precedes every `stc t2, N(gp)`. (Count corrected: `r14lp`
has **8 `delin`s total** = 2 glue copies, `_start` and `__test_reentry`, x {gp, t2, t2, sp},
for **2** cap-table entries — not "8 delins plus delin gp".) **Cap-table entries are NONLIN, so
the clearing cannot fire on them.**

Two further corrections from the audit, both verified:
* The clearing is **not** LINEAR-only. `core/load_unit.sv:448` (also 545/661/714) fires for
  `{LINEAR, REVOKE, UNINIT, SEALED, SEALEDRET}` on the **loaded value's** type
  (`load_unit.sv:167`). NONLIN (`ariane_pkg.sv:648`) is excluded, so the retraction stands —
  but `capstone-ariane/CLAUDE.md`'s "clearing on LDC … loads a linear capability" is WRONG as
  written and two documents now reason from that sentence.
* Entries are NONLIN on **both** generator paths, not only via `delin`: the copy path
  (`_drop_redundant_delins`) emits no `delin` at all, but `SPLIT` preserves `cap_type`
  (`capstone_unit.anvilh:412-435`) and `sp` is pre-delin'd there, so everything split from it is
  already NONLIN. Independent confirmation that `t2` really was LINEAR at `delin` time: RTL
  `DELIN` (`capstone_dyn_unit.anvil:447`) traps on non-LINEAR, and these domains run.
Also note `func LDC` (`capstone_dyn_unit.anvil:293-352`) contains no clearing logic at all; it
delegates to `cap_load_ri`, so the condition lives in the LSU/cache path and was never read.

This is the third time this session an explanation was built on a documented behaviour whose
precondition was not checked against THIS build (the others: the merged-blob derivation story,
and the computed-address story). The empirical result below stands on its own; the WHY is
open again.

The observed consequence remains a right-address/wrong-bounds or null capability — the measured
`mcause=28 OUT_OF_BOUNDS` and the arms returning 0 — but nothing currently explains how a NONLIN
cap-table entry becomes that after a repeated `ldc`.

**It explains every prior observation, including the ones that defeated the earlier axes:**

* `r14b_app.c`'s own note — four STRAIGHT-LINE entries pass (one `ldc` per literal), twelve
  LOOP-ASSIGNED ones fail (the same `"filler"`/`"fill"` slot reloaded each iteration).
* **The nondeterminism — this explanation is REFUTED.** I wrote that differing register
  allocation/spilling changed whether a reload was emitted, so it "was never nondeterministic
  hardware, it was different codegen". The table 190 lines above kills it: `co :141 = WEDGE`
  (attempt 1) and `co :141 = 0` (attempt 3) are **the same binary, two attempts, two different
  outcomes** — codegen cannot differ between attempts on one image. The nondeterminism is real
  and remains unexplained. (Still worth doing: diff the `:141` arm between `n144` and `co` and
  count static `ldc <slot>(gp)`; equal counts would refute the codegen story for that pair too.)
* The `-O0`-only `strlen` freeze already recorded in `build-sqlite-silicon.sh:245` — at `-O1`
  the pointer stays in a register and the reload disappears.
* Why merging changes behaviour without being the cause: merging puts every literal in ONE
  slot, so N distinct literals become N loads of the SAME slot.

**What `r14hl` actually is — the header comment on it is WRONG.** At `-O0` the "hoisted" locals
are spilled, so `r14hl`'s loop executes the SAME number of dynamic `ldc`s as `r14lp` (2 per
iteration, 8 total). The difference is not "hoisted vs not"; it is **which memory is re-read**:
`ldc a0, 0x0(a0)` from a stack slot (hl) versus `ldc a2, 0x0(gp)` from the cap-table (lp).

**Alternatives the pair still cannot separate** (all consistent with dynamic `ldc`-from-`gp` =
2/2/8 for sl/hl/lp):
1. *repetition* — more than one `ldc` from the SAME cap-table slot;
2. *count alone* — more than two `ldc`s from `gp` anywhere. Discriminate with an UNROLLED
   straight-line arm doing 8x `ldc gp[0]`;
3. *the base register/region* — `gp` (a 32-byte cap-table cap `split` from the top of `sp`) as
   an LDC base versus a stack-derived cap. Discriminate with `movc t, gp` once outside the loop
   then `ldc x, 0(t)` inside: same memory, same repetition, different base capability.

No compiler-side fix should be proposed until one of these is chosen — the earlier
"never emit more than one `ldc` per cap-table slot" recommendation assumed (1) and was written
while the refuted linear-clearing mechanism was still standing.

Caveats kept deliberately, given how many claims were retracted this session: `r14lp` "failed"
means no END marker within 120 s, i.e. a hang, and the mcause has not yet been read for THIS
rung; and the two binaries, while matched on globals and cap-table size, have not been diffed
instruction-by-instruction to confirm the loop form is the only codegen difference. Both are
cheap to close and are the obvious next steps.

**A control that passes is not a control that always passes.** `f10.dom:0` returned 2/2 under
firmware `8686cad424cb`, then WEDGED after `SQ: G/enter` under `8c6f5d30905e`, then returned
2/2 again in later boots of that same firmware. So the control is ~non-deterministic at roughly
1-in-5, and a single control pass is weaker evidence than it looks — but a control FAILURE is
still decisive, and voiding that boot is what kept the `n146` slot honest instead of recording
a stall that had not been established.

## !!!! READ FIRST — A ~90-MINUTE WINDOW OF THIS DOCUMENT IS INVALID (instrumentation bug)

**Between 21:00 and 22:33 on 2026-08-02 every "entry stall" recorded here is an artefact of our
own watchdog, not a board behaviour.**

`board-watchdog.sh` grepped the WHOLE UART log for the last `SHA5`/`SHA6` marker. The console
replays ~548 KB of the PREVIOUS boot's scrollback when the driver connects, so the watchdog
matched a `SHA5` from an earlier boot and killed the runner seconds after `load_image` —
**before the board had booted at all**. Verified on every run in the window:

    run log                lines   SHA after its own load_image   SHA in scrollback
    board-waa.log            170            0                          50
    board-ts{p,q,r}.log    54-60            0                          50
    board-kg{1,2}.log      56-62            0                          50
    sllog-222003-{1,2}.log 57-63            0                          50
    rflog-222930-{1,2}.log 55-59            0                          50
    pzlog-222631-1.log        59            0                          50

13 of 13 checked: zero markers after their own `load_image`. The runs never executed.

### Sections invalidated by this — do NOT build on them

* **"THE BOARD STOPPED ACCEPTING ANY IMAGE AT ~20:40"** — REFUTED. The board was never given a
  chance. It ran a full three-domain ladder at 22:50 (`:0` and `:9` returned, `:10` wedged).
* **"R-16 REMAINS UNEXPLAINED — initramfs bloat REFUTED"** — the disproof rested on
  `sllog-222003-*`, both false aborts. Initramfs bloat is **untested**, not refuted.
* **"RETRACTED: R-16 is build-dependent — THE KNOWN-GOOD IMAGE NOW STALLS TOO"** — rested on
  `board-kg{1,2}`, both false aborts. **That retraction is itself withdrawn**; R-16
  build-dependence returns to *unknown*, neither established nor refuted.
* **"THE WORKAROUND RELIABLY TRIGGERS R-16 — 5/5"** — `waa`/`wab`/`wac` were false aborts AND
  byte-identical to each other (see below). At most `st10`/`sb10` remain, i.e. 2 samples.
* **"carve count does NOT predict … 182 stalls 8/8"** — `tsp`/`tsq`/`tsr` were false aborts and
  byte-identical; the 8/8 count is not supportable.

### A second, independent defect: "perturbed draws" were the SAME binary

`CAPSTONE_TS_PAD` / `CAPSTONE_WA_PAD` are referenced nowhere in `capstone/benchmarks/`, so the
`-D` flags were dead and the "independent draws" were duplicates:

    9b0e5331d62392ed  tsp.dom  tsq.dom  tsr.dom
    8ff50a38d6a8b977  waa.dom  wab.dom  wac.dom

So "five independently built binaries" was **two**, and "eight out of eight" was **four**.

### What is NOT affected

* Anything recorded **before 21:00**.
* **Every control-validated result**, because each compares a control and a failure *inside one
  boot*: `f10:0`+`f10:9` return / `f10:10` wedges; `r110:0` returns / `r110:110` wedges;
  `r110:0` returns / `r110:111` wedges.
* **C-16** (the `memset` AS0 tag-strip) — found, fixed, ladder 6/6, board-free reproducer.
* The **SQLite blocker**, re-confirmed 22:50 on a freshly reflashed board with the fixed
  watchdog: `sqlite3RegisterBuiltinFunctions` wedges while `sqlite3MallocInit` +
  `sqlite3PcacheInitialize` return in the same boot.

The watchdog now scans only bytes written after it starts, requires `load_image` in the current
run before any stall verdict, defaults to 180 s, and checks liveness before signalling.

---

## R-16 REMAINS UNEXPLAINED — initramfs bloat REFUTED, and everything else too  `INVALID — evidence was false aborts, see READ FIRST`

The firmware grew 15.4 -> 30.0 MB across the session as 24 dead probe images accumulated in
buildroot's **target** dir (pruning the overlay alone does nothing — the target dir is what is
packed). That looked like a strong self-inflicted explanation, and the entry failures did begin
around 26-28 MB.

Pruning the target dir by explicit name brought the firmware back to **17466376 bytes —
byte-identical to the known-good Aug-1 image**. It still entry-stalls, 2/2.

**So initramfs size is not the cause.** Full list of hypotheses now eliminated by measurement:

    domain image identity     r110 entered 3/3 at 19:05, stalls now -- same binary
    firmware generation       fresh rebuild, no domain changes -- stalls
    thermal / power state     300 s powered off, then retest -- stalls
    initramfs / firmware size back to the known-good byte count -- stalls
    bitstream                 nvbit matches working-caplifive-captype-fixed.bit throughout
    carve count               181 images both enter and stall; 182 correlation was confounded
    dom_data geometry         byte-identical across the entering/stalling divide

Note the power cycle already reconfigures the FPGA from flash, so plain FPGA configuration
state is covered by the thermal test and is also not it.

**R-16 is therefore unexplained, and since ~20:40 the board has not entered a domain under any
condition tried.** Everything after that timestamp carries no information about domains.

### What is NOT affected

Every control-validated result, because each compared a control and a failing case **inside one
boot**:

    f10:0 rc=0 | f10:9 rc=0 | f10:10 WEDGE     the SQLite blocker
    r110:0 rc=0 | r110:110 WEDGE                R-14 variant A
    r110:0 rc=0 | r110:111 WEDGE                R-14 variant B

### Remaining untried remedy

A **bitstream reflash** is the only untried remedy; it was offered and declined, and is
ask-first under CLAUDE.md, so it has not been done. Until the board enters domains again, no
further silicon measurement is possible and additional boots produce zero-information runs.
The productive work left is offline: the `cincoffset`-consumes-linear-`rs1` mechanism and its
two-derivation test, which is written up and ready to run the moment the board recovers.

## RTL RULES THAT BEAR ON THE BLOCKER — one candidate, with its own counter-evidence

Read from `capstone_dyn_unit.anvil` while the board ran. Recording all three, including the
one that argues against the candidate, so the next session does not re-derive them.

### 1. `stc` through an UNINIT destination REJECTS a non-zero immediate  (`:378`)

    } else if((rs1.metadata.cap_type==CAP_TYPE_UNINIT)&&(imm!=64'd0)){
        call raise_exception(data.trans_id,ex_code::ILLEGAL_OPERAND_VALUE)

This maps onto the observed struct-vs-scalar split exactly. Minimal codegen, `-O0`:

    struct kv v;  v.z=..; v.y=..     ->  stc a1, 0(a0)  ;  stc a1, 16(a0)   <- imm 16, NON-ZERO
    const char *p, *q;               ->  stc a1, 0(a0)  ;  stc a1, 0(a2)    <- both imm 0

Struct fields produce non-zero store immediates; separate scalars each get their own address
computation and store at offset 0. That is precisely R-14's "needs the struct element type".

### 2. `stc` NULLS the stored capability — but ONLY through an UNINIT destination (`:54-56`)

    else if(rs1_v.metadata.cap_type==CAP_TYPE_UNINIT){ ...
        if(rs2_v.metadata.cap_type!=NOT_CAP){ let rcnull = create_cnull(); ... }

In the normal path the result pack is `(rs1_v, rs2_v)` — unchanged. So a store does NOT consume
the stored capability unless the destination is UNINIT. Combined with (1), an UNINIT destination
would break the minimal sequence twice over:

    ldc a1, 0(gp)  ;  stc a1, 0(a0)  ;  cincoffsetimm a1, a1, 6   <- a1 nulled by the store

### 3. The counter-evidence: UNINIT comes only from `REVOKE`, and stage 9 works

`modify_cap_type(..., CAP_TYPE_UNINIT)` appears once, inside `func REVOKE` (`:43-67`): revoking
a capability that lacks write permission yields UNINIT. A freshly carved domain stack should not
be UNINIT.

Decisively: **stage 9 RETURNS `rc=0`**, and `sqlite3MallocInit` writes memsys5 zone headers all
over the 256 KB heap — necessarily with non-zero store immediates. If the domain's stack or heap
were UNINIT, stage 9 would trap too. It does not.

**So (1) and (2) are real ISA rules that fit the shape, but the precondition (UNINIT
destination) is contradicted by a control that passes.** Not a resolved cause — a candidate
whose precondition must be demonstrated, not assumed.

### The test that discriminates it, already built

Minimisation ladder stage **146** (four plain scalar pointers — every store `imm=0`) versus
**145/144** (struct fields — non-zero immediates). If 146 returns while the struct arms wedge,
the non-zero-immediate path is implicated whatever the destination type turns out to be. If both
wedge, immediates are not the axis at all.

## REFUTED FROM THE RTL: `cincoffset` does NOT consume a linear rs1 — and a correction to C-16's write-up

Read directly from `capstone_flu_unit.anvil:29-68` (SOURCE, primary):

    func CINCOFFSETIMM(data){
        if(data.cap_rs1.metadata.cap_type==cap_type_t::NOT_CAP){
            call raise_exception(data.trans_id,ex_code::UNEXPECTED_OPERAND)   // <-- DOES check
        } else { ...
            let rs1 = data.cap_rs1;
            let new_cursor = rs1.cursor + val;
            let rd = call create_capability(rs1.metadata,new_cursor);
            let result = call create_result_pack(...,rs1,rd);   // rs1 passed through UNCHANGED

### 1. The consume hypothesis is dead

The RTL passes `rs1` through the result pack **unchanged**. There is no `rs1 != rd` test, no
linearity check, and no nulling anywhere in `CINCOFFSET` or `CINCOFFSETIMM`. The
consume-on-non-copyable behaviour exists **only in QEMU's helper**
(`op_helper.c`: `if (rs1 != rd) { ... if(!captype_is_copyable) *rs1_v = NULL; }`).

So "silicon treats the cap-table capability as LINEAR and the first derivation nulls it" is
**REFUTED**. Straight-line vs loop codegen (`rd != rs1` vs `rd == rs1`) cannot matter on
silicon for this reason, and the 128/129 probe should show no difference. The asymmetry runs
the OPPOSITE way to what I assumed: QEMU is the stricter model here, not the RTL.

### 2. Correction to the C-16 write-up

I wrote, in the C-16 sections and in `ISSUES.md`, that "nothing in the RTL requires a tagged
`cincoffset` base, so the untagged pointer is silently used". That is **wrong for the immediate
form**: `CINCOFFSETIMM` explicitly raises `UNEXPECTED_OPERAND` on a `NOT_CAP` base.

It is right for the **register** form, but only because the check is commented out:

    func CINCOFFSET(data){
        // FIXME: wait for the non-cap instructions to set metadata properly
        // if((data.cap_rs1.metadata.cap_type==cap_type_t::NOT_CAP)|| ... ){
        //     call raise_exception(data.trans_id,ex_code::UNEXPECTED_OPERAND)
        // } else
        if((...UNINIT)||(...SEALED)){ raise } else { ...proceeds... }

So on silicon an untagged base **traps** through `cincoffsetimm` and **is silently accepted**
through `cincoffset`. C-16 remains a real bug (QEMU asserted, the MIR showed the tag being
stripped, the fix changed the emitted instruction), but the claim about how it manifests on
hardware must be stated per-form, not blanket.

### What this leaves for the SQLite blocker

Still open. The straight-line construct wedges with controls returning; the mechanism is NOT
cincoffset consumption. A trap is now a live candidate — an untagged base reaching
`cincoffsetimm` raises `UNEXPECTED_OPERAND`, and R-5 records that illegal capability ops wedge
rather than report — but nothing yet explains what would untag the base in the first place.

## MECHANISM HYPOTHESIS FOR R-14 / THE SQLITE BLOCKER: `cincoffset` CONSUMES A LINEAR rs1

Found by diffing the codegen of the two shapes (offline, no board). Merged string constants
give ONE blob capability per cap-table slot; every literal is a `cincoffsetimm` from it.

**Straight-line (WEDGES on silicon at N>=4, control-validated):**

    ldc            a1, 0(gp)        ; blob capability, loaded ONCE
    cincoffsetimm  a2, a1, 6        ; rd=a2 != rs1=a1
    stc            a2, 16(a0)
    cincoffsetimm  a2, a1, 11       ; reuses a1
    cincoffsetimm  a2, a1, 17       ; reuses a1 ... 8 derivations from one a1

**Loop-assigned (the shape that is not known to wedge):**

    ldc            a0, 0(gp)        ; reloaded in the loop body
    cincoffsetimm  a3, a0, 41       ; rd != rs1
    cincoffsetimm  a0, a0, 48       ; rd == rs1   <-- writes back into itself

### The rule that separates them

`helper_cscincoffset` (`op_helper.c`):

    if (rs1 != rd) {
        *rd_v = *rs1_v;
        if (!captype_is_copyable(rs1_v->val.cap.type))
            *rs1_v = CAPREGVAL_NULL;          // CONSUMES rs1
    }

The consume is gated on **`rs1 != rd`**. The straight-line form uses a fresh destination each
time, so if the cap-table capability is **LINEAR (non-copyable)** the FIRST `cincoffsetimm`
nulls `a1` and the remaining seven derive from a nulled register — producing garbage pointers,
which is precisely the observed corruption. The loop form writes back into the same register,
so nothing is ever consumed.

**Why QEMU never reproduces it:** on QEMU the cap-table capability is NONLIN (copyable), so the
consume branch never fires. If the RTL treats it as LINEAR, silicon diverges exactly here —
and the RTL does not check a `cincoffset` base, so it produces a value and keeps going.

### Why this fits every control-validated observation

* straight-line wedges, loop-from-table does not (R-14 variants A/B vs C)
* N as low as 4 fails — only TWO derivations are needed for the second to read a nulled base
* `sqlite3RegisterBuiltinFunctions` is straight-line over ~200 entries: same shape, same fault
* QEMU-clean at `-O0` and `-O1`, silicon-wedging — the exact asymmetry seen

### NOT established

The cap type on silicon has not been read. This is a codegen+ISA-semantics hypothesis that
explains the data; it is not a measurement. Two tests would settle it, neither run yet:

1. **Two-derivation probe**: `rd != rs1` twice from one `ldc` (predict WEDGE) versus the same
   two derivations chained `rd == rs1` (predict RETURN). Tiny, no added globals.
2. Read the cap type of the `gp[i]` capability directly (`lcc` zimm=1) — but `lcc` probes have
   been unreliable here, so (1) is the better instrument.

**If confirmed, the fix is compiler-side and small**: never derive twice from a cap-table
capability with `rd != rs1` — chain through the same register, or reload per use.

## THE BOARD STOPPED ACCEPTING ANY IMAGE AT ~20:40 — a fresh firmware does NOT restore it  `REFUTED 2026-08-02 — watchdog artefact, see READ FIRST`

    fresh firmware rebuild (no domain changes), r110.dom unchanged:
      attempt 1  ENTRY-STALL
      attempt 2  ENTRY-STALL

So the firmware-generation hypothesis is **not** confirmed either: rebuilding and reflashing the
firmware did not bring back the behaviour `r110` had at 19:05-19:20.

### The full timeline, which is now the primary evidence

    ~19:00-20:16   r110 ENTER 3/3 | n112 ENTER 3/3 | f10 ENTER (controls returned)
    ~20:40 -> now  sb10, waa, wab, wac, t120, tsp, tsq, tsr, u120, L126, r110(x2), r110 on
                   FRESH firmware (x2)  ->  13+ attempts, 7 distinct images, ZERO entries

Nothing has entered the domain since roughly 20:40, across every image tried, including one that
had entered three times earlier and one built from a freshly rebuilt firmware. **This is a
board-level state change, not a property of any domain image or firmware build.**

Candidates not yet separated: thermal/power state after ~25 power cycles today; the resident
bitstream drifting into a bad state; or accumulated JTAG/debug-module state (two JTAG failures
were already seen today: "Timed out after 120s waiting for busy to go low", "Failed to read priv
register", "Protocol error with Rcmd").

### What this invalidates and what it does not

Already retracted above: the per-image model, the carve-count pre-flight rule, and the
static-builtins attribution. Add to that: **any cross-boot comparison from today's later
session is unsafe**, because the board's acceptance of images changed underneath it.

**Unaffected, and worth restating**: every result taken WITHIN a single boot, where a control
returned alongside the failing case.

    f10:0 ret rc=0 | f10:9 ret rc=0 | f10:10 WEDGE      -- the SQLite blocker
    r110:0 ret rc=0 | r110:110 WEDGE                     -- R-14 variant A
    r110:0 ret rc=0 | r110:111 WEDGE                     -- R-14 variant B

Those are internal to one boot and one firmware and survive the board-state change entirely.
This is the strongest practical argument for the control discipline: it is the only thing from
today's board work that is not now in doubt.

### Next action requires a decision, not another experiment

The obvious step is a **bitstream reflash** to clear board state. That is explicitly an
ask-first action under CLAUDE.md (irreversible / outward-facing), so it is NOT being done
unilaterally. Until the board accepts images again, no further silicon measurement is possible
and additional boots would only add more zero-information runs.

## !! RETRACTED: "R-16 is build-dependent" — THE KNOWN-GOOD IMAGE NOW STALLS TOO  `RETRACTION WITHDRAWN — kg runs were false aborts`

    r110.dom, which ENTERED 3/3 earlier today (~19:05-19:20, control returned rc=0):
      21:46  attempt 1  ENTRY-STALL
      21:48  attempt 2  ENTRY-STALL

Same binary, same selector, same position. **So the per-image model is wrong**, and with it:

* **"R-16 is build-dependent"** — RETRACTED. It rested on `r110` entering 3/3 while `x101`/`r112`
  stalled; `r110` now stalls, so image identity does not determine entry.
* **">=182 carves stalls 8/8" as a predictive rule** — WITHDRAWN as a pre-flight filter. Every
  182 image was built late in the session; the correlation is confounded with time/firmware
  generation, not established as causal.
* **"5/5 static-builtins images stall"** — the observation stands, but the attribution to the
  static-builtins configuration does not: those five were all built in the same late window.

### What actually correlates: FIRMWARE GENERATION, not domain image

    ~19:00-20:16   r110 ENTER 3/3, n112 ENTER 3/3, f10 ENTER (controls returned)
    ~20:40 onward  EVERY image stalls: sb10, waa, wab, wac, t120, tsp, tsq, tsr, u120, L126,
                   and now r110 itself -- 11+ attempts, 6+ distinct images, 0 entries

The firmware was rebuilt repeatedly across that boundary (w113, sb, ts, u, L builds). **The
monitor and the entry glue live in the firmware**, and the stall is precisely a
monitor-hands-off/domain-never-returns failure — so a firmware generation is a far more
plausible carrier of this than the domain binary.

### Consequence

Every conclusion drawn today from "image X enters / image Y stalls" needs re-reading as
possibly "firmware generation N was healthy / generation N+k is not". The *control-validated
in-domain results* are unaffected — `r110:0` returned while `r110:110` wedged in the SAME boot,
and `f10:0`/`f10:9` returned while `f10:10` wedged in the SAME boot. Those comparisons are
internal to one boot and one firmware, which is exactly why the control matters.

**So the SQLite blocker finding survives**: `sqlite3RegisterBuiltinFunctions` wedges with two
controls returning alongside it. What does NOT survive is the R-16 model built around it.

### Next test

Rebuild the firmware and re-run `r110` unchanged. If it enters again, firmware generation is
the variable and R-16 is a property of the build/flash cycle rather than of any domain.

## THE N-THRESHOLD IS <= 4: CLAMPING THE BUILTIN COUNT CANNOT WORK

I built a six-point N sweep (4/8/16/32/48/64) to find the largest straight-line struct array
that still returns on silicon. **That was unnecessary — the answer was already in
control-validated data.**

    r110.dom:0    CONTROL trivial return                    RETURNED rc=0
    r110.dom:111  variant B -- FOUR straight-line entries   IN-DOMAIN WEDGE

Variant B materialises only `a[0]`..`a[3]` straight-line and fills the remaining 60 entries in
a loop. It wedges, with its control returning in the same boot. So **four straight-line
struct-field materialisations are already enough to wedge**, and there is no N small enough to
be useful — SQLite needs far more than four builtins.

**Clamping the builtin count is therefore not a viable path**, and the sweep images
(`t120`/`tsp`/`tsq`/`tsr`/`u120`) were board time spent re-deriving a known result. The lesson
is procedural: before designing a new experiment, check whether an existing control-validated
run already answers it.

### Where that leaves the three candidate shapes

    straight-line local        R-14 in-domain wedge, at N as low as 4      (validated)
    static initialised global  R-16 entry stall, 5/5                        (validated)
    loop from a static table   NEVER TESTED against real SQLite            (variant C shape)

Only the third is untried, and there is a specific reason to think it is worth one attempt:
the R-16 entry stall correlates with carve count (>=182 stalls 8/8), and every probe image that
tried variant C's shape carried the probe harness's own globals on top. **The real
`sqlite_silicon` image sits at 179 carves and enters**, so adding variant C's small static
name table (+1 carve -> 180) would still land below the 182 line that has stalled every time.

That is the one remaining shape with a plausible path to running SQLite on this silicon, and it
is exactly the workaround `ISSUES.md` R-14 recommended before any of this was measured:
**build `aBuiltinFunc` in a loop from a static table instead of straight-line.**

## THE WORKAROUND RELIABLY TRIGGERS R-16 — 5/5 static-builtins images entry-stall  `OVERSTATED — 2 distinct binaries, 3 false aborts`

    image                     behaviour                 .data
    r110 r111 f10 n112        ENTER                     15088
    sqlite_silicon            ENTER                      5872
    sb10 st10 waa wab wac     STALL (static-builtins)   24304    <- 5 of 5
    r112 x101                 STALL                     15088

**Every image built with `SQLITE_STATIC_BUILTINS=1` entry-stalls: 5 for 5, across five
independently built binaries** (including three deliberately perturbed draws, `waa`/`wab`/`wac`,
built specifically to get a different outcome). That is not a lottery that can be won by
redrawing.

### Why this is mechanistically unsurprising

The workaround makes `aBuiltinFunc` a compile-time-initialised **global** instead of a
straight-line local. That adds ~9 KB of initialised `.data` (24304 vs 15088) whose capability
leaves the entry glue must copy from the blob and initialise via `__capstone_cap_init` — at the
domain's FIRST entry, which is precisely where R-16 stalls. The workaround moves the work out
of the main run and into the one place that is already failing.

**So the fix for the SQLite blocker cannot be validated on silicon in its current form**: it
converts an R-14 in-domain wedge into an R-16 entry stall. QEMU shows none of this because QEMU
does not reproduce R-16 at all.

### What this does NOT establish

`.data` size alone does not discriminate — `r112` and `x101` stall at 15088, the same value as
four entering images. So "bigger initialised data" is a correlate within the static-builtins
family, not a general rule, and R-16 stays unexplained. The honest statement is: *the
static-builtins configuration is reliably associated with the entry stall*, mechanism plausible
but unproven.

### Consequence for the SQLite path

Both routes are now blocked by different faults:

    keep the local array   -> R-14 in-domain wedge at sqlite3RegisterBuiltinFunctions (validated)
    make it a static       -> R-16 entry stall, 5/5 (this section)

A third shape is needed that avoids both: something that neither materialises the array
straight-line at run time NOR adds a large initialised global. R-14 variant C (fill a local in a
LOOP from a small static table) is the obvious candidate — it was the pre-fix
board-validated-correct shape, and its static table is far smaller than the whole `FuncDef`
array. It has never been tried against real SQLite.

## CORRECTION: carve count does NOT predict the R-16 entry stall  `the 182=8/8 count is unsupportable — duplicates + false aborts`

Earlier I wrote that removing one capability-bearing global (182 -> 181 carves) "flipped an
image from stalling to entering" and called it the first mechanistic handle on R-16. The full
table refutes the general claim:

    182 carves   STALL 4/4     r112, r113, v110, w113
    181 carves   MIXED         ENTER: r110, r111, f10, n112
                               STALL: sb10, st10, x101
    179 carves   ENTER         sqlite_silicon

So **182 has always stalled, but 181 does not predict entry** — four 181-carve images enter and
three stall. The `n112` experiment was a real controlled comparison (same source, one arm
removed, entered 3/3 where the 182 builds stalled 4/4), but it does not generalise: carve count
is not the discriminator, and **R-16 remains unexplained**.

Every structural attribute checked so far fails to separate entering from stalling images:
carve count, `.text` size, merged-string bytes, and dom_data geometry (byte-identical across the
divide). Whatever selects an image is not visible in any of them.

### Practical consequence for the workaround

`SQLITE_STATIC_BUILTINS=1` — the one change that removes the exact construct the blocker was
pinned to — **still has no silicon measurement**, because both images built with it (`st10`,
`sb10`) entry-stalled before executing any code (`sb10` 3/3). This is R-16 blocking the
validation of the fix for R-14/the SQLite blocker.

Since R-16 is per-image and unpredictable, the only available lever is to draw a different
image. With the watchdog's live entry-stall abort a losing draw now costs ~30 s instead of
~600 s, which makes drawing several builds affordable — that is the approach in flight.

## THE SQLITE BLOCKER, CONTROL-VALIDATED: it is `sqlite3RegisterBuiltinFunctions`

Every earlier SQLite verdict on silicon was recorded WITHOUT a control. This one has two, in
the same image and the same boot (`f10.dom`, fixed compiler, no workaround):

    :0   CONTROL trivial `return 0`              RETURNED rc=0
    :9   sqlite3MallocInit + PcacheInitialize    RETURNED rc=0
    :10  + sqlite3RegisterBuiltinFunctions       IN-DOMAIN WEDGE

`stage 9` is the strong control: it does real allocator work (memsys5 zone headers in the
256 KB heap) and returns cleanly. The ONLY delta to stage 10 is
`sqlite3RegisterBuiltinFunctions`. So:

**The SQLite silicon blocker is `sqlite3RegisterBuiltinFunctions`, and it survives C-16.**

### It is the same fault as R-14, and both ends are now control-validated

    R-14 variant A (standalone shape)   control returns, A wedges    VALIDATED
    R-14 variant B (standalone shape)   control returns, B wedges    VALIDATED
    SQLite stage 10 (real code)         2 controls return, 10 wedges VALIDATED

`sqlite3RegisterBuiltinFunctions` builds a straight-line struct array of string constants —
exactly the R-14 variant-A/B shape. Three independent control-validated measurements now agree,
where before today the evidence was a mix of uncontrolled wedges and (as it turned out) two
artifacts.

### What is NOT yet shown

* Variants C and D still have no valid silicon measurement, so "both ingredients required"
  (struct type AND straight-line materialisation) remains unproven — the shape could be
  narrower or broader than R-14's pre-fix table claims.
* Attribution is still open: this is a construct that QEMU executes correctly and silicon does
  not, which is consistent with hardware but does not prove it (C-16 was exactly that pattern
  and turned out to be ours).

### Test in flight: does the workaround actually work on silicon?

`SQLITE_STATIC_BUILTINS=1` deletes the straight-line local entirely (compile-time-initialised
static instead). It is QEMU-green — stage 10 returns `rc=0` — but has **never been validated on
silicon**: the earlier `st10` image hit the R-16 entry stall before running any code, so it
yielded nothing. Running it now with the same `:0` / `:9` / `:10` control ladder. If stage 10
returns, SQLite has a working silicon path for the first time.

## VALIDATED: variant A's wedge IS variant-specific — control returns in the SAME image

    r110.dom, position 1, one boot:
      sel=0    CONTROL trivial `return 0`   RETURNED rc=0
      sel=110  variant A                    IN-DOMAIN WEDGE

This is the check that was missing when the variant-D result was recorded, and here it
**passes**: the trivial control returns from the same image, in the same boot, moments before
variant A wedges. So `r110`'s entry, glue reentry, marker write and return path are all sound,
and variant A's wedge is a property of **variant A**, not of the image.

**"R-14 variant A still wedges on silicon after C-16" therefore STANDS.**

### And it explains the n112 divergence

    r110   control RETURNS  -> image sound   -> its variant verdicts are interpretable
    n112   control WEDGES   -> image broken  -> its variant verdicts are void (5 runs, all
                                               selectors incl. `:0` and no-selector at all)

Two different failures that both presented as "in-domain wedge". The control is what separates
them, and it costs nothing. Restating the rule, now demonstrated in both directions:

> **Run `:0` first on every staged image.** A returning control makes that image's wedges
> evidence; a wedging control makes them noise.

### Status of the R-14 post-fix table

    variant A (110)   WEDGES on silicon, control-validated      QEMU: returns 16
    variant B (111)   wedge recorded, control validation IN FLIGHT
    variant D (112)   VOID -- only ever measured on the broken n112 image
    variant C (113)   never measured on silicon; its static table is the object whose
                      presence flips an image into the R-16 entry stall (4/4 vs 2/2)

So the "both ingredients required" claim still cannot be settled: A is confirmed, B is pending,
and C/D have no valid silicon measurement at all.

## RETRACTED: "variant D wedges on silicon" — THE CONTROL WEDGED TOO

    n112.dom (181 carves, stage-113 static table removed), position 1:
      sel=112  variant D   ENTERED then IN-DOMAIN WEDGE   (3 runs, reproducible)
      sel=0    CONTROL     ENTERED then IN-DOMAIN WEDGE   <-- trivial `return 0`, NO array code

Selector 0 executes `if (stage <= 0) return 0;` in `run_sqlite_staged` — no struct, no array,
no string literals. It reached `SQ: G/enter` and wedged exactly like variant D.

**Therefore every in-domain wedge from this image is void as evidence about the variant it was
supposedly testing**, including the reproducible "variant D wedges 3/3" recorded above. D's
result said nothing about D. Withdrawn.

The control was available at ZERO build cost the whole time: `if (stage <= 0) return 0;` is
**not inside any `#if`**, so **selector 0 is live in every staged image ever built**. It should
have been the first selector run on any new image, before any variant verdict was recorded.
Making that the standing rule:

> **Run `:0` first on every staged image.** If the trivial control does not return, nothing
> else that image reports is interpretable.

### What still stands from that image

**The R-16 entry result is unaffected**, because it is decided *before* any domain code runs:
`n112` reached `SQ: G/enter` in 3/3 runs (both shares complete), against 4/4 entry stalls for
the 182-carve builds. Entry and post-entry behaviour are independent — that is the whole point
of the classification rule in this document.

### Live suspect for the post-entry wedge

Every selector path has one thing the no-selector path lacks: the domain reads
`hostcall_metadata->opcode` from the shared region at entry to pick its probe. If that read is
what wedges on silicon, the runtime-selector mechanism is unusable on hardware — QEMU-green but
board-fatal, the same asymmetry that hid C-16. Test in flight: run `n112.dom` with NO selector,
which runs the same variant D via the compile-time constant and never touches the shared region.

## DOM_DATA GEOMETRY IS IDENTICAL BETWEEN ENTERING AND STALLING IMAGES — and "enters reliably" was n=1

### Geometry does not discriminate

    image             group   blob    captable  storage   STACK    globals_off
    r110              ENTER   75392     2912     354576   211088   0x150000
    r112              STALL   75392     2912     354576   211088   0x150000
    r113              STALL   75392     2912     354576   211088   0x150000
    v110              STALL   75392     2912     354576   211088   0x150000
    r111              ENTER   75120     2896     354320   211904   0x150000
    st10              STALL   75120     2896     354320   211904   0x150000

`r110` (entered) and `r112`/`r113`/`v110` (stalled) have **byte-identical dom_data geometry**;
so do `r111` (entered) and `st10` (stalled). The carve loop and cap-init work against exactly
the same layout in both groups, so **size/layout is ruled out** as the cause of the entry
stall — as are carve count and `.text` size, which also fail to separate the groups (181 carves
appears on both sides; `st10` stalls with *smaller* `.text` than the entering `r110`).

### Correction: the "entering" images have n=1

I wrote that `r110`/`r111` "enter reliably". That was too strong and is corrected here:

    x101   STALLED 6/6      (strong)
    r112   STALLED 3/3      (moderate)
    r110   entered 1/1      (n=1)
    r111   entered 1/1      (n=1)
    v110   stalled 1/1      (n=1)

The stalling side has real repetition behind it; the entering side does not. So
"build-dependent" is supported for the *stalling* images and **assumed** for the entering ones.
Given that `r110` and `r112` share identical geometry and differ only by ~1.6 KB of compiled
stage code, the alternative — that entry is a per-BOOT coin toss and `r110` simply got lucky
once — is not yet excluded.

**Test running:** `r110` three times at position 1. 3/3 entering supports a genuine per-image
split; any stall means "build-dependent" is wrong and the stall is boot-level, which would make
every single-sample entry/stall attribution in this document unsafe.

## THE ENTRY STALL IS NOW THE PRIMARY BLOCKER — IT DEFEATS THE RUNTIME-SELECTOR WORKAROUND

The runtime selector was built precisely to dodge the build-dependent entry stall: put every
probe in ONE image that is known to enter, and select at run time. It works perfectly under
QEMU — all four R-14 variants dispatch correctly from a single image:

    QEMU  sel=113 -> 16    sel=112 -> 16    sel=111 -> 16    sel=110 -> 16

On the board, that image (`v110.dom`) **entry-stalled at position 1** and the run ended with
zero variants measured.

### Why this matters more than the variant table

The workaround assumed there is a stable set of "images that enter". There is not. `r110` and
`r111` enter reliably; `v110` is the *same source* rebuilt with two extra stages compiled in,
and it stalls. So:

* **Any rebuild is a fresh draw.** You cannot carry the "it enters" property across a build.
* **The selector cannot rescue a stalling image** — the stall happens before any domain code
  runs, so run-time selection never gets a chance.
* **Reusing an old entering image is not general**: `r110` was built when only stages 110-111
  existed, so selectors 112/113 fall through its compiled `if` range into the real SQLite path
  and would silently measure the wrong thing. An old image can only answer the probes that were
  compiled into it.

### Consequence for planning

**The entry stall is now the primary blocker for the whole measurement campaign** — ahead of
R-14 and ahead of SQLite itself. Every remaining question (the R-14 ingredient split, the
store-vs-load fork, anything else) is gated behind "can this image enter", and that is currently
a coin toss that cannot be influenced by retrying, reordering, or restructuring the probe.

Today's position-1 tally, by whether the domain's own code ever ran:

    ENTERED   sqlite_silicon, f10, r110, r111
    STALLED   st10, x101 (6/6), r112 (3/3), r113, v110

**Root-causing the SHA5 entry stall should now take priority over further variant work.** It is
a monitor-hands-off/domain-never-returns failure at the FIRST entry, which is where the glue
builds the cap table and runs `__capstone_cap_init` — and unlike R-14 it is not QEMU-visible.

## THE ENTRY STALL IS BUILD-DEPENDENT, NOT A FLAT RANDOM RATE — POOLING HID THE STRUCTURE

Section 0a10 measured a slot-1 `SHA5` entry-stall rate of 2.8% over 107 launches and called it
a "residual floor". That number is real but the *model* behind it is wrong: it pools many
different binaries, and the per-binary behaviour is bimodal, not uniform.

Position-1 launches, grouped by binary:

    ENTER RELIABLY          sqlite_silicon, f10, r110, r111   -> reached SQ: G/enter
    ENTRY-STALL RELIABLY    x101   6/6 stalls
                            r112   3/3 stalls
                            st10   stalled

A binary that stalls does so repeatedly; a binary that enters does so repeatedly. A flat 2.8%
process cannot produce 6/6 and 3/3 on specific images while others never stall.

### Two consequences, both practical

* **Retrying the same binary is close to futile.** The `r112` retry loop spent three boots
  re-drawing the same losing ticket. Retry is the right response to an *infra flake*; for an
  entry stall the right response is to **change the binary or the order**.
* **The runner stops at the first failure**, so a stalling domain at position 1 permanently
  masks everything behind it. `r113` never got a single turn while `r112` sat in front of it.
  Reordering is free and was the move that should have been made after the first stall, not
  the third.

### What this does NOT establish

Why a given image stalls. Nothing structural has separated the stalling from the entering
builds — carve count, `.text` size and merged-string bytes were all checked and none
discriminate (see the section on structural signatures). So "build-dependent" here means
"reproducible per image", not "explained".

**Revised reading of 0a10:** the slot-1-vs-slot-2 comparison (2.8% vs 32%) still stands as a
measurement, but neither number should be used as a per-run failure probability for a
*particular* domain. For planning, assume a given image either enters or does not.

## CLASSIFY BEFORE RECORDING: an entry stall is NOT a result about the domain

Two board failures look identical in a summary and mean opposite things. Always read the LAST
MARKER, never just "did not return":

    last marker = SHA5:xxxx      ENTRY STALL. The monitor handed off and the domain never came
                                 back from its FIRST entry. The domain's own code never ran.
                                 -> tells you NOTHING about the domain. RETRY.

    last marker = SQ: G/enter    IN-DOMAIN WEDGE. The domain entered and hung in its own code.
                                 -> a genuine result about the domain under test.

    SQ: obs=<n>                  RETURNED. A number, always usable.

This was not academic. On 2026-08-02 the variant-D **control** (`r112`, expected to return 16)
came back `NO RETURN, last=SHA5:00000000`. Recorded naively that reads "variant D fails on
silicon", which would have destroyed the entire A/B-vs-C/D comparison the experiment exists to
make — D is the control that isolates "struct element type" as a necessary ingredient. The
domain had not executed a single one of its own instructions. Both controls had already passed
the QEMU gate (112 -> 16, 113 -> 16), so the domains themselves were sound.

**The runner's own "FIRST FAILURE" summary does not make this distinction** — it reports "did
not return" for both. The distinction has to be made when reading the scoped log.

### Retry discipline, now automated

    ENTRY-STALL(SHA5)     domain never ran        -> retry, up to 3x
    RETURNED:<n>          real result             -> stop
    NO-RETURN(last=...)   real in-domain wedge    -> stop

Same principle as the `__CAPSTONE_INFRA_FLAKE__` retry: **a failure that happened before the
thing under test began is not evidence about the thing under test.** Two separate results were
nearly recorded as findings today for want of that rule — the QEMU infra flake on `r14b -O0`,
and this entry stall on `r112`.

## R-14 STILL WEDGES ON SILICON AFTER C-16 — and the watchdog proved the board was really working

    QEMU (fixed compiler)   stage 111 (variant B) -> 16   stage 110 (variant A) -> 16   asserts=0
    BOARD (fixed compiler)  pos1 r111.dom -> NO RETURN, last marker SQ: G/enter

So the R-14 construct still fails on hardware with C-16 fixed, while the *same source* returns
the correct 16 under QEMU. `r110` never ran — the runner stops at the first failure, by design.

**One honest caveat on comparability.** Pre-fix, variant B *returned 4*; here it does not return
at all. Do NOT read that as "the fix made it worse": these are different binaries. The old
variant B was a standalone fpga-repro domain built from `strline_struct_repro.c`; `r111` is the
same *shape* embedded as a staged probe inside the SQLite amalgamation, with entirely different
surrounding code and globals. The shapes match; the binaries do not. A like-for-like comparison
needs the standalone repro rebuilt with the fixed compiler.

### The watchdog earned its place on this run

    QUIET  420s .. 525s   no UART for 75s .. 180s   (limit 240s)
    ALIVE  540s  +4047B
    ALIVE  555s  +1985B
    GONE   555s  runner PID no longer running
    ENDED  555s

This is exactly the information that was missing before: through the 180 s of silence the
watchdog kept reporting *how long* the board had been quiet against the limit, so "wedged
domain" was distinguishable from "runner died" from "still working" **while it was happening**,
not afterwards. The run then resumed (the runner writing its summary) and ended cleanly.

Standing rule going forward: **every board session gets the watchdog as a second process.**
`bash capstone/tests/rtl-smoke/board-watchdog.sh <uart-log> <idle-limit> <runner-pid>`.

## THE REMAINING BLOCKER IS SILICON-ONLY AND IS *NOT* C-16 — TWO RESULTS THAT SETTLE IT

### 1. Stage 10 still fails on silicon with the fixed compiler

    2026-08-02, f10.dom (stage 10, FIXED compiler, NO workaround), position 1
      pos1 f10.dom   NO RETURN   last = SQ: G/enter

So C-16 did **not** fix the stage-10 silicon failure. The construct still stalls on hardware
after the domain enters. Combined with the full-SQLite run (also silent after `G/enter`), both
of today's board runs agree.

### 2. R-14 variant A is QEMU-CLEAN and silicon-wedging

R-14 variant A is now a proper QEMU-gated rung (`silicon-ladder/r14a_app.c` + `r14a_host.c`,
oracle 16), reduced verbatim from the fpga-repro:

    r14a  -O0   PASS  retval = 16
    r14a  -O1   PASS  retval = 16

against a board result of **WEDGE**. And C-16 provably cannot explain it:

    struct kv { const char *z; const char *y; };   /* 2 capabilities = 32 B, NO tail padding */
    struct kv a[64];                                /* uninitialised, assigned element-by-element */

No tail padding means no padding-`memset`; no aggregate initialiser means no initialiser
`memset` at all. C-16's trigger is absent by construction.

### What follows

* **R-14 is a genuinely separate defect from C-16**, and is the prime candidate for the
  remaining SQLite blocker. Marking it "PARTLY SUPERSEDED" rather than closing it was correct.
* **QEMU cannot see it.** The rung passes at both optimisation levels with the fixed compiler,
  so this is not an untagged-capability-arithmetic bug — QEMU asserts on those, which is
  exactly how C-16 was caught.
* **That still is not proof of a hardware defect**, and the registry's existing caution should
  stand. QEMU passing rules out the class of codegen faults QEMU models; it does not rule out
  codegen whose effect QEMU's memory model happens to tolerate. R-1 (the FPGA-only load/store
  hazard, likewise never reproduced under QEMU) is the obvious neighbour and may be the same
  underlying thing.

### The best next experiment

`r14a` is now a **one-command QEMU rung and a board domain from the same source**, which the
old fpga-repro was not. Run variants B/C/D the same way: **variant B is the valuable one** — it
returns a WRONG VALUE (4 instead of 16) rather than wedging, so it converts a hang into a
number, which is exactly the "make every run RETURN" method that this project's own debugging
rule prescribes. A wrong value can be bisected; a wedge cannot.

## BOARD RESULT AFTER THE C-16 FIX: STILL BLOCKED — QEMU GREEN, SILICON SILENT

    2026-08-02, fixed compiler, no workaround, position 1
      SQ: A/dom-ok ... B/mkregion1 ... C/mkregion2 ... D/mapped
      SQ: E/share1 (SHA0..SHA6, ECSZ)   SQ: F/share2 (SHA0..SHA6, ECSZ)
      SQ: G/enter
      <UART idle 600 s>   -> ActionTimeout, run aborted, board released

**C-16 did not unblock silicon.** State this plainly: the fix is real and verified (codegen,
reproducer, stage 10 non-static, full QEMU gate, ladder 6/6), but **SQLite still produces no
rows on the board**. Both shares now complete and the domain enters, then nothing for ten
minutes.

600 s of total silence is not the "legitimately slow" case — a working run emits its first row
well before that, and the idle budget was raised to 600 s specifically so that slowness could
not be mistaken for a stall. So this is a genuine stall (or a broken output path), not
impatience on the runner's part.

### What this means, and what it does not

* **It does NOT invalidate C-16.** That bug was proven independently: QEMU asserts on it, the
  MIR shows the tag being stripped, and the fix changes the generated instruction. It was a
  real silent-corruption bug on hardware regardless of whether it was the *only* one.
* **It DOES mean there is at least one more fault**, and it is silicon-only — QEMU executes the
  same domain end-to-end. Candidates, in rough order of prior plausibility: R-1 (the FPGA-only
  load/store hazard, which QEMU has never reproduced), R-14 variant A (the unpadded 2-pointer
  struct that C-16 explicitly does NOT explain), or the `SHA5`-class stall appearing later in
  the run rather than at entry.
* **The QEMU-vs-silicon asymmetry is now a known, named hazard**: QEMU asserts on untagged
  capability arithmetic, the RTL accepts it silently. A QEMU pass is necessary and NOT
  sufficient, and this run is the proof.

### Next step, and it is cheap

Re-run the **staged ladder** on the board with the fixed compiler — stage 10 first, since that
is the stage that previously failed on silicon and the one C-16 was supposed to fix. If stage 10
now returns, C-16 fixed the entry-side fault and the remaining blocker is later in
`sqlite3_initialize`/`open`; if it still stalls, the remaining fault is in the same construct and
C-16 was only part of it. Then stages 2 and 3. That is one boot for a decisive split.

## REGRESSION STATUS AFTER THE C-16 FIX

The fix is in generic `SelectionDAG` code, so it was gated before shipping:

    ladder (DOMAIN_OPT_LEVEL=-O1, QEMU)
      matmult_int  PASS      beebs_prime      PASS      beebs_bs   PASS
      beebs_cover  PASS      beebs_aha_mont64 PASS      strarray   PASS
      => 6 passed, 0 failed

`strarray` is the new C-16 regression test; the other five are pre-existing rungs and are
unchanged by the fix, which is the point of running them.

Note `matmult_int` passes here at `-O1` under QEMU — that says nothing about R-1, which is an
FPGA-only load/store hazard that QEMU does not reproduce. Do not read this row as R-1 progress.

## BLOCKER RESOLVED UNDER QEMU — the original shape passes with NO workaround

    stage 10 NON-STATIC (sqlite3RegisterBuiltinFunctions, local aBuiltinFunc array)
        -> RETURNED stage=10 rc=0x00, asserts=0

    FULL SQLite QEMU gate, SQLITE_STATIC_BUILTINS unset
        -> exit 0, __CAPSTONE_SQLITE_SILICON_PASSED__, asserts=0

Stage 10 is the construct that has blocked this campaign, and it now returns cleanly with the
workaround OFF. The full five-marker gate passes on the same build.

**The blocker was our compiler, not the silicon.** Restated plainly for the record: every
symptom previously attributed to the board miscomputing a pointer was `memset` writing 15 bytes
of struct tail padding through an **untagged, garbage pointer**, once per array element. It is
silent on hardware only because the RTL does not check a `cincoffset` base — `SPLIT`, `LDC` and
`STC` all validate their operands, but `cincoffset`/`cincoffsetimm` do not.

### What still needs doing (in order)

1. **Ladder regression under QEMU** — the fix touches generic `SelectionDAG` code, so the
   ladder rungs must be re-run before anything ships. Serialise: QEMU suites share the
   `rootfs.ext2` write-lock.
2. **Board run** — firmware rebuilt with the fixed, no-workaround domain.
3. Re-examine every "silicon miscompute" claim in this document and in `ISSUES.md` against this
   root cause; several are likely the same bug.
4. Consider whether `SQLITE_STATIC_BUILTINS` should be deleted outright now that the real fix
   exists, rather than left as a knob.

### Caveat that must travel with this

QEMU passing is necessary, not sufficient: QEMU **asserts** on an untagged `cincoffset` base
while the RTL silently accepts it, which is exactly why this bug survived so long on the board
while never being seen under QEMU — because the staged probes were never run under QEMU. The
board run is still the deciding test.

## !!!!! FIXED AND VERIFIED (QEMU): memset destination typed in AS0 instead of AS200

### The defect

`SelectionDAG::getMemset` (`llvm/lib/CodeGen/SelectionDAG/SelectionDAG.cpp:9380`) built the
destination argument type with `PointerType::getUnqual(Ctx)` — an **addrspace(0)** pointer.
Here AS0 is a 64-bit integer address while the real destination is an AS200 **128-bit
capability**, so the declared argument type is narrower than the value and call lowering
inserts a `TRUNCATE` of the pointer.

The MIR showed it with the correct case immediately adjacent:

    %8:gpr  = PseudoTRUNC_CAP %5      ; truncate the array base -- TAG GONE
    %9:gpr  = ADDI killed %8, 49      ; tail-padding address
    $x10    = COPY %9                 ; passed as memset's destination
    ...
    %13:gpr = CIncOffsetImm %5, 64    ; next element -- CORRECT, tag preserved

The IR was correct throughout (`getelementptr inbounds i8, ptr addrspace(200) %0, i128 49`,
no `ptrtoint`), which is what proved the bug was in the backend and not the frontend.

### The fix

Take the address space from `DstPtrInfo`, which was already in scope and already used for
`checkAddrSpaceIsValidForLibcall`:

    Type *DstPtrTy = PointerType::get(Ctx, DstPtrInfo.getAddrSpace());

For AS0 targets this is exactly what `getUnqual()` returned, so it is a no-op for every other
target.

### Verified

    generated code   before: 8x `addi ..., 49`   after: 0x  ->  `cincoffsetimm a0, a0, 49`
    reproducer       before: helper_cscincoffsetimm assertion
                     after:  PASS, retval = 420 (matches the native oracle)

`strarray_app.c` + `strarray_host.c` are committed as the regression test:
`DOMAIN_OPT_LEVEL=-O0 bash run-ladder-qemu.sh strarray`, ~1 minute, no board.

### Status of the workaround

`SQLITE_STATIC_BUILTINS=1` remains OFF by default and should stay a workaround, not the fix —
it worked only because it deleted the local aggregate initialiser. With the compiler fixed, the
non-static path is the one to validate and ship.

## !!!! ROOT CAUSE FOUND: AGGREGATE-INITIALISER TAIL PADDING IS ADDRESSED WITH `addi`, STRIPPING THE TAG

**Minimal reproducer, no board, no SQLite, no monitor: 4552-byte domain, 8 array elements.**
`capstone/tests/runtime-qemu/silicon-ladder/strarray_app.c` (+ `strarray_host.c`, oracle 420).
Run with `DOMAIN_OPT_LEVEL=-O0 bash run-ladder-qemu.sh strarray` — fails in about a minute:

    qemu-system-riscv64: op_helper.c:655: helper_cscincoffsetimm: Assertion `rs1_v->tag' failed.

### The generated code

For each element of a local `struct fd { const char *z; void *p1, *p2; unsigned char f; }`
(three capabilities at 0/16/32, `f` at 48, **15 bytes of tail padding**):

    sb    a4, 112(a5)      ; store the flags byte
    mv    a0, a0
    addi  a0, a0, 49       ; <-- INTEGER add computes &tail_padding
    jalr  a3               ; memset(dest, 0, 15)

`addi` is integer arithmetic: applied to a capability register it produces an **untagged
scalar**. That scalar is passed to `memset` as the destination, and `memset`'s `p++` is
`cincoffsetimm` on an untagged base — which QEMU asserts on and the RTL does not check.

**The bug: the aggregate initialiser's tail-padding zero-fill computes its destination address
with `addi` instead of `cincoffsetimm`, so the capability tag is stripped before the pointer is
used.** The correct lowering is `cincoffsetimm a0, a0, 49`, which preserves the tag.

### Why this is the blocker

* **On silicon there is no check.** `SPLIT`/`LDC`/`STC` all validate their operands and raise
  exceptions, but nothing in the RTL requires a tagged `cincoffset`/`cincoffsetimm` base. So the
  untagged pointer is used, `memset` writes 15 bytes **through a garbage address**, and
  execution continues. That is a silent memory corruption once per array element.
* It needs **only a struct with tail padding in an aggregate initialiser** — which is exactly
  what `sqlite3RegisterBuiltinFunctions` builds, and exactly the R-14 shape.
* It is **size-independent** (N=8 through 56 all fail) and **not** register pressure, matching
  the controls.

### What this retires

* The blocker is a **compiler bug, not an RTL/silicon defect**. Every "the board miscomputes"
  claim in this document should be re-read with that in mind.
* It explains the corrupted string pointers without any wrong-cursor mechanism, and the
  wrong-cursor measurements were themselves invalid (see the INVALIDATED section).
* `SQLITE_STATIC_BUILTINS=1` works **because it removes the local aggregate initialiser
  entirely** — stage 10 with it returns clean under QEMU (`rc=0x00`, 0 asserts). It is a
  genuine workaround, but it treats the symptom; other struct-array initialisers with tail
  padding remain affected.

### Next

Fix the lowering (tail-padding memset destination must be `cincoffsetimm`), then re-run the
QEMU gate and the ladder. The reproducer above is the regression test.

## !!! ROOT CAUSE CANDIDATE: THE ARRAY CONSTRUCTION ITSELF DOES `cincoffset` ON AN UNTAGGED REGISTER

This reattributes the blocker from silicon to **our codegen**.

### The bisection

Stage 103 was added for exactly this question: it builds the probe array and returns `n`,
touching **no pointer** — no `lcc`, no inline asm, no pointer arithmetic in the probe. Under
QEMU:

    sel=103  (build array, return n)      -> ASSERT   helper_cscincoffset: rs1_v->tag
    sel=104  (delta via integer casts)    -> ASSERT
    sel=105  (neighbour, integer casts)   -> ASSERT

**103 asserting settles it.** The fault is in the ARRAY CONSTRUCTION, not in the probe's read
path. The instrument was also broken (0a, previous section), but fixing the instrument does
not make the assert go away, because the array is built before any measurement happens.

The QEMU helper is unambiguous (`op_helper.c:615-640`):

    assert(rs1_v->tag);                       // the BASE of cincoffset must be tagged
    capaddr_t offset = rs2_v->tag ? ... : rs2_v->val.scalar;   // the OFFSET may be a scalar

So the generated code performs capability arithmetic whose **base** has no tag.

### Why this explains everything the "wrong cursor" story explained, and better

* **QEMU asserts; the RTL does not check.** `SPLIT`, `LDC` and `STC` all validate operands and
  raise exceptions, but nothing in the RTL requires a tagged `cincoffset` base — so on silicon
  the instruction produces a value and execution continues with a **garbage pointer**. That is
  precisely the "pointer into the merged string blob is wrong" symptom, without needing any
  silicon defect at all.
* **It is a compiler bug, not an RTL defect.** Everything attributed to "the board miscomputes
  the cursor" should be re-examined under this hypothesis first.
* **It is consistent with the non-monotone N-dependence** (N=48/52/60 clean, N=56 bad at entry
  55, 0a9): if the untagged base arises from register allocation / spilling, whether it happens
  at all — and at which element — depends on register pressure, which does not vary monotonically
  with array size.

### CONFIRMED by controls: the array block is the trigger, and it is size-independent

    stage 0    (same staged harness, 100-105 block NOT compiled, no array)  RETURNED rc=0x00, asserts=0
    stage 103  PROBE_FD_N=8                                                 ASSERT
    stage 103  PROBE_FD_N=32                                                ASSERT
    stage 103  PROBE_FD_N=48                                                ASSERT
    stage 103  PROBE_FD_N=56                                                ASSERT

Two things follow, and both matter:

* **The staged harness is exonerated.** Stage 0 runs the identical entry/return/marker path
  with no probe array and is clean, so the fault is not in the staging machinery, the runtime
  selector, or the shared-region write.
* **It is NOT register pressure and NOT a size threshold.** N=8 asserts exactly like N=56.
  This kills the spill hypothesis that the disassembly scan suggested, and it also means the
  earlier non-monotone stage-94 pattern (N=48/52/60 "clean", N=56 "bad") was measuring
  something else entirely — those builds were all emitting the same broken construct.

**A straight-line local array of structs whose first member is a string literal makes the
compiler emit `cincoffset` with an untagged base, at any size from 8 elements up.** QEMU
catches it in about a minute with no board involved.

### Not yet established

* **Where** the untagged value comes from. A static scan of the `x100` disassembly found 6
  stack slots written only with `sw` and later read with `ldc`, which would lose the tag — but
  that scan resolves stack bases heuristically and may alias, so it is a LEAD, not evidence.
  The honest statement is: QEMU proves the base is untagged; the mechanism producing it is
  unidentified.
* Whether the same construct appears in real `sqlite3RegisterBuiltinFunctions` codegen, or only
  in the probe's synthetic array.
* An N-sweep (8/32/48/56) is running to test whether the assert is register-pressure driven.

### Immediate consequence

If this reproduces in a small standalone case, it is a **self-contained LLVM bug report** that
needs no board, no monitor and no FPGA — which is a far better artefact than anything this
campaign has produced so far, and it is reproducible under QEMU in ~60 seconds.

## !! INVALIDATED: THE STAGE-100 "CURSOR IS OFF BY 57 BYTES" MEASUREMENT

**The probe that produced it is malformed.** Run under QEMU, the *original* `x100.dom` — the
exact binary that returned `0x09` on the board — reaches `SQ: G/enter` and then dies on:

    qemu-system-riscv64: target/riscv/op_helper.c:627:
      helper_cscincoffset: Assertion `rs1_v->tag' failed.

i.e. the probe performs `cincoffset` on an **untagged** register. Verified pre-existing, not
introduced by the runtime-selector change: the pre-change binary straight out of the staged
overlay does it too.

### Why this invalidates the measurement rather than merely annotating it

QEMU *asserts* on capability arithmetic against an untagged value; the RTL does not check it
and simply produces a value. So on silicon the probe computed and returned **whatever
untagged `cincoffset` yields**, and `0x09` is consistent with garbage from the probe's own
read path rather than with the array slot's cursor. The two explanations are not
distinguishable from the data we have, and the probe cannot arbitrate between them because
the probe is the thing that is broken.

Consequently the following, all built on stages 95-102, are **withdrawn as measurements**:

* "the cursor delta is off by −57 bytes" (0a3) — the headline direct measurement;
* "the cursor's raw low byte is `0x00` where it must end in nibble 2" (0a) — same probe family;
* "the bad slot holds a valid capability with correct bounds but a wrong cursor" — the tag and
  bounds readings come from the same `lcc` sequence in the same broken block.

What survives is only the *stage 94* family, which returns an index rather than doing
capability arithmetic: N=48/52/60 clean, N=56 reproducibly bad at entry 55 (0a9). That
remains the strongest evidence that something is genuinely wrong with the array, and it is now
the ONLY surviving evidence of it.

### What has to happen before any cursor claim is made again

1. Find the untagged `cincoffset` in the stage 100-102 block and fix it — the probe must run
   clean under QEMU before it is trusted on silicon. **QEMU-gate every probe from now on**;
   these were built and shipped to the board without ever being run under QEMU, which is how a
   broken instrument produced four sessions of "measurements".
2. Re-run the delta and neighbour-control probes with the fixed instrument.

This is the most expensive error in this thread: it was not a misread of a log or a stale
figure, it was trusting an instrument that had never been checked against the reference
implementation. The check took one QEMU run.

---

## BOARD-SIDE DEGRADATION 2026-08-03 00:31-01:00 — four stages, ending with GDB not starting

Not a domain problem and not a firmware problem. Recorded so the next session recognises it
instead of re-debugging the compiler.

    00:31   load_image=2  downloaded=0  SQ_after=6    ran, produced some output
    00:37   load_image=2  downloaded=0                JTAG transfer moved nothing (silent)
    00:41   load_image=2  downloaded=1  SQ_after=0    transferred OK, board never reached userspace
    00:47+  load_image=0  gdb_timeout=1               gdb_start() times out; load_image NEVER issued
    00:53-01:00  6/6 attempts NO-TRANSCRIPT           same, after a full power-off + unlock + hold

`ActionTimeout: timed out waiting for event 'gdb_state'` means the debug session never comes up,
so no image is ever transferred. That is upstream of the driver: retrying only re-times-out.

**A full board reset does NOT clear it.** Tried explicitly: take the lock, power OFF, hold 45 s,
release the lock, disconnect, wait 30 s, reconnect. The bitstream still reads correctly
(`working-caplifive-captype-fixed.bit`) and the console lock still works — only the GDB/OpenOCD
side is dead. Consistent with the on-board debug stack (OpenOCD/FTDI) being wedged, which a
power toggle of the FPGA does not restart.

**Remedy is physical/owner-side**: restart the debug adapter (USB replug) or whatever restarts
the gdb server behind the console. Earlier in the same session `LIBUSB_ERROR_NO_DEVICE` appeared
directly, so the adapter has dropped off USB at least once tonight.

### How to tell these apart quickly (all are NOT results)

    gdb_state timeout          -> debug session never started; load_image count is 0
    LIBUSB_ERROR_NO_DEVICE     -> adapter gone from the USB bus
    load_image but no download -> transfer silently moved nothing
    download but no `buildroot login` / `SQ:` after load_image -> board never reached userspace

`board-watchdog.sh` now aborts the last two within ~15 s (`NO-BOOT`), keyed on the login prompt
rather than the bootrom banner — healthy runs show ZERO banners after their own `load_image`,
because the banner prints at power-on.

## SUMMARY — current best understanding (2026-08-03, rewritten end of session)

Three separate failures. Keep them apart; earlier drafts conflated them repeatedly.

| # | Failure | Where it stops | Status |
|---|---------|----------------|--------|
| 1 | **Rev-node pool exhaustion** | before `share1` (`pre-share`) | **SOLVED.** 1020-node bump allocator, no reclamation; ~182 splits/domain -> 5.5 runs/boot vs measured 6/5/5/5 |
| 2 | **`SHA5` entry stall** (R-16) | monitor hands off; domain never returns from its FIRST entry | **OPEN, attribution NOT established.** Intermittent; ~10x worse at slot 2 than slot 1 |
| 3 | **The SQLite blocker** (= R-14) | passes both shares, reaches `G/enter`, wedges in the main run | **OPEN, but LOCATED:** `sqlite3RegisterBuiltinFunctions` |

### Established, control-validated (each taken WITHIN one boot, control returning alongside)

* **The SQLite blocker is `sqlite3RegisterBuiltinFunctions`.** Confirmed 3x independently,
  the last on a freshly reflashed board with rebuilt firmware:
  `f10:0 = rc0 | f10:9 = rc0 | f10:10 = WEDGE`. Stage 9 does real allocator work
  (memsys5 zone headers across 256 KB) and returns, so the delta is that one function.
* **R-14 variants A and B wedge**, each with `:0` returning in the same boot.
* **Minimal repro:** four straight-line assignments of distinct string literals into a
  two-capability struct array — ~10 lines, no SQLite. `r14a_app.c` / `r14b_app.c`, board
  selectors `:110` / `:111`.
* **C-16 — a REAL compiler bug, found and FIXED today.** `SelectionDAG::getMemset` typed its
  destination argument in addrspace 0, truncating a 128-bit capability and stripping the tag.
  Fixed by taking the AS from `DstPtrInfo`; ladder 6/6; board-free reproducer `strarray_app.c`
  (oracle 420). **C-16 is NOT the SQLite blocker** — the blocker survives it.

### Mechanism: what the RTL rules out, and the one live candidate

* **`cincoffset` does NOT consume a linear `rs1`** — REFUTED from `capstone_flu_unit.anvil:29-68`;
  `rs1` passes through unchanged. That consume exists only in QEMU's helper.
* **`stc` rejects a non-zero immediate through an UNINIT destination** (`dyn_unit:378`) and
  **nulls the stored capability through an UNINIT destination** (`:54`). This fits the
  struct-vs-scalar split exactly (struct fields -> non-zero store immediates; separate scalars ->
  all `imm=0`). **Counter-evidence:** UNINIT is produced only by `REVOKE`, and stage 9 passes
  while doing non-zero-immediate stores — so the precondition is unproven.
* Discriminating test built and QEMU-gated: minimisation ladder `140`-`146`.

### Refuted — do not revisit

* "The ceiling is SPLB"; "the SPLB fix caused the SHA5 stall"; "SQLite needs 1059 carves";
  "stage 10 and the probe stall are one fault"; "the reproducer has a 49-byte unaligned stride"
  (`sizeof`=64, `_Alignof`=16); size/threshold theories; stages 11-15 as evidence (pre-date the
  unaligned-copy fix); "cincoffset consumes a linear rs1".
* **The stage-100 cursor measurements (-57 bytes etc.)** — the probe itself did untagged
  capability arithmetic. Withdrawn.
* **Everything recorded 21:00-22:33 on 2026-08-02** — our own watchdog matched a `SHA5` from
  replayed console scrollback and killed runners before boot. 13/13 checked runs have 0 SHA
  markers after their own `load_image`. That window's "board stopped accepting images",
  "initramfs bloat refuted" and "R-16 is build-dependent" all fall with it; those questions
  return to **unknown**, not answered.

### Working rules (learned expensively)

* **Run `:0` first on every staged image.** Free in any staged image (`if (stage <= 0) return 0;`
  is outside every `#if`). Control returns -> that boot's results are evidence; control wedges ->
  discard them all.
* **Classify before recording.** `SHA5`-without-`SHA6` = entry stall (domain never ran);
  `LIBUSB_ERROR_NO_DEVICE` = JTAG adapter gone; `__CAPSTONE_INFRA_FLAKE__` = QEMU boot flake.
  None are results. The runner's "FIRST FAILURE" line collapses them into "did not return".
* **Never scan a whole board log.** The console replays the previous boot's scrollback on
  connect; always split at the run's own `load_image` / `booted once`.
* **Firmware rebuild order is `A=linux-rebuild` THEN `A=opensbi-rebuild`.** Buildroot does not
  track the overlay->cpio dependency; skipping the first step relinks around a stale initramfs
  and the image does not change at all.
* **Never edit a running script**; never wait on a sentinel without watching the producer PID
  (`wait-for.sh`); never reuse an output filename (a stale transcript reads as a live result).

---

## 0aa. TODAY'S POSITION-1 RUNS DO NOT MATCH THE HISTORICAL SLOT-1 RATE

Three position-1 attempts on 2026-08-02, after the SPLB revert:

    sqlite_silicon.dom   passed both shares, reached SQ: G/enter, then silent
    st10.dom             STALLED at SHA5:00000000  (first entry)
    x101.dom             STALLED at SHA5:00000000  (first entry)

Two stalls in three attempts, against a corpus base rate of **2.8%** for slot-1 stalls over
107 historical launches (0a10). Under that rate, 2-of-3 has probability ~0.2%. So slot 1 is
**not** behaving today the way it behaved across the corpus.

That is a real discrepancy and it is NOT explained by anything established so far:

* it is not the SPLB fix — that is reverted, and `st10`/`x101` stalled *after* the revert;
* it is not the pool — these are the FIRST domain in their boots, `head` is fresh;
* it is not position — position 1 is as good as it gets.

**Candidates, none tested:** the firmware rebuilt today differs from the historical one in
ways beyond the monitor (kernel/initramfs regenerated twice); the corpus base rate is pooled
over many different domain builds and may not apply to these particular ones; or the board
itself is in a different state (it has been power-cycled far more times today than on any
previous day).

**Consequence for planning:** `x101` has now failed to execute in **6 of 6 attempts** across
three sessions, so the store-vs-load question remains unmeasured and should not be assumed
answerable cheaply. Anyone picking this up should budget several boots for it, or design a
probe that answers the same question from a domain that *does* reliably enter — the full
`sqlite_silicon.dom` is currently the only build that has entered reliably at position 1.

## 0ab. NO STRUCTURAL SIGNATURE SEPARATES STALLING BUILDS FROM ENTERING ONES

Checked, because "the probe builds have more/bigger globals so cap-init has more to do" is the
obvious explanation and it is wrong:

    build            carves   .text      merged_strs (symbols / bytes)   behaviour
    sqlite_silicon     179    1307584    6 / 21211                       ENTERS (pos 1)
    wd71               182    1321068    5 / 19965                       ENTERS (reliably)
    st10               181    1320392    5 / 19965                       STALLED at SHA5
    x100 / x101        181    1328644    5 / 20235                       STALLED at SHA5

`wd71` and `st10` have **identical** merged-string totals (19965 bytes, 5 symbols) and nearly
identical `.text`, yet one enters reliably and the other stalled. Carve count does not separate
them either — 179 enters, 182 enters, 181 stalls.

So the amount of cap-init work is NOT what decides it, and there is no cheap static predictor
of which build will stall. Do not spend another session looking for one on these axes.

### Design consequence: make the probe a RUNTIME choice, not a build-time one

Every probe today is a separate binary (`-DCAPSTONE_SQLITE_STAGE=N`), so each measurement
re-enters the stall lottery with a *different* image, and `x101` has now lost that lottery 5
times running. The fix is to stop varying the binary:

* build ONE domain — ideally `sqlite_silicon.dom`, the only build that has entered reliably at
  position 1 — and have it read a probe selector out of the shared region at entry, then
  dispatch to the requested measurement;
* the host already writes the shared region, so selecting a probe costs no rebuild, no
  firmware relink, and no new image;
* a boot that enters can then run *several* probes in sequence rather than one, which also
  sidesteps the under-two-domains-per-boot limit from 0a10.

This is the single highest-leverage change available for the measurement campaign, and it is
untried. It does not fix the blocker; it makes the blocker measurable.

## 0. READ FIRST — WEDGES ARE POSITION-DEPENDENT: ~6 DOMAIN RUNS PER BOOT, THEN IT WEDGES

Measured by running the TRIVIAL control (`wd71`: one walk, one return, no SQLite) repeatedly
inside one boot, four boots:

    boot1: 6 correct then WEDGE
    boot2: 5 correct then WEDGE
    boot3: 5 correct then WEDGE
    boot4: 5 correct then WEDGE
    ---------------------------------------------
    21 correct, 4 wedges over 25 runs

**This is NOT a uniform 16% failure rate.** Positions 1-5 are **21 successes, 0 failures**; the
wedge lands at position 6 or 7 in EVERY boot. A random 16% process would give wildly varying
run lengths (1, 12, 3, 8...). The regularity means **something is exhausted after ~6 domain
runs in a boot**, and the domain under test does not matter — these were 25 runs of the same
trivial image.

### The rule this gives, and it is simple

* **A wedge at position <= 5 in a boot is MEANINGFUL** — background failure there is 0/21.
* **A wedge at position >= 6 is SUSPECT** — that is where the control fails too.
* **Never put the domain under test late in a batch.** Put it FIRST after a single control.

### Which recorded results this affects

Re-checked across every batch log. Results whose first wedge landed at position >= 6, i.e. in
the exhaustion zone, and which therefore prove nothing about the domain:

    sqlite-fence.txt   pos 6   fn60.dom
    sqlite-n69b.txt    pos 6   wd69.dom
    sqlite-rep.txt     pos 6   wd52.dom
    sqlite-wdb.txt     pos 6   wd54.dom

Everything that wedged at position 2-5 STANDS, including the blocker: `wd10`/`mt10` wedged at
**position 2** in five separate boots (`boot1`, `boot2`, `boot3`, `goal2`, `mcause`,
`ra-mt10`), where the control has never failed. **The stage-10 blocker is real.**

### A caveat on the obvious cross-check

Tabulating "wedge position" across MIXED batches is circular: a wedge ends the session, so the
wedged domain is always last by construction. Only the identical-domain repetition above gives
usable position data. (An earlier draft of this section reported a flat "16% background rate"
from that circular view and over-generalised; corrected here.)

### RETRACTED: the ceiling is NOT SPLB. (That conclusion read REPLAYED HISTORY.)

An earlier entry claimed "all four rate boots end with `SPLB:0000E006`, so the background wedge
is the exact-fit region spin". **That was wrong.** Splitting each log at the runner's
`booted once` marker:

    board-rate1..4.log:  SPLB before 'booted once' (replayed history) = 1
                         SPLB after  'booted once' (the actual run)   = 0

Every SPLB occurrence was console history replayed on connect, not output from the run. This is
the accumulated-buffer trap already documented in section 9 — hit a second time in one day, on
a conclusion rather than a result.

**Independent confirmation that the ceiling is not SPLB:** the SPLB fix was enabled and verified
active in the built firmware (`0xe007` present, `0xe006` absent from the regenerated
`sbi_capstone_dom.c.S`), and the control STILL wedged at run 6:

    with the fix:  ok ok ok ok ok W    (5 correct, wedge at run 6, ZERO SPLB in the run segment)

So the SPLB fix is a genuine defect fix — the exact-fit spin no longer happens — but it does
NOT remove the ~6-run ceiling.

### What the ceiling actually looks like

Last markers of the RUN segment (history excluded), both with and without the fix:

    ... SQ: G/enter  SQ: H/return  SQ: X/fail  SQ: obs  SQ: A/dom  SQ: id  SQ: libc
        SQ: self  SQ: B/mkregion1  [SQ: C/mkregion2]   <-- ends here

The previous domain completed normally (`G/enter` -> `H/return`), and the NEXT domain dies in
**host-side region creation**, between `B/mkregion1` and `C/mkregion2` — with **no monitor tag
at all**: no SPLA, no SPLB, no RGNO, no SHAB. That is a different site from every named one.

Note `SQ: X/fail` appears before each new domain starts: the host reports the previous run's
marker as a failure because a control returning `0x45` is not the host's expected success value.
That is cosmetic and unrelated.

### Next

Instrument the host's `mkregion1`/`mkregion2` path (the ioctl that creates a region) rather than
the monitor — the monitor never gets to report, so the failure is on the Linux/host side or in
the ioctl entry before any capstone_report site. Reading `dmesg`/driver state after the wedge
would be the cheapest next probe, and needs no monitor change.

### What is exhausted after ~6 runs

Unknown, and now the sharpest question in this document — it is measurable with no SQLite in
the picture at all. Monitor-side candidates that grow monotonically per domain run: region ids
(`rgid` observed climbing 12 -> 17 -> 23 -> 25 -> 29 within one boot), domain ids,
`CAPSTONE_MAX_REGION_N` array slots, CPMP register slots, and rev-node allocations. Read
`region_n` / `dom_n` at the wedge, or instrument the monitor to report them per run.



### Consequences — these apply to EVERY result in this document

* **Any single-sample wedge means nothing.** Most wedges recorded here are single samples by
  construction (a wedge ends the board session), so most "X wedges" entries are consistent with
  pure background.
* **Every A-passes/B-fails pair is suspect**: `wd66`/`wd85`, `wd77`/`wd78`, guarded vs
  unguarded, stage 85 vs 86, the ballast ladder. All were single samples per image.
* **The measured unit must be a RATE**, with n reported. "X failed" is not a result; "X failed
  k of n" is.
* **The stage-10 blocker is probably still real**: it wedged in 3 of 3 separate boots, and at a
  1/6 background rate that is p ~ 0.5%. But its rate has never been measured either, and it
  deserves the same treatment.
* A control that PASSES still proves the board booted and the image loaded — that use remains
  valid. A control that passes does NOT prove the following wedge is meaningful.

### What this explains

The "unexplained build-to-build sensitivity" running through this whole document — identical
logic behaving differently in different binaries, guarded vs unguarded, with and without
padding — is, at least in part, this background rate sampled once per image. The mechanisms
proposed and retracted (walk count, first-walk anomaly, accumulator loss, cap-init threshold,
layout, instruction placement) were all inferred from exactly that kind of single-sample
comparison.

### The one experiment that should precede all others

Measure the background rate properly: run `wd71` alone, many times, across several boots, and
report k/n. Everything else in this document should then be re-stated as a rate against that
baseline, and anything not exceeding it should be struck.

## 1. The blocker in one paragraph

SQLite does not run on the FPGA. The failure is inside `sqlite3RegisterBuiltinFunctions`:
staged probes show stages 0/1/7/8/9 returning `rc=0` (entry+return, `sqlite3_config(HEAP)`,
MutexInit, MallocInit/memsys5, PcacheInitialize all work), while **stage 10 wedges in 3/3
separate boots** — the only wedge in this campaign established across multiple boots rather
than from a single sample. A "wedge" means the domain emits no marker and the board session
must be torn down.

## 0a. FIRST DIRECT READING OF THE BAD SLOT: tag INTACT, cursor WRONG  `ARCHIVED 2026-08-02`

* INVALIDATED — the probe that produced these cursor readings was itself doing
  `cincoffset` on an untagged register (see the INVALIDATED section).
* Full text: `history/02-08-2026_18-05-00_ARCHIVED_superseded-blocker-sections.md`

## 0a2. BOUNDS LOOK RIGHT, CURSOR IS WRONG — the fault is the offset, not the derivation  `ARCHIVED 2026-08-02`

* INVALIDATED — same broken probe; and per RTL, bounds are DERIVED from the cursor,
  so "bounds look right" was never independent evidence.
* Full text: `history/02-08-2026_18-05-00_ARCHIVED_superseded-blocker-sections.md`

## 0c. BUILT AND STAGED, NOT YET RUN (console went down mid-experiment)  `ARCHIVED 2026-08-02`

* STALE — those staged experiments have since been run or superseded.
* Full text: `history/02-08-2026_18-05-00_ARCHIVED_superseded-blocker-sections.md`

## 0a6. THE SHA5 WEDGE IS POSITION-DEPENDENT, AND POSITION 2 REGRESSED BETWEEN AUG 1 AND AUG 2

### Measured: a SQLite-derived domain at position 1 enters cleanly

Running `sqlite_silicon.dom` FIRST, with no control ahead of it, it passed **both** shares
(`SHA5:00000000` / `SHA6:00000000` on each) and reached `SQ: G/enter` (MEASURED, 2026-08-02).

That settles the fork left open in 0a5: explanation **(a)** holds. SQLite-derived domains do
NOT wedge in `share1` as such — they wedge in `share1` **at position 2**. The domain content
is not what decides it.

### And position 2 has not always been broken — it regressed

The Aug-1 baseline, re-checked from the logs rather than memory:

    boot1, boot2, boot3, goal2   wd10.dom at POSITION 2   -> reached SQ: G/enter
    mcause                       wd10.dom at POSITION 2   -> reached SQ: G/enter
    ra-mt10                      mt10.dom at POSITION 2   -> reached SQ: G/enter

Six of six, all at position 2, all passing both shares. Against Aug-2:

    z100_2, z101_1, z101_2, z102_1, z102_2, static/st9   position 2  -> stopped at SHA5

So position 2 was healthy on Aug-1 firmware and is broken on Aug-2 firmware.

### The prime suspect is a regression I introduced

The one firmware change in that window is the **SPLB exact-fit fix** in the monitor
(`sbi_capstone.c`, mtime 2026-08-01 22:42), which on an exact fit does `region_n -= 1`
instead of spinning. It sits in the region path that `share` consumes.

Worth stating plainly: that fix was written to cure the ~6-run ceiling, and the "the ceiling
is SPLB" conclusion was **later retracted** as a misreading of replayed console history. So
the likely situation is a fix for a non-problem that broke domain slot 2.

**This is not yet established.** The correlation is temporal and **confounded**: the Aug-2
domains are different builds from the Aug-1 ones. Two things that do NOT support it:

* region ids do not discriminate — `r1=17 r2=19` appears in both a working run (`z100_1`) and
  three wedged ones;
* the `INTERP_DOMAIN_MTVEC` glue gate, the other change in the window, is **OFF in every
  built domain** (0 `csrw mtvec` occurrences in `x100`, `st9`, `wd71`) — checked, not assumed.

**The clean discriminator is one load**: run the *Aug-1* `wd10.dom` (still on disk) at
position 2 on *today's* firmware, monitor otherwise unchanged. Reaches `G/enter` → the
firmware is fine and the Aug-2 domains differ. Stops at `SHA5` → today's monitor is the
regression, and reverting it restores ~6 usable domain slots per boot.

### Practical consequence, independent of the cause

Until this is resolved, **put the domain under test at position 1**. Every slot-level probe
result that cost six boots was paid for by a convention — control first — that is now known
to be the expensive part.

## 0a7. FULL SQLITE AT POSITION 1: ENTERS, THEN SILENT — AMBIGUOUS, NOT YET A WEDGE

Run 2026-08-02, `sqlite_silicon.dom` built with `SQLITE_STATIC_BUILTINS=1`, at position 1:

    SQ: A/dom-ok ... B/mkregion1 ... C/mkregion2 ... D/mapped ... E/share1 ... F/share2
    SQ: G/enter
    <silence for the full 300 s budget>

Then the per-domain timeout fired and took the session, so `st10` and `st9` never ran.

**Do not record this as a wedge.** Silence between `G/enter` and the first row is exactly the
stretch that `run_sqlite_baked_fpga.py:126-135` warns about: since the linear-safe string
primitives landed the core EXECUTES there, and a previous run was aborted on that stretch and
nearly read as a wedge. 300 s may simply be too short on silicon for open + CREATE + INSERT.

The two are distinguishable and neither has been done for this build:

* `probe_sqlite_wedge.py` with `PROBE_STEPI=1` — an advancing pc means alive, not dead;
* simply raising `SQLITE_STAGE_TIMEOUT` well past 300 s.

### The real cost problem this exposes

With position 2 broken, **a boot yields exactly one domain result**. The staged-probe method
— whose entire value is N hypotheses per boot — is currently delivering N=1, which is the
regime CLAUDE.md's batch-variants rule exists to prevent. Restoring position 2 is therefore
worth more than any single further probe, because it is a ~5x multiplier on every experiment
after it.

Action taken: firmware rebuilt with `CAPSTONE_SPLIT_EXACT_FIT` commented out (both monitor
copies edited, generated `.c.S` removed so the wrapper actually regenerates). The next load
puts `st10` at position 1 — so the R-14 test still lands even if the revert changes nothing —
followed by `st9` and `wd71` at positions 2-3 as the regression check, and full SQLite last
with a much longer budget.

## 0a8. PROVENANCE: STAGES 11-14 ARE A *RESOLVED* BUG — DO NOT RE-THEORISE ON THEM

Every staged marker ever returned was re-decoded and traced back to its log and position.
The string-literal stages tell a clean before/after story that is easy to misread if you only
look at the numbers:

    stage 11  rc=0x01  pos=1  sqlite-stages111213.txt   strlen("capstone_probe_string") = 1
    stage 12  rc=0x01  pos=2  sqlite-stages111213.txt   same, via a local struct slot
    stage 13  rc=0x07  pos=3  sqlite-stages111213.txt   4 names, sum of strlens (expect 36)
    stage 14  rc=0xf0  pos=1  sqlite-stage14.txt        bitmap of s[1..8] non-zero
    stage 14  rc=0xf0  pos=2  sqlite-stage14.txt        (control domain, same result)
    ---- the unaligned-copy fix lands here ----
    stage 14  rc=0xff  pos=1  sqlite-unalign.txt        sqlite_stage14fix.dom -- ALL BYTES GOOD
    stage 13  rc=0x24  pos=1  sqlite-burst.txt          = 36 decimal -- CORRECT

`0xf0` means `s[1..4]` were zero while `s[0]` and `s[5..8]` were fine — a four-byte hole, not
the "a little copied, rest zero-filled" tail that the stage-14 comment predicted. That hole
was the **unaligned copy in the entry glue's blob copy**, and it is FIXED: the same probe
returns `0xff` afterwards, and stage 13 returns the correct 36.

**Consequence for anyone reading the numbers table:** stages 11-15 are pre-fix artefacts.
They are not evidence about the current blocker and a theory built on them is a theory about
a bug that no longer exists. The live evidence for the wrong-cursor fault is stages 95-102
only, all of which post-date the unaligned-copy fix.

(Recorded because the raw marker table lists `stage 14 rc=0xf0` and `rc=0xff` side by side
with no indication that a fix separates them, which is exactly the shape of the three
summary-vs-artefact errors made earlier in this thread.)

## 0a9. THE N-DEPENDENCE IS NON-MONOTONE, AND THE REPRODUCER MAY NOT MODEL SQLITE FAITHFULLY

### Stage 94 across array sizes (index of first bad entry; 0xFF = none bad)

    N=48   idx48.dom  rc=0xff  none bad          pos 2
    N=52   ix52.dom   rc=0xff  none bad          pos 2
    N=56   idx56.dom  rc=0x37  entry 55 = LAST   pos 3  }  same domain, two positions,
    N=56   idx56.dom  rc=0x37  entry 55 = LAST   pos 4  }  same answer -- REPRODUCIBLE
    N=60   ix60.dom   rc=0xff  none bad          pos 3

**N=56 fails and N=60 passes.** So this is NOT a size threshold, and any "arrays larger than
X break" statement is refuted by its own data — consistent with the threshold retractions
already recorded in section 1a. Whatever selects entry 55 of a 56-entry array does not select
entry 59 of a 60-entry one.

Note also that the two N=56 runs sit at positions 3 and 4 and agree exactly, which is a
useful independent check: the wrong-cursor result is stable across position even though the
SHA5 *entry* wedge is position-sensitive. The two faults really are separate (0a4).

### RETRACTED IMMEDIATELY: "the reproducer's struct has a 49-byte unaligned stride"

I read `addi a0, a0, 0x31` in the per-entry code as the array stride, concluded the struct
was 49 bytes with byte-aligned capability slots, and wrote that the reproducer might be
exercising an unaligned-store path SQLite never takes. **That is wrong.** Asked directly,
the compiler says (MEASURED, `aligncheck.c` compiled with the real target flags):

    sizeof(struct probe_fd) = 0x40 = 64      _Alignof(struct probe_fd) = 0x10 = 16
    sizeof(void *)          = 0x10 = 16      _Alignof(void *)          = 0x10 = 16
    sizeof(struct two_ptr)  = 0x20 = 32

The ABI aligns capabilities correctly and the struct is 64 bytes: three capabilities at
offsets 0/16/32, `flags` at 48, 15 bytes of tail padding. `0x31` is 49 = the address just
past `flags`, computed as the argument to the `jalr` on the very next instruction — a
tail-padding zero-fill call, not an array advance. The real stride never appears as an
immediate; each entry's base comes from a per-entry offset loaded from a spill slot
(`ldc a7, -0x790(a1)`, `-0x780`, ...) and added with `cincoffset`.

So there is **no alignment concern**, the reproducer is not disqualified on those grounds,
and no aligned-struct variant is needed. What survives from this section is only the
non-monotone N-dependence above, which stands on its own.

(Fourth correction in this thread, and the first one caught *before* acting on it rather than
after — the check that caught it was compiling a five-line file instead of reasoning about an
immediate.)

## 0a10. THE WHOLE CORPUS, TABULATED: SLOT 2 STALLS ~10x MORE OFTEN THAN SLOT 1

Every domain launch in every run-scoped log was re-parsed (274 launches), classified by its
last marker, and bucketed by position. This replaces every impression-based statement about
position in the sections above.

    pos 1: n=107   RETURNED=99 (93%)   SHA5-stall= 3 (2.8%)   entered-no-return=5
    pos 2: n= 96   RETURNED=48 (50%)   SHA5-stall=31 ( 32%)   entered-no-return=17
    pos 3: n= 30   RETURNED=28         SHA5-stall= 1          entered-no-return=1
    pos 4: n= 22   RETURNED=16         SHA5-stall= 3          pre-share=2
    pos 5: n= 13   RETURNED=11         SHA5-stall= 2
    pos 6: n=  6   pre-share=6 (100%)

### What is sound here, and what is not

**Sound:** positions 1 and 2 are both *unconditioned* — every boot reaches them — so the
comparison is fair. **The second domain launched in a boot stalls at `SHA5` about ten times
more often than the first** (32% vs 2.8%). That is far too large to be noise at n=96/107.

**NOT sound:** positions 3-5 look healthy, but they are **survivorship-biased**. A stall ends
the session, so a position-3 sample exists only in boots where position 2 already succeeded.
Their low rates say nothing, and any claim of the form "only slot 2 is bad, slot 3 is fine"
is unsupported by this table. An earlier draft of this section drew exactly that conclusion
before noticing the conditioning.

**Position 6 is a different failure entirely**: 6 of 6 fail *before* `share1` (`pre-share`),
which is the rev-node pool exhaustion of 0a4, not the `SHA5` stall.

### Consequences

* `SHA5` stalls are **not exclusive** to slot 2 — 3 of 107 happen at slot 1 (`wk9`, `wd55`,
  `wk0`). So "run it at position 1" reduces the failure rate roughly tenfold; it does not
  eliminate it. Any single position-1 result still needs a repeat before it is load-bearing.
* Correspondingly, the section 0a6 framing "the SHA5 wedge is position-dependent" is right in
  direction but was stated too strongly from a handful of runs. The honest version is a
  10x rate difference between slot 1 and slot 2, with a residual ~3% floor at slot 1.
* Because a stall ends the boot, the *expected* number of usable results per boot is roughly
  1 + 0.68 + ... — i.e. under two. That, not the 1020-node ceiling, is what actually limits
  throughput today.

## 0a11. REFUTED: THE SPLB EXACT-FIT FIX IS NOT THE CAUSE — THE MONITOR IS EXONERATED

The A/B test ran. `CAPSTONE_SPLIT_EXACT_FIT` was commented out in **both** monitor copies, the
generated `.c.S` was deleted so the wrapper really regenerated, and the rebuild was verified
from the regenerated assembly: the split path now emits `SPLB`/`0xe006` (the original spin)
and the `EXACT_MID` path is gone. `diff` against the Aug-1 backup shows only 3 hunks, all
SPLB, all now inert — i.e. the monitor is functionally what ran on Aug 1.

Result (2026-08-02, reverted firmware):

    pos 1   st10.dom   last marker = SHA5:00000000     STALLED

**The stall survives the revert.** So the SPLB exact-fit fix is NOT the cause of the `SHA5`
stall, the "position 2 regressed because of my monitor change" hypothesis from 0a6 is
**refuted**, and the monitor is exonerated for this failure.

Recorded as a hypothesis killed by its own test, which is the cheap outcome — it cost one
firmware rebuild and one boot, and it removes the most plausible-looking suspect.

### What the same run says about position

`st10` stalled at **position 1**, where the corpus base rate for a `SHA5` stall is 2.8%
(0a10). One sample is not proof, but hitting a 1-in-35 event on the first attempt is at least
suggestive that this particular build is not drawing from the same distribution — i.e. that
`st10` is genuinely more prone to stalling than the corpus average, rather than unlucky.

Note this sits awkwardly beside the other static-builtins result: `sqlite_silicon.dom`, built
the same way, **passed both shares at position 1** and reached `G/enter`. So "static builtins
stalls in cap-init" is NOT a clean rule either. Both are n=1 at position 1 and both need
repeats before anything is built on them.

### The open mechanism question this sharpens

The static-builtins workaround does not remove the straight-line array construction so much as
**relocate** it: as a `static`, `aBuiltinFunc` becomes a global whose capability leaves are
written by `__capstone_cap_init` at the domain's FIRST entry — which is `share1`, exactly where
`st10` now stalls. Under the old non-static build the same data was built on the stack during
the main run, and `wd10`/`mt10` stalled after `G/enter`. That is a suspicious coincidence:

    non-static builds  -> stall AFTER G/enter (main run)
    static build st10  -> stall AT share1     (cap-init, first entry)

If that holds up, the R-14 workaround moves the fault rather than fixing it, and QEMU passing
end-to-end says only that QEMU does not enforce whatever the RTL enforces. **Not established:**
`sqlite_silicon` is the counter-example above, and no repeat has been run.

## 0a12. RTL FACT: THE CURSOR IS EXACT, THE BOUNDS ARE DERIVED FROM IT — SO "BOUNDS OK" PROVES NOTHING

From `ariane_pkg.sv:714-748` (SOURCE):

    function automatic fat_cap_t decompress_cap(input xlen_t cursor, input xlen_t metadata);
      return '{ metadata: decompress_cap_metadata(metadata, cursor), cursor: cursor };

    function automatic fat_cap_metadata_t decompress_cap_metadata(metadata, cursor);
      ...
      bounds = decompress_bounds(metadata.bounds, cursor);   // <-- bounds depend on cursor

Two consequences, and the second one costs us an argument we were relying on.

**1. A wrong cursor is NOT compression rounding.** The cursor is carried in full 64 bits and
is not part of the compressed metadata; only the bounds are compressed, CHERI-style. So the
tempting explanation — "the container is big enough that the cursor gets rounded to a
granule, and −57 bytes is inside one granule" — is wrong at the format level. Checked before
being written down anywhere as a conclusion.

**2. "The bounds are consistent with the container" is not independent evidence.** Sections 0
and 0a report the bad slot as having *correct bounds* but a *wrong cursor*, and treated the
correct bounds as showing the capability was otherwise intact. But the bounds are
**reconstructed from the cursor at read time**. For a small cursor error the decoded bounds
come out unchanged by construction, so `lcc` zimm=3/4 readings cannot corroborate anything —
they are a function of the cursor we already know is wrong.

What survives: the *cursor* readings (raw low byte `0x00`; delta `0x09` where `0x42` is
correct) and the *tag* being intact (`lcc` did not trap). What does not survive: any claim
that the bounds were independently verified as correct. Sections 0a and 0b should be read
with that correction.

## 0a5. THE SHA5 WEDGE TRACKS POSITION-2 + SQLITE-DERIVED, AND R-14 HAS A QEMU-GREEN FIX

### The R-14 workaround now exists and passes QEMU

`sqlite3RegisterBuiltinFunctions` builds the exact wedging shape because
`build-sqlite-capstone.sh:75` strips `static` from

    static FuncDef aBuiltinFunc[] = { ... }

turning a compile-time-initialised global into a **stack array constructed straight-line at
run time**, then copied element-wise into a separate static. That de-static is ours, not
upstream SQLite's.

`SQLITE_STATIC_BUILTINS=1` (`build-sqlite-silicon.sh:94`) puts it back. Verified this session:

* patch applies cleanly — 0 surviving `capstoneBuiltinFunc` references, copy loop removed;
* carve count **unchanged at 179** (it removes one zero-init static and adds one initialised
  one, so the pool budget is unaffected);
* **QEMU silicon config passes end-to-end: `__CAPSTONE_SQLITE_SILICON_PASSED__`** (MEASURED).

It is still **unproven on the board** — that is the run in flight.

### The SHA5 wedge correlates with position 2 AND with being SQLite-derived

Tabulating every domain launch whose stopping point is known:

    position 1   wd71 (trivial control, 182 carves)     returns 0x45     never failed
    position 2   x100 / x101 / x102 / st9 (SQLite-derived, 179-181 carves)
                                                        stopped at SHA5  6 of 7

The single exception is `x100` in boot `z100_1`, which completed a full run at position 2 and
returned `0x09`. So it is not a hard rule, but 6/7 is far from noise.

Two explanations remain, and they are cleanly separable:

* **(a)** position 2 is bad for these domains, or
* **(b)** SQLite-derived domains wedge in `share1` regardless of position, and `wd71` is fine
  only because it is trivial.

Nothing measured so far distinguishes them, because **a SQLite-derived domain has never been
run at position 1** — every batch put the control there. That is a gap created by the
batching convention itself, not by the hardware.

**The discriminator is one run**: put the SQLite domain FIRST, with no control ahead of it.
Under (a) it enters; under (b) it stops at SHA5 exactly as at position 2. That run doubles as
the deliverable, since a full `sqlite_silicon.dom` that enters and prints its rows IS the
existence proof.

## 0a4. RETRACTION + the rev-node pool arithmetic, now CONFIRMED against the RTL

### RETRACTED: "the five failures died in region-share between B/mkregion1 and C/mkregion2"

That is what section 0a3 says and it is **wrong**. It came from the runner's coarse
`entered=False` classifier, not from the logs. Re-reading the five run-scoped captures, every
one of them gets much further, and all five stop at the **same** point:

    z100_2, z101_1, z101_2, z102_1, z102_2   last line = "SHA5:00000001"   (5/5 identical)

Five identical stopping points is not a capture artifact. And `SHA5` is decisive
(SOURCE, `sbi_capstone.c:111` and `:1020-1026`):

    #define CAPSTONE_TAG_SHA5 ... "about to leave M-mode for the domain"
    #define CAPSTONE_TAG_SHA6 ... "the domain returned from the share entry"
    capstone_trace(CAPSTONE_TAG_SHA5, dom_id);
    d = __domcallsaves(d, CAPSTONE_DPI_REGION_SHARE, r);
    capstone_trace(CAPSTONE_TAG_SHA6, dom_id);

The comment at that site anticipated exactly this reading: "SHA5 followed by silence means the
domain never came back and the monitor is exonerated". So:

* **The monitor is exonerated.** It completed the whole share and handed off.
* **The wedge is INSIDE the domain**, on the domain's *first* entry — `E/share1` is an entry,
  not merely a table update — i.e. while the entry glue runs its carve loop and cap-init,
  well before the main `G/enter` run.

The earlier "no monitor tag at all" description belongs to the separate ceiling failure between
`B/mkregion1` and `C/mkregion2`; conflating the two was the error.

### CONFIRMED (SOURCE + MEASURED): the ~6-runs-per-boot ceiling is rev-node pool exhaustion

Every `split` allocates a revocation node, the allocator is a monotonic bump with no
reclamation, and the pool is 10 bits:

    capstone_dyn_unit.anvil:135   send rev_node_ep.init_req(rs1.metadata.revnode_id)   <- SPLIT
    capstone_rev_node.anvil:77-78 set node_id := #{20'd0,*head}; set head := *head+10'd1;
    capstone_rev_node.anvil:160   set head := 10'd3          (start)
    capstone_rev_node.anvil:168   reg head : logic[10]       (10 bits)
    capstone_rev_node.anvil:217   if *head == 10'd1023 { overflow_flag := 1'b1 }

Only `drop_req` touches a node and it merely clears `valid` (`:61-68`) — it never lowers
`head`. So a boot has **1023 − 3 = 1020** allocations before `head` wraps and starts reusing
live ids.

Measured carve counts (`.capstone_gp_initdesc` header `count`, offline):

    wd71.dom  182     x100.dom 181     x101.dom 181     x102.dom 181

The glue builds the table on the first entry only (`start-gp-captable-interp.S:275-280`,
idempotent entry), so a domain run costs ~182 splits plus a handful of monitor allocations:

    1020 / ~185  =  5.5 domain runs per boot

and the measured ceiling was **6, 5, 5, 5** over four boots (21 correct / 25 runs). The
arithmetic and the measurement agree. This was previously INFERRED as R-12; it is now
confirmed from both ends.

Two consequences that matter more than the ceiling itself:

* **It does NOT explain the position-2 probe wedges.** After one control domain `head` is only
  ~190 of 1020. Those failures are a genuinely different fault, and per the retraction above
  they are in-domain on first entry.
* **RETRACTED within the hour: "full SQLite needs 1059 carves and overflows the pool".**
  1059 is the count *without* string merging. The silicon build enables
  `-capstone-merge-string-constants=true` by default (`build-sqlite-silicon.sh:210`), and the
  actually-staged domain measures **179 carves**:

        179 carves   .../overlay/test-domains/sqlite_silicon.dom      (MEASURED, offline)

  179 of 1020 is not close to the pool limit. **Pool exhaustion is NOT the SQLite blocker**,
  and trimming the carve count is not the fix. I asserted the opposite from a stale figure in
  a source comment instead of measuring the artefact — the same mistake, in the same file,
  that the comment at `:186` records having cost "a long detour" once already.

  I then wrote that SQLite (179 carves) sitting in the same regime as the probes (181) was
  "corroboration that the SQLite blocker and the position-2 probe wedge are one fault, not
  two". **That is also withdrawn** — checked against the logs immediately afterwards, the two
  stop in *different places*:

      stage 10 (wd10/mt10)   share1 SHA0..SHA6, share2 SHA0..SHA6, ECSZ, "SQ: G/enter", then
                             silence  -- wedges in the MAIN run, having completed both shares
                             (6 boots: boot1, boot2, boot3, goal2, mcause, ra-mt10 -- all 6
                             identical, DETERMINISTIC)

      probes x100/101/102    stop inside the FIRST share at "SHA5:00000001", never reaching
                             SHA6 -- and NOT deterministic: x100 completed the whole run in
                             one boot and wedged at SHA5 in another, same image, same position

Different entry, different determinism. Equal carve counts say only that both are in the same
regime; they are not evidence of a shared fault, and the differing stopping points are weak
evidence *against* one. Treat them as two open faults until something actually links them.

**Three corrections in a row on this thread** (Family A classification, the 1059 figure, this
one), all from asserting past the evidence rather than reading the artefact. The pattern is
worth naming: each one came from a *summary* — my own classifier, a source comment, a carve
count — where the primary log or binary was one command away.

## 0a3. DIRECT MEASUREMENT: the cursor is off by 57 bytes — and Family A now blocks the work  `ARCHIVED 2026-08-02`

* INVALIDATED — the "cursor is off by 57 bytes" headline came from the broken probe.
* Full text: `history/02-08-2026_18-05-00_ARCHIVED_superseded-blocker-sections.md`

## 0b. NEAR-MISS WORTH RECORDING: mcause=24 does NOT imply the code under test trapped

Attempting to read the bad slot's capability type (`lcc` zimm=1 on `arr[55].zName`), the domain
wedged and the cleared trap latch showed **`mcause = 24` (UNEXPECTED_OPERAND)**. `LCC` raises
exactly that when its operand is `NOT_CAP` (`capstone_dyn_unit.anvil:171-173`), so the obvious
reading was "the stored pointer lost its tag" — the hypothesis stated in advance.

**That reading is wrong.** Checking the marker trail for that domain:

    w95   reached G/enter = FALSE   H/return = FALSE
          last markers: SHA2 ... BASE:815FF000 ALEN:00001000 SHA3 SHA4 SHA5

It died in the REGION-SHARE path, before the domain ran. The `lcc` never executed. The
`mcause=24` is a **Family-A** fault (section 5) — the same signature as `pad73`/`pad74` — and
says nothing about the slot.

**Rule, since this nearly produced a false root cause:** `mcause=24` is ambiguous between
Family A (region-share, before `G/enter`) and any in-domain `UNEXPECTED_OPERAND`. **Always
confirm `SQ: G/enter` appears for that domain before attributing a trap to the code under
test.** This is the third time in this campaign that a marker-trail check overturned a
conclusion drawn from a register reading.

The slot-content question — pointer wrong / tag lost / data wrong — therefore remains OPEN.
`w96` (cursor) and `w97` (data via the container) never ran, because the wedge ended the
session.

## 1. WHAT IS ACTUALLY ESTABLISHED (audited 2026-08-02; read this before any ladder below)

An adversarial audit re-read the raw logs rather than this document, and the ladder as
previously written mixed two different probe programs and under-sampled every point except one.
Corrected table, with sample counts:

| N | evidence | verdict |
|---|---|---|
| 48 | `g48` 48/48 correct; `idx48` -> `0xFF` | PASSES (2 builds) |
| 52 | `ix52` -> `0xFF` | passes (1 build, 1 run) |
| **56** | `g56` 55/56 (one bad); `idx56` first-bad=55 **twice**; `d56` WEDGED | **FAILS — 3 independent builds, 2 different probe programs, 3 boots** |
| 60 | `ix60` -> `0xFF` | passes (1 build, 1 run — UNREPLICATED) |
| 64 | `g64` **RETURNED** (rc=0xC0); `ix64` **WEDGED** | **CONTRADICTORY — 2 builds, opposite outcomes** |
| 72, 96 | wedged | 4 builds |

**Robust:** N=56 corrupts exactly one entry — the last — reproducibly, across independent
builds and probe programs. That is the finding.

**NOT established:** any size law. "N=64 wedges" was `ix64` only; `g64` returned. The
non-monotonic "56 fails / 60 passes" inversion rests on ONE unreplicated N=60 run. Both the
"threshold" story and the "non-monotonic" story are under-sampled.

**Supported instead:** the same N can give opposite outcomes in different builds (N=64, two
builds). Outcome tracks the BUILD, not the entry count.

### Control reliability (measured, corrects an earlier alarm)

`wd71` across all logs: **60 returns, 2 wedges**. The earlier "~1-in-6 background rate" was an
artefact of measuring inside the exhaustion zone; at slots 1-5 the control is highly reliable,
so a failure at slots 1-3 IS meaningful. All ladder data sits at slots 1-3 and is uncontaminated.

### Ruled out offline by the audit (do not re-propose)

* **Bounds-compression unrepresentability.** `compress_bounds`/`decompress_bounds` were
  reimplemented verbatim from `ariane_pkg.sv:656` and `:749` (including the cursorless branch
  and the +-2^(E+14) corrections) and round-tripped for all five builds' `(stor, off)` pairs over
  every 16-aligned base in a 1 MiB window: **zero failures**. For these lengths `E=0` and the
  encoding is exact.
* **An "index >= 32 switches to runtime cincoffset" discontinuity.** `idx48` entries 32..47 use
  that form and all read back correctly.
* **A layout/mod-32 predictor.** One was found that fit all five stage-94 builds
  (`total mod 32` = 16,16,0,16,0 -> PASS,PASS,FAIL,PASS,WEDGE) and was then **refuted by the
  audit itself** against the six stage-92 builds already on disk (g4 and g32 have `total%32 = 0`
  and are correct; g72 has 16 and wedges).

### There is NO instruction-level difference at the failing entry

Disassembled across all five builds: the last entry's code is the same shape AND the same frame
offset (`s0-0x4700`) in every one; only the string's container offset changes, by exactly
`container_size - 5` as the container size requires. The reader is identical too. So whatever
differs is address/layout or timing — **not code**.

## 1a. RETRACTION FIRST: there is NO size threshold. The failure is BUILD-dependent.

Filling in the ladder destroyed the threshold claim recorded in 1b:

    N=48   ALL CORRECT
    N=52   ALL CORRECT
    N=56   index 55 BAD        (2/2 in one boot -- deterministic FOR THAT BINARY)
    N=60   ALL CORRECT         <-- passes, ABOVE the size that fails
    N=64   WEDGED
    N=72   WEDGED
    N=96   WEDGED  (3 builds)

**Non-monotonic: 56 fails, 60 passes.** So "correct at N<=48, fails above ~48" is WRONG and is
withdrawn, as is the "threshold between 48 and 72". Size is not the determinant.

What the data actually supports is the pattern seen all session: **a given BUILD either works or
does not, deterministically for that build, and size only loosely correlates.** `idx56` failing
2/2 in one boot is real for `idx56`; it says nothing about "N=56" as a class, because `ix60`
with more entries is fine.

**Everything in 1b that depends on a threshold is therefore unsupported**, including the
inference that SQLite's 72 entries put it "above the threshold". The SQLite link is now only:
`RegisterBuiltinFunctions` builds this shape, and this shape fails in some builds.

### What DOES survive, and it is still the best artefact in this campaign

* A local array of N structs with distinct string literals can leave its LAST entry's `zName`
  unreadable while every earlier entry is fine (`idx56`: first bad index 55, 2/2).
* That is a WRONG ANSWER, not a wedge, so it can be sampled repeatedly within a boot -- the only
  failure in this campaign with that property.
* A validated control exists (`idx48` -> `0xFF`, all correct) and the probe reports an index, an
  encoding that cannot alias correct with incorrect.

### What this means for the method

Per-N single samples cannot establish a size effect while build-to-build variation is
uncontrolled. To claim any threshold, each N needs several INDEPENDENT BUILDS (e.g. vary a
no-op) and a failure RATE per N -- not one build per N. That is a large amount of board time
and should be planned as such rather than inferred from a ladder.

## 1b. ROOT CAUSE LOCALISED: a straight-line struct array of ~72 entries wedges the core  `ARCHIVED 2026-08-02`

* SUPERSEDED — the real root cause is C-16 (memset destination typed in AS0),
  plus a separate, still-open silicon-only fault (R-14).
* Full text: `history/02-08-2026_18-05-00_ARCHIVED_superseded-blocker-sections.md`

## 2. Root cause  `ARCHIVED 2026-08-02`

* SUPERSEDED — this said "NOT FOUND". A root cause WAS found: C-16. A second,
  silicon-only fault remains open.
* Full text: `history/02-08-2026_18-05-00_ARCHIVED_superseded-blocker-sections.md`

## 3. Refuted BY MEASUREMENT — do not revive without new evidence

| hypothesis | how it died |
|---|---|
| `cincoffset` consumes its source | SOURCE: `capstone_flu_unit.anvil:43,:62` return `rs1` unchanged |
| `STC` clears its source register | SOURCE: `capstone_dyn_unit.anvil:427` returns `rs2_v` unchanged |
| carve / rev-node pool exhaustion at entry | MEASURED: 183 carves vs ~1000 budget |
| `LDC` consumes its memory slot | MEASURED: stage 57/58 = 7 (two reads, both non-NULL and equal) |
| the SHA5 wedge is self-inflicted | MEASURED: UNGUARDED `wd51` returned `0xB1`, unchanged |
| array identity ("the Nth array is broken") | MEASURED: `wd60/61/62`, one shared array, only the multi-walk shape failed |
| granule/carve-base misalignment is the cause | MEASURED: `ga60 = 0xC1`, identical with granule-aligned glue |
| "the first data-dependent walk fails" | MEASURED: `wd66 = 2` inverts it; `wd71` bare walk passes 3/3 |
| store ordering / missing fence | MEASURED: `fence rw,rw` before `domain_main` changed nothing |
| binary layout | MEASURED: passing and failing binaries have identical carves, symbol vaddr, and the SAME 21 loop instructions |
| instruction placement | MEASURED: +24/+56 byte padding, identical failure |
| walk COUNT (1 ok / 2 partial / 3 wedge) | MEASURED: confounded — two of the four "3-walk wedges" never entered the domain, and `wd63` runs FOUR walks and RETURNS |
| the dyn unit is blocked on a rev-node query | MEASURED: `wrev=1`/`memwait=1` are ALSO set in the healthy control (`0xd5`); they are resting state |
| rev-node allocator exhaustion at the wedge | MEASURED: `head=413`, `overflow=0` (healthy: 222) |

## 4. Established and reproducible

* **Livelock, not a hang, for at least one probe.** Stage 51 returns `0xB1` — the domain runs
  and RETURNS. MEASURED.
* **The emitted pointers are correct.** `__capstone_cap_init` derives literals at
  `0x6da/0x6e0/0x6e6` — deltas of exactly 6 — across 1544 straight-line instructions with zero
  calls/branches; the one reused register is correctly spilled and reloaded. SOURCE
  (disassembly). Note: proves what is EMITTED, not runtime values.
* **`wd66` is a deterministic reproducer** (7 samples, all `2`): same element walked twice
  through the same pointer, first walk overruns, second terminates; the two loops are
  byte-identical (23 instructions each). MEASURED.
* **`wd71` is a deterministic control** (6+ samples, all `0x45`). Use it in every session.
* **Results are NOT always reproducible.** `wd63` returns `0x0E` and `0x0F` on identical
  back-to-back runs in one boot. Any single-sample conclusion is unsafe. MEASURED.

## 5. TWO wedge populations — never merge them again

MEASURED across every board log:

    sw=225   sw=255                        n    what fails
    0x84     0x98 = trap_seen=1 mcause=24  12   dies in REGION-SHARE, never enters the domain
    0x95     0x89 = trap_seen=1 mcause=9   13   dies INSIDE the domain (mcause 9 = stale entry ECALL)
    0xd5     0x8f                           1   HEALTHY (wd71 returned)

`mcause 24` is a real capability exception (`UNEXPECTED_OPERAND`, `capstone_unit.anvilh:289-291`;
cause `= 23 + code`, `cva6.sv:1357`). So **capability faults DO latch** — the instrument works,
it was pointed at the wrong family. Family A is a genuine fault taken with `mtvec = 0`, hence
silent. Family B latches no new trap.

Every "the blocker wedges N times" count written before 2026-08-01 mixes these.

## 6. Current hypotheses, ranked  `ARCHIVED 2026-08-02`

* SUPERSEDED — hypothesis 2 (capability-compression aliasing) is refuted: the RTL
  carries the cursor in full 64 bits and only the bounds are compressed.
* Full text: `history/02-08-2026_18-05-00_ARCHIVED_superseded-blocker-sections.md`

## 7. CONTESTED — do not cite either side as settled

Does a 256-byte global's capability really span >= 1 MiB?

* MEASURED (stage 77, 2/2): `lcc` zimm 3/4 gave `end - start >= 1 MiB`.
* SOURCE: the carve is exactly 256 bytes (`start-gp-captable-interp.S:446-449`; `SPLIT` narrows
  the parent in the same instruction, `capstone_dyn_unit.anvil:140-144`), and ordinary `lbu` IS
  bounds-checked (`load_store_unit.sv:970-971`, cause 28).

Both attempts to settle it (`wd78`, `wd79`) WEDGED. **Settling test, no new instrumentation:**
rerun stage 76 with offset `1024*1024 + 512` instead of `1024*1024`. A fault confirms
compression aliasing; another `0x77` supports the over-grant reading.

## 8. Real defects found along the way (report separately; none is proven to be THIS bug)

* **No timeout/abort on rev-node queries.** `get_node_query_validity`
  (`capstone_dyn_unit.anvil:106-112`) is `send >> recv` with no abort; `get_rev_node`
  (`capstone_rev_node.anvil:36-41`) likewise blocks on `recv mem_ch.read_res`. Any unanswered
  query is an unrecoverable machine hang by construction.
* **`REVOKE_NODE` walks unbounded** — no visit limit, no cycle detection; only exits on
  `depth <= depth_bound`, and an invalid node does not stop it (`capstone_rev_node.anvil:13-34`).
  If it parks, every later query hangs.
* **The rev-node allocator wraps silently.** 10-bit bump allocator, no reclamation; overflow
  drives only a debug LED (`cva6.sv:1185,1652`).
* **Carve base granule misalignment.** idx 170 (`sqlite_heap`, 256 KB, granule 512),
  `base%g = 64`, `len%g = 0`. Simulation: granule-align OFF -> 1 unrepresentable carve, ON -> 0.
  The 2026-07-31 revert note had the failing END backwards. Knob `INTERP_GRANULE_ALIGN=1`.
* **QEMU is STRICTER than the silicon on ordinary loads.** QEMU keeps fat capabilities with
  exact bounds and checks ordinary loads (`trans_rvi.c.inc:286-292` -> `op_helper.c:1107`), so
  spatial violations that land on an alias boundary pass on silicon and trap in emulation. Also
  `RISCV_EXCP_CAP_OOB` (`cpu_bits.h:697`) is defined and never raised — QEMU's OOB `mcause` will
  not match the RTL's 28.
* **`mtvec = 0` in domains** means an in-domain fault has no handler and cannot print. Upstream
  design question; not to be patched unilaterally.

## 8b. RETRACTED — the "cap-init store threshold" was a FOUR-WAY CONFOUND

**The 1222..1263 threshold is WRONG. Do not cite it. Do not rebuild on it.**

Verified against the primary logs and the monitor source, not argued:

**(a) `pad200` never ran cap-init at all.** In `board-pad.log` its last marker is
`SQ: C/mkregion2SPLB:0000E006` — it died during REGION CREATION, with no `D/mapped`, no
`E/share1`, no `G/enter`. In `board-bis.log` the last domain (`SQ: id=4`) shows
`SHA5:00000004` and no following `G/enter`. In neither run did `__capstone_cap_init` execute a
single `stc`. A threshold in cap-init stores cannot be measured by a domain that never reaches
cap-init.

**(b) The failure site is a DELIBERATE MONITOR SPIN, not silicon.**
`sbi_capstone.c:494-504`:

        if (base + len == region_end) {
            if (base == region_base) {
                // matching region. We don't support this for now
                capstone_report(CAPSTONE_TAG_SPLB, CAPSTONE_ERR_SPLIT_EXACT);
                ... capstone_uart_flush();
                while(1);

`CAPSTONE_TAG_SPLB` = "split_out_cap: exact-fit region unsupported" (`:44`),
`CAPSTONE_ERR_SPLIT_EXACT = 0xe006` (`:58`). The predicate is a HOST MMAP ADDRESS COINCIDENCE —
the requested region exactly matching an existing one. It is an UNIMPLEMENTED CASE in our own
monitor, upstream of the domain and upstream of cap-init.

**(c) Store count does not determine the outcome.** Direct counterexample, measured on the
artifacts: `wd77`, `wd78`, `wd79` all have **exactly 1048** cap-init stores and are three
distinct binaries (md5 `9c9bd9c3…`, `fbaef16d…`, `c59b396a…`). `wd77` RETURNS; `wd78` and
`wd79` WEDGE. Same count, opposite outcomes.

**(d) The ladder confounded pad count with SEQUENCE POSITION.** `pad200` was always run LAST and
always had the highest domain id (`SQ: id=3`, then `id=4`). Monitor region ids grow monotonically
through a boot (`rgid=12 → 17 → 23 → 25 → 29`). No low-pad build was ever run last, and `pad200`
was never run first.

**What actually survives:** nothing about cap-init store counts. The `sb0` "corroboration" (1257
stores, wedges at entry) is the same confound — it also fails before the domain runs.

**Method failure to learn from:** the ladder looked like a controlled experiment because ONE
variable was varied deliberately. Three others varied with it (image size, sequence position,
host allocation state), and the outcome was never checked against the marker trail to confirm
the domain even reached the code under test. **Always confirm the failure SITE (marker trail)
before attributing a wedge to the thing the probe varies.**

---

## 8c. What the same investigation established instead (offline, verified)

* **RTL fixed-capacity structures are ELIMINATED as the mechanism.** Verified inventory:
  scoreboard 8, store buffer 4/4, `MaxOutstandingStores` 7, wbuf 8, AXI MetaFifo 4, dcache 2048
  lines, icache 1024, rev-node pool 1024, dom-switcher 67, TLB/PMP 16. A sweep of `core/` and
  `corev_apu/` finds NO depth in 1222..1263 and no byte bound in the corresponding range.
* **Tag memory is ELIMINATED.** `ariane_pkg.sv:586-590` and `wt_axi_adapter.sv:148-149`:
  `tag_addr = TAG_MEM_BASE + ((paddr - DATA_MEM_BASE) >> 4)` — a pure function of physical
  address, with no counter, allocator or high-water mark, so store COUNT cannot advance it.
  The arithmetic also shows the shadow tag region ends exactly at `CAP_REVNODE_MEM_BASE`
  (`0xBC3C_0000 + ((0xBC3C_0000-0x8000_0000)>>4) = 0xBFFF_C000`), abutting the node table with
  zero slack — it cannot overflow into it.
* **A real codegen shape change exists but is on the WRONG side of the boundary.**
  `__capstone_cap_init` grows `0x2c98 → 0x41f4` and `ldc` count `111 → 1413` between `pad150`
  and `pad175`, i.e. the register allocator collapses into per-leaf spill reloads. But `pad175`
  is already fully in the new shape and RETURNS, so it is not the wedge boundary. Its real
  consequence is that the ladder rungs are NOT one monotone family.
* **NEW, and independently actionable: `split_out_cap` cannot handle an exact-fit region.**
  That is a genuine unimplemented case in our monitor (`sbi_capstone.c:496-504`) that hangs the
  board whenever the host allocator happens to hand back an exactly-matching region. It is
  almost certainly responsible for a share of the "random" wedges in this campaign, and it is
  OURS to fix, not the hardware's.


(HISTORICAL, RETRACTED — kept only so the reasoning can be audited.) Stage 80 is an
entry-and-return domain that touches one array element and returns. The claim WAS that the only
variable across these builds is how many capability leaves `__capstone_cap_init` must store
(`CAPINIT_PAD=N` adds N initialised pointers). No SQLite code runs at all — no strings, no
walks, no hash tables, no allocator traffic.

    domain    cap_init stores   result
    wd71            1048        rc = 0x45   (independent control)
    pad1            1015        rc = 0x61   RETURNS
    pad120          1134        rc = 0x61   RETURNS
    pad200          1263        WEDGED
    pad260          1381        (never ran -- the wedge ended the session)

**There is a threshold between 1222 and 1263 capability stores in cap-init.** (CONFIRMED in a
second, independent boot — see the bisection below.)

    pad120   1134   rc=0x61   PASSES
    pad150   1184   rc=0x61   PASSES
    pad175   1222   rc=0x61   PASSES
    pad200   1263   WEDGED    (wedged in TWO separate boots)

`pad200` wedging twice, in different sessions, clears the single-sample caveat: this is the
only wedge in the campaign reproduced across boots other than stage 10 itself.

Cross-check: `sb0` (STATIC_BUILTINS at stage 0) has **1257** stores and wedges AT ENTRY — inside
the same band, from a completely different source change. Two independent routes to the same
region.

This is the first mechanism in the campaign with a NUMBER attached and no SQLite logic in the
path, which also makes it the first that could be handed over as a hardware-side reproducer.

**Caveats, stated plainly:**
* `pad200`'s wedge is a SINGLE sample — a wedge ends the session, so it cannot be repeated
  within a boot. Confirm across separate boots before quoting the number.
* The bound is wide (1134..1263). Narrow it before reporting.
* It does NOT explain the in-domain (`0x95`) family on its own: `b10n0` wedges at stage 10 with
  only 1017 stores, below the passing 1134. So either there are two mechanisms, or store count
  is a proxy for something else (total leaf bytes, a specific leaf, an address pattern).
* 1024 — the rev-node pool size — is NOT the boundary: 1134 stores passes.

**Next:** bisect 1134..1263 with intermediate pads, repeat `pad200` across boots, and then ask
what is exhausted at that count. Candidates: a fixed-depth structure in the store path, a
tag-cache capacity, or total bytes rather than store count (vary leaf SIZE at constant count to
separate those two).

CORRECTION on that last point: with this probe design, store count and capability BYTES are
proportional by construction — every cap-init leaf is one 16-byte capability store — so they
cannot be separated by varying the pad. What the ladder DOES isolate is stores from CARVES: the
pad is a single array, i.e. ONE extra carve carrying N extra stores, and the carve count is
therefore constant across the ladder. The threshold is in the store count, not the number of
globals.

## 8d. Session close-out 2026-08-01 (post-reflash)

**Board reflashed and verified.** The resident NV bitstream read as `None` and the board came up
on a STOCK OpenPiton+Ariane design — no Capstone bitstream resident. Reflashed to
`working-caplifive-captype-fixed.bit` (90 s), verified by re-reading rather than trusting the
flash call, and it persisted across reconnect. **Check the resident bitstream at the START of a
session; do not assume continuity from the previous one.** The runners' hard-stop would have
caught it, but only after a wasted build.

**Post-reflash health is good:** `wd71` returns `0x45`.

**`wd66`'s decode is STILL UNVALIDATED.** `wd81` — the probe built to return the two raw guard
values instead of a bitmap — WEDGED, so the question stands: `wd66 = 2` has been read as "first
walk overran, second succeeded" on the strength of a bitmap decode nobody has checked, and
`rc = 2` is equally consistent with a clobbered accumulator. **Do not cite the first-walk anomaly
as established.** `wd81` differs from `wd66` only in trivial ways (loops back-to-back, clamp at
the end instead of a conditional between them) and wedges where `wd66` returns — another
instance of the unexplained build-to-build sensitivity.

**The SQLite blocker is NOT the SPLB monitor spin.** Every stage-10 run reaches `SQ: G/enter`
and then wedges in-domain, so it is a genuine Family-B failure. SPLB is a SEPARATE, ours-to-fix
defect (`sbi_capstone.c:496-504`, exact-fit region unsupported -> `while(1)`) that corrupted the
`pad200` results and plausibly other "random" wedges in this campaign.

**Operational error to avoid repeating:** a cleanup glob `sb*.dom`, written for the `sb0`/`sb10`
probes, also matched **`sbi.dom`** — a package-installed domain. Deleting those desyncs
buildroot's stamps and previously caused six consecutive boot failures. Recovered by clearing
`.stamp_target_installed` for `capstone-sbi-domain` and `capstone-test-domains` and rebuilding.
**Enumerate probe names explicitly; never prefix-glob in the staged tree.**

## 8e. MINIMAL REPRODUCERS (copy-pasteable)

All builds go through `build-stage-probes.sh`, which prints per-artifact hashes and a
distinct-hash count so a silently-cached build cannot pass as fresh. Common preamble:

    cd <REPO-ROOT>            # the llvm-capstone checkout
    source capstone/tests/capstone-test-env.sh
    export FPGA_URL="$(cat ~/.claude-c/secrets/fpga-console-url)"
    export FPGA_FW="$PWD/capstone/caplifive-system/sw/buildroot/build/build/opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin"

**Before any run:** confirm the resident bitstream is `working-caplifive-captype-fixed.bit`.
It was found to be `None` (stock OpenPiton+Ariane) at the start of a session.

### R1 — THE BLOCKER: stage 10 wedges (3/3 separate boots)

    PROBE_DEST=/tmp/capstone/repro bash capstone/benchmarks/sqlite/build-stage-probes.sh 10
    # stage into the initramfs, rebuild firmware, then:
    export SQLITE_STAGE_DOMS="/test-domains/wd71.dom,/test-domains/wd10.dom"
    export SQLITE_STAGE_TIMEOUT=180 PROBE_SCOPED_OUT=/tmp/capstone/r1.txt
    python3 capstone/tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py

Expected: `wd71` -> `rc=0x45`; `wd10` -> WEDGED, having printed `SQ: G/enter` (it DOES enter the
domain). Source: `sqlite_capstone_domain.c`, stage 10 = `sqlite3MallocInit()` +
`sqlite3RegisterBuiltinFunctions()`. Stage 9 (`MallocInit` + `PcacheInitialize`) returns `rc=0`.

### R2 — the deterministic wrong-answer reproducer (7 samples, all `rc=2`)

    PROBE_DEST=/tmp/capstone/repro bash capstone/benchmarks/sqlite/build-stage-probes.sh 66
    export SQLITE_STAGE_DOMS="/test-domains/wd71.dom,/test-domains/wd66.dom,/test-domains/wd66.dom"

Stage 66 walks `capstone_probe_lit[1]` ("rtrim") TWICE through the same pointer and returns a
2-bit map. Observed `rc=2` on every sample. **The decode is UNVALIDATED** — `2` reads as "first
walk overran, second terminated", but is equally consistent with a clobbered accumulator. The
two loops were verified byte-identical (23 instructions each, `0x36994` / `0x36a40`).
`wd81`, built to return the raw guard values instead, WEDGES.

### R3 — the health control (6+ samples, all `rc=0x45`)

    PROBE_DEST=/tmp/capstone/repro bash capstone/benchmarks/sqlite/build-stage-probes.sh 71

Stage 71 walks the same element ONCE and returns `0x40 | index-of-NUL` = `0x45`. Put it FIRST in
every batch; a value other than `0x45` means the session is bad, not the experiment.

### R4 — SPLB: our monitor hangs on an exact-fit region (OURS TO FIX)

Not yet reduced to a deterministic trigger — it fires when the host allocator returns a region
exactly matching an existing one, which is layout-dependent. Signature in the UART log:

    SQ: C/mkregion2SPLB:0000E006

Source: `sbi_capstone.c:494-504` — `if (base + len == region_end) { if (base == region_base) {
... while(1); } }`, tagged `CAPSTONE_TAG_SPLB` ("split_out_cap: exact-fit region unsupported",
`:44`) with `CAPSTONE_ERR_SPLIT_EXACT = 0xe006` (`:58`). Any wedge whose last marker is
`SPLB:0000E006` is THIS, not silicon — check for it before attributing a wedge to hardware.

### R5 — the wedge-triage procedure (do this for EVERY wedge)

1. Find the last UART marker for that domain. `SPLB:0000E006` -> R4, ours. `SHA5` with no
   `SHA6` -> region-share (Family A). `SQ: G/enter` with no `SQ: H/return` -> in-domain
   (Family B).
2. Read the debug mux at the wedge — the runner does this automatically (`sw=255`, `224`, `225`,
   `249`-`254`) and clears the trap latch before each domain.
3. **Compare against the HEALTHY values, never against other wedges:** healthy `sw=225 = 0xd5`,
   `sw=224 = 0xff`. `wrev`/`memwait` are SET in the healthy state — they are resting values and
   mean nothing on their own.
4. `sw=255` bit7 = trap_seen, bits[6:0] = mcause. `24` = a real capability exception
   (`UNEXPECTED_OPERAND`); `9` = a stale ECALL from domain entry, i.e. no new trap.

## 8f. RETRACTED: the "first walk overruns" anomaly. The walks were always fine.

Measured in ONE boot, on freshly-reflashed hardware:

    wd71  control                              rc = 0x45
    wd82  walk 1 ONLY                          rc = 0x45   guard = 5   CORRECT
    wd83  walk 2 (after a discarded walk 1)    rc = 0x45   guard = 5   CORRECT
    wd66  the two-walk bitmap probe            rc = 0x02

Both walks of `capstone_probe_lit[1]` ("rtrim") terminate at the NUL, index 5, exactly as they
should — walk 1 included. **`lit[1]` was never broken, and there is no first-walk anomaly.**
Every statement in this campaign of the form "the first walk overruns", "lit[1] never
terminates", or "walk N fails" is WITHDRAWN. They all trace back to `wd66`'s bitmap, which was
never validated.

### What `wd66` actually reproduces (still real, still deterministic)

With both walks provably correct, `wd66` should return `3`. It returns `2` on every one of 7+
samples. So the lost bit is the ACCUMULATOR update, not the walk:

    guard = 0; while (z[guard]) { if (++guard > 64u) break; }
    if (guard <= 64u) m |= 1u;      <-- this update is LOST (bit0 never set)
    guard = 0; while (z[guard]) { if (++guard > 64u) break; }
    if (guard <= 64u) m |= 2u;      <-- this one survives (bit1 set)

`guard` is demonstrably 5 at both points, so the predicate is true both times. A deterministic
loss of the FIRST read-modify-write to a local, with the SECOND surviving, is a much sharper
and smaller phenomenon than a string-walk failure — and it is a genuine miscompute, not a probe
artefact, because stages 82/83 confirm the inputs.

**Do not re-derive the walk story.** The next step on this reproducer is to instrument `m`
itself: return `m` after the first update only, and separately return `guard` and `m` from the
same domain, to establish whether the OR, the store, or the reload is what is lost.

### Why this went unnoticed for so long

`wd66` was deterministic (7 samples), which was mistaken for "trustworthy". Determinism only
means the probe reports the same thing every time; it says nothing about whether the ENCODING
is read correctly. The decode was never checked against a probe that returns the raw quantity —
and when one finally was built (`wd81`, both guards at once) it WEDGED, which delayed the check
further. Splitting it into one number per domain (`wd82`, `wd83`) settled it immediately.

## 8g. RETRACTED AGAIN: no accumulator bug either. The phenomenon is BUILD-TO-BUILD.

One boot, freshly reflashed hardware:

    wd71   control                                   rc = 0x45
    wd84   walk 1 + the FIRST update only, x2        rc = 0x71   m = 1   CORRECT
    wd85   the FULL wd66 sequence, re-encoded        rc = 0x73   m = 3   CORRECT
    wd66   the same sequence, its own encoding       rc = 0x02           WRONG

`wd85` performs exactly what `wd66` performs — walk, `m |= 1`, walk, `m |= 2` — and returns
**m = 3**, the correct value. So the first update is NOT lost, and the accumulator finding
recorded in 8f is **withdrawn**. `wd84` further shows the first update survives in isolation.

### What is actually left, and it has been present the whole time

**Identical C logic, built into a DIFFERENT BINARY, produces a DIFFERENT RESULT — and each
binary is internally deterministic.** The same pattern, repeatedly:

| pair | same source logic | outcome |
|---|---|---|
| `wd66` vs `wd85` | walk, `m|=1`, walk, `m|=2` | `2` vs `3` (correct) |
| `wd77` vs `wd78`/`wd79` | two `lcc` reads, 1048 stores each | returns vs wedges |
| guarded vs unguarded `wd52`/`wd53` | same probe | wedges vs returns |
| `wd71` vs `wd81` | one walk vs two, trivially different | returns vs wedges |

Every "mechanism" this campaign proposed — the `lit[1]` walk, walk count, the first-walk
anomaly, the accumulator loss — was a different sampling of THIS. Each looked specific because
the comparison was between two binaries, and the binary was never the controlled variable.

**Consequence for method:** a probe that differs from its control in ANY way other than the one
being tested is measuring this phenomenon, not the hypothesis. The only comparisons that have
survived scrutiny all session were within ONE binary (`wd60/61/62` on a shared array) or against
a healthy reading of the SAME register.

**Consequence for the blocker:** the stage-10 wedge may itself be an instance rather than a
distinct bug. It is reproducible across 3 boots, but it has never been compared against a
control binary that differs ONLY in the code under test.

### Next step on this thread

Characterise the build-to-build variation directly instead of chasing its symptoms: take ONE
source, build it N times with only a benign perturbation (e.g. an added no-op global that shifts
nothing semantically), and see how many of the N binaries misbehave. If a meaningful fraction
do, that is the finding, and it subsumes most of this document.

## 8h. The variation is SYSTEMATIC, not random — and padding is not the trigger

Six binaries from ONE source, differing only by N no-ops before the sequence, each run in its
OWN boot with the `wd71` control first:

    BALLAST=0    WEDGED     (control 0x45)
    BALLAST=4    WEDGED     (control 0x45)
    BALLAST=8    WEDGED     (control 0x45)
    BALLAST=12   no output
    BALLAST=16   WEDGED     (control 0x45)
    BALLAST=20   WEDGED     (control 0x45)

**5 of 5 measured binaries wedge.** The control returned `0x45` in every boot, so the board was
healthy and these are real. Yet `wd85` — which performs the SAME sequence (walk, `m |= 1`,
walk, `m |= 2`, return `0x70 | m`) — returns the correct `0x73`.

**This corrects the framing in 8g.** The phenomenon is NOT random build-to-build luck: five
DISTINCT binaries fail the same way. Something specific and deterministic about the stage-86
build fails, while the stage-85 build succeeds. Padding — i.e. pure code placement — is
**excluded**, which is the one thing this experiment does settle cleanly.

### The remaining difference between the two, and it is small

    stage 85:  ... first walk ...  if (stage == 85) { second walk; m |= 2u; }   -> RETURNS 0x73
    stage 86:  ... first walk ...  second walk; m |= 2u;                        -> WEDGES 5/5

Both execute both walks (the stage-85 predicate is true at run time). The difference is that
stage 85 reaches its second walk through a RUNTIME BRANCH while stage 86 falls into it
straight-line. That is the smallest delta yet between a passing and a failing domain, and both
sides are reproducible — `wd85` returned in its run, and stage 86 failed in five.

**This is the most tractable open lead in the document.** It is two builds of one function
differing by one conditional, with a known-good and known-bad side, no SQLite logic, and a
control that passes in the same boots.

### Next

1. Disassemble `wd85` and `bal0` side by side across the whole probe block — not just the loop —
   and diff. The delta should be small enough to read instruction by instruction.
2. Build a stage-87 that is stage 86 plus an always-true runtime branch around the second walk,
   to test the branch-vs-straight-line hypothesis directly.
3. Do NOT generalise from "5/5 wedged" to "all such code wedges" until at least one more
   passing/failing pair is characterised; the sample is one code shape.

## 8i. CORRECTION: the variation is INTRA-BINARY. "Deterministic per binary" is refuted.

    wd71  control                       rc = 0x45
    wd85  has the guard (known good)    rc = 0x73
    wd87  guard restored, run 1         rc = 0x73
    wd87  guard restored, run 2         WEDGED     <-- SAME BINARY, SAME BOOT

`wd87` both RETURNED and WEDGED from one binary in one boot. Two consequences, both correcting
entries above:

1. **"Restoring the branch fixes it" is NOT supported.** The analysis script printed that
   conclusion because it filtered `None` (wedge) out of the sample before testing whether every
   remaining value was `0x73`. A wedge is a RESULT, not a missing datum. The branch hypothesis
   is UNRESOLVED, not confirmed.
2. **"Each binary is internally deterministic" (8g, 8h) is REFUTED.** The same image gives
   different outcomes on consecutive runs in one boot. `wd63` (`0x0E` then `0x0F`, one boot)
   showed this earlier and was not generalised.

### What this means for everything above

Most of this document's history is single-sample comparisons between binaries. If the SAME
binary can both pass and wedge, then:

* every "binary A passes, binary B fails" pair — `wd66`/`wd85`, `wd77`/`wd78`, guarded vs
  unguarded, `wd71`/`wd81`, stage 85 vs stage 86 — may be sampling ONE nondeterministic
  process rather than a difference between the binaries at all;
* the "5/5 wedged" ballast result (8h) is consistent with a high failure PROBABILITY, not with a
  systematic property of stage 86;
* the only claims that survive are those measured with repetition on ONE binary.

**The correct unit of measurement is a RATE, not an outcome.** Any future claim of the form
"X fails / Y passes" must report n and the number of failures for each, from repeated runs of
the same image.

### Immediate next step

Take the two most-used images (`wd71`, which has returned `0x45` in ~10 runs, and `wd85`) and
run each 5-10 times within one boot to establish their failure rates. If `wd71` is genuinely
0/10 and `wd85` is 0/n while `wd87` is ~1/2, the rate differs by image and there is something to
explain. If everything shows a nonzero rate, the phenomenon is global and the entire
per-mechanism framing of this document is the wrong shape.

## 9. Instrument and method traps (all of these bit during this campaign)

1. **Never read a debug register only at the failure.** Read it at a SUCCESS first. Three of
   eight bits in `sw=225` are identical in healthy and wedged states; a "signature" seen at four
   wedges meant nothing.
2. **Never read `board-<tag>.log` for results** — it carries accumulated console scrollback.
   Only `PROBE_SCOPED_OUT` is valid.
3. **A wedge ends the session**, so a wedging domain CANNOT be repeated within one boot. Every
   wedging result is a single sample by construction. Repeat across boots, or build a probe that
   RETURNS a marker instead (the stage-51 watchdog is the model).
4. **A domain earns an early slot only if THAT EXACT BINARY has returned before.**
5. **Never wait on a process by name** — `pgrep -f <pattern>` matches the waiting command
   itself. Three deadlocks, ~50 minutes lost. Sequence steps in one script.
6. **`llvm-objdump --disassemble-symbols` silently truncates**; use `--start-address/--stop-address`
   and check the byte count against the symbol size.
7. **Every generated edit must assert its anchor matched** — a silent no-op `replace` produced a
   probe that did not compile.
8. **Stage N contains stage M for M < N** on the normal path; never order a superset before the
   subset it depends on.
9. Build probe batches with `build-stage-probes.sh` — it prints per-artifact hashes and a
   distinct-hash count, so a cached build cannot pass as fresh.

## 10. Next steps  `ARCHIVED 2026-08-02`

* SUPERSEDED — the BUILTIN_LIMIT clamp is no longer the recommended move; see the
  summary and the entry-stall sections.
* Full text: `history/02-08-2026_18-05-00_ARCHIVED_superseded-blocker-sections.md`

## Workaround attempt 1: clamp builtin registration (2026-08-01)  `ARCHIVED 2026-08-02`

* SUPERSEDED — the clamp workaround predates the C-16 root cause.
* Full text: `history/02-08-2026_18-05-00_ARCHIVED_superseded-blocker-sections.md`

