# ARCHIVED — superseded sections of SILICON-BLOCKER.md

Archived 2026-08-02. These were moved out of `ref/SILICON-BLOCKER.md` because they are
INVALIDATED, SUPERSEDED or STALE. They are kept in full because several record retractions
and dead ends that should not be re-derived — read them as history, never as current truth.

**Current truth lives in `ref/SILICON-BLOCKER.md` and `ref/ISSUES.md` (C-16, R-14).**

---

## WHY ARCHIVED: INVALIDATED — the probe that produced these cursor readings was itself doing
  `cincoffset` on an untagged register (see the INVALIDATED section).

## 0a. FIRST DIRECT READING OF THE BAD SLOT: tag INTACT, cursor WRONG

Reading `arr[N-1].zName` in the failing build (N=56), each probe in its own boot, and with
`SQ: G/enter` checked per domain so Family-A faults cannot be mistaken for results:

    stage 97  (data byte via arr[])   WEDGED, entered=FALSE  -> Family A, no information
    stage 96  (cursor, lcc zimm=2)    returned 0x00, entered=TRUE   <-- the one real reading
    stage 95  (cap type, lcc zimm=1)  WEDGED, entered=FALSE  -> Family A, no information

**Two conclusions from stage 96:**

1. **The tag is INTACT.** `LCC` raises `UNEXPECTED_OPERAND` when its operand is `NOT_CAP`
   (`capstone_dyn_unit.anvil:171-173`). Stage 96 executed `lcc` on that slot and RETURNED
   normally, so the slot holds a valid capability. **The "the store lost its tag" hypothesis,
   which was stated in advance, is REFUTED.**
2. **The cursor is WRONG.** Its low byte reads `0x00`. `"fn55"` sits at container offset
   3906 = `0xF42`, so an aligned container base should give a low byte of `0x42`.

So the failure is **not** a lost store and **not** a lost tag: a well-formed capability is
present in the slot, pointing at the wrong address. That is why dereferencing it yields a byte
that is not `'f'` rather than trapping.

### Caveats

* ONE sample, and only the LOW BYTE of the cursor. `0x00` is consistent with a zero cursor but
  could in principle alias (it would need the container base to end in `0xBE`).
* Stages 95 and 97 produced nothing — both died in region-share before entering. The data-side
  question (is the literal itself intact?) is still open.

### Next

Read more of the cursor and the bounds, one value per domain to avoid the two-value probes that
have wedged: cursor bytes 1/2/3 (`lcc` zimm=2 shifted), and `start`/`end` (zimm 3/4) to see
whether the capability's BOUNDS are also wrong or only its cursor. If bounds are right and only
the cursor is wrong, the fault is in the `cincoffset` that computes the container offset for
that entry, not in the capability's provenance.


---

## WHY ARCHIVED: INVALIDATED — same broken probe; and per RTL, bounds are DERIVED from the cursor,
  so "bounds look right" was never independent evidence.

## 0a2. BOUNDS LOOK RIGHT, CURSOR IS WRONG — the fault is the offset, not the derivation

Reading the bad capability field by field, one per domain, `SQ: G/enter` verified for each:

    CURSOR low byte   0x00    entered=True    (2 samples, both entered)
    END    low byte   0xE0    entered=True
    START             WEDGED  entered=False   -> Family A, no information

**Arithmetic (this is what makes a low byte usable):**

* The merged-string container for N=56 is 3911 bytes; its carve is
  `align_up(3911,16) = 3920 = 0xF50`.
* Carve bases are 16-aligned, so `end = base + 0xF50` must have low nibble 0. Observed `0xE0`
  fits, and implies `base_low = 0x90`.
* `"fn55"` is at container offset `3906 = 0xF42`, so a CORRECT cursor is `base + 0xF42`, whose
  low byte would be `0x90 + 0x42 = 0xD2`.
* Observed cursor low byte: **`0x00`**.

Independently of the base: offset `0xF42` has low nibble 2 and a 16-aligned base has low nibble
0, so ANY correct cursor ends in nibble **2**. `0x00` ends in nibble 0. **The cursor cannot be
`container_base + 3906` for any legal base** — this is not an aliasing coincidence, and it does
not depend on the sample count.

**Conclusion:** the capability in the bad slot is well-formed and its BOUNDS are consistent with
being derived correctly from the container. Only its CURSOR is wrong. Combined with the earlier
finding that the tag survived (`lcc` did not trap), the fault is neither a dropped store, nor a
lost tag, nor a bad provenance — it is the **cursor value** for that one entry.

That points at the `cincoffset` that adds this entry's container offset:

    386c4: addi        a6, a6, -0xbe     # 3906, the container offset of "fn55"
    386c8: cincoffset  a5, a5, a6        # container capability + 3906   <-- the suspect
    386cc: stc         a5, 0x0(a0)       # store into arr[55].zName

with the caveat that this exact instruction sequence is byte-identical in builds that WORK.

### Still open

* `START` was never read (its domain died in region-share). Without it, "bounds are right" rests
  on `END` alone.
* The cursor's other three bytes are unread, so its actual value is unknown — only that it is
  not the correct one.
* A load through it does NOT trap (the original probe read a byte and returned an index rather
  than wedging), so the cursor is still WITHIN the capability's bounds. It is a wrong address
  inside a correct region, not an out-of-bounds pointer.


---

## WHY ARCHIVED: STALE — those staged experiments have since been run or superseded.

## 0c. BUILT AND STAGED, NOT YET RUN (console went down mid-experiment)

Three probes are built, staged and in the firmware; the console became unreachable (TLS
handshake timeout, 3/3) before they produced results. Each answers something the readings so far
cannot. Re-run them first when the board is back — no rebuild needed.

    stage 102   delta(arr[54] - arr[0])   expect 0x3C   the NEIGHBOUR control: validates the
                                                        method and confirms entry 54 is correct.
                                                        Run FIRST -- if this is wrong, "only the
                                                        last entry" is false.
    stage 100   delta(arr[55] - arr[0])   expect 0x42   the first DIRECT measure of the error.
                                                        Self-referencing, so it needs no
                                                        knowledge of the runtime base. The
                                                        deviation from 0x42 IS the fault size.
    stage 101   same slot read TWICE      expect 0xB0   0xB0 = the two reads AGREE -> the value
                                                        was STORED wrong. 0xB1 = they DIFFER ->
                                                        it is corrupted on READ. Nothing
                                                        measured so far separates these two, and
                                                        they are different bugs.

### The two experiments that would close the picture

1. **Read `START`** (`lcc` zimm=3). Its domain died in region-share, so "the bounds are right"
   currently rests on `END` alone.
2. **Test whether SQLite passes stage 10 when this construct is avoided.** This is the only
   experiment that converts "this explains the blocker" into "this IS the blocker", and it has
   never been run. Concretely: build the SQLite domain with the builtin array restructured so it
   is not a straight-line local (e.g. filled by a loop from a static table -- the R-14 variant C
   shape that was already known to behave), and see whether stage 10 returns.

### Honest status of the cause

**Not understood.** What is established is a symptom: in some builds, the LAST entry of a
straight-line struct array holds a valid capability (tag intact, bounds consistent with the
container) whose CURSOR is wrong. Not established: why; whether the error is at store or at
load; whether the larger-N wedges are the same fault; and whether it is what stops SQLite.


---

## WHY ARCHIVED: INVALIDATED — the "cursor is off by 57 bytes" headline came from the broken probe.

## 0a3. DIRECT MEASUREMENT: the cursor is off by 57 bytes — and Family A now blocks the work

    stage 100   delta(arr[55] - arr[0]) = 0x09   expected 0x42   entered=TRUE
                => the stored cursor is 57 bytes BELOW where it should be

This is self-referential — it subtracts two cursors read in the same domain — so unlike the
earlier `0x00` low-byte reading it needs no knowledge of the runtime container base and no
alignment argument. Two independent methods now agree the cursor is wrong:

* raw cursor low byte `0x00`, where any correct cursor must end in nibble 2 (offset `0xF42`
  plus a 16-aligned base);
* delta from entry 0 reads `0x09` where `0x42` is correct.

Both are single samples that ENTERED the domain (verified by `SQ: G/enter`).

### The blocking problem is now Family A, not the slot

Across the last two rounds: **1 of 6 attempts entered the domain.** The other five died in
region-share before `G/enter`, with the `wd71` control returning `0x45` in every one of those
boots — so the board was healthy and this is not exhaustion (all runs were at slot 2).

Consequences:
* `stage 101` (stored-wrong vs read-wrong — the biggest open fork) has NEVER executed.
* `stage 102` (the neighbour control that would validate the whole method) has never executed.
* Any further slot-level work costs ~6 boots per reading at the current entry rate.

**So the next thing worth fixing is the region-share failure itself.** It is distinct from the
SPLB exact-fit spin already fixed: the ceiling work showed a failure between `SQ: B/mkregion1`
and `SQ: C/mkregion2` that emits NO monitor tag at all — not SPLA, SPLB, RGNO or SHAB — so the
monitor never reaches a reporting site. Instrumenting the host-side mkregion path (the ioctl and
driver) is the cheapest way in, and needs no firmware change.


---

## WHY ARCHIVED: SUPERSEDED — the real root cause is C-16 (memset destination typed in AS0),
  plus a separate, still-open silicon-only fault (R-14).

## 1b. ROOT CAUSE LOCALISED: a straight-line struct array of ~72 entries wedges the core

**Standalone reproducer, no SQLite in the path.** A local array of N structs, each holding a
DISTINCT string literal plus two pointers, initialised straight-line:

    struct probe_fd { const char *zName; void *p1; void *p2; unsigned char flags; };
    struct probe_fd arr[] = { { "fn0", 0, 0, 0 }, { "fn1", 0, 0, 1 }, ... };

Measured, each at position 2 with the `wd71` control returning `0x45` in the same boot:

    N=1    stc=126   returned 0xC1     OK
    N=4    stc=141   returned 0xC4     OK
    N=16   stc=177   returned 0xD0     OK
    N=32   stc=225   returned 0xE0     OK
    N=48   stc=282   returned 0xF0     OK
    N=72   stc=362   WEDGED
    N=96             WEDGED  (3 independent builds, all wedged)

**Threshold between 48 and 72 entries.** SQLite's builtin `FuncDef` array has **72** entries —
at the threshold — which is why `sqlite3RegisterBuiltinFunctions` wedges.

### How this was reached (elimination, not hypothesis)

1. Stage 9 (`MallocInit` + `PcacheInitialize`) RETURNS; stage 10 (`+ RegisterBuiltinFunctions`)
   WEDGES at position 2 in five separate boots.
2. Stages 88/89/90: `sqlite3WindowFunctions`, `sqlite3RegisterDateTimeFunctions` and
   `sqlite3RegisterJsonFunctions` each RETURN -> all three exonerated.
3. `BUILTIN_LIMIT=0` skips the `strcmp` loop and inserts ZERO entries, and still wedges -> the
   remaining work in that path is the array CONSTRUCTION.
4. Stage 92 rebuilds only that construction, with nothing else, and reproduces the wedge above a
   size threshold.

This is the R-14 shape ("straight-line materialisation of distinct string constants into a
struct array") finally isolated with a size threshold and no SQLite dependency.

### PINNED: it is the LAST ENTRY of the array that reads back wrong

Stage 94 returns the INDEX of the first bad entry (or `0xFF` if all are correct), so the
encoding cannot alias a wrong answer with a right one:

    idx48   rc = 0xFF   ALL 48 entries correct        <- validates the probe at a known-good size
    idx56   rc = 0x37   first bad entry = index 55    <- 2 of 2 in one boot, deterministic
    idx56   rc = 0x37   first bad entry = index 55

**Index 55 is the LAST element of the 56-entry array.** Entries 0..54 all read back correctly.

This rules out a fixed-index threshold: if the fault were "entries at index >= 48 are corrupt",
the first bad index would be reported as 48, not 55. It is specifically the FINAL element of the
straight-line initialisation, and it is consistent with the earlier count result (55 correct of
56 -> exactly one bad entry).

**Restated minimal claim:** in a straight-line local array of N structs each holding a distinct
string literal, once N exceeds ~48 the LAST entry's `zName` pointer does not read back as
written. At larger N (72, 96) the domain wedges instead of returning a wrong answer.

**How this produces the SQLite blocker:** `sqlite3RegisterBuiltinFunctions` builds exactly this
shape with 72 entries. A corrupt `zName` on the final entry is then hashed by
`sqlite3InsertBuiltinFuncs` -> `sqlite3Strlen30` walks a bad pointer -> the livelock/wedge that
has been the blocker all along. That also explains why every earlier probe kept implicating
"strlen" and "lit[1]": those were downstream of a corrupt pointer, not the fault itself.

### The next questions, now narrow

1. Is it always the LAST entry, or the entry at some fixed OFFSET from the end? Test N=52, 60,
   64 and see whether the reported index tracks N-1 each time.
2. Is the pointer wrong, or the memory it points at? Return the low bytes of
   `arr[N-1].zName` rather than testing its first character.
3. Does it depend on the struct's OTHER fields? Drop `p1`/`p2` and see whether the threshold
   moves -- that separates "too many capability leaves" from "too large a frame".
4. Does the same happen for a `static` array (cap-init path) as for a local (stack path)?

### Refinement: above the threshold the construct MISCOMPUTES or WEDGES, and both occur

    N=56, build "g56"  (count encoding)     RETURNED, count = 55 of 56   <-- WRONG ANSWER
    N=56, build "d56"  (deficit encoding)   WEDGED
    N=48 and below                          RETURNED, correct            (5 boots)
    N=72 / N=96                             WEDGED                       (4 builds)

So the failure has TWO manifestations at the same entry count, depending on the build: a
silent wrong answer (one entry's `zName` not reading back as expected) or a wedge. That is
consistent with one underlying fault whose visible effect depends on layout, and it means
**a returning run is not proof of correctness** for this construct -- the count has to be
checked, which is why the probe now returns a deficit rather than a marker.

The wrong-answer form is the more useful one: it returns, so it can be repeated within a boot,
and the deficit says how many entries were corrupted. Chase THAT, not the wedge.

### A probe-design correction worth keeping

The first version of this ladder returned `0xC0 | (count & 0x3f)`, which WRAPS at 64: `N=64`
correct and `N=72` miscounted-to-64 both render as `0xC0`, and an auto-generated conclusion
("shared literals return, so distinct literals are the variable") was drawn from exactly that
ambiguity and had to be withdrawn. Encode a DEFICIT (expected - actual), never a raw count, so
a wrong answer cannot alias a right one.

### Caveats, stated

* The N=72 wedge is ONE sample; a wedge ends the session so it cannot be repeated in-boot. The
  claim "above threshold wedges" rests on N=72 (1) plus N=96 (3 independent builds) = 4 samples.
  N<=48 returning is 5 samples across 5 boots.
* The threshold is bracketed (48..72), not pinned. Bisect 56/64 to narrow it.
* What scales with N is confounded: entry count, distinct string literals, total stack frame,
  and cap-init/`stc` count all grow together. The next experiments should vary ONE: same entry
  count with SHARED string literals; same count with larger structs; same count spread across
  two smaller arrays.

### Why this matters

It is the first artefact in this campaign compact enough to hand over as a hardware question:
a few dozen lines of ordinary C, a known-good side (N<=48), a known-bad side (N>=72), a control
that passes in the same boot, and no SQLite, allocator, hash table or string walking anywhere in
the path.


---

## WHY ARCHIVED: SUPERSEDED — this said "NOT FOUND". A root cause WAS found: C-16. A second,
  silicon-only fault remains open.

## 2. Root cause

**NOT FOUND.** No mechanism has survived measurement. Do not present any of the below as the
cause.


---

## WHY ARCHIVED: SUPERSEDED — hypothesis 2 (capability-compression aliasing) is refuted: the RTL
  carries the cursor in full 64 bits and only the bounds are compressed.

## 6. Current hypotheses, ranked

1. **Family A (region-share) is a capability exception that is silent because `mtvec = 0`.**
   The monitor never writes `dom_seal[1]` (`sbi_capstone.c:760,782-784`) and slot 1 IS
   `{ctvec,mtvec}` (`csr_regfile.sv:399`). Getting its `mepc`/`mtval` would name the faulting
   instruction. SOURCE + MEASURED (mcause 24 latched).
2. **Capability compression aliasing.** Register capabilities hold compressed metadata whose
   bounds are rebuilt from the CURRENT cursor (`ariane_pkg.sv:692-693`), and `CINCOFFSET` does
   no representability check (`capstone_flu_unit.anvil:41-42`). Effective bounds can slide with
   the pointer at multiples of 2^(E+14). SOURCE + a reimplementation of the arithmetic;
   CONTESTED by one board reading (§7).
3. **Something in Family B that has not been named.** `wrev`/`memwait` are resting state, the
   dyn unit reports `dyn_rdy=1` (idle) at those wedges, and no new trap latches. Genuinely open.


---

## WHY ARCHIVED: SUPERSEDED — the BUILTIN_LIMIT clamp is no longer the recommended move; see the
  summary and the entry-stall sections.

## 10. Next steps

1. **WORKAROUND (highest value for the deadline):** clamp `BUILTIN_LIMIT` in
   `build-sqlite-silicon.sh` and find the largest builtin count that still initialises. A
   minimal existence proof (CREATE/INSERT/SELECT on integers) needs very few builtins. If a
   small limit gets past the wedge, SQLite runs on silicon with a documented limitation.
2. Settle §7 with the `1024*1024 + 512` variant of stage 76.
3. Get Family A's `mepc`/`mtval` — it is a real, latching capability fault and would name the
   faulting instruction directly.
4. Re-take any pre-2026-08-01 conclusion that rests on a single sample.

---


---

## WHY ARCHIVED: SUPERSEDED — the clamp workaround predates the C-16 root cause.

## Workaround attempt 1: clamp builtin registration (2026-08-01)

`BUILTIN_LIMIT=<n>` in `build-sqlite-silicon.sh` clamps how many entries
`sqlite3RegisterBuiltinFunctions` processes. Built limits 1/8/24 at **stage 3** (through
`sqlite3_open`), run with the `wd71` control first:

    wd71   rc = 0x45    control OK
    bl1    WEDGED       (bl8/bl24 never ran -- a wedge ends the session)

**Do NOT read this as "one builtin entry reproduces the bug".** The build script's comment says
`limit=1 wedging -> the construct itself is broken`, but that shorthand assumes the probe is
SCOPED to that function. Stage 3 runs `sqlite3_initialize` AND `sqlite3_open`, i.e. stage 3 is a
superset of stage 10, so a stage-3 wedge at limit=1 is equally consistent with the failure being
somewhere later in `open` that clamping does not touch.

Scoped retest built: `BUILTIN_LIMIT=0` and `=1` at **stage 10**, which stops inside
`sqlite3RegisterBuiltinFunctions`:

* **limit 0 returns, limit 1 wedges** -> a SINGLE builtin entry is a minimal reproducer. That
  would be by far the smallest repro this campaign has produced.
* **limit 0 AND limit 1 both return** -> the builtin construct is fine at small counts and the
  earlier stage-3 wedge is later in `open`; re-bisect there, and the clamp is a viable
  workaround knob.
* **limit 0 wedges** -> the wedge is not in the builtin loop at all; stage 10's boundary is
  reached before any entry is processed, and the whole "RegisterBuiltinFunctions is the wedge
  point" framing needs re-checking.

### CORRECTION: `BUILTIN_LIMIT` was the WRONG KNOB (2026-08-01, MEASURED + SOURCE)

Scoped retest at stage 10: **`BUILTIN_LIMIT=0` WEDGES** (control `wd71` returned `0x45`).
Zero builtin entries processed, still wedged.

That looks like it exonerates the builtin path, and it does NOT. Reading the amalgamation at
the definition (`sqlite3-capstone.c:137217`):

    SQLITE_PRIVATE void sqlite3RegisterBuiltinFunctions(void){
      FuncDef capstoneBuiltinFunc[] = { ... ~72 entries ... };

The array is a **LOCAL** — `build-sqlite-capstone.sh` strips `static` — so it is constructed on
the STACK, straight-line, at run time. `BUILTIN_LIMIT` rewrites only the INSERTION loop bound
(`capstoneI<ArraySize(...)` -> `capstoneI<0`, `build-sqlite-silicon.sh:124-126`). **It never
reduces the construction.** At `limit=0` the full ~72-entry array is still built on the stack
before the (now empty) insertion loop runs.

So `limit=0` wedging is entirely consistent with the STRAIGHT-LINE CONSTRUCTION being the
culprit — which is exactly the R-14 shape (straight-line materialisation of distinct string
constants into a struct array wedges; the same data assigned in a loop, or as a flat pointer
array, is fine).

**Consequences:**
* The "~72 entries / scale effect" theory that motivated `BUILTIN_LIMIT` is untestable with
  that knob and remains neither confirmed nor refuted.
* Stage 10 remains the wedge point, and the suspect narrows to the array CONSTRUCTION rather
  than the hash insertion.
* `SQLITE_STATIC_BUILTINS=1` targets exactly this: it restores `static`, turning the run-time
  stack construction into a compile-time global initialised through `__capstone_cap_init`
  (machinery that already performs 394 capability-leaf stores successfully in this domain). It
  was dismissed earlier as "a regression that breaks even stage 0" — a SINGLE-SAMPLE verdict
  from the period when many single-sample verdicts in this campaign turned out wrong. **It is
  being re-tested at stage 0 and stage 10.**

### Workaround attempt 2: `SQLITE_STATIC_BUILTINS=1` — CONFIRMED REGRESSION (MEASURED)

The patch was verified to apply before spending board time (`static FuncDef aBuiltinFunc[] = {`
present, `capstoneBuiltinFunc` gone — an initial grep for the OLD name returned 0 and would have
read as "patch failed" if trusted).

    wd71  rc = 0x45   control OK
    sb0   WEDGED      STATIC_BUILTINS at STAGE 0

**Stage 0 is entry-and-immediate-return** — the domain runs no SQLite code at all. So restoring
the array to a compile-time `static` breaks the domain BEFORE any code executes; the only thing
that changed is what `__capstone_cap_init` must materialise. The earlier "breaks even stage 0"
verdict is CONFIRMED, not a single-sample error. This workaround is closed.

### The convergence worth chasing

Both ways of materialising the same ~72-entry `FuncDef` array fail:

* as a LOCAL (straight-line stack construction) -> stage 10 wedges;
* as a `static` (cap-init leaves) -> stage 0 wedges, i.e. even earlier.

Measured cap-init cost:

    build             cap_init size   capability stores (stc)   outcome
    sb0   (static)        16248                1257             WEDGES AT ENTRY
    b10n0 (clamp 0)       10768                1017             wedges at stage 10
    wd71  (probe)         10768                1048             RETURNS

Carve counts are ~equal (181/181/182) — the array is ONE global, so it adds one carve, not 72.
Store count alone does NOT predict the outcome (1017 wedges, 1048 returns), so there is no
simple monotonic threshold across different failure points.

**But `sb0` is the cleanest signal in the campaign:** it wedges at ENTRY, where the only work is
cap-init, with ~20% more capability stores than the largest known-good build. That makes
"cap-init fails somewhere above ~1048 stores" a sharp, cheap hypothesis — and unlike the
in-domain wedges it involves no SQLite logic at all.

**Next test (bisection, entry-time only):** build domains whose cap-init store count is varied
between ~1048 and ~1257 (e.g. by adding N dummy initialised globals holding capability leaves to
a stage-0 domain) and find the threshold. Stage 0 is the ideal vehicle: it returns immediately,
so any wedge is attributable to cap-init and nothing else. If a threshold exists, it is a
concrete number to hand over, and it would also explain the region-share (Family A) failures,
which happen before `domain_main` for the same reason.

### Workaround status

**Not yet available.** `SQLITE_STATIC_BUILTINS=1` was tried earlier and is a REGRESSION (it
breaks even stage 0). Builtin clamping has not yet produced a passing configuration. If the
scoped retest shows limit 0 passing, the next question is the largest limit that still passes,
and whether that leaves enough builtins for CREATE/INSERT/SELECT on integers.

---

