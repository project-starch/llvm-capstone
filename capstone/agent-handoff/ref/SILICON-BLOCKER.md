# The silicon blocker — everything known

**Living document.** Update it whenever a claim is added, refuted, or measured. Every entry
must say how it is known: MEASURED (board), SOURCE (quoted file:line), or INFERRED.
Last updated: 2026-08-02.

---

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

## SUMMARY — current best understanding (2026-08-02)

There are **three separate failures**, repeatedly conflated in earlier drafts. Keep them apart.

| # | Failure | Where it stops | Status |
|---|---------|----------------|--------|
| 1 | **Rev-node pool exhaustion** | fails *before* `share1` (`pre-share`), 6/6 at position 6 | **SOLVED.** 1020-node bump allocator, no reclamation; ~182 splits/domain -> 5.5 runs/boot vs measured 6/5/5/5 (0a4) |
| 2 | **`SHA5` stall** | monitor hands off, domain never returns from its FIRST entry | **OPEN.** 32% at slot 2 vs 2.8% at slot 1 (0a10). Monitor exonerated (0a11) |
| 3 | **The SQLite blocker** | passes both shares, reaches `G/enter`, stalls in the main run | **OPEN.** Wrong cursor in a merged-string capability (0a3) |

### What is established

* Pool exhaustion arithmetic, confirmed from RTL **and** measurement (0a4).
* The `SHA5` stall is in the domain, not the monitor: `SHA5` = "about to leave M-mode",
  `SHA6` = "returned"; the stall sits between them (0a4). The **first** entry is where the glue
  builds the 179-entry cap table and runs `__capstone_cap_init`.
* Slot 2 stalls ~10x more often than slot 1, from 274 tabulated launches (0a10).
* SQLite is **179 carves**, not 1059 — the pool is NOT its blocker (0a3).
* The cursor of the bad slot is wrong by a measured, self-referential −57 bytes (0a3).
* The cursor is carried in **full 64 bits**; only bounds are compressed, and they are decoded
  *from* the cursor — so "bounds look right" is not evidence of anything (0a12).

### What is refuted (do not revisit)

* "The ceiling is SPLB" — misread replayed console history.
* "The SPLB exact-fit fix caused the SHA5 stall" — reverted and tested; stall survives (0a11).
* "SQLite needs 1059 carves and overflows the pool" — pre-string-merging figure (0a3).
* "Stage 10 and the probe stall are one fault" — different stopping points (0a4).
* "The reproducer has a 49-byte unaligned stride" — `sizeof`=64, `_Alignof`=16 (0a9).
* Threshold/size theories generally — N=56 fails, N=60 passes (0a9).
* Stages 11-15 as evidence — pre-dating the unaligned-copy fix, which resolved them (0a8).

### Working rules

* **Run the domain under test at position 1**, and repeat any single result — slot 1 still has
  a ~3% stall floor.
* Expected yield is **under two domains per boot**, because a stall ends the session. Budget
  experiments accordingly; this, not the pool, is the throughput limit.
* Always split console output at `booted once` — replayed history has been misread twice.
* Confirm `SQ: G/enter` before attributing anything to the domain's main run.

### Biggest open question

Whether the bad cursor is **stored** wrong or **read** wrong. The RTL says `stc`/`ldc` carry
the cursor verbatim and range/alignment-check it (0a12), so "stored wrong" is the strong prior
and the fault would then be in the address arithmetic. `x101` measures this directly and has
never yet executed.

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

