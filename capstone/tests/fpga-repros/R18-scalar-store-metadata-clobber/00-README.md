# R-18 — a scalar in the upper half of a 16-byte cache row is silently zeroed on silicon

**Status: reproducible silicon defect, REPRODUCED IN RTL SIMULATION (~13 s), with a mechanism that
now fits, and a compiler-side workaround validated in simulation. See `sim/`.**

## The defect in one paragraph

A `-O0` loop that mixes capability traffic with ordinary scalar locals can have one of those scalars
**silently set to zero** part-way through — no trap, no tag violation, nothing in any log. The same
byte-identical binary computes the correct answer under QEMU and the wrong one on the FPGA. Which
variable is hit depends only on where the compiler placed it. If a *loop-control* variable lands in
the affected position the loop runs **extra iterations** instead of producing a wrong value.

> **THIS ISSUE HAS BEEN REPORTED and our side is worked around.** Do not re-open it to record
> mechanism work. A second, DIFFERENT signature found 2026-08-08 — the victim holding
> `compress_cap(NULL) + n` rather than being zeroed — is tracked separately as **R-19**, in
> `../R19-movc-zero-metadata-in-slot/`, together with two corrections to the mechanism described in
> the report that went out. The reproducer, trigger and workaround here are unaffected by those.

## Status, 2026-08-08

**This package is the ZEROING signature**: a scalar in the upper half of a 16-byte D-cache row is
silently written with `0` and then counts up from there. Raw full-width readbacks confirm **no
capability metadata lands in the victim** (`craw` reads `0x00000237`, `graw` and `gztr` likewise).

**It reproduces in Verilator, with a matched control**, in ~13 s each:

| test | verdict |
|---|---|
| `sim/scalar-store-movc-zero.S` | **FAILS** — witness at `+0x10` zeroed |
| `sim/scalar-store-realcap-samegeom.S` | **FAILS** — a *real, valid* capability triggers it too |
| `sim/scalar-store-addi-zero.S` | **passes** — the matched control, one instruction different |

So the trigger is **any non-zero capability metadata on an ordinary store's data register**, not the
null form specifically, and the dual-bank path is exercised in simulation rather than only inferred.

**What does NOT match between simulation and board:** which slot dies. At the geometry we built both
ways, the simulation damages the slot **8 bytes** from the trigger and leaves the one 4 bytes away
exact; the board does the opposite. That is unexplained and is the main open question here.

**A SECOND, DIFFERENT signature — the victim holding `compress_cap(NULL) + n` instead of being
zeroed — is NOT this issue.** It is tracked as **R-19**, in `../R19-movc-zero-metadata-in-slot/`,
and it does *not* reproduce in simulation. Please read that package separately; mixing the two is
what this note exists to prevent.

**Our side is worked around:** `-capstone-int-zero-for-zero-copy` emits an integer move for a copy
from `x0`, so no capability shadow reaches the store. Silicon-confirmed. See
`../../../agent-handoff/design/R18-workaround-movc-zero.md`.

## REPRODUCED IN SIMULATION — start here, it costs 13 seconds

`sim/scalar-store-movc-zero.S` reproduces the defect in Verilator, deterministically, in ~13 s.
Six earlier directed tests had failed to, and this package used to name that as its main gap.

Three tests, identical geometry and loop, differing ONLY in what produced the store's data
register (all ~12810 cycles, none near the 2,000,000-cycle timeout):

| `sim/` test | store's data register | verdict |
|---|---|---|
| `scalar-store-movc-zero.S` | `movc a4, x0` — null-capability shadow | **FAILED**, witness A zeroed |
| `scalar-store-realcap-samegeom.S` | a real, valid capability | **FAILED**, witness A zeroed |
| `scalar-store-addi-zero.S` | `addi a4, x0, 0` — zero shadow | **SUCCESS** |

### The mechanism

An ordinary `sw` at `+0x18` (bank 1, lanes 0-3) corrupts `+0x10` (bank 0, **the same byte lanes**).
`sim/movc-zero-rvfi-trace.dasm` shows only TWO architectural accesses to `+0x10` in the whole run —
the seed writing `0x0a0a0a0a`, and the readback returning `0x00000000`. Nothing wrote zero to it.

1. `issue_read_operands.sv:1140` puts the data register's capability metadata on the store's
   write-user sideband **ungated by opcode**;
2. `wt_dcache_mem.sv:138` — `st_wr_cap = |wr_user_i` — classifies the store as a capability store
   **by the VALUE of that sideband**;
3. `:230-238` a so-classified store asserts **both** banks (`bank_req = '1; bank_we = '1`);
4. `:152-158` the **same byte enable** is applied to both banks.

So the store writes its data into its own slot *and* into the same byte lanes of the other bank.
The splash carries the store's **DATA**, not the metadata — which is why no `0x08000000` is ever
found in memory, and why every raw readback shows a clean count.

> **The `R XOR 8` rule that stood here is WITHDRAWN (2026-08-08, audited).** It is arithmetically
> just "the victim is 8 bytes from the trigger store", and the corpus splits cleanly into
> distance-8 builds where it holds (10) and distance-4 builds where it fails (`rs4`, `ka0`, `gnt`,
> `gz0`, `gzn`, `graw`). **Distance is invariant under base alignment**, so the alignment doubt
> once offered to excuse `gnt` cannot rescue it. Collapsing replicas, it is 2 of 4 distinct
> trigger→victim geometries, and its apparent lack of false positives is vacuous: in every clean
> build the predicted target is an unallocated slot or an unobserved pad, so the rule was never at
> risk of being contradicted there.
>
> Two arithmetic errors of mine are also withdrawn with it: the predicted targets printed for
> `kb12` and `rs4` were computed by subtracting 8 unconditionally instead of XORing bit 3, which
> flips direction when the trigger sits at row offset 0. Corrected, **`kb12` is a match**. And
> `dp0`, which a later note demoted to "inferred", is measured and is a match — its packing lives
> in `fdreg_depth_body`, not in the `fdreg_compute` wrapper the check was run against.
>
> **What survives are two necessary conditions, not a predictor:** the damaged scalar is in the
> trigger store's own 16-byte row (16/16), and it is in bank 1 of that row, at offset 8 or 12
> (16/16). Which of two bank-1 candidates gets hit is **not predicted** — silicon supports that
> discrimination in exactly two builds.

### THE SIMULATION DOES NOT REPRODUCE THE BOARD'S SYMPTOM — read this before citing it

`sim/scalar-store-movc-zero.S` is built at exactly `gz0`'s geometry: a `movc`-sourced `sw` at row
offset 8, a victim at row offset 12 (distance 4), witnesses at row offsets 0 and 4. The two
disagree about which slot dies:

| | slot at distance 4 (row+12) | slot at distance 8 (row+0) |
|---|---|---|
| **simulation** | `0x240` = 576 — **exact** | `0x00000000` — **zeroed** |
| **board** (`graw`, raw, full width) | `0x00000009` — **damaged** | never measured |

So the simulation exhibits a real dual-bank splash — the RVFI trace shows only two architectural
accesses to the zeroed slot — but it damages the slot the board leaves alone and spares the slot
the board destroys. **The simulation confirms a mechanism; it has not been shown to be the board's
mechanism.** Do not cite it as such.

**The cheapest experiment that settles it** costs one boot and one extra return field: add a
witness at `gc+0x0` (distance 8) to the `gz0` domain and read it back raw. If the board zeroes
`gc+0x0` *and* damages `gc+0xc`, there are two effects and everything above describes only one. If
`gc+0x0` is intact, the simulation and the board are different faults.

### The workaround, validated in simulation AND on silicon

`addi a4, x0, 0` in place of `movc a4, x0` — byte-identical otherwise — **passes**. `movc rd, zero`
writes `compress_cap(NULL)` = `0x08000000` into the register's capability shadow; an integer op
writes a zero shadow, `st_wr_cap` is never asserted, and no dual-bank write occurs. Our `-O0`
codegen materialises integer zero with `movc`, which is why the pattern is pervasive in every
failing build.

**Silicon confirmation.** `c8` (movc) and `c8fix` (addi) are the same source at the same frame
geometry — frame 80, rmw `[20,24,28]`, accumulator still at the damaged row offset 12 — differing by
one instruction. One boot, control first: `k800` 4 OK, `c8` qc=**567** (its 15th consecutive boot at
that value), `c8fix` qc=**576**. Cycles 44116 vs 44075, so it is not cured by doing less work.

**Scope it honestly: this removes the COMMON case, not the class.** Any value reaching a store's
data register from a capability-producing op still carries a non-zero shadow and still splashes. A
complete fix is on the hardware side — classify by opcode rather than by the sideband's value, or
gate the metadata onto the sideband by opcode at issue.

## THE TRIGGER (2026-08-08) — read this first

A plain `sw` whose **data register was produced by `movc rd, zero`** corrupts a *different* scalar in
the same 16-byte D-cache row. Nothing else about the store matters.

Four arms, one boot each, identical victim and row-mate addresses, identical store counts, same
region, a passing `k800` control and the `c8` anchor in every boot:

| arm | the row-mate's per-outer-pass reset store | outer loop | victim |
|---|---|---|---|
| `gz0` | `movc a0, zero; sw a0, 0x8(a1)` | short | **9 — DAMAGED** |
| `gzn` | `movc a0, zero; sw a0, 0x8(a1)` + 2 nops | padded to match a clean arm | **9 — DAMAGED** |
| `gzl` | `ldc; lw; sw` — stores the **value zero**, from a load | padded | 576 — clean |
| `gzs` | `lui; addi; sw` — stores a nonzero | padded | 576 — clean |

Correct is 576 in every arm. What this excludes, by measurement rather than by argument:

* **the stored VALUE** — `gzl` stores zero and is clean;
* **the store COUNT** — identical across all four;
* **the outer-loop instruction count** — `gzn` is padded to a clean arm's length and still fails;
* **region** — the victim here is a GLOBAL; the same effect appears on the stack (`c8`);
* **the victim's address, cache set and bank-row** — unchanged across the gap arms (`s0-0x34`);
* **the capability store's distance and row-adjacency** — `rmC` fails with it two rows away;
* **the compiler** — the emitted victim RMW is a plain `lw`/`addiw`/`sw` on a fixed offset.

### The full path, misclassification AND corruption

`movc` is a capstone-FLU op, so `commit_stage.sv:279` writes `result_metadata` into the cap-metadata
regfile under the **integer** GPR write-enable (`issue_read_operands.sv:1663-1665`, `.we_i(we_pack)`,
not `cap_we_pack`); `:1140` takes `cap_data.cap_metadata_b` **ungated by opcode**; it flows through
`load_store_unit.sv:1013` -> `store_unit.sv:345` -> `store_buffer.sv:173` -> `wt_dcache_mem.sv:138`,
`st_wr_cap = |wr_user_i`. So an ordinary `sw` is classified as a capability store **by value**.
`compress_cap` of a null capability is `0x08000000` (`ariane_pkg.sv:754-772`).

**What the misclassification then does** is the dual-bank write at `:230-238` with the same byte
enable at `:152-158`, confirmed in simulation. One point deserves emphasis because it cost us a
retraction: **the splash carries the store's DATA, not its metadata.** We spent a session looking
for `0x08000000` in memory, not finding it, and concluding the dual-bank path was innocent. Raw,
unmasked readbacks:

| probe | reads | raw value |
|---|---|---|
| `craw` | the stack victim at `c8`'s geometry | `0x00000237` — a clean count |
| `graw` | the global victim | `0x00000009` — a clean count |
| `gztr` | the row-mate itself | `0x00000009` — a clean count |

No metadata anywhere. The victim is written with the store's data — zero, for a `movc`-zero store —
and counts up from there. Those readbacks are consistent with the mechanism, not evidence against
it, and reading them as the latter was our error. (It does still refute a write-buffer
8-byte-merge candidate, whose prediction was `twin = 0x08000009`.)

It also explains `sn8` below: an accumulator seeded to 1,000,000 returns 567 because the splash
*overwrote* it with the store's data (0), after which it counted up normally.

**Two earlier mechanisms remain refuted** and should not be revisited: a writeback-forwarding
validity gate (the RTL has the same expression in both ternary arms, so the change is a no-op), and
a "bank 0 at the same byte lanes only" rule (its own control came back damaged).

Build flags per arm are recorded in the staged `.qemu-pass` markers; all four are
`FDREG_STAGE=37, GVICT=3, GTWIN=2`, differing only as the table above describes.

## What is established

* **Reproducible and deterministic.** Fifteen frozen, checksummed artifacts in `src/`. `c8` returned
  67699255 on **fourteen** consecutive boots (cycles 44013–44111).

  *Correction (2026-08-07).* An earlier revision said the failing arm and its control "differ only
  in where the `-O0` allocator placed the accumulator". That was wrong — they also differ in entry
  VA, read straight off the artifacts:

  | image | entry VA | result |
  |---|---|---|
  | `c8`  | `0xf0000` | 567 — damaged |
  | `c0`  | `0x30000` | correct |
  | `sn8` | `0x30000` | 567 — damaged |
  | `sn0` | `0xf0000` | correct |

  Taken pair-by-pair that is a confound. Taken together it is the opposite: the VAs are **crossed**
  between the two pairs, so each VA hosts one damaged and one correct image, and base VA is
  *excluded* as the cause while `FDREG_SHIFT` — the accumulator's row offset — tracks the damage
  across both. The corrected claim is stronger than the one it replaces, but it has to be stated
  this way rather than by asserting a single variable that was not in fact held.
* **Necessary condition:** the victim is always in the **upper 8 bytes of its 16-byte cache row**
  (row offset 8 or 12, never 0 or 4) — 9 of 9 builds where the victim was measured directly.
  **It is necessary, not sufficient:** roughly 10 *undamaged* upper-half scalars appear across the
  same dataset, so this constrains the search rather than explaining anything.
* **The slot is overwritten, not skipped.** With the accumulator initialised to a sentinel of
  1,000,000 it returns **567**, not 1000567 — so the location is written and counted up from there.
  *(Caveat: N=1. Worth repeating with several sentinel values.)*
* **QEMU is correct** for every variant.

## What we RULED OUT (so you need not spend time on these)

| ruled out | how |
|---|---|
| an over-wide capability store | a witness immediately above the store reads back bit-exact after 576 stores; `extract_transfer_size` pins STC at 8 bytes/one beat |
| ~~store misclassification via the write-user sideband~~ — **THIS ROW IS WITHDRAWN. It is the mechanism.** | `scalar-store-cap-operand.S` passing was a GEOMETRIC artifact: it stores at `+0x1c`, and at that offset the splash lands on a slot it did not check. At identical geometry (`sim/scalar-store-realcap-samegeom.S`, store at `+0x18`) a real capability corrupts exactly as a null one does. Ruling the family out on that test was wrong. |
| any single-address anchor for the victim | fitted against all builds; best 13/19 |
| distance from the capability store | same value reproduced at 3× the distance, different row, different frame size |

We also do **not** recommend gating the WB-port metadata forward on validity: `issue_read_operands.sv:765`
already has `cap_result.result_metadata` in **both** arms of its ternary, so that change would be a
**no-op**. We nearly sent that as a fix and withdrew it.

## The outer-pass lead — TESTED 2026-08-08, and it was right

All three measured reset points — **9, 72, 558** — are multiples of the inner trip count (9), i.e.
they land on **outer-pass boundaries**, pointing at something happening once per outer pass rather
than once per iteration. That is now confirmed and localised: the once-per-outer-pass event is the
row-mate's **reset store**, and the trigger table at the top of this file is the controlled test of
it. The lead was correct; what was wrong was every mechanism proposed to explain it.

## The "null vs valid" tension — RESOLVED 2026-08-08, and it was our test's fault

This section used to say we could not reconcile `scalar-store-cap-operand.S` passing with a real
capability while `movc rd, zero` failed on the board, and called it the sharpest open question.
It was not an asymmetry. That test stores at `+0x1c` and the new one at `+0x18`, so the two differ
in **geometry** as well as in the operand. At identical geometry a real capability corrupts exactly
as a null one does (`sim/scalar-store-realcap-samegeom.S`, FAILED, witness A zeroed).

**The trigger is ANY non-zero capability metadata on an ordinary store's data register**, not the
null form specifically. `movc rd, zero` is simply the pervasive source, because that is how `-O0`
materialises an integer zero.

## The one arm the rule does not fit

`gnt` puts its row-mate at `+8` and its victim at `+12`, so the rule predicts the splash lands on
`+0` and the victim should be clean. It was damaged. The candidate is that `gc`'s runtime alignment
is not what the source requests: the interp entry glue **ignores the descriptor's `align` field**
(it loads `+0x0` and `+0x10` at stride 24 and never `+0x8`) and carves every global at `sp.END`
minus multiples of 16, so `gc`'s row offset is whatever `sp.END` happens to be mod 16. That is
inference, not measurement. It is free to measure — return `&gc[0] & 0xF` in a spare nibble — and
it does not affect the simulation result, which uses an explicitly aligned buffer.

### The older builds, re-examined against the rule (2026-08-08)

Resolving the ACTUAL `movc`-zero store offset in each build — rather than assuming which counter is
reset — and comparing the rule's predicted splash target (`R XOR 8`) with the recorded victim:

| build | movc-zero store | predicted splash | recorded victim | delta | |
|---|---|---|---|---|---|
| `c8` | s0−0x3c | s0−0x34 | s0−0x34 | −9 | **match** |
| `rs8` | s0−0x3c | s0−0x34 | s0−0x34 | +9 | **match** |
| `dp0` | s0−0x3c | s0−0x34 | s0−0x34 | −9 | **match** |
| `rs4` | s0−0x38 | s0−0x30 | s0−0x34 | −72 | **mismatch, and it is real** |
| `ka0` | s0−0x34 | s0−0x3c | s0−0x38 | −558 | victim INFERRED — see below |
| `kb12` | s0−0x40 | s0−0x48 | s0−0x38 | −9 | victim INFERRED — see below |

> **CORRECTION 2026-08-08, same day.** The paragraph that stood here said `ka0` and `kb12` return a
> single masked number so their victims were "inferred". **That was wrong in both directions.** The
> check behind it looked for stage-19's packing (`slliw 0x14` + `slli 0x10`) and false-negatived
> stage-26's four-scalar packing (`slliw 0x18` + `slli 0x34`). Re-checked by counting the `or`
> instructions in each build's compute function:
>
> | build | packing | victim |
> |---|---|---|
> | `c8`, `rs4`, `rs8` | 2 `or`, shifts 0x10/0x14 | **MEASURED** |
> | `ka0`, `kb12` | 2 `or`, shifts 0x18/0x34 | **MEASURED** |
> | `dp0` | 0 `or`, single value | **inferred** |
>
> So `ka0` and `kb12` are genuine counterexamples, and `dp0` — which the table above counted as a
> match — is the one that is not evidence either way. **The rule's fit on the older corpus is
> therefore weaker than stated above, not stronger.** An audit of the true fit across every build
> with a measured victim is in progress; treat the row-by-row table above as provisional until it
> lands, and treat the MECHANISM (which is confirmed in simulation, not fitted) as the solid part.
>
> Note also that predicting a build's victim from ONE `movc`-zero store is questionable methodology
> in the first place: `c8` alone has seven trigger sites, so several splashes land in several
> places, and "the" victim is not a well-posed question without enumerating all of them.

The `+333`/`+330` builds belong to the separately documented extra-iteration fault, not to this one.

## Why it took so long to reproduce in simulation

Six directed tests failed to reproduce this, and for a long time that was recorded here as the
package's main gap, with the reasoning that the failing code runs inside a capability domain after
`capenter` on a monitor-carved stack and could not be reduced to bare metal. **That reasoning was
wrong.** It reduces to bare metal fine. What the six earlier tests lacked was not the domain
context but a store whose *data register carried capability metadata* — every one of them stored a
value produced by an integer op, so `wr_user_i` was zero and the trigger was never created. The one
test that did put a capability there stored it at a geometry whose splash target it did not check.

The lesson worth carrying: those six clean tests were read as evidence the hardware was innocent.
They were evidence that the tests did not exercise the condition.

---

## Relationship to `RTL-store-user-metadata/` — read that first

That package already established the RTL routing, and this one does **not** rediscover it:

> every store routes the capability-metadata shadow of its **data** register into the dcache
> write-user sideband, ungated — `load_store_unit.sv:1003-1020`, `store_unit.sv:344-346`,
> `store_buffer.sv:172-176`; and the write buffer tracks `data` per byte but `user` as **one flat
> field with no per-byte mask** (`wt_dcache.sv:70-79`).

It also identified bit 27 as `bounds.cursorless` in `cap_metadata_t`
(`core/include/ariane_pkg.sv:609-637`).

Its status line reads *"code-level RTL observation, NOT a demonstrated software-visible defect"*,
and it left one thing explicitly open:

> could **not** trace a path from `data_wuser` into a plain `lw`'s returned data … That remains
> UNRESOLVED and would need the `wt_dcache_mem.sv` fill/writeback merge path.

This package was originally written claiming to CLOSE that open question via the store side. **That
claim is retracted** (see the box above): no path from a scalar store's `wr_user_i` to a non-zero
value has been demonstrated anywhere. The open question above is still OPEN. What this package
contributes is the **software-visible measurement**, not an explanation. The routing is prior work.

## What the repro shows

**17 frozen images** in `src/`, all instrumentation **mode 0** (`fdreg_fpga_app.c` sets
`LADDER_INSTR_MODE 0`) — mode 4 is a confirmed miscompute trigger and a previous `0x08000000`
sighting was traced to it, so a defect repro must not carry it. Verified: zero `minstret` reads.
Every one was QEMU-verified before it was ever boarded, and every board run carried a passing
`k800` control first and the `c8` anchor second.

**The original four** — the accumulator's row offset is the only variable:

| image | accumulator lands | expected on silicon |
|---|---|---|
| `c0.dom` | row offset 4 — **lower** half | `0x04090240` (p=64, k=9, qc=576) **correct** |
| `c8.dom` | row offset 12 — **upper** half | `0x04090237` → qc=**567** |
| `sn0.dom` | lower half, accumulator starts at 1,000,000 | `1000576` **correct** |
| `sn8.dom` | upper half, accumulator starts at 1,000,000 | **567** |

**The localization set** — which scalars share the victim's 16-byte row. `rmB` vs `rmC` is the
single-variable pair: same frame, same victim address, same `p`, capability store two rows away in
both; only `k`'s row membership differs:

| image | `k` | victim row-mates | silicon |
|---|---|---|---|
| `rg16.dom` | out of the row | none | 576 **correct** |
| `rg32.dom` | out of the row | none | 576 **correct** |
| `rmB.dom` | out of the row | `p` only | 576 **correct** |
| `rmC.dom` | **in the row** | `p` and `k` | qc=**567** |

**The trigger set** — a GLOBAL victim, identical addresses and store counts, differing only in what
produced the row-mate's reset store (this is the four-way control):

| image | reset store | silicon |
|---|---|---|
| `gz0.dom` | `movc a0, zero; sw` | victim **9** — damaged |
| `gzn.dom` | `movc a0, zero; sw` + 2 nops (loop length matched) | victim **9** — damaged |
| `gzl.dom` | `ldc; lw; sw` — stores the VALUE ZERO from a load | 576 **correct** |
| `gzs.dom` | `lui; addi; sw` — nonzero | 576 **correct** |

**The raw-readback probes** — same builds, returning the slot unmasked, because every earlier
number in this investigation masked to 16 bits and so could not tell "lost increments" from
"overwritten with metadata":

| image | reads | raw |
|---|---|---|
| `craw.dom` | stack victim at `c8` geometry | `0x00000237` |
| `graw.dom` | global victim | `0x00000009` |
| `gztr.dom` | the row-mate itself | `0x00000009` |

`gvf0.dom` / `gvf6.dom` are retained from the superseded stack-vs-global experiment; see the trail.

`sn8` is the decisive one. The accumulator is initialised to **1,000,000**; if increments were
merely being lost it would return 1000567. It returns **567** — the slot was *overwritten* and
counted up from there.

QEMU computes the correct value for all four.

## Reproducing

    source capstone/tests/capstone-test-env.sh
    bash capstone/tests/fpga-repros/R18-scalar-store-metadata-clobber/run.sh

The images are **frozen and checksummed** (`SHA256SUMS`) because the effect depends entirely on
where the `-O0` allocator places the accumulator; rebuilding can move it and cure the fault.
`run.sh` verifies the checksums before doing anything. To inspect a layout:

    python3 capstone/tests/runtime-qemu/silicon-ladder/extract-frame-layout.py src/c8.dom src/c0.dom

### Rebuilding the frozen images from source

Verified 2026-08-07: this reproduces `c8.dom` **byte-for-byte** (`sha256` starts `9ecd8c6f9eb2b23d`).

    source capstone/tests/capstone-test-env.sh
    cd capstone/tests/runtime-qemu/silicon-ladder
    DOMAIN_GLUE=interp DOMAIN_BASE_VA=0xf0000 \
      DOMAIN_EXTRA_CFLAGS="-DFDREG_STAGE=19 -DFDREG_SHIFT=8 -DFDREG_GAP=0" \
      bash build-ladder-domain.sh fdreg_fpga_app.c /tmp/c8.dom

Per-image parameters: `c8` = SHIFT 8 @ `0xf0000`; `c0` = SHIFT 0 @ `0x30000`; `sn8` = SHIFT 8
sentinel @ `0x30000`; `sn0` = SHIFT 0 sentinel @ `0xf0000`.

Three things about this recipe are worth stating because each one cost a rebuild:

* **`DOMAIN_GLUE=interp` is load-bearing.** Without it the build takes the generated-prologue path,
  which emits `lla fdreg_defs` for the large-RO copy of the 1296-byte `fdreg_defs` array. That
  symbol is `static`, so it has local binding and the link fails with
  `ld.lld: error: undefined symbol: fdreg_defs`. The generator's guard only rejects `.L` symbols,
  not local-binding ones — a real latent bug in `gen-gp-captable-glue.py`, filed here rather than
  worked around silently.
* **The base VAs above are the ones in the artifacts.** An earlier write-up of this recipe gave
  `0x30000` for `c8` and `0x60000` for `c0`; neither matches, and following it does not reproduce
  the frozen images.
* **`src/fdreg_kernel.h` is now the header the images were built from.** It previously carried no
  stages 30+, so `gvf0`/`gvf6` — checksummed into this package — could not be rebuilt from it.

## Supporting evidence, from the full trail

* The victim is in the **upper half of its row in 9 of 9** builds where it was measured directly
  (row offset 8 or 12, never 0 or 4). Undamaged builds also carry upper-half scalars, so this is a
  real constraint rather than an artifact of where slots land.
* One build returns **`0x08000237 = 0x08000000 + 567`**, and `0x08000000` is exactly
  `compress_cap` of a **null capability** — so that value is metadata-shaped, not a number.
  *(A `clobber + (576 − reset)` decomposition of every victim was also recorded; it is an
  arithmetic identity with two free parameters per observation and has no predictive content.
  Do not cite it.)*
* **All three measured reset points — 9, 72, 558 — are multiples of 9**, i.e. they land on
  OUTER-PASS boundaries. That is the most interesting unexplained fact here, and it points at
  something that happens once per outer pass rather than at the victim's own store.
* A loop-**control** variable in the affected slot produces **extra iterations** instead of a wrong
  value, and cycle counts confirm the extra iterations really executed (69081 vs 44001).

## What this is not

* **Not reproduced in Verilator.** `verif/tests/custom/capstone/stc-neighbour-load.S` and
  `stc-counter-pair.S` pass at both RTL revisions, cycle-for-cycle identical, across five rounds of
  added fidelity. They are bare-metal M-mode and we could not construct a directed test that
  produces stale WB-forwarded metadata on a scalar store's `rs2`. **The clean simulation means the
  trigger was never created — it neither confirms nor refutes the chain.** This is the single
  biggest gap.
* **Not a stable rate.** One clobber in 576 iterations in most builds, 558 in one. No account of
  why; it is the strongest argument that more than one thing may be involved.
* **Not the mode-4 harness artifact.** All images here are mode 0, verified.

## Bitstream

Measured on `caplifive_65536_nodes.bit`. The chain is present at `capstone-ariane` HEAD
`458982093` and at `7aac52f93` (the commit this bitstream is built from); `git diff` between them
touches none of the files involved.

## Fix

**Hardware, the real fix.** Classify capability stores **by opcode** rather than by
`|wr_user_i` (`wt_dcache_mem.sv:138`), and/or gate the metadata onto the write-user sideband by
opcode at issue (`issue_read_operands.sv:1140`). Either breaks the chain at its root. Needs a
reflash and is the project lead's call.

**Software, available now and validated in simulation.** Emit an integer op instead of
`movc rd, zero` when materialising an integer zero. `movc rd, zero` writes `compress_cap(NULL)` =
`0x08000000` into the register's capability shadow; an integer op leaves it zero, so `st_wr_cap` is
never asserted and no dual-bank write happens. `sim/scalar-store-addi-zero.S` is byte-identical to
the failing `sim/scalar-store-movc-zero.S` except for that one instruction, and it **passes**. Our
`-O0` codegen uses `movc`, which is why the trigger is pervasive in every failing build.

**Scope the workaround honestly: it removes the COMMON case, not the class.** Any value that reaches
a store's data register from a capability-producing op still carries a non-zero shadow and still
splashes. It is a mitigation to ship while the hardware fix is decided, not a substitute for it.

**Do NOT** gate the WB-port forward on validity "matching the scoreboard-port version" — an earlier
version of this file suggested it and it is a **no-op**: `issue_read_operands.sv:765` has
`cap_result.result_metadata` in both arms of its ternary and does not sanitise to zero.

## What is still open — none of it needs the board

Items 1 and 2 of the previous list are **done**: the directed Verilator test exists
(`sim/scalar-store-movc-zero.S`) and the witnesses at victim±8 are built into it, which is how the
splash target was identified. What remains:

1. **One board arm does not fit the `R XOR 8` rule.** `gnt` places its row-mate at `+8` and its
   victim at `+12`, so the splash should land on `+0` and the victim should be clean; it was
   damaged. Suspected cause: the domain's globals are not 16-byte aligned at runtime, because the
   interp entry glue **ignores the descriptor's `align` field** (it loads `+0x0` and `+0x10` at
   stride 24 and never `+0x8`) and carves at `sp.END` minus multiples of 16. Free to measure —
   return `&gc[0] & 0xF` in a spare nibble. Unmeasured, so treated as inference.
2. **`rs4` (−72) and `ka0` (−558) are unexplained by this mechanism.** They may be a second fault.
   The `+333`/`+330` builds are the separately documented extra-iteration fault, not this one.
3. **The sentinel result is still N=1.** `sn8` returning 567 from a seed of 1,000,000 is now
   *predicted* by the mechanism (the splash overwrites with the store's data, then it counts up),
   which is corroboration rather than proof. A sweep over ≥3 sentinel values would settle it.
4. **Does this explain the other silicon miscompiles?** `matmult_int` (R-1) survived the C-14 fix
   and has no accepted cause. If it is an instance of this defect, the software workaround clears
   it — a larger result than R-18 alone, and one board boot tests it.

Trail, including all eight retractions and what caused each:
`capstone/agent-handoff/history/07-08-2026_23-55-00_r18-localized-to-row-mate-traffic.md`.
Earlier trail: `capstone/agent-handoff/history/07-08-2026_02-30-00_nested-loop-capability-index-iteration-loss.md`.
A superseded mechanism report is retained at
`capstone/agent-handoff/history/07-08-2026_RETRACTED_scalar-store-metadata-mechanism.md` — **do not
link it**; it is kept only so its conclusions are not re-derived.
