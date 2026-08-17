# Current Capstone state

Minimal snapshot. Read first in every session.

## 2026-08-17 — CURRENT. Anything below dated earlier predates two RTL fixes and a reflash.

* **Bitstream: `caplifive_s07diag.bit`** (S-06 fix `25035c4c0` + S-08 fix `9fd5507b` + the mtval
  diagnostic `45bd5a3ee`). Every silicon number taken before it is baseline-invalid. All five
  driver `FPGA_BITSTREAM` defaults are repointed.
* **S-06: FIXED in silicon and verified.** All §1 software workarounds reverted; lit green and
  14/15 QEMU suites, the 15th being the known inverted `static-cap-globals` probe.
* **S-08: FIXED by the RTL lane and verified on silicon.**
* **S-07: the ONE open silicon issue.** SQLite's full extended workload completes with no software
  workarounds in roughly two runs out of three; the remainder wedge at mcause 25. Deliverable and
  the ask: `tests/fpga-repros/S07-capability-untagged-on-reload/`. Committed, **not pushed — the
  project lead pushes, and it reaches the RTL lane only then.**
* **MicroPython QEMU census complete for direct single-interpreter files.** EXTRA+MPZ executed all
  917 direct tests in upstream's default base directories: `565 PASS / 338 FAIL / 12 FAULT /
  0 HANG / 2 UNSCORED`. A second run attempted all 200 direct optional files from `cmdline`,
  `float`, `import`, `io`, `thread`, and `unicode`: `27 PASS / 161 FAIL / 0 FAULT / 0 HANG /
  12 UNSCORED`. Patch 0012 gives stream ioctl a port-configurable carrier; the Capstone port uses
  `void *`, preserving seek capabilities and changing exactly `io_bytesio_ext.py`,
  `io_stringio1.py`, and `io_stringio_base.py` from FAULT to PASS across a full 1,117-file rerun.
  The resumable chunk runner gets past domain-fatal faults without changing tests.
  The 529 other Python files are fixtures, runner utilities, benchmark/differential inputs, or
  require multi-instance, network, hardware, port, or architecture-specific harnesses; they are
  not ordinary one-source interpreter cases. See `plans/micropython-domain-compilation.md`.
* The invariant, the "no software probe can fire" result, and the "mtval unreadable by every
  channel" measurement are in `state/current-next-step.md` §0 — read that before planning any
  S-07 work.

## S-06/S-08 CONFIRMED ON SILICON; S-07 investigated (2026-08-15)

Board lane reports S-06 and S-08 fixes VERIFIED on caplifive_s06fixs08fix.bit: s06agg 5->15,
s06aggcap 7->15, s06aggwide 237->255, EXCX 0 (was 4/4), and SQLite completes through finalize
with NO software workarounds — a first. S-07 survives.

S-07 (a capability read back by LDC arrives NOT_CAP, mcause 25) — sim/RTL done; an mtval
DIAGNOSTIC INSTRUMENT shipped (no fix — cause not sim-reachable). NEW (2026-08-16): the board
lane reframed the wedge site as likely the SECOND fault (on :memory:, sqlite3OsRead is only
reached after an upstream sqlite3_step failure; readDbPage asserts !MEMDB), open question H1
(real tag loss) vs H2 (legitimate NULL pMethods). Instrument (commit 45bd5a3ee): on cause 25,
mtval carries the faulting operand's rs1 cursor, so the monitor's MTVL dump discriminates
H1 (nonzero) from H2 (zero) — one boot of the failing workload, no reproducer needed.
Validated 4 ways; source pre-check predicts H1 (memjournal pMethods is static-const non-null).
Fold into next synth. Earlier findings:
- CONFIRMED consequence: an LDC bypassed to LOAD_WB erases the capability (wb[2].cap_data
  tied '0). New board-free invariant in scoreboard.sv, positive-controlled (fires on forced
  bypass), silent across the full 77-test sweep. Commit a114313aa on fpga-testing-dev-s06fix.
- REFUTED cause: the one-deep syncer tracker is never overwritten and CANNOT be — the dyn
  unit serializes cap loads at issue (capstone_dyn_ready backpressure stalls the next cap
  load's whole issue). Overlap and hit-under-miss both architecturally impossible in sim;
  two directed tests (s07-ldc-overlap-displace) confirm. The reporting lane's proposed
  8-entry-vector fix would be DEAD CODE.
- Handed back per the proposed split (board owns the trigger). Remaining candidates: the
  registered capstone_dyn_ready handshake under silicon timing; the shadow-tag DRAM refill
  race (A-2, not yet probed); the hostcall/domain-boundary path. Board datum needed to
  localize: the faulting LDC's writeback port / trans_id.
  Answer: tests/fpga-repros/S07-capability-untagged-on-reload/rtl/ANSWER-FROM-THE-RTL-LANE.md

## S-08 ROOT-CAUSED: an S-06 P4 width bug; FIXED, needs re-synthesis (2026-08-15)

The `caplifive_s06fullfix.bit` regression (S-08: userspace ecall undelegated, monitor takes
the trap) was a bug in the S-06 fix itself — NOT the reporters' lane-packing hypothesis
(packing verified aligned from the generated switcher). P4 made every dom-switch context
store an unconditional 16-byte granule write; the scalar CSR context rows are 8-byte-stride
slots, so each save clobbered the NEXT slot with zeros before the sequential exchange read
it — medeleg restored 0, delegation died. Reproduced in sim with a positive-controlled
killing test (a real sealed-context CALL whose callee checks the restored CSRs: medeleg
zeroed pre-fix, sentinel intact post-fix; CAPENTER provably does not drive the switcher,
which is why no green test could ever see this). Fix: dom-switch width honors the
switcher's per-row metadata_en. Commit `9fd5507be` on `fpga-testing-dev-s06fix` (one on top
of squashed `25035c4c0`), sweep evidence `verif/sim/s06-s09fix.txt` (call-hot and
revocation RESTORED to baseline signatures). Answer to the reporting lane:
`tests/fpga-repros/S08-.../rtl/ANSWER-FROM-THE-S06-AUTHOR.md`. **Next: the project lead
pushes the branch; the board owner re-synthesizes from `9fd5507be`.** The S-06 acceptance
rungs remain untested on silicon either way.

## S-06 FIXED IN RTL — sim-validated, awaiting synthesis (2026-08-14)

The untagged-`ldc`/`stc` high-half loss (S-06) is fixed in the RTL, not worked around. Branch
`fpga-testing-dev-s06fix` on `capstone-ariane`, delivered as ONE squashed commit `25035c4c0`
(the phased P1–P6 history with per-phase gate evidence is archived locally on branch
`s06fix-phases-archive`; the squashed tree is byte-identical to the phased HEAD). A real 1-bit capability tag rides beside every compressed-metadata lane (64→65,
dom-switch 128→129); untagged `ldc`/`stc` is now a verbatim 128-bit copy with the tag cleared,
and tag-setting is opcode/tag-driven, never inferred from `|metadata|` — this also closes the D7
live-forgery hazard. Verilator-validated against a pinned 73-test baseline: semantics-neutral
phases bit-identical; the S-06 family flips to assert the fix (`untagged-ldc-stc-128` passes,
`s06-lowhalf-zero{,-swap}` FAIL→SUCCESS); two security directed tests pass with firing controls.
Two adversarial audits + a QEMU-differential confirm the contract holds and could not break the
scope. Trail: `history/14-08-2026_18-30-00_s06-rtl-fix-p0-p6.md`; handoff:
`tests/fpga-repros/S06-untagged-ldc-stc-high-half/` (00-README STATUS block).

**BLOCKED on two external steps, in order:** (1) `git push` of the branch is 403 — refined
diagnosis: the stored credential pushes `llvm-capstone` fine but lacks WRITE access to
`project-starch/capstone-ariane` specifically (per-repo permission, not a dead token). Fix =
grant that account write on capstone-ariane, or supply a token that has it. The squashed RTL
commit is LOCAL-ONLY until then. (2) hardware
synthesis + flash (board owner). After flash: acceptance boot, then SQLite with
`SQLITE_LDC_HIGH_HALF_FIXUP=0` and no `-capstone-guard-cap-granule-copies`.

**Bears on S-07** (below): S-07 is "a capability read back from memory arrives untagged" — the
same tag-through-memory path S-06's fix rebuilds. The fix MAY subsume S-07 on silicon; not
asserted, to be checked on the flashed bitstream. One residual NOT closed: AMO tag resurrection
(I4), documented with a repro, tracked as a separate follow-up.

## SQLITE RUNS ON SILICON — ~77% of executions complete (2026-08-14)

The headline changed today. SQLite's basic workload — CREATE, three INSERTs, a SELECT returning all
three rows, finalize — runs in a pure-capability domain on the FPGA and returns the correct rows,
in **10 of 13 genuine executions**. Measured over three boots (control + eight repetitions each, all
controls passing), not inferred from a lucky run.

The other 3 are **S-07** and all three landed at the **same instruction**, `output_text+0xdc`, from
two different physical placements: a capability read back from memory arrives untagged, mcause 25.
The site is fixed per image; only whether it fires is sporadic. Reproducer package, ready to hand
over as a single link: `tests/fpga-repros/S07-capability-untagged-on-reload/`.

The **extended** workload still wedges (`sqlite3DbMallocRawNN`, same defect). `output_text` is our
own harness, not SQLite. No timing number is admissible from these runs — the S-06 workarounds,
both confirmed ON in the measured binary, add ~33 KB of `.text` and a branch per granule.

Numbers: `ref/fpga-silicon-measurements-for-paper.md` §4e (which supersedes §4c, "the domain does
not complete"). Trail: `history/14-08-2026_18-30-00_s07-wedge-rate-and-fault-site.md`.

## TWO SEPARATE SIGNATURES, R-18 AND R-19 -- both handed over, both worked around (2026-08-08)

**Do not merge them.** R-18 is the ZEROING form: the victim is written with 0 and counts up, and
raw readbacks show NO metadata anywhere. R-19 is the METADATA-IN-SLOT form: the store's own slot
returns `compress_cap(NULL) + n`, e.g. `0x08000A31` = `0x08000000` + 2609. They share a trigger
class and ONE workaround clears both. Whether they are one defect or two is **UNKNOWN** and neither
package asserts it. Packages: `capstone/tests/fpga-repros/R18-scalar-store-metadata-clobber/` and
`.../R19-movc-zero-metadata-in-slot/`; each is a self-contained link handed out on its own.

**What simulation shows, stated at the strength the evidence supports.** A dual-bank splash is
DEMONSTRATED in Verilator with a one-instruction matched control -- but the slot it damages is not
the slot the board damages (sim: 8 bytes away and in the LOWER half of the row; board: 4 bytes away,
UPPER half). Treat it as a real mechanism demonstrated, NOT as the board symptom reproduced. R-19's
form does not reproduce in simulation at all, and the test that "passes" there fires its trigger
ONCE where the reproducing test fires it 64 times and fails to reproduce even the R-18 splash -- so
that pass rules nothing out. A 13-second `-DTRIG_IN_LOOP=1` run would settle it and has not been done.

    issue_read_operands.sv:1140   metadata onto the store's write-user sideband, UNGATED by opcode
    wt_dcache_mem.sv:138          st_wr_cap = |wr_user_i -- classified by VALUE, not opcode
    wt_dcache_mem.sv:234-237      such a store asserts BOTH banks
    wt_dcache_mem.sv:153-155      the same byte enable to both banks
    wt_dcache_mem.sv:156-158      bank 1 takes wr_user_i, bank 0 takes wr_data_i  <- NOT confirmed

`R XOR 8` is **WITHDRAWN** as a board rule: it holds in ten builds whose victim lies 8 bytes from
the trigger and fails in six whose victim lies 4 bytes away, and distance is invariant under base
alignment. It still holds in simulation.

**Bitstream: `caplifive_65536_nodes.bit` is the REFERENCE SILICON.** Project lead's decision
2026-08-08: no reflash. Every R-18/R-19 measurement, old and new, was taken on it. Whether either
signature appears on `caplifive_fixed_forward.bit` is UNTESTED and is not going to be tested --
do not re-open this, and do not describe any result as a cross-bitstream comparison.

**Simulation, measured 2026-08-08 with a working positive control.** `movc-zero-self-clobber.S`
fires its trigger ONCE and passes; at `-DTRIG_IN_LOOP=1` it fires 64 times and FAILS (tohost 3,
1974 cyc) with witness A zeroed. So trigger count is the differentiator and a single-shot directed
test proves nothing -- but the check IS now known to fire, and with it firing the store's own slot
still read back CLEAN. R-18's splash reproduces in simulation; R-19's metadata-in-slot form does
not, even at matched trigger count.

**WORKAROUND, ours, CONFIRMED ON SILICON UNDER A CONTROLLED 2x2 (2026-08-08) and needing no
bitstream:** emit `addi rd, x0, 0`. The old justification (c8 567 -> c8fix 576, "same geometry, one
instruction") was CONFOUNDED -- those two are linked 0xc0000 apart on a layout-sensitive defect.
A 7-arm boot crossed the workaround against the link address and settled it: damaged at 0xf0000 and
0x70000 both return 67699255; cured at 0x30000 and 0x50000 both return 67699264; R-19's fdd/fdw at
0xb0000/0x90000 give 0x08000A31 vs 2609. Effect tracks the INSTRUCTION, not the layout.
instead of `movc rd, zero` for integer-zero materialisation. `movc rd, zero` writes
`compress_cap(NULL)` = `0x08000000` into the register's capability shadow; an integer op leaves it
zero, `st_wr_cap` is never asserted, and no dual-bank write happens. Our `-O0` codegen uses `movc`,
which is why the trigger is pervasive. **This removes the COMMON case, not the class** -- any value
reaching a store's data register from a capability-producing op still splashes.

**NOT YET DONE, and the highest-value next step:** several documented "silicon miscompiles",
`matmult_int` (R-1) above all, may be THIS bug. If so the workaround clears them too, which is a
larger result than R-18 alone. One board boot tests it.

Handover package (board images + `sim/` directed tests + RVFI trace + rebuild recipe):
`capstone/tests/fpga-repros/R18-scalar-store-metadata-clobber/`. Message drafted for the board
owner at `/tmp/capstone/boardowner-msg-R18.md` (uncommitted, by policy).

Trail, including all eight retractions and what caused each:
`history/07-08-2026_23-55-00_r18-localized-to-row-mate-traffic.md`.

## WHERE THE PROJECT ACTUALLY IS (2026-08-07) -- read this before anything below

**Bitstream: `caplifive_65536_nodes.bit`.** Board results before 2026-08-04 were measured on an
older bitstream and must be re-checked before being relied on.

**FIXED and silicon-proven since the last update**
* **C-14, destructive `movc` of a live scalar.** The fix asks per SITE via ReachingDefAnalysis
  over ALL reaching defs, not per function. `locfl3` returns its oracle 26 on silicon where the
  pre-fix build wedged, control green in the same boot; lit 47/47; QEMU ladder clean. It also
  clears `loc1`, `locfl8` and `matmult_int`. NOTE it is provably INERT at -O0 (matmult_int is
  byte-identical with the fix on and off), so it does not cover anything in the SQLite domain,
  which builds -O0.
* `matmult_int` STAYS on the NOT-controls list -- see
  `history/06-08-2026_21-10-00_matmult-int-is-not-cleared-by-the-c14-fix.md`.

**TWO OPEN SILICON DIVERGENCES. Neither is root-caused. They are NOT the same fault.**

> **UPDATED 2026-08-07 (boots 51-55 + RTL simulation). Divergence B has been re-characterised
> from the ground up, and FIVE conclusions were RETRACTED in the process. Read
> `history/07-08-2026_02-30-00_nested-loop-capability-index-iteration-loss.md` before acting on
> anything about B written above or below this box.**
>
> **B is TWO faults with DIFFERENT victims, not one.** Stage 19 returns all three counters packed
> (`p<<20 | k<<16 | qc`) instead of qc alone, which every earlier rung did:
>
> | rung | p | k | qc | cycles | implied iters |
> |---|---|---|---|---|---|
> | c0 shift0 | 64 | 9 | 576 correct | 44001 | 575 |
> | c4 shift4 | 64 | 9 | **909** | **69081** | ~904 |
> | c8 shift8 | 64 | 9 | **567** | 44074 | 576 |
>
> * **k, the inner index, is knocked BACKWARDS mid-loop** (c4). The inner loop runs ~904
>   iterations instead of 576 and qc faithfully counts them. Both loops still exit at k=9, p=64
>   because they terminate on their own conditions -- a transient corruption is invisible in the
>   final values. ONLY THE CYCLE COUNT EXPOSES IT.
> * **qc, the accumulator, LOSES STORES** (c8). Cycles confirm exactly 576 iterations ran and nine
>   increments never landed.
>
> **THE CYCLE COUNT IS A FREE DISCRIMINATOR AND WAS UNUSED FOR THE WHOLE INVESTIGATION.** It is
> printed on every RESULT line (`ladder_perf_domain.h`). cycles/76.58 = iterations actually
> executed, which separates "lost an increment" from "ran extra passes" independently of the
> returned value. Read it on every arm.
>
> **RETRACTED -- do not resurrect:**
> 1. "A 16-byte capability store corrupts memory above its footprint, up to ~12 bytes"
>    (`635d2d4ea894`). Refuted: wp0 reproduces the same value at 3x the distance, different row,
>    different frame size.
> 2. **OVER-WIDE CAPABILITY WRITE.** Refuted directly: a witness at sp+0x10 written before the
>    loop and read back with a LOOP-FREE UNGUARDED load returns 0xA5A50000 BIT-EXACT after 576
>    capability stores. Memory above the store is NOT damaged.
> 3. **STALE-METADATA STORE MISCLASSIFICATION.** Refuted: a `movc a0,zero / movc a1,zero` barrier
>    against a same-size `nop` control -- byte-identical images bar two instructions -- BOTH
>    return 567. (The RTL bug is real: st_wr_cap = |wr_user_i, wt_dcache_mem.sv:138. It is not
>    what bites here.)
> 4. **`value = f(k bits[3:2])`.** Refuted: mispredicts gp16/gp32.
> 5. **"The defect needs two RMW slots exactly 8 bytes apart"** (`2a9ef7a255ac`). Refuted:
>    separations of 12 and 20 also fail, and shift0 has separation 8 and is CORRECT.
>
> Checked against all ten builds, NO single geometric variable is a function of the outcome. An
> independent disassembly of every artifact found builds with BYTE-IDENTICAL frame geometry
> returning DIFFERENT values (906 vs 909, correlating with FDREG_GUARD) -- there is a hidden
> variable no geometric law contains.
>
> **RTL SIMULATION NOW WORKS** -- see the `rtl-sim` skill. Directed tests in ~13s with the RVFI
> trace giving every load's address AND returned value. `stc-neighbour-load.S` and
> `stc-counter-pair.S` (submodule, branch `capstone-bootstrap`) do NOT reproduce at either RTL
> revision, across four rounds of added fidelity (pair 8 bytes apart, nested loop, resetting
> index, faithful -O0 sequence with three capability loads). HEAD 458982093 and `7aac52f93` are
> indistinguishable on them, cycle-for-cycle. **(Corrected 2026-08-08: `7aac52f93` is NOT "the
> board's" revision. The resident bitstream is `caplifive_65536_nodes.bit`, and at `7aac52f93`
> the pool is `reg head : logic[10]` = 1021 nodes; the 65536-node pool arrives at `91ea10837`.
> The synthesis commit is UNRECORDED, bounded below by `91ea10837`.)** The untested difference is the DOMAIN
> context: capenter, the monitor-carved stack, CPMP.
>
> **NEXT:** stage 20, the role swap -- the same three frame slots holding different variables, to
> separate "damage follows the SLOT" from "damage follows the VARIABLE". `p == k+4` and "the
> accumulator is the upper slot" have never been varied in any build.


> **CORRECTION 2026-08-08, hours after the entry below was written.** The paragraph that follows
> presented the precall result as new and called it "the split the evidence could not previously
> make". **It was already known.** Commit `cf2ab143aceb` is titled "The entry glue is not the SQLite
> fault: it completes in four seconds", and `ref/SILICON-BLOCKER.md:246` records
> `sq_pc.dom:0 -> obs=40465 = 0x9E11, returned in 4 s` — measured on 2026-08-06 **with `k800`
> passing as a control in the same boot**, which today's rerun did not have. So a board boot was
> spent reproducing a published result, and the failure was not checking `SILICON-BLOCKER.md`
> before building. The reproduction stands as an independent N=2 and nothing more.
>
> **AND THE BRACKET ITSELF WAS INVALID.** `sqlite_silicon.dom:0` has NO staged dispatch --
> `CAPSTONE_SQLITE_STAGE` is undefined in every current build, so the `0x5A6E` marker is absent and
> selector `:0` is **inert**. Verified byte-wise across `sq-base`, `sq-precall`, `sq-postcall` and
> the staged `sqlite-silicon` copy: `0x5A6E` count 0 in all of them. So `:0` ran the FULL DATABASE,
> and "entry+return only wedges" was never a fact. This is recorded verbatim at
> `ref/SILICON-BLOCKER.md:197-206`, and `build-sqlite-silicon.sh:269-284` carries a preflight gate
> added "after this exact mistake cost several board sessions". It cost another one.
>
> **The bracket was closed anyway, long ago.** `sq_qr` returns `obs=0x9E33` in 4 s -- "ENTERS, RUNS
> C, WRITES `res`, RETURNS" (`SILICON-BLOCKER.md:162-177`). The call transfer, `domain_main` body,
> `res` write, return and glue unwind all work. An `INTERP_RETURN_POSTCALL` probe was built before
> this was known; it is redundant and was not run.
>
> **THE ACTUAL FRONTIER**, three localizations further in:
> `sqlite3_initialize` -> `sqlite3PcacheInitialize` -> `sqlite3RegisterBuiltinFunctions` -> the
> **qr15/qr16 pair**. Levels 15/18/19/20/23 RETURN, 16/21/22 WEDGE. Level 20 inlines the exact
> `sqlite3InsertBuiltinFuncs` body and RETURNS; level 16 calls the real function and WEDGES. So the
> union write, the branch, the real global and the search are all cleared. 21/22/23 are explicitly
> UNATTRIBUTED (three changes at once, images shifted 372-480 B); "an argument capability wedges" is
> REFUTED by fdreg stage 4. **The pair that carries weight is 15 vs 16: two bytes at 0x242C6-7,
> identical file sizes and section offsets, `lvl` runtime so both paths compile into both images at
> the same addresses.**
>
> **R-18 IS EXCLUDED FROM THE SQLITE BLOCKER (2026-08-08, board, control green).** The link recorded
> below is CLOSED, in the direction that rules R-18 out. A minimal 13 KB off-SQLite repro isolated
> the one trigger class that survives the codegen workaround inside `sqlite3InsertBuiltinFuncs` -- a
> store whose data register came from `ldc` rather than `movc`:
>
> | arm | trigger in the loop | victim |
> |---|---|---|
> | `k800` | -- | 4, control OK |
> | `gz0ref` | `movc`-from-zero | **9** of 576 -- DAMAGED |
> | `gldcfix` | **`ldc`-sourced only** (workaround on) | **576** -- CLEAN |
>
> `ldc`-sourced stores do NOT trigger R-18. Re-scored against that, the SQLite hang path holds
> **five** measured-triggering sites, ALL `movc`-from-zero, and the workaround removes all five --
> yet the wedge persists unchanged. **R-18 cannot be SQLite's blocker.**
>
> It also corrects the exposure figures: `sqlite_silicon.dom` has 4948 raw sites but only **2333**
> of the measured triggering class; 985 are `ldc`-sourced and harmless, and the remaining ~1630
> (`movc` register-to-register, `cincoffset*`) are UNTESTED. The earlier "5494 sites, 44% removed"
> framing over-stated exposure by counting every capability producer as a trigger.
>
> **A LIVE LINK TO R-18, measured 2026-08-08:** `sqlite3InsertBuiltinFuncs` contains **2** R-18
> trigger sites in the baseline -- a `movc`-from-zero at `0x13ca30` and an **`ldc`-sourced store at
> `0x13ca98`**. The codegen workaround removes the first and CANNOT reach the second. So R-18 is
> half-excluded there, not excluded: the survivor is exactly the class the flag does not cover, and
> it sits in the one function the whole bisection points at.
>
> What today did add, on the SQLite track: **R-18 is excluded as the entry blocker** (a build with
> the workaround, hang-path trigger sites 6 -> 1, wedged identically), the **carve count is 179**
> so R-12 pool exhaustion does not bind this build, and the **driver guard is fixed** — it had
> hard-stopped on the 0x9E11 sentinel *twice*, once on 2026-08-06 (recorded there as "harness
> artefact", never fixed) and again today.

**A -- THE ENTRY GLUE IS NOT THE FAULT (2026-08-06; reproduced 2026-08-08).** `INTERP_RETURN_PRECALL`
returns sentinel `0x9E11` from immediately before `domain_main`, after the carve loop and after
`RUN_CAP_INIT`. On silicon it **RETURNED**: `SQ: obs=40465`, with `ENT2:00009E11` in the monitor
entry trace and the full sequence `A/dom-ok`, both region shares (SHA6 x4), `G/enter`, `H/return`.
So **the carve loop AND cap-init both complete on hardware; the fault is inside `domain_main` or in
reaching it.** This was established on 2026-08-06; see the correction above.

Two things eliminated the same day:
* **R-12 (rev-node pool exhaustion) does NOT apply to this build.** The carve count read from the
  `.capstone_gp_initdesc` header is **179**, not the 1059 the plan assumes; string merging collapsed
  it. The plan's Step 1 (trim `SQLITE_OMIT_*` under 1000) is aimed at a dead constraint.
* **R-18 is not the entry blocker.** A build with the `movc`-zero workaround (hang-path trigger
  sites 6 -> 1, `movc`-from-zero 2333 -> 18) wedged identically at stage 0.

Note the driver hard-stopped on this run claiming the domain "was not staged" -- a FALSE ALARM, its
guard only knew staged `0x5A6E` markers. Fixed. The `.dom` was verified byte-present in
`rootfs.cpio` before the boot and the UART shows the domain plainly ran.

**A (historical) -- the SQLite blocker, a HANG.** Wedges in sqlite3_initialize ->
sqlite3RegisterBuiltinFunctions -> sqlite3AlterFunctions -> sqlite3InsertBuiltinFuncs.
MINIMAL REPRO, layout-proof: `/tmp/capstone/sqlite-qr15` returns and `/tmp/capstone/sqlite-qr16`
wedges, and the two images differ by exactly TWO BYTES -- one immediate at 0x332c4
(`li a1,0xf` vs `li a1,0x10`). `lvl` is a runtime variable so both code paths are compiled into
both images at identical addresses; layout is excluded by construction. Level 15 links a static
FuncDef clone array with an inline loop; level 16 hands the SAME array to the real
sqlite3InsertBuiltinFuncs, which is a SINGLE loop. Both created AND entered. NO off-SQLite
reproducer. NO mechanism.

**B -- an iteration/value divergence, WRONG NUMBERS not a hang.** Deterministic, and it returns
values, so it is bisectable where A is not.
* INSIDE SQLite the trigger is a FOUR-WAY CONJUNCTION, with nine controls each removing exactly
  one condition and each exactly correct against a MEASURED QEMU oracle: a NESTED loop, AND a
  CAPABILITY access in the inner body, AND an index that is the INNER counter, AND that index
  RESETTING each outer pass. L31 has all four: board 567 of 576 on SIX runs, QEMU 576.
* B is NOT A. `sqlite3InsertBuiltinFuncs` is a single loop; level 39 is exactly that shape and
  returns 576.
* OFF-SQLITE, a 20 KB rung now diverges too (`lfa`/`lfb`/`lfc`, boot 39-41) -- but with a
  DIFFERENT signature: a CONSTANT +330 that tracks the expected value (576->906, 288->618,
  144->474). The loop counts correctly and the accumulator starts at 330 rather than 0, i.e. a
  stale stack slot. Whether that is the same fault as L31's deficit of 9 is NOT established.
* Full control table: `history/07-08-2026_02-30-00_nested-loop-capability-index-iteration-loss.md`

**WHAT TO DO NEXT**
1. Stage 9 (`init9.dom`, built) returns the accumulator IMMEDIATELY after `unsigned qc = 0`.
   0 means the initialisation lands; 330 means it does not. This is the single most decisive
   queued probe.
2. `lf0.dom` (built) is the TRUE control for `lfa` -- stage 7 at LEAVES=0, same base VA, same
   guard, same build path. fdreg7 is NOT that control: it differs in five ways at once.
3. Sweep the global count down from 43 to find the smallest reproducing rung, then freeze it
   into `capstone/tests/fpga-repros/`.
4. Only a SIMULATION can answer what differs on a pass that skips its inner body. A 20 KB
   reproducer makes that feasible for the first time; S01 has an open request for exactly this.

**TRAPS THAT COST BOARD TIME THIS WEEK -- all now mechanised**
* The staged runner hard-stopped on any WRONG value, because PROBE_SENTINELS can only list
  CORRECT ones. Eight consecutive boots ran their control plus ONE probe. Fixed to match the
  0x9E marker FAMILY (`82bb2f338fcd`).
* The runner reported an R-16 ENTRY STALL as "created and entered". Entry is `SQ: G/enter`,
  NOT `SQ: A/dom-ok` (`9860bd962c87`).
* Firmware over 32 MB is rejected AFTER the board is locked and power-cycled; preflight now
  gates on it.
* `grep` here is ugrep and silently returns NOTHING on UART captures containing control bytes.
  Scan them with python3.
* A probe that adds capability-bearing globals moves the capability sections and entry-stalls;
  reuse existing globals.
* Rungs with ~170 cap-table entries entry-stall; 43-107 enter reliably.

## HOW PROGRAMS REACH THE BOARD CHANGED (2026-08-03) — UART TRANSFER IS RETIRED

**Never ship a program to the board over the UART console again.** Bake it into the buildroot
image (`overlay/test-domains/` AND `build/target/test-domains/`, then `A=linux-rebuild`
followed by `A=opensbi-rebuild`) and invoke it from the shell. Measured: a ~10 KB domain took
MINUTES to transfer (16 chars per socket.io emit, each an HTTPS round trip), while the same set
baked in ran **10 domains in ONE boot in ~5 minutes** — the JTAG upload happens anyway.

* Sanctioned drivers: `fpga_driver/run_baked_rungs_fpga.py` (ladder rungs),
  `fpga_driver/run_sqlite_stages_fpga.py` (SQLite / staged probes).
* DEPRECATED, each now carrying a header saying so: `run_ladder_perf_fpga.py`,
  `run_sqlite_fpga.py`, `run_ladder_base_fpga.py`. They still transfer.
* **The baked driver does not reboot between programs**, so everything after the first
  failure is collateral: at most ONE unknown per boot, last, after a known-good control.

Full rules: `ref/HOW-TO-LAUNCH-ON-FPGA.md` §"UART TRANSFER IS RETIRED". Older sections of THIS
file that describe `fast_xfer`/tier-1 transfer as current (below, ~lines 350-365) are history.

## Latest (2026-07-28) — bare-metal baseline works; ALL overheads revised UP

**Read `history/28-07-2026_02-30-00_RESULTS-bare-metal-baseline-works-*` before any
paper-facing work.** Paper-facing source of truth:
`ref/fpga-silicon-measurements-for-paper.md`; open issues: `ref/ISSUES.md`.

### I-2 is FIXED by removing the OS from the baseline

The baseline now runs bare-metal as an S-mode OpenSBI payload. Proof: the `ctrsanity`
control (identical 5-instruction loop both sides) reads **600,041 cyc bare vs 600,309
capability — ratio 1.000**, where Linux gave 728,727 (1.21x). Quality went from 1/15
passes tied at min instret to **15/15, spread 0**.

### The table — every number rose

| rung | opt | **cycles** | was | **instr** |
|---|---|---:|---:|---:|
| `rv8_primes` | −O0 | **1.263x** | 1.050 | 1.130x |
| `beebs_cnt` | −O1 | **1.353x** | 1.165 | 1.319x |
| `beebs_bs` | −O1 | **1.530x** | 1.274 | 1.058x |
| `beebs_prime` | −O0 | **1.683x** | 1.032 | — |
| `beebs_recursion` | −O1 | **1.955x** | 1.801 | 1.458x |
| `ctrsanity` (control) | −O1 | 1.000x | — | 1.000x |

**Pervasive spatial safety costs 26–96 % in cycles, not 3–5 %.**

### §3 "overhead is ABI, not hardware" is REFUTED

`rv8_primes` cycles grow **1.263x** against instructions **1.130x**; CPI **RISES**
1.762 → 1.970. The old "CPI falls" was interrupts inflating the baseline's CPI (~14
cyc/instr vs real code's ~1.8). **Do not claim enforcement is free per instruction.**

### Paper is now STALE and needs approval to update

`tab:spatialcost` still shows `beebs_bs` 1.274x (should be **1.530x**) and the prose
still says 27 % cycles / 5.8 % instructions. The "ABI not enforcement" paragraph was
already removed, which turned out to be right. **Per CLAUDE.md, ask before editing.**

### Superseded — the min-of-16 era (2026-07-28, earlier)

### BOARD RESULTS 2026-07-28 — ONE defensible row; the 3.2/5.0/80 headline is WITHDRAWN

**Read `history/28-07-2026_01-30-00_*` and `history/28-07-2026_00-10-00_*` before any
paper-facing work.**

Measuring each baseline **16x** and keeping the least-disturbed pass (min instret; the
count of passes tied at that minimum is the cleanliness evidence) changed the table
materially, and every change ran the same way — **our overheads are larger than claimed**:

| rung | ties | old | **new** |
|---|---|---:|---:|
| `beebs_bs` | **15/15**, 45-cyc spread | 1.181x | **1.274x cyc / 1.058x instr — CLEAN, the only defensible row** |
| `beebs_prime` | 5/15 | 1.073x | **>=1.605x** (was published as 1.032x — WITHDRAWN) |
| `beebs_cnt` | 1/15 | 0.773x | 1.165x (the impossible sub-1.0 was interrupts) |
| `rv8_primes` | 1/15 | 1.051x | 1.055x — **uncorrected, too long for a clean pass** |

**Cause (I-2, confirmed):** the Linux baseline services timer interrupts inside the
bracket. A control kernel compiling to the identical 5 RISC-V instructions on both
targets runs at a metronomic **6.003/6.001 cyc/iter in the domain** vs **7.29** under
Linux; the excess scales 3.9x for 4x the work at ~14 cyc per extra instruction.

**Section 3 ("overhead is ABI, not hardware") is SUSPENDED — it may invert.** Removing
the calibrated interrupt load takes `rv8_primes` from cycles 1.055x / instr 1.103x
(CPI *falls*) to cycles 1.280x / instr 1.132x (CPI *rises*). Only a **bare-metal
baseline** can settle it.

**min-of-16 works as a function of kernel length:** clean below ~2k cycles, large but
uncertified 10k-170k, **useless above ~700k**. Bare-metal baseline still required for
long kernels.

**Paper updated** (`old-parts/evaluation.tex`, commit `51479d8`, local only): table cut to
the single `beebs_bs` row, the ABI-not-enforcement paragraph removed rather than softened,
CPI footnote corrected to the in-domain range 1.20-2.58.

### Superseded — board results 2026-07-27 (13 boots)

**`beebs_bs` is a new, clean 4th row: 1.181× cycles / 1.058× instructions** (capability
2,258 cyc / 875 instr vs −O1 warm baseline 1,912 / 827; CPI 2.31 → 2.58). Reproduces
across two sessions and a power cycle.

**`beebs_cnt` is silicon-CORRECT but NOT publishable as a cycle row** — 1.138×
instructions yet 0.684× cycles, i.e. it would claim capabilities are 32 % *faster*.
Uncontrolled confound, logged as **I-2**; it also puts a question mark on `beebs_prime`'s
1.032×. `beebs_fibcall` miscomputes at ~94 % of baseline instructions; `beebs_fac` and
`beebs_duff` hang.

**R-1's same-object clause is CONFIRMED, not refuted.** `beebs_cnt` is the cross-object
control and it passes exactly as R-1 predicted. The repro package needs no correction.

> **⚠ A mid-session report that R-1 was refuted, and that "an ordinary rebuild flips a
> passing rung", were BOTH WRONG and are withdrawn.** Cause: the sweep was accidentally
> run at −O0 — `run_ladder_perf_fpga.py` rebuilds by default and `LADDER_OPT` was set only
> on the pre-build. Logged as **I-1** with the rules that prevent it. It was caught solely
> because a known-good rung was in the sweep as a control.

Trail: `history/27-07-2026_22-40-00_RESULTS-two-new-silicon-rungs-and-an-O-level-procedure-bug.md`.

**Paper:** `tab:spatialcost` should gain the `beebs_bs` row (needs approval per CLAUDE.md;
not done). `beebs_cnt` must NOT go in until I-2 is resolved.

### Board queue (2026-07-27) — superseded by the results above

`beebs_bs` is **silicon-correct** (887447230 = oracle, 2264 cyc) and needs only its
**baseline** boot to become the 4th measured row; both halves are now registered.
Four further rungs — `beebs_fibcall`, `beebs_fac`, `beebs_cnt`, `beebs_duff` — are
built, QEMU-green, oracles fixed, `-O1`, registered in **both** the build script and
`ladder_base_ctl.c`'s dispatch table. All four are **predicted PASS under R-1**, and
the predictions are recorded in `ref/ISSUES.md` before the board runs. `cnt` and
`duff` are the load-bearing pair: they are the first **cross-object** controls in the
whole investigation (every failing rung to date reads and writes ONE array through
two derived capability registers). If they pass, the measured table goes 3 → 8 and
R-1 stays narrow; if either fails, R-1 is wider than written and both the registry
and the repro README must be corrected before the package goes to the board owner.

**Trap that already cost one boot:** `ladder_base_ctl.c` keeps a hand-maintained
name→function dispatch table, separate from the `RUNGS` list in
`build-ladder-base-fpga.sh`. A rung in one but not the other builds clean and then
reports `--` for every column. Add to both.

Two further results that upgrade the paper: measured **CPI 2.0–3.2** (the draft
assumed 1, so `tab:appoverhead`'s SQLite figures roughly halve), and on
`rv8_primes` **+10.2 % instructions but only +5.6 % cycles with a *lower* CPI** ⇒
capability enforcement is near-free per instruction; the overhead is the
gp-captable ABI. That is one benchmark — caveat it.

### 2026-07-27 — two blocked rungs UNBLOCKED by compiler fixes (board-free)

`beebs_crc32` and `beebs_insertsort` now **build at −O0/−O1/−O2 and pass the QEMU parity leg**,
taking the ladder from 3 to 5 *buildable* rungs. **They were then measured on the board and BOTH
FAIL — the measured set stays at 3.** `beebs_crc32` hangs at −O1; `beebs_insertsort` returns
957879052 against an oracle of 271779359 with only 560 retired instructions, i.e. the compute
never ran. Both were already wrong on silicon at −O0 in the 25-07 sweep, so the compiler fixes
were **necessary but not sufficient** — they removed the build blocker and exposed the same
unexplained silicon divergence underneath. Not a regression. Trail:
`history/27-07-2026_15-48-02_RESULTS-the-two-newly-buildable-rungs-fail-on-silicon-too.md`.
The fixes themselves remain worth having:

1. **`beebs_crc32` was never a compiler bug.** The kernel generates its CRC table at runtime to
   avoid a large initialized global; −O1+ **constant-folds the loop** and re-materialises a
   2048 B *private* constant `.L.crctable`, which the cap-table glue cannot deliver (over the
   12-bit unrolled path, and the large-RO copy path needs a *linkable*, non-`.L` symbol visible
   from the glue's separate TU). Fixed by making the polynomial opaque to the optimizer — one
   line, no runtime change. **Generalises: any hand-rolled table meant to dodge the large-RO
   limit can be silently undone at −O1+, SQLite included.**
2. **`beebs_insertsort` — the clang crash was hiding a real defect.** Guarding an
   `APInt::getSExtValue()` assert in `SelectionDAGAddressAnalysis` exposed
   `Constant:i128<0xFFFFFFFFFFFFFFFC>` — **CodeGenPrepare zero-extends a negative address
   offset** into the pointer carrier (`AddrMode.BaseOffs` is `int64_t`, `ConstantInt::get`
   defaults to `IsSigned=false`). Invisible on ≤64-bit-pointer targets; on a 128-bit capability
   `−4` becomes a huge positive offset. It was producing a **wrong address**, caught only by our
   backend's fatal guard. Latent for any wide-pointer target, CHERI included.
3. **`i128 = and` was unlowerable** — the dispatch `return`ed the constant-mask helper
   unconditionally, so its bail left the node unlowered instead of falling through to the general
   path OR/XOR use.

**RV8 is NOT fixed — do not quote "0/7 → 5/7".** Five RV8 benchmarks now *compile* at −O1/−O2,
then **fail 10/10 at runtime** (3 silent hangs; `sha512`/`norx` take deterministic capability
faults, cause 5 OOB and cause 24, same PC at both levels). −O0 controls all pass. These are not
regressions — you cannot regress code that never compiled — they are pre-existing −O1+ codegen
defects newly exposed, and root-causing them is the next real compiler task.

**Regression status: clean.** Capstone lit 41/41, BEEBS 82/82, CoreMark, authority 32/32, RV8
−O0 5/5, full X86 + RISCV lit. The only failures are 6 `emutls*`/`tls-android` tests, **verified
pre-existing** by stashing the changes, rebuilding `llc`, and reproducing them identically.
Trail: `history/27-07-2026_12-59-35_three-codegen-fixes-unblock-two-ladder-rungs-and-rv8-at-O1.md`.

### The other 4 rungs

| rung | status on silicon |
|---|---|
| `matmult_int` | **HANGS** the `cscall` at −O1/−O2 — no result at any reachable config |
| `coremark_matrix` | **HANGS** at −Os and at −O0 @32 KiB — localized to `core_init_matrix` (#66) |
| `beebs_crc32` | cannot **build** at −O1+ (2048 B folded table overflows a 12-bit store offset) |
| `beebs_insertsort` | **crashes clang** at −O1 |

### RETRACTED — do not carry these forward

The 2026-07-25 sweep table below this section reported **4 rungs miscomputing**
under an "array-store-with-live-accumulator" framing. **Both the framing and that
rung classification are withdrawn:**

- The rungs contain **zero `shrink`** instructions, so the documented
  `shrink`→store root cause cannot apply; and `beebs_recursion` has no array at
  all. Bounds-representability is refuted too (the rung with the *largest* global
  passes). **Do not escalate the shrink story to the board owner.**
- **"Scalar rungs pass, array rungs fail" is too strong.** A controlled A/B showed
  two builds of the same rung differing only in `domain_main` — *with* the minstret
  instrumentation `beebs_prime` returns 1087631800 (wrong, deterministic across two
  sessions); *without* it, 582955588 = the oracle. **Four instructions, none inside
  the computation, flip a passing rung.** A passing rung is not stable ground —
  re-gate on the oracle after ANY domain change.
- **"Domain-entry fault" is dead** (#63, `LADDER_INSTR_MODE=7`): the entry path runs
  and both hanging rungs complete a full domain round-trip when the compute is
  branched over. The domain-boundary `fence.i` (#61) is therefore the wrong layer.
- **"Fragile `bne` loop exits" is dead** (#65). It was observed statically that
  `matmult_int` at −O1 emits 8 conditional branches **all `bne`** while −O0 emits 8
  **all `blt`**, suggesting one fault whose symptom the branch kind selects. A −O1
  build with ordered exits forced — verified 0 fragile / 8 ordered, QEMU-correct
  through the *same* controller — **still hangs, identically.** The codegen split is
  real but is a **correlate, not the cause**. Do not restate "one fault, two
  symptoms".

**The pattern to inherit:** two hypotheses died in two days, both by promoting a
strong *static* correlation to a *mechanism* before a board test could speak. At
~2.5 min/boot with days left, prefer a **bisect that needs no mechanism guess**
(mode 7 and #66 paid off; #65 did not).

### RESOLVED 2026-07-27 (board #67a–#67f) — `delin` in domain code wedges the RTL

**`coremark_matrix`'s first fault is NAMED, with a size-matched control.** Six boots,
each build QEMU-correct through the identical controller first. Full trail:
`history/27-07-2026_04-33-58_RESULTS-delin-wedges-the-RTL-controlled-and-second-fault-isolated.md`.

| probe | delta | board |
|---|---|---|
| #67a | while loop only | **RETURNS 9** |
| #67c | + **`delin`** (one instruction) | **HANGS** |
| #67e | #67c with `addi x0,x0,0` **instead** (size-matched) | **RETURNS 9** |
| #67f | `B = A + N*N`, **no `delin`** | **RETURNS 9** |
| #67d | **full** benchmark, **no `delin`** | **HANGS** |

1. **The `delin` opcode is the fault, not code layout.** #67c and #67e differ only in a
   4-byte instruction's *encoding* — same position, same `"+r"(A)` plumbing. This control
   was mandatory: the 26-07 A/B showed 4 added instructions can flip a rung.
2. **Not "`delin` is unimplemented".** The glue `delin`s several caps in *every* domain and
   passing rungs work. The difference is the operand: glue delins a cap **fresh from
   `split`**; domain code delins one **loaded by `ldc` from the cap-table** — which the glue
   already delin'd before `stc`, so on a type-preserving machine it is **NONLIN→NONLIN**.
   That is exactly the case `capstone-qemu` `f4d416c265` patched to be idempotent
   *"rather than faulting"*. Same QEMU-permissive / RTL-enforces shape as `C_GEN_CAP`.
   **Caveat:** instrumented QEMU reports that operand as **LIN**, so QEMU and the glue
   disagree about type after `stc`→`ldc`. Which side is wrong is a **board-owner question**.
3. **Dropping the `delin` is safe but insufficient.** #67f returns (the `rd != rs1`
   derivation does not consume `A` on hardware) and QEMU still gives 14343 — but the full
   rung still hangs (#67d). **≥2 independent faults.** Fault 2 is in the **seeding loop or
   later**, which revives the surviving static candidate: `coremark_matrix` is the only rung
   doing **narrow (`sh`) accesses through the block cap**. Next: phase-bisect inside the
   seeding loop, or widen `MATDAT` to 32-bit.
4. **`matmult_int` has no `delin` at all** — fault 1 cannot explain it. Still possibly two bugs.
5. **A minimal silicon repro now exists** (two 4-byte instructions, both QEMU-correct) — the
   *paper-acceptable* outcome: a documented hardware limitation, not an unexplained one.

### What survives, cumulatively

- The hang is **inside the compute**, not at domain entry.
- For `coremark_matrix` it is inside **`core_init_matrix`** — bisected against mode 7
  at the same −O0 @32 KiB config: entry-only **RETURNS**, entry + `core_init_matrix`
  **HANGS**, everything **HANGS**. That is one ~40-line function. Two candidates
  remain, **not yet separated**: the dimension loop
  `while (j < blksize) { i++; j = i*i*2*4; }` (`bgeu` `0x10428` / `mulw` `0x10444`),
  and the N×N seeding loop running `seed = ((order*seed) % 65536)` per element,
  writing `A[]`/`B[]` **through the gp-delivered block capability**.
- It is **not** the loop-exit condition, and **not** discriminated by instruction
  mix (M-extension ops included, re-checked properly), code size, global count, or
  `.bss` size.
- **Do not assume the two hanging rungs share a mechanism.** `matmult_int` has no
  data-dependent bound at all; `coremark_matrix` is built around one.
- **Three further framings refuted board-free (2026-07-27, lane C** —
  `history/27-07-2026_02-45-07_core_init_matrix-codegen-audit-three-framings-refuted.md`**):**
  (a) *"an extra capability load/store in a loop is the trigger"* — `rv8_primes`
  reloads its block cap from the cap-table **and** stores through a dynamically
  derived cap in its hottest loop, and is silicon-correct; (b) *"the block cap gets
  round-tripped through memory"* — the **passing** `beebs_prime` spills and reloads
  its block cap; (c) *"a redundant NONLIN→NONLIN `delin` faults on the RTL"* —
  instrumented QEMU shows **zero** redundant delins in the whole coremark run, so
  the cap is genuinely LIN at that site and the in-kernel `delin` is necessary.
  Also: at −O0 `core_init_matrix` keeps **no** live capability across the loop — it
  reloads **both** `A` and `B` from stack slots every iteration.
  **Surviving candidate, for `coremark_matrix` only:** it is the sole rung doing
  **narrow (`sh`/`sb`/`lh`/`lb`) accesses through the block capability** (4 stores +
  9 loads at −Os); all three passing rungs use word-or-wider only, and `matmult_int`
  has none — so it cannot be a shared mechanism. Treat as a candidate, not a cause.
  **⚠ Probe #67 as specified is a 3-way, not a 2-way:** the `delin` + `B = A + N*N`
  derivation block sits *between* the dimension loop and the seeding loop, so
  "return `N` before the seeding loop" leaves two candidates on its HANG branch.
  Move the split point before the `delin`, or make it 3-way.
- The corruption is a **silicon divergence** — QEMU runs the identical binaries
  correctly — and is **NOT proven a compiler bug**. If our code is ISA-legal and
  QEMU-correct, this is an RTL divergence to hand to the board owner with a minimal
  repro: a **paper-acceptable** outcome (documented hardware limitation).

Trail: `history/27-07-2026_00-58-47_RESULTS-65-falsified-66-localizes-hang-to-core_init_matrix.md`,
`history/27-07-2026_00-28-51_loop-exit-condition-splits-hang-from-miscompute.md`,
`history/26-07-2026_23-56-07_the-hang-is-in-the-compute-not-at-domain-entry.md`,
`history/26-07-2026_17-43-17_controlled-ab-four-instructions-flip-a-passing-rung.md`,
`history/23-07-2026_17-30-00_gp-captable-silicon-array-loop-miscompute-OPEN.md`.
Memory `project_gp_captable_codegen`.

### Tooling traps that silently corrupt this analysis

- **The Capstone-triple disassembler cannot decode M-extension instructions.**
  Domains build `-Xclang -target-feature -Xclang +m`, but `llvm-objdump` on a
  `capstone64` binary prints every `mul`/`div`/`rem` as `<unknown>`. Any
  mnemonic-keyed analysis must pass `--triple=riscv64 --mattr=+m`. (Re-run properly,
  the "no discriminating instruction" conclusion still **stands** — a trap, not a
  retraction.)
- **`<sym+0xNN>` in disassembly is not a branch target.** Regexes grabbing the last
  hex number on the line invert forward/backward branch classification. Strip `<...>`.
- **At −O0 clang emits a forward exit test plus an unconditional `j` backedge.** A
  metric counting only *conditional backedges* reports zero for every −O0 build.
- **A domain that hangs reports nothing at all** — the controller prints `res[]` only
  after the `cscall` returns, so "write a marker and read it back" probes are unusable
  on a hang. Design probes around *does it return at all*.

---

## Latest (2026-07-26) — xlang cross-language repro corpus (separate track from the board work)

**14 of 15 rows reproduce; 12 of 15 reproduce the temporal-borrow class the
benchmark is about.** All 15 `run.sh` pass and assert their expected outcome.
Stock toolchain only — no Capstone compiler, no QEMU fork, no board.

- Artifacts: `xlang/` (start at `xlang/README.md`).
- Full state, evidence and open decisions:
  `history/26-07-2026_18-04-21_xlang-phase1-state.md`.
- **Do not quote "14/15" for the temporal benchmark** — rows 6 and 11 reproduce as
  *spatial* heap-buffer-overflows, not UAFs, which contradicts the companion note's
  "every defect here is a temporal borrow" claim. Row 7 does not reproduce and
  appears not to exist as specified.
- Corpus is no longer monolingual: Lua↔Rust 2/2 and Rust→C 1/1 now reproduce, so
  the paper's "two subjects" framing is backed by artifacts for the first time.

## Superseded (2026-07-25) — silicon-ladder perf sweep, original table

**Kept for provenance. Its rung classification and explanation are RETRACTED by the
2026-07-27 section above — read that first.**

| rung (fresh dom) | silicon | oracle | mcycle | verdict as reported then |
|---|---:|---:|---:|---|
| rv8_primes | 99991 | 99991 | 17,283,292 | ✅ PASS |
| beebs_prime | 582955588 | 582955588 | 47,804 | ✅ PASS |
| matmult_int | 1166210317 | 774662735 | 76,498 | ❌ reported miscompile |
| beebs_crc32 | 1568735421 | 1703161001 | 311,902 | ❌ reported miscompile |
| beebs_insertsort | 255001740 | 271779359 | 10,463 | ❌ reported miscompile |
| beebs_recursion | 2095861164 | 1579141629 | 30,263 | ❌ reported miscompile |
| coremark_matrix | — | 14343 | — | transfer never landed |

Each was verified on a dom rebuilt after the 24-07 memcpy fix (`d078839`) and each
was **QEMU-correct** with that same fresh binary — that part stands, and it is why
this is a silicon divergence rather than a build artifact. `beebs_insertsort`'s
255001740 coinciding with the pre-fix memcpy signature was a **red herring**.

Two process findings from that sweep, both still valid:

1. **The runner could run stale binaries** — it reused pre-built `.dom`s and read a
   different dir than the build script wrote. Now rebuilds-by-default + hard-fails on
   stale (`4be78cb`/`bd03316`). It did not explain any of the miscompiles.
2. **Board transfer improved** (`fast_xfer`: Ctrl-C resync to escape the `> `
   continuation prompt a dropped char leaves; catch the wedge timeout and escalate
   instead of aborting; third slower tier). This recovered 2 of 3 previously
   unverifiable rungs. `coremark_matrix` was later shown **not** transfer-blocked.

Full table + mechanics + correction trail:
`history/25-07-2026_03-58-47_fpga-ladder-perf-sweep-results.md`.

Runner: `tests/rtl-smoke/fpga_driver/run_ladder_perf_fpga.py` — one full
power-cycle + JTAG reload per rung (each rung runs as first domain / clean icache;
warm `reset halt` does NOT re-enter OpenSBI), tier-1 `fast_xfer.fast_put`
transfer, `insmod /capstone.ko` (UP image doesn't auto-load it). The `-b` LLVM was
rebuilt from scratch with `-capstone-gp-captable` (system `/usr/bin/clang++`,
`RISCV;Capstone`); all 7 perf domains build `cjalr=0 ldc-gp≥1`.

## Latest (2026-07-24) — CoreMark matrix on the silicon ladder (QEMU)

CoreMark 1.01's **matrix** benchmark now runs as silicon-ladder **rung 7** in a
pure-cap domain on QEMU: domain crc16 `14343` == native `cc -O0` oracle, static
gate `cjalr=0 ldc-gp=1`, `__CAPSTONE_LADDER_COREMARK_MATRIX_PASSED__`. Files in
`tests/runtime-qemu/silicon-ladder/coremark_matrix_{kernel.h,app.c,host.c}` +
`run-coremark-matrix-qemu.sh`. Matrix only (list/state CRCs are pointer-size-
dependent → wouldn't match a native oracle); driven standalone with CoreMark's
validation-run matrix params (N=9). Built `-Os` (pinned in the wrapper): CoreMark
matrix is ~4.7 KiB `.text` at `-O0` and overflows the 4 KiB PCC window; ~1.5 KiB
at `-Os`. **Note:** the `-b` clang is stale (predates the merged
`-capstone-gp-captable` flag); validated with a sibling checkout's current clang
driving the `-b` runtime — the `-b` LLVM build config was restored to shared +
`clang;lld` but the rebuild is deferred. Trail:
`history/24-07-2026_14-14-09_coremark-matrix-silicon-ladder-rung.md`.

## Latest (2026-07-22) — gp-free domain bring-up (silicon-shaped ABI)

On branch **`capstone-gp-free`** (off `capstone-bootstrap`; not merged/pushed): a
real globals-using integer app now runs **correctly** in a pure-capability domain
**gp-free / cjalr-free** on QEMU with the `gp = PCC(cursor 0)` fabrication
**disabled** — `gp` is an image-covering data cap the **monitor** delivers via the
cscratch stack region (board owner's confirmed channel; same as `capstone-c`).

- **Compiler `-capstone-gp-free`** (default off, byte-identical off; lit 40/40):
  plain `jal`/`jalr` calls/returns within PCC (no `cjalr`); global data via `SCC`
  (absolute in-bounds cursor) not `cincoffset gp` (which needs the unrepresentable
  cursor 0). Files: `CapstoneAsmPrinter.cpp`, `CapstoneISelDAGToDAG.cpp`
  (`selectCall`), `CapstoneExpandPseudoInsts.cpp` (`expandCapGlobalBase`).
- **Monitor** `create_domain` mints `gp` with `C_GEN_CAP` + stashes it at the
  cscratch region top slot; **glue** `start-gpfree-cscratch.S` loads it. **QEMU**
  `op_helper.c` gates the 4 gp-fabrication sites behind `CAPSTONE_GP_FABRICATE`
  (default on) + a `CAPSTONE_GP_STANDIN` monitor stand-in.
- Proof + repro: `tests/runtime-qemu/gp-free-domain/` (`build-and-run.sh` →
  `__CAPSTONE_GPFREE_DOMAIN_PASSED__`); default domains still pass with the rebuilt
  monitor. Trail: `history/22-07-2026_16-09-12_gp-free-domain-bringup-qemu-proof.md`;
  guidance memory `project_silicon_gp_delivery_boardowner_guidance`.
- **Remaining:** same `create_domain` change on the FPGA (caplifive-system) copy +
  board image rebuild + a silicon smoke/cycle run (Experiment A). QEMU + monitor
  submodule edits kept as local experiments (no submodule-source commits).

## Latest (2026-07-15) — read this first; sections below predate it

Since 2026-07-03 the active work shifted from C1/C2 to the **performance
reframe** (2026-07-13): eager CHERI matches our temporal security, so the
separating axis is **performance**. That comparison is now **DONE** and in the
paper.

- **CHERI-vs-Capstone temporal-safety perf comparison — DONE (QEMU-to-QEMU, two
  workloads).** Eager CHERI (the config that matches our security) pays
  **~14–17 M instr per free** (address-space sweep); our revoke-at-free is
  **O(1), +5 instr/op**; async CHERI is 1.9–6.4× but blocks **0/11** UAF at the
  contract point. Paper `evaluation.tex` §`sec:eval-perf-compare` filled
  (`tab:perfcompare` microbench + `tab:perftree` real-workload BST). CHERI stack
  is fully local at `~/cheri` (`tests/cheri-perf/`, `tests/cheri-baseline/`).
  Full report: `history/15-07-2026_00-20-00_cheri-capstone-perf-comparison.md`;
  plan `plans/perf-cheri-vs-capstone-qemu.md`.
- **`-O2`/`-O1` capability-select ICE — FIXED (2026-07-15).** `lowerSELECT`
  crashed on an i128 cap select with non-null constant arms; fixed in
  `CapstoneISelLowering.cpp` (rematerialize constant arms as `li` via
  CopyToReg). This unpinned the Capstone BST tree probe from `-O0`; it now builds
  **and runs clean at `-O2`** (revoke-at-free +5, matching the microbench).
  Backend lit 39/39, clang 6/6, authority **26/26**. Trail:
  `history/15-07-2026_03-43-21_cap-select-o2-ice-fixed.md`.
- **Nightly orchestrator added:** `capstone/tests/run-nightly.sh`
  (build → lit → QEMU suites serially → report to `/tmp/capstone/`).
- **Corrections to the sections below:** the authority suite is now **26 domains**
  (not 20); `-capstone-shrink-stack` is **default ON** since 2026-07-03 (covering
  varargs save-area + dynamic alloca, so those are no longer "not yet"); the
  task-005 FastCC-i128 and revoke-intrinsic-DCE codegen defects are **resolved**.
- **Standing next step:** the Capstone **RTL cycle-accurate** number
  (human-in-the-loop; **postponed** pending the board owner's answer on automation).

## SQLite in-memory bring-up

SQLite 3.53.3 compiles, links, **and runs end to end** as a
`capstone64-unknown-elf` pure-capability domain using memsys5 over the static
arena and the runtime-initialized SQLite VFS skeleton. `run-sqlite-memory.sh`
executes `CREATE TABLE` / `INSERT` / `SELECT` and the domain returns correct rows
(`row name=alpha value=11 / beta=22 / gamma=33`, `__CAPSTONE_SQLITE_MEMORY_PASSED__`).
The pinned fetch/build/run workflow is in `capstone/benchmarks/sqlite/README.md`.

**Bring-up is complete — all 8 gaps resolved:**
- Gaps 1–2 (compiler): `CapstoneCapGlobalInit` recurses nested global aggregates
  (#71); clang memcpy-from-private-template of cap aggregates handled (#72).
- Gaps 3–4 (QEMU): untagged `ldc`/`stc` made bit-preserving over the full 128-bit
  word, enabling a tag-preserving `memcpy` (#73/#74).
- Gap 5 (compiler ISel): `cscincoffset` int+ptr operand order (#79).
- Gap 6 (SQLite alignment): 16-align `sqlite3NestedParse`'s `saveBuf` so the
  tag-preserving `memcpy` fast path carries Parse-tail caps (#80).
- Gap 7 (compiler): materialize interior-pointer capability globals
  (`&global[N]`) — `sqlite3aLTb/aEQb/aGTb` (#81).
- Gap 8 (SQLite alignment): 16-align the `BtCursor` embedded by `allocateCursor`
  (#82).

Full per-gap detail in `history/` (dated notes) and
`design/sqlite-gap6-memcpy-tag-preservation-proposal.md`. Follow-ups: the SQLite
8-byte-alignment class (gaps 6/8) may surface more instances under wider workloads.

**In-domain cap-fault delivery — abort retired (2026-07-03).** QEMU no longer
aborts on an in-domain capability fault: `riscv_cpu_do_interrupt`'s
`assert(env->priv < PRV_C)` is replaced (for `env->priv == PRV_C`) by a clean halt
— a structured `[CAPSTONE] domain halted by capability fault: cause=…` line then
`fflush`+`exit(0)`. This preserves the domain's serial output (`abort()` didn't
flush stdio — the gaps 8/9 "no serial output" cause) and turns a SIGABRT into a
named halt. The monitor host-trap path (`priv < PRV_C`) is unchanged. Validated:
full authority suite all-PASS, SQLite base+extended PASS, no abort in logs. Step A
proved the `ctvec` horizontal-trap path can't deliver this (a domain installs no
`ctvec`). **Return-to-host** delivery (domain terminates, host continues) is the
remaining, monitor-side step — see
`design/domain-fault-delivery-proposal.md` + `history/03-07-2026_00-00-03_*`.

## Verified baseline

All of the following pass on the `capstone-bootstrap` branch:

- LLVM Capstone backend builds the sample domain; `ld.lld` links native `EM_CAPSTONE`
- `capstone/caplifive-buildroot/build/local.mk` present — keeps the image on the Capstone-enabled OpenSBI path
- All HostCall probes pass: shared-region, stdout, filewrite, fileread, full file-handle
  lifecycle (open/write/read/sync/stat/truncate/close), path ops, combined file-object
- `run-nullblk-baseline.sh`, `run-nullblk-split-io.sh`, and
  `run-nullblk-split-rmmod.sh`
- `run-hostcall-all.sh`, `run-nullblk-all.sh`, and `run-all-beebs.sh` provide
  aggregate gates for reproducible full reruns; keep individual wrappers as the
  diagnostic entry points. The HostCall, `null_blk`, and full BEEBS aggregates
  have passed end to end; BEEBS has also passed with `RUN_ALL_BEEBS_JOBS=4`.
  `run-all-beebs.sh` is serial by default
  (`RUN_ALL_BEEBS_JOBS=1`) and has opt-in isolated parallelism via
  `RUN_ALL_BEEBS_JOBS=N`. It keeps child output in per-benchmark logs by default
  and prints compact pass/fail lines; set `RUN_ALL_BEEBS_VERBOSE=1` for streamed
  child output. It retries structured QEMU infra flakes before benchmark
  execution twice by default (`RUN_ALL_BEEBS_BOOT_RETRIES=0` disables this) and
  caps aggregate boot-to-login waits at 90 seconds by default
  (`RUN_ALL_BEEBS_LOGIN_TIMEOUT`), but does not retry benchmark marker failures.
- QEMU runtime smoke tests use snapshot mode, so repeated runs do not mutate `rootfs.ext2`
- Buildroot getty is pinned to `ttyS0`, avoiding intermittent boot-to-login hangs through `/dev/console`
- QEMU runtime smoke tests force `-smp 1`, avoiding intermittent boot stalls under the current OpenSBI/QEMU setup
- `run-coremark.sh` - all three algorithms, "Correct operation validated."; CoreMark now uses
  compiled C `domain_main`, not `coremark_domain_entry.S`
- `capstone/benchmarks/beebs/run-beebs-fac.sh` - first BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-insertsort.sh` - second BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fibcall.sh` - third BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-cnt.sh` - fourth BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-bubblesort.sh` - fifth BEEBS benchmark runs end to
  end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-prime.sh` - sixth BEEBS benchmark runs end to
  end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-recursion.sh` - seventh BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-janne-complex.sh` - eighth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-tarai.sh` - ninth BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-cover.sh` - tenth BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-duff.sh` - eleventh BEEBS benchmark runs
  end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-levenshtein.sh` - twelfth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-jfdctint.sh` - thirteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fdct.sh` - fourteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-strstr.sh` - fifteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ndes.sh` - sixteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arraybinsearch.sh` - seventeenth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-queue.sh` - eighteenth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-listinsertsort.sh` - nineteenth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-listsort.sh` - twentieth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-expint.sh` - twenty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-aha-compress.sh` - twenty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-md5.sh` - twenty-third BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-cast128.sh` - twenty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-matmult.sh` - twenty-fifth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-crc32.sh` - twenty-sixth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-mergesort.sh` - twenty-seventh BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-stringsearch1.sh` - twenty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-bs.sh` - twenty-ninth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fir.sh` - thirtieth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-lcdnum.sh` - thirty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ns.sh` - thirty-second BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ud.sh` - thirty-third BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nsichneu.sh` - thirty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arraysort.sh` - thirty-fifth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arrayheapsort.sh` - thirty-sixth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arrayquicksort.sh` - thirty-seventh
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-dllist.sh` - thirty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-hashtable.sh` - thirty-ninth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-aes.sh` - fortieth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-picojpeg.sh` - forty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-sha256.sh` - forty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-huffbench.sh` - forty-third BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-rijndael.sh` - forty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-crc.sh` - forty-fifth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-statemate.sh` - forty-sixth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-arcfour.sh` - forty-seventh BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-des.sh` - forty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-aha-mont64.sh` - forty-ninth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-dijkstra.sh` - fiftieth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ctl-stack.sh` - fifty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ctl-vector.sh` - fifty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-edn.sh` - fifty-third BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ctl-string.sh` - fifty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-qrduino.sh` - fifty-fifth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-rbtree.sh` - fifty-sixth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-miniz.sh` - fifty-seventh BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-slre.sh` - fifty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-wikisort.sh` - fifty-ninth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-trio-sscanf.sh` - sixtieth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-compress.sh` - sixty-first BEEBS
  benchmark runs end to end and validates its adapted LZW-state checksum marker
- `capstone/benchmarks/beebs/run-beebs-cubic.sh` - sixty-second BEEBS
  benchmark runs end to end with the soft-float/libm runtime and root oracle
- `capstone/benchmarks/beebs/run-beebs-sqrt.sh` - sixty-third BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ludcmp.sh` - sixty-fourth BEEBS
  benchmark runs end to end with the local const-array source workaround
- `capstone/benchmarks/beebs/run-beebs-minver.sh` - sixty-fifth BEEBS
  benchmark runs end to end and validates its adapted matrix checksum marker
- `capstone/benchmarks/beebs/run-beebs-frac.sh` - sixty-sixth BEEBS
  benchmark runs end to end with shared soft-float/libm support
- `capstone/benchmarks/beebs/run-beebs-st.sh` - sixty-seventh BEEBS
  benchmark runs end to end with correctly-rounded software `sqrt`
- `capstone/benchmarks/beebs/run-beebs-nbody.sh` - sixty-eighth BEEBS
  benchmark runs end to end with correctly-rounded software `sqrt`
- `capstone/benchmarks/beebs/run-beebs-qsort.sh` - sixty-ninth BEEBS
  benchmark runs end to end with a widened 1-indexed array and sorted-region hash
- `capstone/benchmarks/beebs/run-beebs-qurt.sh` - seventieth BEEBS benchmark
  runs end to end and validates all three quadratic root cases
- `capstone/benchmarks/beebs/run-beebs-select.sh` - seventy-first BEEBS
  benchmark runs end to end with a widened 1-indexed array and return-value oracle
- `capstone/benchmarks/beebs/run-beebs-newlib-sqrt.sh` - seventy-second BEEBS
  benchmark; self-contained `__ieee754_sqrtf`, upstream exact verifier with
  `exp[]` moved to `static const` (Bug #9), soft-float builtins only
- `capstone/benchmarks/beebs/run-beebs-newlib-exp.sh` - seventy-third BEEBS
  benchmark; self-contained `__ieee754_expf`, oracle tail vs host reference
- `capstone/benchmarks/beebs/run-beebs-newlib-log.sh` - seventy-fourth BEEBS
  benchmark; self-contained `__ieee754_logf`, oracle tail vs host reference
- `capstone/benchmarks/beebs/run-beebs-newlib-mod.sh` - seventy-fifth BEEBS
  benchmark; self-contained `__ieee754_fmodf`, oracle tail vs host reference
- `capstone/benchmarks/beebs/run-beebs-stb_perlin.sh` - seventy-sixth BEEBS
  benchmark; 3-D Perlin noise, self-contained oracle (`benchmark()` compares a
  10x10 plane against a `static const` table and returns 0 on full match);
  only external dep is `floor`, added to the shared soft-float libm
- `capstone/benchmarks/beebs/run-beebs-matmult-float.sh` - seventy-seventh BEEBS
  benchmark; `matmult` source built `-DMATMULT_FLOAT` (float[10][10]), soft-float
  builtins only, FNV-1a checksum of the global `ResultArray` vs a host reference
  (`--gc-sections` drops the dead `values_match`/`frexpf`/`fabsf`)
- `capstone/benchmarks/beebs/run-beebs-whetstone.sh` - seventy-eighth BEEBS
  benchmark; classic Whetstone over the shared libm (added `atan`); built
  `-DPRINTOUT` with a capturing `POUT` that FNV-folds every module's outputs,
  compared exactly to a same-libm host reference

Most BEEBS correctness-marker wrappers now share `beebs_simple_domain.c` and
`beebs_simple_host.c`. Keep separate per-benchmark domain/host files only when
the marker ABI or host behavior is genuinely different; currently the older
`fac`, `fibcall`, and `insertsort` wrappers keep custom markers.

Most Capstone-specific benchmark source adaptations live in explicit `.c` files
under `capstone/benchmarks/beebs/adapted/`; shell scripts generally orchestrate
fetch/build/link/run rather than embedding C source. Full-replacement adapted
files (bubblesort, prime, cnt, duff, janne_complex, tarai, levenshtein,
recursion) are compiled directly. Prefix/tail files (crc32) and tail-append
files (strstr, insertsort, jfdctint, fdct, aha-compress, nettle-md5,
nettle-cast128, nettle-arcfour, nettle-des) are concatenated with the stripped
upstream source at build time. `huffbench` uses checked-in adapted C snippets
for its freestanding prefix and RNG replacement. `aha-mont64` uses a checked-in
rewrite helper for constant hoisting. `ndes` uses a checked-in rewrite helper
for pointer-based aggregate passing and explicit table delinearization.
`ctl-string`, `qrduino`, `miniz`, `slre`, and `trio-sscanf` are generated as
scratch sources under `$CAPSTONE_TMP_ROOT/beebs-build` because their adaptations
are local include/stub/allocation/verifier rewrites rather than reusable
replacement translation units.  `slre` additionally uses a checked-in tail file
(`adapted/beebs_slre_capstone_tail.c`) to avoid the `char *regexes[]` global
pointer array that would require caprelocs.  `wikisort` uses a checked-in tail
file to keep the upstream prefix while replacing the Range/sort/test tail.
`trio-sscanf` strips hosted includes, builds with `TRIO_SSCANF`,
`TRIO_EMBED_STRING`, float/file/dynamic-string features disabled, a minimal set
of embedded `triostr` helpers, and checked-in freestanding libc stubs.
`compress`, `cubic`, `minver`, `qsort`, `qurt`, and `select` use adapted
oracle tails because the upstream verifiers return `-1`. FP benchmarks use
compiler-rt soft-float builtins and, where needed, the shared
`adapted/beebs_softfloat_libm.c` domain libm.

`build-beebs-simple-capstone-common.sh` now supports `BEEBS_EXTRA_DEFINES`
(array of `-D` defines, e.g. `BEEBS_EXTRA_DEFINES=(QUICK_SORT)`),
`BEEBS_STRIP_FROM_REGEX` plus `BEEBS_ADAPTED_TAIL_SRC` for single-source
tail-replacement adaptations, and includes `-fno-jump-tables` unconditionally
(jump tables use raw integer addresses which fault on Capstone since loads
require capabilities).

## Resolved blocker

The 2026-06-09/10 split `null_blk` unload blocker is resolved. The hang was
diagnosed as lost timer progress after split-domain activity: QEMU traces showed
that the final timer H-interrupt was taken while `mie.MTIP` was disabled, after
which OpenSBI did not reprogram the timer and RCU/percpu-ref progress stopped.

The fix is in `capstone/capstone-qemu`:

- Capstone H-interrupt selection in `riscv_cpu_local_irq_pending()` now considers
  only interrupts enabled by `env->mie`.
- `rmw_mie64()` calls `riscv_cpu_check_interrupts()` after `mie` changes so a
  pending H-interrupt becomes deliverable when software reenables it.

The split null_blk package also keeps the safer fixes found during investigation:
metadata is borrowed per domain call instead of permanently shared, and
`null_validate_conf()` copies back only validated scalar configuration fields.

All temporary Linux/OpenSBI/QEMU trace and printk diagnostics were removed before
the verified run.

## Important distinction

The validated path is the **split host/domain runtime path**, not a full hosted
`capstone64-unknown-linux-gnu` Linux userspace. The helper is ordinary guest Linux;
the domain is a Capstone-loaded domain.

## Known backend bugs (stable workarounds in place)

The prologue frame-lowering bug is fixed and validated. Three remaining LLVM backend
workarounds from CoreMark bring-up stay in `capstone/benchmarks/coremark/build-coremark-capstone.sh`
and should only be removed after focused root fixes. Details: `plans/backend-compiler-fixes.md`.

The `va_list` capability-tag-loss backend bug is fixed and validated: `va_start`/
`va_arg`/`va_copy` now lower with capability ops (`stc`/`ldc`, 16-byte `cincoffset`
stride). The CoreMark `ee_printf_asm.S` trampoline is removed — `ee_printf` uses a
standard C `va_list` and CoreMark still validates. This unblocks the `va_list`
prerequisite for `trio`.

The `sub i128` pointer-decrement backend blocker is fixed and validated:
`ptr - integer` and `ptr + (-offset)` now lower through `cincoffset` with a
negated XLEN offset.

The `sub i128` pointer-difference backend blocker is also fixed and validated:
`ptr - ptr` now lowers by extracting both capability cursors with `lcc ..., 2`,
subtracting the XLEN cursor values, and sign-extending the integer result back
through the `i128` carrier when needed. `ctl-string` is the proof benchmark.

Stack-passed capability arguments are fixed: a function with >8 args whose extra
args are pointers had its stack-slot address computed with an integer `ISD::ADD`
(→ `addi`, tag-stripping), delivering the callee an untagged capability.
`CapstoneTargetLowering::LowerCall` now uses a capability `CIncOffset` for the
slot address (test `stack-cap-arg.ll`; repro `tests/runtime-qemu/stack-cap-arg-repro/`).
This unblocked RV8 `norx` and is the same class as the `va_list` fix.

The i128 non-vector-shift assertion (Bug #3) is fixed (`lowerScalarI128Shift`
general constant-shift fallback). **Capability globals are now auto-tagged**: the
`CapstoneCapGlobalInit` ModulePass synthesizes a per-module `__capstone_cap_init`
(called from `my_first_domain/start.S` before `domain_main`) that materializes
initialized capability globals in place at runtime — a tag cannot live in the
static image. Validated via `static-cap-typed-load-repro` + lit
`static-cap-global-init.ll`. Design:
`design/capability-globals-init-decision.md`.

## Capability granularity & provenance (C1/C2 — paper track)

After the three benchmark suites completed, work pivoted to the paper's security
contributions. **An external audit (2026-06-29,
`history/29-06-2026_15-08-22_granularity-provenance-audit.md`) reviewed this whole
direction; its findings are folded in below — read it before paper-facing work.**
Current state on `capstone-bootstrap`:

- **Bounds model** (`design/capability-bounds-model.md`): the narrowing op is
  **`SHRINK`** (`int_capstone_cap_shrink`); `SPLIT`/`SHRINKTO` exist in the ISA
  but are unwired. **Audit correction:** the `<4 KiB exact / grain-above`
  representability rule is **spec-derived, NOT measured** — this QEMU keeps exact
  fat bounds in a side table (`cm_map`) and restores them on load, so observable
  `SHRINK` is **exact at all sizes**. Un-narrowed bounds are segment-granular
  (single `PT_LOAD` ≈ whole image).

- **C1 object-granularity narrowing — INITIAL SLICES (not a spatial-safety
  theorem; broad `gp`/`sp` roots remain, permissions stay RWX):**
  - **Globals** — `selectLGA` (`CapstoneISelDAGToDAG.cpp`) narrows each sized data
    global to `[&g, &g+sizeof(g))`. Flag `-capstone-shrink-globals` (**default on**);
    functions / unsized externs not narrowed.
  - **Heap** — NOT a libc policy: only **two benchmark-local allocators**
    (`rv8_malloc.c`, dtoa `malloc_beebs`) `cap_shrink` returns; trio left
    un-narrowed (its `realloc` over-reads); CoreMark uses stack storage. Do not
    call this "heap default-on."
  - **Stack** — fixed stack objects narrowed to `[&obj, &obj+size)` via the
    shared `narrowToFrameObjectBounds` helper, now covering **both** the
    bare-`FrameIndex` address **and** interior pointers / load-store bases
    (`materializeFrameIndexAddrBase`), flag `-capstone-shrink-stack`
    (**still default off** pending the empirical default-on matrix). Not yet:
    varargs save-area, dynamic `alloca` (variable-size + spill slots excluded by
    design). Object- not subobject-granularity.
  - Validation is **functional only**: **CoreMark ✓, RV8 7/7 ✓, BEEBS 82/82 ✓**
    with global+heap on; stack-on smoke = CoreMark + 9 stack-heavy BEEBS ✓. Found
    a **real OOB bug**: rijndael wrote 8 bytes through a `char r[4]` (patched).
    **Code-size overhead measured across all 90 domains (CoreMark + 7 RV8 + 82
    BEEBS, 2026-07-01):** globals narrowing costs a near-constant **~15.6 bytes
    per narrowed global**; as % text, **median 1.83%, mean 4.17%, range 0%
    (no sized globals) – 46% (`statemate`, generated WCET tables)**; no
    correctness regression — matrix + full table in
    `design/c1-coverage-matrix-and-overhead.md`. **Runtime/cycle overhead still
    NOT measured** (functional QEMU, no cycle-accurate path) — don't claim it.
  - **Negative pointer difference fixed:** exact signed element scaling now
    restores `srai` after narrowing the i128 pointer-difference carrier to XLEN;
    genuine logical shifts remain `srli`. Positive and negative runtime probes
    pass, including `low - high == -7`.

- **Provenance/authority evidence suite** (`capstone/tests/capstone-authority/`,
  `run-authority-suite.sh`): 20 domains pinning runtime behavior (source + asm +
  QEMU trap/no-trap vs an oracle). forge/ptr→int→ptr **tag-fault**; global/heap/
  stack edge/index `_oob` **bounds-fault**; positive/negative pointer differences
  and last-valid-byte controls pass. A struct-field over-read is
  **no-trap-today**, confirming the subobject-bounds gap. The additive opt matrix
  passes all 12 eligible domains at `-O1/-O2/-O3`; 8 assembly-verified O0-only
  probes are explicitly skipped. Runtime fact:
  a domain-mode capability fault currently **aborts the QEMU model** (a
  `riscv_cpu_do_interrupt` assertion) after emitting the diagnostic.

- **Regression tests:** lit `cap-shrink-globals.ll`, `cap-shrink-stack.ll`
  (on/off A/B), `ptr-diff-signed.ll`, and updated
  `static-cap-global-init.ll`. Full Capstone lit suite green (32 tests).

- **C2 (provenance verifier) — REDESIGNED (v2, 2026-07-01), awaiting reviewer
  sign-off before implementing.** The audit found v1 (`UNKNOWN`-accepting,
  opcode-only) was a hygiene checker, not a proof. The redesign in
  `design/c2-provenance-verifier-proposal.md` §"Design (v2)" folds in all three
  fixes: no permissive `UNKNOWN` (`ROOT`/`CAP`/`INT`/`TAINTED` lattice, TAINTED-as-
  authority flagged), IR→MIR intent + calling-convention arg/return seeding,
  precise per-opcode transfer functions (LDC propagates memory tag; tied-operand
  ops inherit+validate; integer-as-base is a fault not a forge), two separated
  properties (P1 non-forging / P2 preservation), and a small hand-proved formal
  model with the corpus as validation. v1 retained in the doc for history. Do NOT
  implement until the reviewer signs off on v2.

- **Audit's strategic reframing (for the reviewer):** object bounds re-derive
  CHERI; Capstone's novelty is linearity/revocation/`SPLIT`/**root-elimination**.
  Proposed stronger frame: **provenance + attenuation + root-elimination** (trusted
  `SPLIT` removes the ambient broad root from application code). A
  research-direction decision, not yet acted on.

## Where to go next

- Next milestone: `state/current-next-step.md`
- Test entry points: `ref/testing-matrix.md`
- Deep design docs: `design/`
