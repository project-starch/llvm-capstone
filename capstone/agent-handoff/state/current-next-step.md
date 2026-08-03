# Current recommended next step

> **THE BLOCKER HAS A DEDICATED LIVING DOCUMENT:**
> `capstone/agent-handoff/ref/SILICON-BLOCKER.md` — everything known, with every claim tagged
> MEASURED / SOURCE / INFERRED, the refuted table, the two wedge families, **minimal
> reproducers (section 8e)** and the wedge-triage procedure. Read it before this file; several
> conclusions recorded below were later retracted there.


## 2026-08-02 — C-16 FOUND AND FIXED (a COMPILER bug), but SILICON IS STILL BLOCKED

### Read this before anything else in this file

**The SQLite blocker was largely our own codegen, not the RTL.** `SelectionDAG::getMemset`
typed its destination argument in **addrspace 0** (`PointerType::getUnqual`), while the real
destination is an AS200 **128-bit capability**. Call lowering therefore truncated the pointer
(`PseudoTRUNC_CAP` + `ADDI`), stripping the tag, and `memset` then did capability arithmetic on
an untagged base — writing 15 bytes of struct tail padding **through a garbage address, once
per array element**. Silent on hardware because the RTL does not check a `cincoffset` base.

Fixed by taking the address space from `DstPtrInfo`. No-op for AS0 targets.
Registered as **C-16 FIXED** in `ref/ISSUES.md`.

    reproducer   silicon-ladder/strarray_app.c (+ _host.c, oracle 420)
                 DOMAIN_OPT_LEVEL=-O0 bash run-ladder-qemu.sh strarray   -- ~1 min, NO BOARD
    verified     codegen addi x8 -> cincoffsetimm; reproducer PASS
                 stage 10 NON-STATIC rc=0x00; full SQLite QEMU gate PASSES with no workaround
                 ladder 6/6

### But the board still does not run SQLite

Fixed compiler, no workaround, position 1: both shares complete, `SQ: G/enter` reached, then
**600 s of silence** and abort. So **at least one more fault exists and it is silicon-only**.
Do not present C-16 as having unblocked the board.

### The methodological rule this session established

**QEMU-gate every probe before shipping it to the board.** The staged probes were built and
sent to the board for four sessions without ever being run under QEMU — the one tool that
asserts on untagged capability arithmetic. A broken instrument produced four sessions of
"measurements", and the stage-100 "cursor is off by 57 bytes" result was INVALIDATED because
of it. QEMU passing is necessary and NOT sufficient (QEMU asserts, the RTL accepts silently) —
but QEMU *failing* is free and immediate.

### Next steps, in order

1. **Staged split on silicon with the fixed compiler** (f10/f2/f3 built and staged; run may be
   in flight). Stage 10 first: if it now returns, the remaining blocker moved later into
   `sqlite3_initialize`/`open`; if it still stalls, C-16 was only part of the same construct.
2. ~~**Re-run the four R-14 variants with the fixed compiler.**~~ **DONE 2026-08-03 — R-14 does
   NOT close as a C-16 duplicate.** It is reproduced on silicon and now looks like an **RTL
   defect**: the failing capability store is measurably in bounds (type NONLIN, cursor ≥ start,
   cursor+16 ≤ end, 1312 B of headroom), the identical binary is correct under QEMU, and every
   compiler-side mechanism proposed was refuted on the board. Reproducer packaged for hand-over
   at **`capstone/tests/fpga-repros/R14-frame-pad/`** (`k800` returns 4, `k1200` never returns,
   sources differ only in the size of a dead `volatile` pad).
   **Remaining, cheapest first:** read `lcc` field 5 (`perm`) at the failing address — a missing
   write permission would be a legitimate reason to fault and would point back at the glue;
   then read `lcc` off the faulting `stc`'s own base register; then get `mcause`/`mepc` for
   `k1200` itself by adding the debug-mux read (already in `run_sqlite_stages_fpga.py`) to
   `run_baked_rungs_fpga.py`.
3. Re-read every "silicon miscomputes" claim in `SILICON-BLOCKER.md` and `ISSUES.md` against
   C-16 — several are plausibly the same bug.
4. Decide whether to delete `SQLITE_STATIC_BUILTINS` now that the real fix exists.

---

## 2026-08-01 (early) — READ THIS FIRST: results on this board are NOT always reproducible

### The finding that changes how everything else must be read

Determinism had never been tested in this campaign. It has now. Running the SAME binaries
repeatedly inside ONE boot:

    wd66  x3   rc = 2, 2, 2          DETERMINISTIC
    wd63  x2   rc = 0x0E, 0x0F       NON-DETERMINISTIC
    fn63  x2   rc = 0x0F, 0x0E       NON-DETERMINISTIC (confirms it)

**`wd63` returns different answers on identical back-to-back runs.** Consequences:

1. **"The first walk succeeds, later ones fail" is RETRACTED** — it was built on `wd63 = 0x0E`,
   and the next run gave `0x0F` (array 0 overran too).
2. **Every single-sample conclusion in this campaign is unsafe**, including
   `stage 52 = 0xC1` ("`lit[1]` is the bad one"), which drove days of bisection. A result seen
   once is a sample, not a fact.
3. **Run every probe at least 3x from now on.** The runner accepts the same `.dom` repeated in
   `SQLITE_STAGE_DOMS`; repetitions inside one boot are nearly free.

### What is genuinely established (and reproducible)

* **It is a LIVELOCK, not a hang.** Stage 51 returns `0xB1` — the domain runs and RETURNS.
  Every hypothesis requiring the core to stop is retired.
* **The emitted pointers are provably correct.** `__capstone_cap_init` derives the literals at
  `0x6da / 0x6e0 / 0x6e6` — deltas of exactly 6 — matching the merged `.rodata` container.
  1544 straight-line instructions, zero calls, zero branches; the one reused register is
  correctly spilled and reloaded. (Proof about EMITTED code, not about runtime values.)
* **`wd66` is a deterministic reproducer** (5 samples, all `2`): the same element walked TWICE
  through the SAME pointer — first walk overruns, second terminates correctly. Its two walk
  loops were verified byte-identical (23 instructions each, `0x36994` / `0x36a40`, only branch
  targets differ). Use `wd66` as the vehicle for any further bisection; it is the only stable
  failing case found.

### Refuted BY MEASUREMENT (do not revive without new evidence)

| hypothesis | how it died |
|---|---|
| `cincoffset` consumes its source | `flu_unit.anvil:43,:62` return `rs1` unchanged |
| `STC` clears its source register | `dyn_unit.anvil:427` returns `rs2_v` unchanged |
| carve/rev-node pool exhaustion | 183 carves measured against a ~1000 budget |
| `LDC` consumes its memory slot | stage 57/58 = 7 (two reads, both non-NULL and equal) |
| the SHA5 wedge is self-inflicted | UNGUARDED `wd51` returned `0xB1`, unchanged |
| array identity ("Nth array is broken") | `wd60/61/62`, one shared array, only the loop failed |
| granule misalignment is the root cause | `ga60 = 0xC1`, identical with granule-aligned glue |
| "first walk succeeds" | `wd66 = 2` inverts it; `wd63` varies anyway |
| store ordering / missing fence | `fence rw,rw` before `domain_main`: `fn66 = 2`, no change |

### Real but LATENT (fix on its own merits, NOT this bug)

* **Carve base granule misalignment.** idx 170, `sqlite_heap`, 262144 B, granule 512,
  `base % g = 64`, `len % g = 0`. Simulation over the real descriptor: granule-align OFF -> 1
  unrepresentable carve, ON -> 0, for every plausible region top. **The 2026-07-31 revert note
  had the failing end backwards** (it blamed the length; the length is fine). `ga60` shows
  enabling it does not fix the livelock, so it is a separate correctness issue.
  Knob: `INTERP_GRANULE_ALIGN=1`.
* **`wd65` wedges where `wd62` returns 5** — same array, same single walk, differing only in a
  `volatile` pointer load, and the failure modes differ (domain death vs overrun-and-return).
  Open thread; do NOT assume it shares a cause with the livelock.
* Domains run with `mtvec = ctvec = 0` (no monitor writes `dom_seal[1]`) — upstream design
  question, deliberately not patched unilaterally.

### Next step

Re-take the foundational bisection results WITH REPETITION, starting with stage 52, and treat
any result that varies as unusable until characterised. `wd66` is the stable vehicle for
narrowing the livelock itself.

### Tooling and traps (all of these bit during 31-07/01-08)

1. **Never read `board-<tag>.log` for results** — it carries the accumulated console
   scrollback, so it returns markers from EARLIER runs. Only `PROBE_SCOPED_OUT` is valid.
2. **Never pattern-match a string your own command line contains.** `while pgrep -f "make
   build LINUX_PAYLOAD"` matched itself and deadlocked six shells for ~50 minutes while
   reporting false progress. Use a bracket pattern (`"[m]ake ..."`).
3. **A domain earns an early slot only if THAT EXACT BINARY has returned before.** Guarded
   `wd53` and `wd65` were placed early as "controls" on source-level identity after the binary
   changed underneath; each wedge ended its run.
4. **`llvm-objdump --disassemble-symbols` silently truncates** (~470 of 9088 bytes, stopping at
   a local `.Lpcrel` label). Use `--start-address/--stop-address` and check the disassembled
   size against the symbol size.
5. **Prune only your OWN staged domains** — never package-installed ones (`fib`, `sbi`,
   `smode`, `thread`), which desyncs buildroot's stamps (six boot failures). Keep
   `sqlite_silicon.dom` and `sqlite_host.user` for the freshness gate.
6. **Each staged block's statics land in EVERY build** unless `#if`-guarded — `stage` is a
   function parameter and probes build at `-O0`, so nothing folds. Guards are in place for
   stages 51-66; keep adding them.
7. Build probe batches with `build-stage-probes.sh` — it prints per-artifact hashes and a
   distinct-hash count, so a silently-cached build cannot pass as fresh.
8. Image size: 10.5-15.4 MB boot fine; 26 MB and 46 MB do not.

## STRUCTURAL LIMIT ON SAMPLING (learned 2026-08-01, affects how to read every result)

**A wedge ends the board session, so a WEDGING domain can never be repeated inside one boot.**
Every "n=2/n=3, deterministic" figure recorded above is therefore necessarily from a
RETURNING domain (`wd66`, `wd61`, `wd62`, `wd63`). Every wedging result — `wd10`, `wd52`,
`wd53`, `wd65`, `wd67` — is a SINGLE sample by construction, not by choice. "Wedges
consistently" has never actually been established for any of them.

Consequences for method:

1. To repeat a wedging case, use **separate boots** (~5 min each). Batch the returning probes
   within a boot; batch the wedging ones across boots.
2. **Prefer a probe that RETURNS a marker over one that wedges** — that is what the stage-51
   watchdog achieved (silence -> `0xB1`) and it is what made any of this measurable. When
   bisecting a wedging stage, build the bounded/early-return variant FIRST.
3. Do not describe a wedge as reproducible without naming how many boots it was seen in.

Also: **stage N ⊃ stage M for M < N on the normal path.** Stage 3 (`sqlite3_open`) contains
`sqlite3_initialize`, which contains `sqlite3RegisterBuiltinFunctions` (stage 10). Ordering
stage 3 before stage 10 guarantees the run dies before reaching 10. Order staged probes so a
superset never precedes the subset it depends on.

Also: **never wait on a process by name.** Three separate deadlocks were caused by a command
polling `pgrep -f <pattern>` where its own command line contained the pattern — including once
where a bracket pattern (`"[b]uild-..."`) still matched because the same script later invoked
the real string. Six shells hung ~50 minutes on the first occurrence while reporting false
progress. Sequence steps inside ONE script instead of polling for another task.

## The blocker is SOLID, not intermittent (3 separate boots, 2026-08-01)

Because a wedge ends its session, stage 10 was sampled across three SEPARATE boots, each
running `wd66` first as a liveness control:

    boot1: WEDGED    boot2: WEDGED    boot3: WEDGED
    samples=3  successes=0  ->  ALWAYS FAILS

**`sqlite3RegisterBuiltinFunctions` fails every time.** This closes the possibility raised by
`wd63`'s run-to-run variation that SQLite might sometimes get through — there is no retry path
to an existence proof. It is also the first wedge in this campaign established as reproducible
rather than assumed from a single sample.

Note what this does and does not say: the BLOCKER is deterministic across boots, while some
PROBES (`wd63`) vary within a boot. Both are true; they are different quantities. The
non-determinism does not rescue the blocker, and it does not excuse the earlier single-sample
conclusions either.

### Where to resume

`wd66` remains the only stable failing reproducer (7 samples, all `2`): the same element walked
twice through the same pointer, first walk overruns, second terminates, with the two loops
verified byte-identical. Narrowing that is the live thread — it is small, deterministic, and
sits in the same code family (data-dependent string walk) as the blocker.

Do NOT resume by re-deriving "lit[1] is broken": that came from `stage 52 = 0xC1`, a single
sample that could not be re-taken (the guarded rebuild wedges), and `wd62`/`wd59` both show
`lit[1]` walking correctly in isolation.

## RETRACTED AGAIN: it is not "the first walk". It is the BINARY LAYOUT.

Same-binary baseline, run in one boot (this is the like-for-like comparison that was missing
every previous time):

    wd70  counted loop, NUL test in body   rc=0x45  x4   correct, deterministic
    wd71  BARE walk of lit[1], nothing before it   rc=0x45  x3   CORRECT, deterministic
    wd66  same walk, as the first of two    rc=2     x7   first walk overruns, deterministic

`wd71` performs exactly the operation `wd66` calls "the first walk" — same element, same array,
same `while (z[guard])` source — and it TERMINATES CORRECTLY at index 5, three for three.

**So "the first data-dependent walk fails" is retracted.** Walk ordering is not the variable.
Each binary is internally deterministic and different binaries with identical C semantics give
opposite answers, so **the variable is the LAYOUT of the built domain.**

This is the thread flagged repeatedly all session without a name:

* guarded `wd52`, `wd53`, `wd65` wedge where their UNGUARDED builds returned;
* `wd54`/`wd55` wedge while logically identical probes return;
* `wd66` fails where `wd71` succeeds on the same operation.

All one phenomenon: **whether a given domain's string walk works is decided by how that domain
was laid out, and is then stable for that binary.**

Caveat, do not over-unify: `wd63` varies WITHIN a single binary across runs in one boot
(`0x0E` / `0x0F`). Layout cannot explain that. There are at least two effects here, and they
must not be merged on convenience.

### What this makes worth doing next

The question is now well posed and cheap to attack offline: **what differs in the LAYOUT
between a passing binary (`wd71`) and a failing one (`wd66`)?** Both contain
`capstone_probe_lit`; compare, for that symbol and its carve, the address, the alignment, the
carve base/length the glue computes for it, and the granule those imply. That is a static
diff over two artifacts already on disk — no board time — and it is the first time in this
campaign the comparison has been between two binaries that differ ONLY in outcome, not in
what they are testing.

### ...and the LAYOUT explanation is refuted too (offline, no board time)

Static diff of the PASSING (`wd71`) and FAILING (`wd66`) binaries:

* **Identical layout.** Both: 182 carves; `capstone_probe_lit` at vaddr `0x169c70`
  (`addr%16 = 0`, `addr%512 = 112`); the six 256-byte carves have identical indices, storage
  sizes, relative bases and blob offsets. Nothing about the data placement differs.
* **Identical loop code.** `wd66`'s first walk and `wd71`'s bare walk share the **same 21
  instructions** — `lui cincoffset cincoffsetimm ldc lui cincoffset cincoffsetimm lwu
  cincoffset lbu beqz j lui cincoffset cincoffsetimm lw addiw sw li bltu j` — i.e. the entire
  loop body (load, test, increment, bound-check, branch). They diverge only AFTER the loop:
  `wd66` breaks and falls through, `wd71` returns `0xB6` (`cincoffsetimm li sw j j`).

So the previous entry's conclusion ("the variable is the binary layout") is **withdrawn**. The
layout is the same, the executed loop is the same, and the results are still opposite and
still stable per binary.

**What actually differs between them, and is therefore all that is left:**

1. the ADDRESS of the loop (`0x36994` in `wd66` vs `0x36be4` in `wd71`) — i.e. instruction
   placement / I-cache line, not data;
2. the post-loop code path;
3. the surrounding stage code compiled into each binary.

That is a precisely posed question and a cheap one: build ONE binary containing both shapes
(the two-walk form and the bare form) so the comparison is within a single image, then vary
only the loop's alignment (e.g. `.balign` padding before it) and see whether the outcome
follows the address. If it does, this is instruction placement, which is a very different
class of bug from everything chased so far.

**Do not** re-derive a data-side explanation without first refuting the address hypothesis:
the data side is now excluded by direct comparison of the two artifacts, not by argument.

## Instruction placement REFUTED; a reproducible WALK-COUNT effect emerges

One binary containing BOTH shapes (bare walk, then the paired walks), with only the paired
walks' address moved by padding. Three SEPARATE boots, `wd71` control first in each:

    wd72  pad   0 bytes  @0x36ab4   WEDGED   (control wd71 = 0x45)
    wd73  pad +24 bytes  @0x36acc   WEDGED   (control wd71 = 0x45)
    wd74  pad +56 bytes  @0x36aec   WEDGED   (control wd71 = 0x45)

**Instruction placement is refuted** — the outcome does not follow the address. The control
returned correctly in all three boots, so the board was healthy and these are real results.

### The pattern that IS supported

Counting DATA-DEPENDENT WALKS performed by the domain:

| walks | binaries | result | samples |
|---|---|---|---|
| 1 | `wd71` | `0x45`, correct | 6 |
| 2 | `wd66` | returns `2` | 7 |
| 3 | `wd67`, `wd72`, `wd73`, `wd74` | **WEDGE** | 4 binaries, 4 boots |

`wd67` (three walks of one element) and `wd72/73/74` (one bare + two paired) are different
code in different binaries and wedge independently at the same walk count. One walk always
works; three walks always wedge. This is the first pattern in the campaign reproduced across
multiple binaries AND multiple boots.

It also subsumes the older observations without needing them to be about `lit[1]`: stage 52
walked 16 elements, stage 51 walked 16, stage 63 walked 4 — all high-count, all failing.
Stages 61/62 did ONE walk each and both passed.

**Shape of the mechanism:** monotonic degradation with the number of walks points at something
CONSUMED per walk and never released, rather than at any property of the data, the pointer, the
array, the layout or the code address — all of which are now excluded by direct measurement.
The rev-node pool is a fixed-size BUMP allocator with no reclamation
([[project_fpga_silicon_measurement_status]]), which is the right shape; note the earlier carve
count (183 vs ~1000 budget) only measured carves at ENTRY, not per-walk consumption at runtime.

### Caveat that must not be smoothed over

`wd66` reports `rc=2` = bit0 clear, bit1 set, i.e. "first walk failed, second succeeded". That
is the OPPOSITE order from a consumption story, and it has never been explained. Either the bit
encoding is being misread or something clobbers `m`. **Do not build on `wd66`'s bit order**
until a probe returns the two guard VALUES rather than a pass/fail bitmap.

### Next

1. Probe walk-count directly: N walks for N = 1,2,3,4 in one binary, returning the count
   completed before failure rather than a bitmap.
2. Have the probe return the raw `guard` value of each walk, to settle the `wd66` bit-order
   question.
3. If consumption is confirmed, read the rev-node allocator state via the debug mux
   (`rev_node_head` / overflow, sel `11001`/`11010`) before and after a walk.

## CORRECTION (multi-agent audit, verified against primary logs): the walk-count ladder was CONFOUNDED

Two claims recorded above are wrong. Both were checked directly, not argued.

**1. "3 walks always wedge — 4 binaries, 4 boots" is really 2 binaries, 2 boots.**
`wd73` and `wd74` NEVER ENTERED THE DOMAIN. Counting `SQ: G/enter` in the run-scoped files
(each contains the `wd71` control first, so the control accounts for one):

    sqlite-pad72.txt  G/enter=2, F/share2=2, SHA6=4, last line "SQ: G/enter"  -> wd72 ENTERED, wedged in-domain
    sqlite-pad73.txt  G/enter=1, F/share2=1, SHA6=2, last line "SHA5:00000001" -> wd73 died in region-share
    sqlite-pad74.txt  G/enter=1, F/share2=1, SHA6=2, last line "SHA5:00000001" -> wd74 died in region-share

They executed **zero walks**, so they are not data points about walk count at all. The real
support is `wd67` and `wd72` — n=2, each a single sample by the record's own rule.

**2. "Instruction placement REFUTED" is wrong, and probably backwards.** `wd72` (0 pad),
`wd73` (+32B) and `wd74` (+64B) are the same source; the UNPADDED one reached the domain body
and the two PADDED ones did not. Padding changed *where* the failure happened. That is
placement mattering, recorded as placement being ruled out.

**3. `wd63` falsifies a monotone walk-count/ldc threshold outright.** Its inner `break` exits
only the `while`, so all four iterations run: **four walks, and it RETURNS** (`0x0E`/`0x0F`,
seen in two separate boots). At `-O0` that is hundreds of `ldc` from one stack slot — far more
than the ~18 the 3-walk story implies is fatal.

**What actually survives:** `>= 3 walks through the SAME pointer` fails (n=2), while 4 walks
through FOUR DIFFERENT pointers return. "Same pointer" vs "count" has never been separated,
and the walks also differ in iteration count (a terminating walk is ~6 iterations, an
overrunning one is 65), so the x-axis was never controlled.

## THE HIGHEST-VALUE FINDING: a "wedge" is probably an UNTRAPPED EXCEPTION, not a stalled core

Verified against primary sources:

* The monitor zeroes all sealed slots and writes only 0, 2, 3 —
  `sbi_capstone.c:760` (`dom_seal[i] = 0`), `:782`, `:783`, `:784`. **Slot 1 is never written.**
* Slot 1 *is* `{ctvec, mtvec}` and *is* swapped in on domain entry — `csr_regfile.sv:399`
  (`7'd1: dom_switch_reg_resp_o = {ctvec_q, mtvec_q};`).
* Capability faults are ordinary traps (`ex_stage.sv:469`, `cva6.sv:1357`, cause `23 + code`;
  `capstone_unit.anvilh:289-296`), and with `mtvec = 0` they vector to pc=0, which is outside
  PCC and re-faults forever — silently, because the monitor's `EXCX` report is unreachable.

So an in-domain fault and a hung core are INDISTINGUISHABLE today, and "wedge" has been read as
"hang" throughout this campaign without evidence. This is a measurement defect, and fixing it
is worth more than any single hypothesis: it converts every future wedge into a returnable
marker and removes the sampling limit that makes wedging probes unrepeatable within a boot.

**Cheap confirmation available now, no code change:** run a wedging domain (`wd67`) and then
read the trap latch via the debug mux with `probe_wedge_regs.py` (switches=255 trap latch;
clear via switches=191 first). A latched trap proves the wedge is an exception.

## Latent bugs worth their own ISSUES.md entries (NOT this bug)

* `capstone_dyn_unit.anvil:302` sends `cap_load_ri.init` BEFORE the `NOT_CAP` check at
  `:303-306`, and it is the only error branch with no `abort_accumulation_load` — leaves
  `req_set` sticky.
* `scoreboard.sv:320-324` hardwires `wb[1..3].cap_data = '0`, forwarded at fixed priority
  (`issue_read_operands.sv:786-807`).

## Trap-latch read on the REAL blocker (stage 10), latch cleared per domain — 2026-08-01

    clear failures: 0
    sw=255 TRAP LOG   0x89  trap_seen=1  mcause=9   (ECALL from S-mode)
    sw=224            0x5d  excommit=0 ldsync=1 stsync=0 lsu_rdy=1 dyn_rdy=1 flu_rdy=1
                            flush=0 privM=1

**The clear worked but cannot isolate a domain fault**, because the domain is ENTERED via an
S-mode ECALL, which re-latches cause 9 straight after the clear.

**However, this is weak evidence AGAINST the untrapped-capability-fault hypothesis.** A
capability fault inside the domain occurs AFTER the entry ecall, so it would OVERWRITE the
latch (which keeps the most recent nontrivial trap, `cva6.sv:1077-1083`). The latch still
reads 9. So either no committed capability exception occurred, or capability faults do not
reach `ex_commit.valid` with a non-zero cause on this path.

Caveat: this argument depends on capability faults actually reaching `ex_commit.valid`
un-filtered. Verify that before treating hypothesis #1 as refuted — do NOT record it as
refuted on this alone.

**`privM = 1` at the wedge** says the core is in MACHINE mode, i.e. not executing domain code,
which is consistent with being stuck in the monitor or at pc=0 — and is NOT consistent with a
plain in-domain livelock. `excommit = 0` (that bit is the exception-valid bit, `cva6.sv:500`)
means no exception is being signalled at the sampling instant.

### What this changes about the next step

The trap latch cannot answer the hang-vs-fault question while the entry ecall keeps
overwriting it. Two ways forward, in order:

1. **Give the domain a real `mtvec`** so a fault REPORTS rather than vanishing. This is the
   force multiplier: it converts every wedge into a returnable marker with cause and epc, and
   removes the sampling limit that makes wedging probes unrepeatable within a boot. Either set
   it in the entry glue (our code — first check whether the domain may write `mtvec` at its
   privilege) or have the monitor populate `dom_seal[1]`. **The monitor route is a design
   decision and must be proposed, not applied unilaterally.**
2. **Sample `privM` and `mepc` repeatedly at the wedge**, not once. A single sample cannot
   distinguish "stuck at pc=0 in M-mode" from "sampled during a monitor entry". The mepc log
   (`recent_nontrivial_mepc_log_q`) is available on its own selector.

## ROOT-CAUSE LOCALISATION: every wedge is the DYN UNIT BLOCKED ON A REV-NODE QUERY

Route A (give the domain an `mtvec` so faults report) was built and run. It did NOT convert
wedges into returns — and that is the decisive result, because it REFUTES the untrapped-fault
hypothesis rather than confirming it.

    mt71 (control, flag ON)  rc=0x45   handler does not perturb a working domain
    mt67 (3 walks)           WEDGED    with a valid mtvec handler in the domain image
    mt10 (the real blocker)  WEDGED    same

With `mtvec` pointing at a reachable handler, a genuine trap WOULD have been caught. It was
not. **The wedge is not an exception.** (Board-confirmed prerequisite: a domain can write
mtvec — stage 75 wrote 0x40 and read 0x40 back.)

### The signature, identical across FOUR wedges in FOUR binaries

    sw=225 {tbe,wstore,wload,wrev,domsw,stall,memwr,memwait} = 0x95 = 1001 0101
      tbe=1  wstore=0  wload=0  wrev=1  domsw=0  stall=1  memwr=0  memwait=1

Seen in `board-ra-mt67.log`, `board-ra-mt10.log`, `board-mcause.log` (wd10) and
`board-pad72.log` (wd72) — same value every time.

* **`wrev = 1`** — `waiting_for_rev_res`, set immediately before the blocking `recv` in
  `get_node_query_validity` (`capstone_dyn_unit.anvil:106-112`):
  `set waiting_for_rev_res := 1'b1 >> send rev_node_ep.query_req(revnode_id) >>
   let vali = recv rev_node_ep.query_res >> ...` — **there is no abort or timeout path**, so an
  unanswered query blocks the unit forever.
* **`memwait = 1`** with `wrev = 1` is, per the debug-mux decoder, *blocked on the node-table
  D$ access*.
* `stall = 1`, and `excommit = 0` on the companion read (`sw=224`, `ex_commit.valid` is the
  exception bit, `cva6.sv:500`) — consistent with a stalled pipeline and no exception pending.

**So a "wedge" is: a capability load issues a revocation-node validity query, the query is
never answered, and the dyn unit waits forever.** That is a hardware stall, which is why no
trap handler can catch it and why the core never advances.

This is consistent with everything that survived: it is not the data, the pointer, the array,
the layout, the code address, store ordering, or walk count — all excluded by measurement. It
also explains why `mtvec` (Route A) could not help, and why the failure looks deterministic per
binary yet varies with unrelated-seeming changes: what varies is whether a given execution
reaches a query that goes unanswered.

### What this does NOT yet establish

WHY a query goes unanswered. Candidates, in order: the node-table D$ access never completes;
the rev-node unit is itself blocked elsewhere; or a request/response mismatch loses the reply.
`capstone_rev_node.anvil` and the node-table memory path are where to look next.

### Next steps

1. Read `rev_node_head` and the overflow flag at the wedge (selectors `11001`/`11010`) to see
   the allocator state when the query hangs. Add them to the runner's wedge read.
2. Trace the query path in `capstone_rev_node.anvil`: what makes it not answer a `query_req`?
   Look for a state where it is waiting on memory that never returns, or an ordering rule that
   drops a request.
3. Route B (`dom_seal[1]` in the monitor) is NOT needed for this: the wedge is not a trap, so a
   trap vector cannot help. Do not spend the firmware risk on it for this purpose.

Route A stays in the tree, gated OFF (`INTERP_DOMAIN_MTVEC`), verified byte-identical when off
(wd71 sha 27477e88aa49297e both before and after). It is still the right tool for any FUTURE
fault-vs-stall question, which is exactly what it settled here.

## Rev-node state at the wedge: allocator is HEALTHY; serving_idx cannot answer the question

Read at the stage-10 wedge (selectors verified against `cva6.sv:1184-1189`, not taken from the
probe's labels):

    rev_node_head = 413      overflow = 0
    serving_idx   = 0        init_seen = 0   mrev_seen = 0   (all four bytes zero)

**Allocator exhaustion is REFUTED as the trigger.** head is 413 of a ~1021 pool with
overflow=0, so the bump allocator had not wrapped and ids were not being reused. That retires
the R-12 pool-exhaustion story for this failure — it has been re-proposed repeatedly across
this campaign and is now excluded by direct measurement.

**CORRECTION — `serving_idx` does not mean what the probe's label implies.** It is assigned
ONLY in the `rev_req` (revoke) handler, `capstone_rev_node.anvil:131-134`
(`} else try msg = recv ep.rev_req { ... set serving_idx := msg; ...`), and never in the
`query_req` path at `:55-61`. So `serving_idx = 0` means "no revoke request has ever been
served", NOT "node 0 is being queried". An earlier decode here read it as "serving node 0,
below the id space" — that was wrong and is withdrawn. The debug word is
`{mrev_seen, init_seen, serving_idx[29:0]}` (`:215`).

**Consequence: there is no debug register that exposes the QUERIED node id.** The queried id
lives only in the `query_req` message. Reading it would need an RTL change and therefore a
bitstream reflash, which is not available here.

### So do it from software instead — the next probe

The queried id is the `revnode_id` in the metadata of the capability being loaded. A domain
can read that with `lcc` BEFORE performing the load that hangs, and return it as a marker:

* sane id, `3 <= id < head` -> the capability is well-formed and the hardware failed to answer
  a legitimate query => RTL defect.
* `id == 0` or `id >= 1024` -> the capability carries a bogus revnode_id => our side produced
  it (cap-init, a stale/uninitialised slot, or a load of untagged memory), and the RTL's only
  fault is hanging instead of erroring.

Note `lcc` encoding: funct7 is **0x04** with the zimm in the rs2 field (0x08 was wrong and cost
a build). Prefer a C-level read if one exists before hand-rolling `.insn`.

### Independent of the trigger: report this defect

`get_node_query_validity` (`capstone_dyn_unit.anvil:106-112`) is
`send query_req >> recv query_res` with **no timeout and no abort path**, and the rev-node
unit's `get_rev_node` (`capstone_rev_node.anvil:36-41`) likewise blocks on
`recv mem_ch.read_res`. Any unanswered query is therefore an unrecoverable machine hang by
construction: no trap, no diagnostic, board dead until power-cycle. That is a robustness defect
worth its own ISSUES.md entry regardless of what triggers it, and it is the reason this
campaign produced silence instead of error codes for days.

## RETRACTED: "the wedge is a rev-node query stall". The signature was never controlled.

A HEALTHY baseline was finally read — the same registers after `wd71`, which RETURNS `0x45`
(6/6 samples) — using `probe_wedge_regs.py`, which reads registers even when the run completes.

    sw=225 {tbe,wstore,wload,wrev,domsw,stall,memwr,memwait}
      healthy (returned) : 0xd5 = tbe=1 wstore=1 wload=0 wrev=1 domsw=0 stall=1 memwr=0 memwait=1
      wedge   (mt10)     : 0x95 = tbe=1 wstore=0 wload=0 wrev=1 domsw=0 stall=1 memwr=0 memwait=1

**`wrev = 1` and `memwait = 1` in BOTH.** They are the RESTING state of those bits, not evidence
of a blocked query. The claim that "every wedge shows the dyn unit blocked in
get_node_query_validity while the rev unit waits on the node-table read" rested on reading
those bits at four wedges and never once at a success. It is **withdrawn**.

This is the same missing-control error as the earlier retractions (guarded-vs-unguarded, walk
count, instruction placement): a signature compared only against other instances of itself.

### What the controlled comparison actually shows

The only bit that differs in `sw=225` is **`wstore` (1 -> 0)**. On `sw=224`:

      healthy : 0xff = excommit=1 ldsync=1 stsync=1 lsu_rdy=1 dyn_rdy=1 flu_rdy=1 flush=1 privM=1
      wedge   : 0x5d = excommit=0 ldsync=1 stsync=0 lsu_rdy=1 dyn_rdy=1 flu_rdy=1 flush=0 privM=1

differing in **`stsync` (1 -> 0)**, `excommit` (1 -> 0) and `flush` (1 -> 0). So what changes at
a wedge is STORE-side, not rev-query-side.

**Caveat on the healthy `sw=224 = 0xff`:** all-ones is suspicious as a reading — it may be a
bus-idle/default rather than real state, in which case the `sw=224` deltas are meaningless too.
`sw=225 = 0xd5` is not all-ones and is therefore the more trustworthy of the two. Do not build
on the `sw=224` deltas until 0xff is confirmed to be real state.

### What SURVIVES from the rev-node work

* `get_node_query_validity` (`capstone_dyn_unit.anvil:106-112`) and `get_rev_node`
  (`capstone_rev_node.anvil:36-41`) genuinely have **no timeout and no abort path** — read from
  source, independent of any board reading. Any unanswered query IS an unrecoverable hang by
  construction. Still worth an ISSUES.md entry as a robustness defect; it is simply not
  established that it is what bites us.
* `rev_node_head = 413, overflow = 0` at the wedge: the allocator had not wrapped. Exhaustion
  remains refuted as the trigger (healthy read: head = 222, overflow = 0).
* Route A stands: with a reachable `mtvec` handler the wedge was unchanged while the control
  returned, so it is not an ordinary trap. (Pending the auditor's check on whether capability
  faults use `mtvec` at all.)

### Rule this cost yet another retraction to learn

**Never read a debug register only at the failure.** Read it at a SUCCESS in the same session
first. A "signature" seen at four wedges means nothing without the healthy value; three of the
eight bits here were identical in both states.

## THERE ARE TWO WEDGE POPULATIONS, NOT ONE — and one of them IS a capability exception

An adversarial audit found that this campaign has been treating two distinct failures as one.
Verified directly over every board log in `/tmp/capstone`:

    sw=225   sw=255                          count  runs
    0x84     0x98 = trap_seen=1, mcause=24     12   pad73, pad74, fence, wda..wdg, rep, n69b
    0x95     0x89 = trap_seen=1, mcause=9      13   mt10, mt67, wd67, boot1..3, goal2, pad72
    0xd5     0x8f                               1   wd71 (HEALTHY, returned)

`mcause = 24` is a REAL capability exception: `ex_code` numbering is
`UNEXPECTED_OPERAND = 24` (`capstone_unit.anvilh:289-291`), and the cause is formed as
`64'd23 + exception_code[3:0]` (`cva6.sv:1357`, `ex_stage.sv:469`).

**This also closes the caveat left open earlier:** capability faults DO reach
`ex_commit.valid` with a non-zero cause, so the latch does record them. The earlier
"weak evidence against a fault" reasoning was applied to the wrong population.

The split lines up exactly with a fact established independently by counting `SQ: G/enter`:
`pad73`/`pad74` are the runs that **never entered the domain** (they died in region-share) —
and they are the mcause-24 family. The in-domain wedges (`mt10`, `mt67`) are the `0x95`
family, whose only latched trap is the stale entry ECALL.

So:

* **Family A — `0x84` / mcause 24 (UNEXPECTED_OPERAND).** Fails during REGION-SHARE, before
  `domain_main`. A genuine capability exception, taken with `mtvec = 0`, hence silent.
* **Family B — `0x95` / mcause 9 (stale).** Fails INSIDE the domain. No new trap latched.
  `wrev`/`memwait` are set here but are ALSO set in the healthy control (`0xd5`), so they are
  resting state and say nothing.

**Do not merge these again.** Every "the blocker wedges N times" count in this file mixes them.

### Other findings from the audit, verified

* **Bit order is CORRECT** — `cva6.sv:1107-1116` is MSB-first and the runner packs `states[i]`
  to bit `i` and prints MSB-first. Independent calibration: `0x89` -> mcause 9 (ECALL from
  S-mode, exactly what domain entry emits); reversed would be 17, reserved. So the decode is
  sound; the error was interpretation, not decoding.
* **`dyn_rdy = 1` at every `0x95` wedge** (`sw=224 = 0x5d`). `capstone_dyn_ready_o` is
  `!valid & ready_q` (`ex_stage.sv:837-838`) and `send ep.rtr(1'd1)` happens once at the top of
  the dyn unit's main loop (`capstone_dyn_unit.anvil:480`). A unit parked inside
  `get_node_query_validity` could not be re-asserting `rtr`. This is further evidence against
  the (already retracted) "dyn unit blocked forever" reading.
* **Provenance error of mine:** `head=413/overflow=0` came from `board-revstate.log`, a
  different, later run than the four logs quoted for the `0x95` signature — the two were
  presented together as if from one wedge. The rev-node selectors are absent from those four
  logs entirely.

### The positive control the mtvec experiment never had

`mt71` returned `0x45`, i.e. it never faulted, so nothing ever entered `.Ldomain_trap`. No run
has yet shown the handler catching a fault. Build a domain that **deliberately** faults (an
`ldc` past the end of a bounded capability -> `OUT_OF_BOUNDS`, `capstone_dyn_unit.anvil:322-325`)
with `INTERP_DOMAIN_MTVEC=1`:

* it returns via `.Ldomain_trap` -> "the wedge is not an exception" becomes an earned inference
  for Family B;
* it wedges -> the Route A result collapses and the handler never worked.

Order in one boot: deliberate-fault domain, then `wd71`, then the wedger last.

## Route A remains UNVERIFIED — and the control turned up something else

The positive control (stage 76: dereference 1 MB past `capstone_probe_lit`, a 256-byte
cap-table global, built with `INTERP_DOMAIN_MTVEC=1`) returned **rc = 0x77**, i.e. **no fault
fired**. `mt71` returned `0x45` in the same session, so the board was healthy.

The load really did execute — it is not dead code eliminated by the compiler (the pointer is
`volatile`), confirmed in the disassembly of `mt76.dom`:

    36970: lui        a1, 0x100        ; +1 MiB
    36974: cincoffset a1, a0, a1       ; offset the capability by 1 MiB
    3698c: lbu        a0, 0x0(a0)      ; the load -- present and executed
    369a4: li         a0, 0x77         ; returned normally

**Consequences:**

1. **Route A is NOT verified.** No run has ever shown `.Ldomain_trap` catching a fault, so
   "mt10 still wedged with a handler installed => the wedge is not an exception" remains an
   ASSUMPTION. Do not cite it as established. Needs a fault the hardware definitely raises.
2. **A byte load 1 MB beyond a 256-byte global's capability did not fault.** Either the storage
   capability handed out for that global is much wider than the object, or bounds are not
   enforced on this path. Both are worth chasing on their own merits — the first would mean
   cap-table carves over-grant, the second would be a spatial-safety hole. Neither is
   established yet; the next step is to read the capability's own base/length (via `lcc`
   fields, NOT the validity query, which uses the hanging channel) and compare against the
   256-byte object.

**Better fault candidates for a retry of the control** (pick one the RTL definitely raises):
`UNEXPECTED_CAP_TYPE` by using a scalar where a capability is required; `INSUFFICIENT_PERMISSION`
by storing through a read-only capability; or the region-share path itself, which already
demonstrably latches mcause 24 (the `0x84` family) — that family is the natural positive
control, since it is a fault that IS occurring.

### Method note

This is the fourth time a probe was invalid rather than the hypothesis being wrong. The build
caught this one (`capstone_probe_lit` undeclared, because a guard-widening `replace` silently
matched nothing — it had no assertion). Every generated edit in this file must assert its
anchor matched.

## MEASURED: the capability used for a 256-byte global spans >= 1 MiB (over-grant)

Stage 77 reads the capability's own bounds with `lcc` zimm 3 (start) and zimm 4 (end)
(`capstone_dyn_unit.anvil:186-187`), using the in-tree encoding
(`.insn r 0x5b, 0x1, 0x4, rd, rs, x<zimm>`, `start-gpfree-captable.S:34`). Deliberately NOT
zimm 0 — that is the validity query, on the channel that hangs.

    wd71 (control)  rc = 0x45
    wd77            rc = 0x4F   class 4  -> end - start >= 1 MiB      (2/2 samples)

`capstone_probe_lit` is **256 bytes**. The capability reaching it covers **at least 1 MiB**.

**This resolves stage 76 and inverts its reading.** Bounds ARE enforced; the 1 MiB overshoot
did not fault because it was genuinely IN BOUNDS. There is no enforcement hole — there is an
authority-granting bug.

**Why it matters beyond this investigation.** The whole point of `-capstone-gp-captable` is one
narrow capability per global. If every global's storage capability spans >= 1 MiB, then
per-object spatial safety is not actually being delivered by the ABI as built: an overflow of
any global can reach ~1 MiB of neighbouring globals without a fault. Any claim in the write-up
about per-global bounds must be checked against this measurement before it is made.

**Not yet distinguished** (a source-side trace is running):
1. the glue's carve/`split` produces an over-wide storage capability (so every global is
   affected), or
2. the compiler is not using the per-global storage capability at all here and instead reaches
   the object through a wider region capability (so the carve is fine and codegen is at fault).

Both are real defects; they need different fixes. Check the glue's `split(t2, sp, t1)` against
`SPLIT` in the RTL (does the result inherit the parent's `end`?), and check what the cap-table
entry actually holds.

**Measurement limitation to fix on the next pass:** stage 77 clamps `log2(len)` at 15, so the
reading only proves `len >= 1 MiB` and cannot say how much larger. Return the raw `start` and
`end` (or `log2` unclamped) to size the over-grant exactly — 1 MiB vs the whole data region are
very different findings.

## CONTESTED: the "capability spans >= 1 MiB" measurement is NOT settled

Two independent lines disagree and neither has won:

**Board (stage 77, 2/2 samples):** `lcc` zimm 3/4 on the pointer to `capstone_probe_lit`
returned class 4, i.e. `end - start >= 1 MiB` for a 256-byte global.

**Source trace (quoted, reproducible):** the carve is EXACTLY 256 bytes —
`stor = max(16, align_up(size,16))` (`start-gp-captable-interp.S:446-449`), and `SPLIT` narrows
the parent in the same instruction, writing it back (`capstone_dyn_unit.anvil:140-144`,
`commit_stage.sv:280-281`), so carve *i* ends where carve *i-1* began. No over-grant is
possible from that path. It also shows ordinary `lbu` IS bounds-checked
(`load_store_unit.sv:970-971`, cause 28).

Its alternative explanation for stage 76's non-fault is **capability compression**: register
capabilities hold 64-bit compressed metadata whose bounds are reconstructed from the CURRENT
cursor — `ariane_pkg.sv:692-693`:

    b = xlen_t'({cursor >> (E+14), B[13:0]}) << E;
    t = xlen_t'({cursor >> (E+14), T[13:0]}) << E;

so an offset that is a whole multiple of 2^(E+14) decodes to a window that has SLID with the
pointer. Re-implementing that arithmetic predicts: `+0x100` faults, `+0x200` faults,
`+0x100000` does NOT (stage 76's exact offset), `+0x100200` faults again. `CINCOFFSET` performs
no representability check (`capstone_flu_unit.anvil:41-42`).

**Attempts to settle it both WEDGED**, so nothing was learned:

    wd78 (unclamped log2 of end-start)                 WEDGED
    wd79 (start at base vs start at base+1MiB)         WEDGED
    wd71 control returned 0x45 in both sessions

Note `wd77` — nearly identical code, two `lcc` reads — RETURNS. `wd78` adds only a loop and
wedges. That is the same unexplained build-to-build sensitivity seen all session, and it means
these wedges are not evidence about bounds.

### Which reading to prefer, and why

**Treat the over-grant claim as UNPROVEN and lean toward the compression explanation** until
measured. The source trace quotes the mechanism, reimplements it, and predicts a PATTERN across
offsets; the board result is a single quantity read at one cursor, from which a security
conclusion was generalised. An earlier commit here stated the over-grant as fact and drew a
consequence for the write-up's spatial-safety claims — that consequence is withdrawn pending
evidence.

### The test that settles it, and it needs no new instrumentation

Run stage 76 (the plain out-of-bounds read that returned `0x77`) with the offset changed from
`1024*1024` to `1024*1024 + 512`:

* **faults / behaves differently** -> compression aliasing confirmed; the capability never
  covered that memory, and the defect is that EFFECTIVE bounds move with the cursor at
  alias-period offsets — subtler than an over-grant and invisible to inspecting a capability
  at rest.
* **also returns `0x77`** -> the range really is accessible and the over-grant reading stands.

Prefer this over more `lcc` probes: it reuses a domain shape that has already RETURNED, whereas
the two `lcc`-based deciders both wedged.

### Independent of the above, and worth reporting on its own

QEMU keeps FAT register capabilities with exact bounds and DOES bounds-check ordinary loads
(`insn_trans/trans_rvi.c.inc:286-292` -> `op_helper.c:1107`), so it would have faulted where the
silicon did not. Emulation is STRICTER than hardware here. Any spatial violation landing on an
alias boundary passes on silicon and traps under QEMU — a silicon-only gap no amount of
emulation testing can surface. Also: `RISCV_EXCP_CAP_OOB` (0x1c) is defined in
`capstone-qemu/target/riscv/cpu_bits.h:697` and never raised; the OOB path raises
`LOAD_ACCESS_FAULT` instead, so QEMU's mcause will not match the RTL's 28.
