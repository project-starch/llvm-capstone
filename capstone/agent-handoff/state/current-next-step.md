# Current recommended next step

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
