# Decisions waiting on the lead — 2026-09-10

Everything below is blocked on a call only you can make. Nothing here is a status update; each item is
a question with the evidence needed to answer it, what happens either way, and what it costs. Read in
order — **item 1 gates four of the others.**

Nothing is edited, committed or synthesised while these are open. Both lanes are holding.

---

## 1. The `end` convention — ONE TOKEN EACH SIDE, no convention change

**GATES: R-30, R-31, the firmware change, the QEMU change, the synthesis run, and the flash.**

### The question I asked first was wrong, in both directions

I put it as *"is `end` inclusive or exclusive, project-wide"*, you ruled **exclusive**, and executing
that ruling immediately produced a contradiction. The RTL lane refused the edit rather than make it,
which is how this surfaced.

There is no project-wide convention to rule on. The two documents use **opposite** conventions and each
is internally consistent **except for one instruction copied from the other**:

| | convention | verified by | the ONE odd instruction | the fix |
|---|---|---|---|---|
| **RTL** | **exclusive** (`end` = one past the last byte) | LSU faults when `ea + size > bound_end` (`load_store_unit.sv:985`); SPLIT sets `rs1.end := val` and `rd.start := val` (`dyn:141-142`), which only partitions without overlap if exclusive | `INIT`'s check `cursor <= end` (`capstone_flu_unit.anvil:139`) — inclusive arithmetic | `<=` → `<` |
| **spec** | **inclusive** (`end` = the last byte) | aliasing closed over `[base, end]` (`prog-model.adoc:119`); `end = INIT_*_END - 1` (`ctrl-status-insn.adoc:79-80`); SHRINKTO `end = cursor + imm - 1` (`cap-man-insn.adoc:269`) | the STORE BOUND `[base, end - CLENBYTES]` (`mem-access-insn.adoc:93`) — exclusive arithmetic | `end - CLENBYTES` → `end - CLENBYTES + 1` |

### Why this resolution is right

Fix one token on each side, change **no** conventions, and the two agree byte for byte. For a region of
bytes `S..S+63`:

* **RTL** `end = S+64`; stores permitted at `S, S+16, S+32, S+48` (`cursor <= end-16`); the last leaves
  the cursor at `S+64`; amended `INIT` accepts `cursor >= end` ✓
* **spec** `end = S+63`; amended bound permits the same four positions; cursor ends at `S+64`; the
  **existing** `cursor > end` accepts it ✓
* **QEMU** is exclusive throughout and already accepts this.

### ⚠ One route on the table would open a bounds hole — do not take it

Changing the **RTL's** store bound to `end - 15` (correct for the spec's inclusive `end`) permits, under
the RTL's exclusive `end`, a 16-byte store whose last byte lands **at** `end` — **one byte past the
region, on the store path.** The RTL lane proposed it in good faith from the spec's arithmetic and
withdrew it after reading the RTL's own; both of us made the same class of error, three hours apart, in
opposite directions.

### What each answer authorises

* **Adopt the resolution (recommended):** the RTL's one token, the spec's one token, both fixes ship
  with the firmware change, then synthesis. **Q-07 also closes** as a side effect (see item 5).
* **Adopt full exclusive as originally ruled:** a four-site spec revision (`:421`, SHRINKTO `:269`,
  `ctrl-status:79-89`, `prog-model:119`) plus two RTL stragglers. Real revision, not an amendment
  riding a bitstream.
* **Adopt full inclusive:** the RTL's access paths and SPLIT all change. Largest option; not
  recommended by anyone.

**My recommendation: the resolution.** Smallest, makes both documents self-consistent, no convention
moves, and the RTL becomes conformant rather than deviant.

---

### ⚡ SYNTHESIS IS NOT BLOCKED BY THIS DECISION — the RTL change is the SAME under every surviving route

Checked against the branch: the RTL diff versus the flashed revision is **exactly two operators**, plus
two new directed tests and a testlist entry.

```
- if(rsp == 1'd1 || ((rs1.metadata.perm&3'd2)==3'd2)){     R-31 polarity
+ if(rsp == 1'd1 || ((rs1.metadata.perm&3'd2)!=3'd2)){
- if(rs1.cursor <= rs1.metadata.end){                      R-30 precondition
+ if(rs1.cursor <  rs1.metadata.end){
```

**Both surviving routes produce that identical RTL change.** The resolution (spec fixes its store
bound, RTL fixes `INIT`) and the original full-exclusive ruling both give `<=` → `<` at `flu:139`. Only
the **spec** edit differs between them — one token versus four sites — and a spec edit is a document,
not a bitstream. The one route that WOULD have changed the RTL differently, moving the store bound to
`end - 15`, is refuted as a bounds hole and is not on the table.

**So the bitstream can be synthesised now and the spec decision can follow it.** What synthesis cannot
precede is the **FLASH**, which needs the firmware change (item 2) because the pair must not ship
RTL-only. Synthesis produces a hash; flashing spends it.

### ⛔ THE BUILD IS BLOCKED ON A PUSH, SEPARATELY FROM ANY AUTHORISATION

The synth lane cannot see the branch, and checked rather than assuming I was wrong to say it exists.
Both facts verified here:

* **`r30-r31-init-revoke` exists locally at `1bfff7776`**, and `66c4e7517` **is** an ancestor of it, so
  the base check will pass once it is reachable.
* **It has never been pushed.** `git ls-remote --heads origin r30-r31-init-revoke` returns nothing.

**Why, and it is not an oversight.** The branch is not on the push allowlist — that file is the lead's
and no lane may add to it — and the agent credential has historically lacked write access to
`capstone-ariane`. That is why `fpga-testing-dev` at `66c4e7517`, the currently flashed build, was
pushed by the lead rather than by a lane. This needs the same.

**So there are TWO things for the lead, not one, and either without the other leaves the build stuck:**

1. **Push `r30-r31-init-revoke` at `1bfff7776`** to `project-starch/capstone-ariane` (or name the remote
   to use instead).
2. **Say the word in the synth lane's own session.** An authorisation given here does not reach them,
   deliberately — they were asked to refuse a relayed one and they did.

**A bundle would move the commit but NOT solve this, and the synth lane is why I am saying so.** I
offered `git bundle` as a way to hand them `1bfff7776` with no push and no allowlist involvement.
They TESTED it rather than reasoning about it — a real bundle, `git bundle verify` reporting okay,
a fetch landing the tip — and it works mechanically. They then refused it on a ground I had not
considered: `collect-synth-artifacts.sh` stamps `PROVENANCE.txt` from the repository HEAD, so a build
from a bundle reports a hash against a commit **that exists on no remote**, and nobody, the lead
included, can resolve afterwards what was actually built. This build's whole value is a
pre-registered comparison against `66c4e7517`; a base nobody can resolve makes that comparison
uncitable even if every number lands inside the predicted range.

**So the push is load-bearing, not paperwork.** The bundle stays available as a fallback if you would
rather not push, and the synth lane will build it while stating prominently that the base is
unpushed — but that is a worse artifact and the choice is yours. One incidental gain from their test:
`git bundle verify` REFUSES a bundle whose prerequisite the receiving repository lacks, so
"`66c4e7517` is an ancestor" would be enforced by git against their copy of the flashed revision
rather than asserted by me.

*(A correction that cost the synth lane a search: this document is in the PARENT repo,
`project-starch/llvm-capstone` on `dev`, not in `capstone-ariane`. I cited it without naming the repo.)*

### Pre-registered prediction for the R-30/R-31 build, written BEFORE the run

Required by the synth lane: a build whose expected reading is not written down first cannot be checked
afterwards. Predictions, for the hash synthesised from `r30-r31-init-revoke`:

* **WNS** within the family's spread, **−11.7 to −15.3 ns** on the `clk_out1_xlnx_clk_gen` intra-clock
  row of `ariane_xilinx_timing_summary_routed.rpt`. The 66c4e7517 build read **−12.425**.
* **Placed LUTs 168.9k–170.5k** from `ariane_xilinx_utilization_placed.rpt`. 66c4e7517 read **169.7k**.
* **Rationale for expecting no measurable movement:** the change is two comparison operators inside
  conditions already being evaluated — no new signal, no new term, no new comparator. That is also why
  it linted at baseline 40 where the R-29 candidate went to 41.
* **What would falsify "this is a free change":** WNS outside that spread, or LUTs outside it, or any
  movement in the UNOPTFLAT loop set. Any of those stops the flash conversation and sends the change
  back, regardless of the functional results.
* **A movement in the loop set falsifies the REASONING, not just the number, and must not be explained
  away.** The prediction rests on the claim that two operators inside already-evaluated conditions add
  no signal, term or comparator — the same claim that predicted lint at baseline 40 and was borne out.
  If the loop set moves, that claim is wrong, and it is the claim that has to be re-examined first,
  before any argument that the movement is harmless. (RTL lane's addition; it is the half of a
  pre-registration that usually gets forgotten.)
* **Independently verified 2026-09-10:** the branch changes exactly two non-comment lines under
  `core/`; everything else in the diff is comments, the two directed tests and their testlist entries,
  none of which is synthesised. The `capstone-spec` submodule is clean at `ca9c84f` — **no spec edit has
  been made in any form**, including the corrected one-token version.
* **Read the intra-clock row, not `eth_rxck`**, and gate every artefact on mtime later than the run
  start — `make clean` leaves the previous run's reports in place.

**This build does NOT settle R-29** (different branch, lint-failing, item 3) and does NOT authorise a
flash (item 2).

## 2. How should the monitor reclaim a revoked region?

**Blocked by item 1. This is the firmware half and it must ship with the RTL fix — never RTL-only.**

Once R-31 lands, revoke returns UNINIT-at-base and **five** monitor sites meet it. The widest,
`split_out_cap`, needs no re-share at all: a revoked region stays `region_live`, so an unrelated later
`create_region` in its range picks it and SPLITs an UNINIT.

| option | what it costs | what it buys |
|---|---|---|
| **1. Zero-fill, then INIT** *(the spec's intended reclaim)* | **one store per 16 bytes** — **256 stores per revoke**, see the correction below | The security property itself. The owner must overwrite the borrower's data before reuse. **The only option that closes anything.** |
| **2. Leave UNINIT, refuse the re-share** | turns a working path into an error return; `__mrev` needs LIN, so callers learn a new contract | cheap |
| **3. Fill lazily in the domain** | changes the shared-region ABI; largest change | matches what the type is for; moves cost to the region's user |

### ⚠ RETRACTION: the number I told you was 256 times too large, and it flips the recommendation

I wrote "~65,536 stores per revoke on SQLite's 1 MiB region" and then wrote that the number IS the
decision. **The region is not 1 MiB. There is no 1 MiB region anywhere in the tree.**

`SQLITE_HC_REGION_SIZE` is **4096** (`benchmarks/sqlite/sqlite_hostcall.h:34`). So is every other
region-size constant I could find — `BORROW_COST_REGION_SIZE`, `CORPUS_REGION_SIZE`, `REGION_SIZE`,
`SQLITE_HIER_REGION_SIZE`, `ROW3_ARENA_SIZE` and thirteen more: **every one is 4096UL**, and the
literal call sites pass `4096` too. A page, not a megabyte. If a megabyte-scale region exists it is
not created through `create_region` and I did not find it.

**4096 / 16 = 256 stores per revoke.** Not 65,536.

What that does to the cost, from numbers already measured on this silicon
(`fpga-silicon-measurements-for-paper.md` §1) rather than from a fresh estimate:

| quantity | value | where it comes from |
|---|---:|---|
| reclaim today (`mrev`+`delin`+`revoke`) | 171 cyc | measured |
| copy rate, 256 B / 1024 B | 3.52 cyc/byte | measured, and linear across both points |
| fill of 4096 B, CEILING (copy rate, so load+store — a fill has no loads) | ~14,400 cyc | derived |
| fill of 4096 B, FLOOR (256 stores at the measured CPI 2.0-3.2) | ~510-820 cyc | derived |
| **reclaim WITH the fill** | **~700 to ~14,600 cyc** | derived |

And against the workload: boundary frequency on speedtest1 is **~1 borrow per ~19k instructions**
(measured), which at the measured CPI is ~47.5k cycles between borrows. Adding 510-14,400 cycles to
each moves the boundary overhead from the measured ~1 % to somewhere between **~2 % and ~31 %**.

**This is a different decision from the one I put to you.** At 65,536 stores option 1 was
self-evidently unaffordable and option 3 was the likely answer. At 256 stores option 1 is a page
memset per reclaim, the cost is bounded above by ~31 % on the one workload we have measured
boundary frequency for, and it is the only option that closes anything.

**Recommendation, restated on the corrected number: option 1.** The per-revoke cost should still be
measured on the board before acceptance, but it is now a confirmation rather than a go/no-go — and
the arm is cheap, because `uninit_init_then_use_ok` (item 5) is already the fill-then-init shape in
seven lines.

**How the error happened, since it is the same class as the others today.** I did not read a
constant; I carried "1 MiB" from somewhere else in my head and multiplied. The check that would have
caught it is one `grep` for the size constant, which is the same check the RTL-versus-spec type
numbering needed, and the same one the QEMU permission mask needed an hour ago.

---

## 3. R-29: spend a synthesis run on a lint regression, or reformulate first?

**Not blocked by item 1. Independent.**

The fix is **functionally correct** — the directed test goes from failing to passing, four controls
unchanged, predictions written first — and **fails the lint gate at UNOPTFLAT 40 → 41**, a new loop
inside the very module whose source warns about exactly that.

* **(a) Synthesise anyway.** Precedent exists: S-10 took the same +1 and was accepted after synthesis
  showed its loop set unchanged and its nets clear of the critical path. ~1 h 20 m plus the collector.
  **Only synthesis can actually answer whether this loop is benign.**
* **(b) Reformulate first**, synthesise only a lint-clean candidate. Cost unknown; three formulations of
  a closely related term all took the same +1.

I have not recommended one. `CLAUDE.md` is explicit that the lint gate must pass and that adding a term
to a cone on the standing UNOPTFLAT list is the highest-risk edit available; spending a build on a gate
failure is a deliberate trade, not a lane's call.

**Context for the trade:** R-29 is a **one-instruction window** (boot sw49: the defect appears only when
the store is immediately before the load; a single intervening instruction clears it), and `W-12` is
already in force. It is the least urgent of the three RTL items.

---

## 4. Q-04 — are scalars exempt from the MOVC consumption rule?

**Genuinely a spec question. I ruled on it, was wrong, and retracted.**

My ruling argued the spec was explicit because *"NOT_CAP is type 0, and 0 != 1"*. **The spec has no
`NOT_CAP` type** — its table is Linear 0, Non-linear 1, Revocation 2, Uninitialised 3, Sealed 4,
Sealed-return 5 (`prog-model.adoc:177-184`). `NOT_CAP = 0` is the RTL's enum, which inserts it at zero
and shifts everything up; the RTL says so itself. I evaluated a spec sentence with RTL constants.

**The ambiguity is real, on four grounds:** MOVC's operands are annotated as capabilities
(`cap-man-insn.adoc:16,23,25`); spec commit `a1db3c2` **removed** MOVC's *"`x[rs1]` is not a
capability"* exception and **left the consumption clause untouched**; `mem-access-insn.adoc:45` uses the
same `(i.e. type != 1)` gloss for a scalar-**excluding** condition; and `:105` writes the guard longhand
as *"is a capability and type is not 1"*.

**So scalar-exemption is a RESTORATION of the clause's original precondition, not an amendment** — the
opposite of what my ruling said, and it makes it the cheaper option.

**What each answer implies:** exempt → QEMU is right, the RTL changes, and **C-32** (not C-14) is the
compiler item it gates. Not exempt → QEMU changes, and the compiler must stop using `movc` where the
source is live. **What stands either way:** the RTL does null a NOT_CAP source, board-confirmed.

---

## 5. Q-07 — does the QEMU fix close the divergence or only narrow it?

**A choice inside the fix, not a fact. Blocked by item 1 only for its precondition form.**

The Q-07 change is **written and deliberately uncommitted** (`capstone-qemu/target/riscv/op_helper.c`):
`csinit`'s three host `assert()`s become guest traps, `csrevoke` gains the spec's permission clause and
puts the cursor at **base**, and the **missing UNINIT cursor advance** is added to the store path — QEMU
had none anywhere, so a monitor fill loop would have been a no-op and the suites would have passed a
monitor that fills nothing.

* `csinit` keeps `cursor == end` → QEMU accepts one value, the fixed RTL accepts `end` and above →
  **narrows**.
* `csinit` takes `cursor >= end` → sets identical → **closes**. This is what is written.

Both implementations hold `end` exclusively, so `>=` is coherent. **It must land together with the
monitor change (item 2)** or `run-nullblk-all.sh` goes red and stays red.

### It did not compile, and I had called it written

`CAP_PERM_MASK_W`, the symbol the new permission clause tested, **exists nowhere in the QEMU tree**.
The change had never been built when I described it as written and reasoned about what it would do.
Corrected to `cap_perms_allow(perms, CAP_PERMS_WO)`, the file's own idiom, which expands to the same
`(perms & 2) == 2` the RTL tests. A second correction in the same pass: the `csinit` comment stated
that the exclusive-`end` ruling was **the lead's 2026-09-10 ruling**. It is not; it is item 1 of this
file and it is still open. The code is written under the *recommended* resolution and the comment now
says so.

### Predictions for the suite run, written BEFORE it

The change is built into `capstone-qemu/build-q07/`, **never into `build/`** — the shared binary is
what every other lane's suite picks up by default, and swapping uncommitted semantics into it is the
2026-09-05 toolchain-rebuild incident in a new coat. The suites take it via `CAPSTONE_QEMU_BINARY`.

**There are TWO different reds here and conflating them would be the whole error.** One is a harness
that encodes the old behaviour; the other is a defect reproducing.

| probe / suite | prediction | the discriminator that makes it that and not the other |
|---|---|---|
| `uninit_use_before_init_fault` | **PASS**, cause 26 | unchanged: the read is refused by TYPE, and the cursor move makes the address *unambiguously* in-bounds, so it tests more than before |
| `uninit_negative_offset_fault` | **PASS, and VOID as evidence** | it reads `db[-1]` *because* revoke parked the cursor at `end`, making that address `end-1` and inside the region. With the cursor at `base` it addresses `base-1`, which is OUT of bounds. It will still report 26, but no longer for the reason its own comment gives. **The probe needs rewriting, not the fix.** |
| `uninit_init_then_use_ok` | **FAIL**, domain faults cause 29 instead of returning `0x1412005e` | it does `revoke` then `cap_init` with **no fill in between**. Cursor at base, `csinit` now demands `>= end`. This is a HARNESS red: the probe encodes the old `cursor = end` shortcut. |
| `run-nullblk-all.sh` | **FAIL** at the monitor's `INIT` after `REVOKE` | `sbi_capstone.c:1196-1197` and `:1340-1341`, same shape, in the monitor rather than a domain. This red **IS M-5 reproduced** and is the intended consequence, not a regression. |
| `run-smoke.sh` | **PASS** | a red here would be a surprise and would say the change reaches beyond the reclaim path |

**`uninit_init_then_use_ok` is M-5 in miniature** — the same revoke-then-init-with-no-rewrite, in
seven lines, in a domain, off the board. That makes it the cheapest possible test bed for whichever
reclaim shape is chosen in item 2, and it costs no board time.

**If `run-nullblk-all.sh` comes back GREEN, the first hypothesis is that it never executed a
revoke-then-init pair — not that M-5 is fine.** A suite that cannot reach the condition reports a
pass, and that is the most expensive mistake available on this project. The positive control is to
confirm from the log that `helper_csrevoke` ran at all before recording any verdict either way.

**The QEMU change will NOT be committed whatever colour comes back.** Its landing condition is
unchanged: it lands together with the monitor change, which is item 2, which is yours.

### THE READING — every prediction confirmed, except the one that mattered most

Run against `build-q07` (positive control: the new-only diagnostic string appears once in the binary
under test and zero times in the shared `build/` one, so any reading here is attributable).

| probe / suite | predicted | actual |
|---|---|---|
| `uninit_use_before_init_fault` | PASS, cause 26 | **PASS, cause 26** |
| `uninit_negative_offset_fault` | PASS, void as evidence | **PASS, cause 26** — and void, as predicted |
| `uninit_init_then_use_ok` | FAIL, cause 29 | **FAIL, cause 29**, `pc = 0x101560264` |
| the four `row11` LINEAR probes | unaffected | **4 PASS** — the change is confined to the UNINIT path |
| `run-nullblk-all.sh` | FAIL at the monitor's reclaim | **PASS** — see below |

**The nullblk prediction was WRONG, and the pre-registered response to a green was the right one.**
I wrote: *"if it comes back GREEN, the first hypothesis is that it never executed a revoke-then-init
pair — not that M-5 is fine."* That is exactly what happened. **The suite never executes a revoke at
all**: `revoke` appears zero times in all three serial logs, no null_blk-side source calls it, and the
guest command is `modprobe`/`insmod`/`dd` with no region revoke anywhere in it. So nullblk is a valid
don't-break-the-split-path control — and it stayed green, which is worth having — but it is **not an
M-5 gate**, and the plan's §5.1 named it as one. The real gate is `uninit_init_then_use_ok`.

Had I written "nullblk green means M-5 is fine" instead of the control, this run would have produced a
clean, confident, entirely void result. That is the sixth instance of that shape this session.

**One probe now needs rewriting, and it is the probe and not the fix.**
`uninit_negative_offset_fault` reads `db[-1]` *because* revoke parked the cursor at `end`, which made
that address `end-1` and INSIDE the region — that is its whole stated point, separating "no read
authority" from "out of bounds". With the cursor at `base` it addresses `base-1`, which is outside.
It still reports 26, because the type check precedes the bounds check, so it passes for a reason its
own comment contradicts. Gated on the same decision as the fix; noted so it is not forgotten.

---

## 6. Smaller, and safe to defer

* **R-4** — retitled RECORD ONLY. Closure as *"not reproducible"* was rejected as overstating: nobody
  ever attempted reproduction. **Confirm or overrule.**
* **C-45** — decided to land this cycle (own commit on top of C-38's, never squashed). Compiler lane is
  holding for your word.
* **`precommit-scan` scope** — proposal written and NOT applied
  (`precommit-scan-removed-lines-proposal.md`): removed and context lines should WARN, not BLOCK.
  Every pattern unchanged; only which lines can FAIL narrows. Six controls listed. It currently
  prevents **C-4** from being archived despite a final status.
* **CLAUDE.md, two proposed sentences**, not added because that file is yours:
  1. *"Name the observation that proves the triggering condition existed, and make the instrument
     refuse a verdict without it."*
  2. *"State which definition a constant or symbol comes from before acting on it; this codebase carries
     three numbering systems and two of them silently disagree."*

---

## What is NOT waiting on you

Done and pushed: R-30/R-31 filed, demonstrated and through all four gates (lint at baseline, a
one-variable functional pair, an auditor pass, 88/88 sweep); R-25/R-26/R-27 archived; the fence.i drop;
the SQLite stock-ness merge with its board validation; boots sw44–sw51; the R-29 folder split; C-5 and
C-38 closed; C-45 and C-46 filed; R-11 run with the positive control it never had; the registry
corrections; and the ID allocator.

**Read this alongside `next-cycle-r29-and-close-out.md`**, whose "STATE AS OF 2026-09-10 NIGHT" section
lists what was retracted today and why — four of my own conclusions among them.
