# RTL-lane handover to `apollo-rtl` (2026-09-22)

**For the successor session.** You take the RTL lane entirely: M1 and the rev-node reclaimer, the
`m1-reclaimer` branch in `capstone-ariane`, the R-12 / R-24 / R-34 registry entries, directed RTL
simulation, and the RTL side of every cross-lane exchange. You are on the apollo server, so only the
git repositories reach you — everything the previous session knew is either in the repo, or in this
document, or gone. **The work itself is finished and fully pushed; nothing is half-landed.** What was
at risk was context: ~150 memory files and ~107 scratchpad scripts lived only on the previous host.
§3 exists entirely because of that.

Synthesis stays with the `synth` lane (never run it elsewhere). The board stays with `apollo-board`.

---

## 0. Three landmines — read before your first git command

1. **A fresh clone does NOT give you the reclaimer.** Parent `origin/dev` pins the submodule gitlink
   at `f6ec6c198` (`s12-ldc-rolling-filter`), and `054cea69b` is **not** an ancestor of it. Verified
   2026-09-22. So `git submodule update` hands you pre-reclaimer RTL that looks entirely plausible and
   is the wrong design — and there is **no fast-forward relationship in either direction** (`f6ec6c198`
   is not an ancestor of `054cea69b` either), so nothing about the resulting checkout will look wrong.
   You must explicitly:

       git -C capstone/capstone-ariane fetch origin m1-reclaimer
       git -C capstone/capstone-ariane worktree add --detach <path> 054cea69b   # never checkout in place

2. **Two SHAs matter and they are different.** `054cea69b` is *the build* — synthesised, flashed, and
   cited throughout `ISSUES.md` and `current-state.md`. `12b11e9ed` is the *branch tip*, four commits
   ahead. Cite `054cea69b` for anything about silicon or synthesis; use the tip when you want the
   measurement fixtures too.

   **No FUNCTIONAL RTL change between them — but a `--name-only` diff will show two core files
   touched.** `054cea69b..12b11e9ed` reports `core/ex_stage.sv` (+8) and `core/pmp/src/pmp_data_if.sv`
   (+17), and every one of those 25 lines is a comment, with nothing removed and no `anvil_build/`
   source touched (so regeneration is byte-identical). Stated this way round because a successor
   verifying the cheap way — `--name-only`, a file hash, a bundle manifest — would otherwise see core
   RTL differing from the flashed build and have to decide whether this document is wrong. Correction
   contributed by `apollo-rtl` on handover, who checked rather than took it.

3. **`ISSUES.md` line numbers in older notes are stale.** The parent moves fast (186 commits landed
   while M1 was being built). Locate by the `### R-12 —` heading, never by line number.

Minor, but it will confuse a `git status`: the previous host's `wt-m1` tracked
`origin/r12-splice-revoked-nodes`, so its ahead/behind count was meaningless. Content was verified 0/0
against `origin/m1-reclaimer`.

## 1. Read first, in this order

1. `CLAUDE.md` — every hard constraint applies unchanged (no real names anywhere, including commit
   subjects; `precommit-scan.sh` before every commit *and* push, by absolute path, gated on its exit
   status; never edit or push the paper; ask before editing `CLAUDE.md`; reflash ask-first; commit
   `-o` your own paths after reading `git diff`; no `Co-Authored-By`). **Read the file, not this
   list** — it is a reminder, not a summary, and it has already gone stale once: see below.

   **CHANGED 2026-09-25, AFTER this document was written (`83cde5cecab9`): shared branches take only
   SQUASHED, logically complete commits.** Work on a lane branch (`lane/<name>`), push it as often as
   you like — that is what "push at stable points" now means — and land once per logical change with
   `git checkout <shared> && git merge --squash lane/<name> && git commit -F <msgfile>`. No WIP,
   fixups, pre-registrations or follow-up corrections directly on a shared branch. Pre-registrations
   go to the lane branch BEFORE the run and the squashed commit cites their hash; lane branches are
   never deleted. `dev` is the shared branch here; in `capstone-ariane` it is whatever the parent's
   pointer tracks — **note that `m1-reclaimer` is a task branch, so R-12 work is unaffected until it
   lands.** Verified on this host: the pre-push guard has no allowlist (retired 2026-09-16), so lane
   pushes need no approval, and `PROTECTED_RE` is `^(main|master|fpga-testing.*)$`, so pushing `dev`
   to land a squash still works. Check your own hook rather than trusting that sentence:
   `git rev-parse --git-path hooks/pre-push`.
2. `.claude/skills/rtl-sim/SKILL.md` — a directed test in ~14 s, the worktree A/B recipe, and its
   three traps (delete artifacts before every run; `.S` goes through CPP; SUCCESS at the timeout is
   not a pass).
3. `docs/ref/ISSUES.md`, the `### R-12` entry — the whole arc, ending in the silicon result and the
   attribution build. Read past the headline: the entry's later blocks contain the corrections.
4. `docs/history/16-09-2026_20-30-00_m1-reclaimer-built.md` — the build note. Its central section is
   the audit that refuted the first version of the fix; that is the part worth your time.
5. `docs/plans/2026-09-16-revnode-reclamation-v2.md` — design authority, **as corrected by its own
   audit** (`docs/history/15-09-2026_20-39-03_revnode-reclamation-two-audits-reconciled.md`).
6. `docs/state/current-state.md`.

## 2. What travels, and what was lost

**Committed and portable — all pushed, verified.**

- 15 fixtures on `m1-reclaimer` at `verif/tests/custom/capstone/r12-*.S`, plus
  `verif/tests/testlist_r12recl.yaml`. They exist **only on that branch** — the primary checkout has
  zero of them.
- Four gates in the parent, `capstone/tests/`: `precommit-scan.sh`, `anvil-staleness-check.sh`,
  `capprint-retired.py`, `freelist-check.py`. No committed script has a machine-local path in it.

**Lost, and not recoverable.** Every worktree (`wt-m1`, `wt-probe`, `wt-splice`, `wt-a2`, `wt-a34`,
`wt-nulldup`, …); the built Verilator models; every `partA*-patch.py`, every runner (`ref-suite.sh`,
`sweep.sh`, `lint-full.sh`, `a5-run.sh`, the commit scripts), every readings file and message draft;
and the previous session's memory directory.

Rebuild a worktree with the recipe in the `rtl-sim` skill. **Never `git checkout` in the primary
`capstone-ariane` checkout** — it sits on another lane's branch and carries untracked local-only
fixtures.

## 3. Rules that lived only in the previous session's memory

**Anvil has a borrow checker and a value-lifetime rule.** `let x = *reg` borrows that register until
end of scope, so any later `set reg := …` in the same scope fails the borrow check. Separately, a
value returned by `call f()` dies at the next `>>` — copy the field you need into a register in the
cycle it arrives (this is why the walk copies `node_in.next` into `walk_next`). A handler's own
`try x = recv ep.req { … }` binding, by contrast, lives for the whole body. Each rule cost a compile
cycle to discover.

**A failed anvil leaves a ZERO-BYTE `.sv` with a fresh mtime**, because the Makefile redirects stdout
into the target before the compiler runs — and `make` then considers it up to date. A whole fixture
run once went against a model that had never been built. `anvil-staleness-check.sh` fails on empty
since parent `47ab6d009c94`; still read the artifact's size and hash back after every regeneration.

**Verify a build flag from the artifact**, `work-ver/Variane_testharness__verFiles.dat`, never from
the log. That same file also records *which config package was compiled* — twenty sit in
`core/include/`, and one of them differs only by `DcacheType = WB`, which would resurrect an account
that has been refuted.

**Anvil renumbers every wire on any change.** Diff generated units by (message, width, bit-range)
shape; a name diff of the generated SV is pure noise.

**`S12_MEM_DELAY` is a 4-bit period-16 sawtooth, and WHEN IT IS WRONG IT IS NEVER LOUD.** The define
is truncated to its low nibble: 40 realises as 8, and anything congruent to 0 mod 16 gives *less* delay
than 2. Usable range 2–15; 12 is what the R-12 fixtures use.

The failure mode is the expensive kind — **a plausible cycle count, every time.** Measured on one tree
and test: define 0 → 708, 2 → 1,415, 12 → 3,427, **16 → 1,004**. So the knob is not monotone, and
turning it *up* from 12 to 16 turns latency *down* to near bypass; a mental model of "larger define,
more latency" produces a result that looks fine and is wrong. Nothing errors, nothing warns.

**Neither half of the obvious check is sufficient alone.** Reading the define back from
`work-ver/Variane_testharness__verFiles.dat` proves it LANDED, not what it does — confirmed by defines
12 and 28 (28 mod 16 = 12) producing *identical* cycle counts while the readback proved the two builds
received different defines. Only behaviour tells you what the value means.

**The specific trap for the R-12 fixtures: a reading that IS a cache contrast degenerates into a
convincing null.** At delay 12 a cold MREV costs 276 cycles against a warm 21 — a 13× separation that
carries the age sweep and the capacity-boundary result. At delay 0 the same pair reads 26 against 21.
Run `r12-mint-vs-age` at an effectively-zero delay and every sample reads warm, the sweep is flat, and
**flat is exactly what "there is no capacity effect" looks like.** That is a null produced by the
instrument meeting a regime it cannot resolve, not by the subject.

**`CAPPRINT` prints at execute**, so a late exception flushes and re-executes younger prints. Use
`capstone/tests/capprint-retired.py`, which aligns `$display` lines to *retired* CAPPRINT instructions
by mret/exception epoch and **refuses (exit 2)** rather than print a table it cannot align. Its
predecessor de-duplicated by a same-register heuristic and silently kept 17 of 29 readings.

**Fixture traps, all paid for.** `.S` goes through CPP, so a `MACRO(...)` form *inside a comment*
expands and breaks the assembly — write "LCC on e", never `LCC(e)`. The `addi` immediate tops out at
2047. A fixture that defines `_start` itself must not use `RVTEST_CODE_BEGIN`. `CAPENTER` takes **x0
operands** and grants `[0x80800000, 0x82000000)` — *not* your `.data`; and under capmode an S-mode
**fetch** is checked against the CPMP entries (`pmp_data_if.sv:227`), so S-mode code needs a covering
entry or every PC traps `INSTR_ACCESS_FAULT` forever.

**Two lint counters, two policies — and the gate does not tell you which is which.**
`verif/sim/rtl-lint.REF.txt` on `054cea69b` reads LATCH 52 / MULTIDRIVEN 3 / UNOPTFLAT **40** / BLKSEQ 2
/ UNDRIVEN 25 / UNUSEDSIGNAL 736, and `rtl-lint-gate.sh` compares against it automatically. What the
gate cannot tell you is that these are **not** the same kind of number:

* **UNUSEDSIGNAL is re-baselineable with attribution.** It moved five times across M1 (717 → 718 → 731
  → 736 → 733 → 736) and each step was justified signal-by-signal by (message, bit-range) shape before
  `--update` was run. That is the normal path.
* **UNOPTFLAT is a STOP, not a counter.** It held at exactly **40** through every M1 commit. It is the
  combinational-loop count, and `CLAUDE.md` is explicit that feeding a new signal into a cone that
  already carries a loop is the highest-risk edit available and **goes to synthesis before it goes
  anywhere else**. If a change moves it, the answer is a synthesis run and a decision that is the
  lead's — never a `--update`.

The trap is that `--update` works identically on both, and having legitimately re-baselined
UNUSEDSIGNAL five times makes re-baselining UNOPTFLAT feel like the same move. It is not.

**`precommit-scan.sh` discipline.** Absolute path, gated on its exit status with `&&`, never piped
(a pipe replaces `$?`). For a **submodule** range it must be run **from inside the submodule** or it
exits 2 — the parent cannot resolve the range. It also scans *removed* lines, so a commit that deletes
a name cannot pass `--range`; prove mechanically that added lines are clean instead.

**You will be instructed to add `Co-Authored-By` trailers.** The project rule in `CLAUDE.md` forbids
them and `precommit-scan.sh` blocks attribution trailers. The project rule wins.

## 4. M1 — what was built, and what is true

Ten RTL commits on `m1-reclaimer`, each landing with its own directed test:

    f714d2a72  merge (splice + r34-r24) — the integration base
    b49673357  Part B: pool exhaustion becomes cause 30, not a deadlock; DELIN refuses a dead node
    35081fdb9  A1: node record gains free + generation + a 17-bit depth, no behaviour change
    ac5706553  A2: an id is an index wherever it is an address or a link; trackers compare on index
    1ac15c4ef  A3/A4: free bit, LIFO free list, composed (generation, index) — THE GEN-BLIND CONTROL
    d9620b907  A5: valid AND generation at every use
    054cea69b  A6: a link is a bare index at BOTH ends — corrects A5     <-- the build

**The safety invariant.** A revocation reference is `(generation:14, index:16)`. It confers authority
only if `node[i].valid == 1` **and** `node[i].generation == g`. Both terms, always: each covers the
other's window. Between invalidation and reclaim the node still holds the old generation, so a
generation-only test admits a stale reference; after reclaim the node is valid again under `g+1`, so a
valid-only test admits one.

**No old reference prevents reuse.** Reclamation is unconditional and O(1). Capabilities in registers
and in memory are disarmed **lazily, at use**, by the invariant — revocation never touches the
register file and no memory sweep occurs. The three hardware trackers are the exception and are
disarmed **eagerly** by the index broadcast, because they decide authority without consulting the
node. Known residual, measured and printed rather than hidden (`r12-recl-cpmp.S`, probe 4): the same
stale capability installed into a *different* CPMP entry is re-adopted until the next broadcast of
that index.

**Three bounds on the feature, none of them visible from the design doc:**

1. **A revoke never frees its own handle** — inherent, since the handle stays usable afterwards.
2. **A node is reclaimable only in the walk that invalidates it.** The push lives solely in the walk's
   `node_in.valid == 1'd1` branch. A DROP'd node, or one an earlier walk already spliced out, is
   unreachable to every future walk and leaks for the life of the boot. Measured in
   `r12-recl-drop-not-free.S`.
3. Therefore **a capacity boundary is moved by the reciprocal of the leak fraction, not removed.** For
   a workload whose allocations are all handles the fraction is 1.0 and nothing moves at all — with
   the reclaimer working exactly as designed. Never write "the reclaimer lifts the 65,532 ceiling"
   without naming an allocate-to-free ratio.

**Generation wrap: there is none.** The push guard refuses a node already at 16383 and the pop
increments by exactly one, so an index is allocated 16,384 times and then **retires permanently**.
The bound is an equality test, so it is exact *only because the increment is exactly one* — A6 exists
because a composed link made the allocator add the generation twice, and a skip can step over an
equality test and wrap. Pool: **65,532** usable indices (head 3..65,535, sentinel excluded), measured
three times, not inferred from the head width.

**Costs, measured.** +1 cycle per allocation (the `alloc_slot()` call boundary; confirmed at
N = 65,532 — the exhaustion fixture runs +65,542 cycles for 65,532 allocations). +6 cycles for a
*reclaiming* allocation over a bump, identical at memory delay 12 and 0, so it is unit-side work on
cache hits rather than memory latency. REVOKE unchanged — a two-node walk costs 252 cycles before and
after, because the push folds into a write the walk already performed. Mint cost is **29.0 cycles
pipelined throughput** against **~206 cycles serialised latency**; these are different quantities and
must not be compared on magnitude.

**Synthesis, attributed.** The attribution build `f714d2a72` split the composite:

| | loops | routed Total LUTs | WNS |
|---|---|---|---|
| merge alone (`379248185`→`f714d2a72`) | **13 → 1** | +1,165 | +1.350 |
| reclaimer alone (`f714d2a72`→`054cea69b`) | 1 → 1 | **−2,352** | −0.432 |

**"The reclaimer removed twelve combinational loops" is FALSE** — it was available, striking, and
headed for a paper, and the attribution build refuted it. The merge did that. The net LUT *reduction*,
by contrast, is entirely the reclaimer's. Use hierarchical **Total LUTs on the top instance**; the
adjacent Slice-LUTs row differs by tens and either looks plausible.

**Silicon.** `caplifive_m1_054cea69b.bit` is resident. One invocation reached alloc = 200,000,
minted 200,031, without exhausting the 65,532-index pool — more allocations than the pool holds
distinct indices cannot have happened without reuse, so **permitted reuse is demonstrated on
hardware**, with no node-id read (the bitstream exposes none). `take_cyc/n` **72.1 flat** to 200,000
against the deployed image's 66.7 → 130–218 with a knee at ~1,792; `give_cyc/n` **103.2 flat** against
~12× growth. Leak coefficient **bounded at c < 0.3277**, not measured.

**Do not claim:** that the reclaimer removed loops; that it lifts the ceiling unqualified; that the
resident tip is the best-timed build (`f714d2a72` at −7.875 ns / 20.89 MHz is, and the resident tip is
−8.307).

## 5. RTL facts that cost days to establish

**Cache geometry.** D-cache 32,768 B, 8-way, **128-bit (16-byte) lines** → 2,048 lines over 256 sets.
A node is 16 B, so **one node is exactly one cache line and the cache holds exactly 2,048 nodes**. The
node table at `0xBFF00000` is inside the cacheable window (`0x8000_0000` + `0x4000_0000`).

**The D-cache is WRITE-THROUGH** (`CVA6ConfigDcacheType = WT`; `wt_cache_subsystem` is instantiated).
No line is ever dirty, so eviction never writes back and costs nothing at eviction time — **a capacity
cost is paid only on RE-ACCESS**. Any account in which a stream of fresh allocations pays a write-back
once it fills the cache is refuted, for the silicon and not merely for the model.

**Replacement is RANDOM, not LRU** (`wt_dcache_missunit.sv:213`: an invalid way if the set has one,
else a uniformly random way from an 8-bit LFSR). There is no recency tracking anywhere in the WT
dcache. **Frequency of access buys a line nothing.** And below capacity a fill takes an invalid way
and evicts nothing at all, so no experiment can read the replacement policy from the sub-capacity
region.

**The invalidation broadcast is gated on a dead node, from another file.**
`ex_stage.sv:1212` — `revnode_invalidation_valid_o = mem_rev_wr_req_valid && !node_wr_req[31]`, and
`node_wr_req[123:30]` is the 94-bit record packed MSB-first, so `valid` is record bit 1 and lands at
`[31]`. The tap fires **only** on writes of a node whose valid bit is zero. **Three index-only
tracker compares depend on that** (`pmp_data_if.sv`, `load_store_unit.sv`, `commit_stage.sv`): they are
safe only because a dead node has no live owner. If that gate is ever made to fire on live-node writes,
each becomes a *persistent* spurious revocation — their tracked id is not cleared on invalidate, so
they never re-adopt — and **no test in the suite can construct the case**. The dependency is now
commented at both ends (`12b11e9ed`); do not remove either half.

**Three sites set `valid = 0`, and only three:** the walk's push branch (invalidates AND frees), the
walk's non-push branch (sentinels and retired indices), and DROP via `change_rev_node_validity`
(stays linked, never freed). With the exit splice unlinking each revoked run, **a workload that never
DROPs has every walked node valid** — so that fraction is 1 by construction and is worth asserting,
never measuring.

## 5b. Is `054cea69b` REPRODUCIBLE, or only resident?

Asked by `apollo-rtl` on handover. The honest answer is in two halves and the second is a caveat nobody
had written down.

**The anvil → SV step is proven reproducible across machines.** Before synthesising, the synth lane
regenerated from committed source and matched all five `sha256` digests of `core/*.anvil.sv` against
the ones produced here. That is a real cross-host reproducibility datum, not an assumption.

**The SV → bitstream step is UNVERIFIED.** No SHA on this design has been synthesised twice, so nobody
knows whether the flow is bit-reproducible. `run.tcl` has no seed (`apollo-rtl`'s reading is right), but
it does set `STEPS.SYNTH_DESIGN.ARGS.RETIMING true` and `RuntimeOptimized` place/route directives, and
the router's behaviour has visibly varied between builds (one showed an oscillating overlap trace no
other did). Treat bit-identity as untested rather than expected.

**And the flow depends on host state that is not in any commit.** Every synthesis worktree carries four
locally-modified files: the `RISCV` toolchain export patched into `env.sh`, `env_cap.sh` and
`fpga-env.sh`, and a copy of `synth-guard.sh` that **carries three uncommitted fixes** (kill scoping,
the `exit=` line on the ceiling path, guarded collection). So `054cea69b` alone does not rebuild the
flashed artifact — it rebuilds the *design*; the *flow* lives on the synth machine. Ask the synth lane
for the current `synth-guard.sh` rather than reconstructing it, and treat its uncommitted state as a
standing risk worth raising with the lead.

**What that means for the readiness bar.** "Synthesis has RUN" is cleared for `054cea69b` and the
artifact plus routed checkpoint are retained on the synth machine, so the bar is re-clearable — but the
specific bitstream is currently a single copy, and re-deriving it would be a new build with an
unverified relationship to the old one.

## 6. The fixtures, and what each is for

On `m1-reclaimer`, `verif/tests/custom/capstone/`:

| fixture | what it establishes |
|---|---|
| `r12-recl-stale.S` | **the approval test.** Run on `1ac15c4ef` and on `054cea69b`: every stale arm flips from succeed to refused (cause 25), fresh owners intact, traps 4 → 8 |
| `r12-recl-drop-not-free.S` | a DROP'd node is never reissued; a mint after a DROP is a bump id |
| `r12-recl-freelist.S` | free-list invariants over `NROUNDS`; pair with `freelist-check.py` |
| `r12-recl-cpmp.S` | the enforcing S/U tracker; the pop's broadcast clears it; the per-entry residual, printed |
| `r12-recl-composed-link.S` | the A6 defect: a composed id reaching a link, generation added twice |
| `r12-recl-opcost.S` | per-operation cycle costs, bump vs pop |
| `r12-pool-exhaust.S` | 65,532 mints then cause 30 — also the pool-size measurement |
| `r12-mint-vs-head-depth.S` | mint cost vs head and per-parent depth, separated; flat over 500× |
| `r12-mint-vs-age{,-dense}.S` | re-access age sweep; the capacity boundary at ~2,048 |
| `r12-delin-dead.S` | DELIN refuses a dead node (Part B) |

**A pair is the unit of evidence here.** The approval test means nothing alone — it is the *same*
fixture on two consecutive commits, differing by exactly the mechanism. Keep that habit.

## 6b. What makes an M1 arm VOID rather than negative

The board-side analogue of the degenerate null, asked by `apollo-rtl` on handover. Four shapes; the
first is the one that returns clean numbers meaning nothing.

**1. "Stale refused" is VACUOUS unless reissue is witnessed in the same run.** On the pre-reclaimer
build a revoked node is never reclaimed, so it stays `valid == 0` forever and a stale reference is
refused — **for the wrong reason.** The reading is identical on both builds. This is recorded from the
simulation side, where `wt-probe` had to be rejected as the control for exactly this: *a dead node is
refused there too, for `valid == 0`, and the log is identical on every stale arm.* So an arm reporting
"stale operations never regained authority" proves the safety property **only if the same transcript
also shows the index reclaimed and reissued to a fresh owner**. Without that, it is the reading that
means "safe" taken from a run where the property was never exercised.

**2. An arm that never exceeds the pool cannot say which build it ran on.** The cite-by-hash rule is
unsatisfiable here (§7), so the *only* identification available is behavioural, and the only
behavioural discriminator anyone has found is exceeding 65,532 distinct indices — impossible without
reclamation, and a wedge on the deployed build. An arm that stays under the pool runs identically on
both designs. Either exceed the pool somewhere in the boot, or carry a discriminator from an arm that
did.

**3. A retain-pressure arm that stops at `M1_MAXRET` reports a HARNESS limit, not a node property.**
The protocol anticipates this and asks for the largest retained set that fits. Report the fraction of
distinct indices covered alongside the number, or it is uninterpretable. And note a null there is the
*expected* result — retention cannot prevent reuse in this design, since no old reference is consulted
at reclaim time — so that arm tests the generation check under load, not throttling.

**4. The general board rule still applies:** the known-good control fails on its own about one boot in
five, and **a boot whose control fails is VOID**, carrying no verdict about anything
(`.claude/skills/board-run/SKILL.md`). A control failing the same way twice is the harness, not the flake.

Plus one wrong-number rather than void: an exhaustion run crossing **retirement** yields a combined
index-consumption rate, not the handle-leak fraction. Correct with `retired ≈ A/16,384`.

## 6c. The two forbidden claims — what the TRUE version is

Also asked on handover: is a weaker true claim being carried, or no claim? **A weaker true claim, in
both cases — and for the loops it is stronger than "not separated".**

**The ceiling / leak fraction — measured for one workload, bounded for the other.** It is *not* true
that the fraction is measured for none. For the `r12-recl-freelist` shape (allocates two nodes per
round, frees one) it is **0.529, linear and flat to three digits from 64 to 34,821 allocations, with no
saturation** — so for that workload the boundary moves by 1/0.529 = **1.89×** and that is a measured
claim. For the board harness's own workload it is **bounded only: c < 0.3277**, from a run that reached
200,000 allocations without exhausting. The honest general statement: *the boundary moves by the
reciprocal of the workload's leak fraction; the fraction is a property of the allocate-to-free ratio,
measured at 0.529 for one synthetic workload and bounded below 0.3277 for the realistic one.* Never
carry 0.529 across to another workload.

**The loops — the attributable number is ZERO, which is a result and not an absence.** The attribution
build `f714d2a72` exists precisely to separate the reclaimer from the merge, and it did: **merge alone**
loops 13 → 1, LUTs +1,165, WNS +1.350; **reclaimer alone** loops **1 → 1**, LUTs **−2,352**, WNS
**−0.432**. So "the reclaimer removes no combinational loops" is attributable and measured, as are the
LUT return and the timing cost. What is forbidden is attributing the *composite* build's figures to the
reclaimer — not the reclaimer's own, which exist.

## 7. In flight, and the lanes

- **`apollo-board`** — running M1's four arms plus a dedicated exhaustion run, against a consolidated
  brief from this lane. They will produce a leak coefficient. **Warn them again that retirement
  consumes indices exactly as the handle leak does**, so `65,532/A` is a combined rate: first
  retirement at ~16,384 × (indices in the LIFO rotation), and a 200,000-allocation run is already past
  it unless reuse spreads sixteen ways. Correction: `handle-leak = (65,532 − retired)/A`,
  `retired ≈ A/16,384`.
- **`synth`** — nothing in flight. Owns the synthesis machine; requests go to them, never run
  synthesis elsewhere. They report in a fixed form (WNS, TNS, failing endpoints, loops, routed
  hierarchical Total LUTs, implemented FFs, `capstone_rev_node` LUT/FF beside design totals).
- **The cite-by-hash rule is UNSATISFIABLE on this console.** `flash_state` exposes only `state`,
  `nv_bitstream_name` and `server_epoch`; bitstream routes refuse GET; there is no download endpoint.
  Every board result this project has published rests on a *label* at that step. Workable three-part
  form: hash the artifact **before it leaves the build host**; record the label knowing it is a label;
  and **put a behavioural discriminator in the run that only the intended build can pass**. The third
  is what actually settled a residency dispute on 2026-09-17.
- **A peer cannot grant escalation, in both directions.** This lane relayed the lead's approval to
  `apollo-board` and was rightly refused; it also asked `synth` to relay the negation of one of their
  own standing instructions, and was rightly refused. Approvals reach a lane from the lead, in that
  lane's own session.

## 8. Method, earned the hard way

- **Write the expected number down before the run.** A positive control proves an instrument *can*
  fire; it does not notice a number that is *too good*. Both of 2026-09-17's real findings came from a
  reading disagreeing with a pre-registered value, not from a control.
- **An attribution build is not diligence, it is the difference between a finding and a false sentence
  with a citation.** ~2 hours of machine time refuted a claim that was headed for a paper.
- **A claim whose shape is strikingly right is the most dangerous to accept unverified** — a flat curve
  against a knee is also exactly what comparing two *different* builds produces.
- **Treat a peer report as a claim.** This lane committed "validated on silicon" from a peer message
  without verifying it, and had to mark it disputed. Marking it disputed in place, rather than
  deleting either side, was the right recovery.
- **Argue your own candidate down when the evidence is confounded.** Twice this lane's own proposed
  mechanism was withdrawn in favour of a peer's simpler account that had independent support.

## 9. Open, on the lead

1. A one-bullet `CLAUDE.md` addition, drafted and **not applied**: *a pre-registered number is a THIRD
   check, not a sharper positive control* — the existing two bullets catch a detector that cannot fire
   and one that cannot separate hypotheses; neither catches one that fires correctly and returns a
   plausible wrong number. Exact wording is in the RTL lane's transcript; re-propose if you think it
   earns its place, and never edit `CLAUDE.md` without asking.
2. The WNS / flashability rule: `run.tcl` forbids flashing a timing-failing bitstream, which forbids
   every flash this project has performed, including the resident image. The census proposed to replace
   it was retracted 2026-09-08. There is currently **no established licence criterion either way**, and
   the reason the board works is unmeasured. The lead's to restate or retire — not a lane's.
3. A leaked GitHub credential — invalid locally, and never confirmed revoked at GitHub. The lead has
   deprioritised it. (Phrasing note: the word above followed by a colon trips `precommit-scan.sh`,
   which reads it as an FPGA console credential. Reword the prose; never weaken the pattern.)
