# R-35 — on silicon a REVOKED capability still reads AND writes the storage its object has given up, at every age, and the access does not trap

**Status (2026-09-21): ROOT-CAUSED, in the board's own RTL, and the cause is NOT a missing tag check.**
REPRODUCED ON THE BOARD four times with controls; provenance SETTLED (the flashed bitstream carries the
R-34/R-24 fix, verified by a cause comparison on the board rather than by the build record). The defect
is in `core/load_store_unit.sv:966-971` of RTL `054cea69b`: the revocation check is real, but it depends
on a **single core-wide tracked revnode id**, and an access presenting a different id **re-adopts itself
as valid**. See "ROOT CAUSE" below. Bounds and permissions survive because they are read from the
capability's own metadata, which is why this path enforces bounds and not revocation.

**CURRENT STATUS (2026-09-23): the fix is `capstone-ariane` `f83fe9342`, branch `r35-m1-revnode-cache`.
It is correct in simulation and is NOT a reflash candidate.** `f83fe9342` WAS SYNTHESIZED 2026-09-23 AND IS NOT DEPLOYABLE. Area: fixed -- 177,669 post-synth LUTs, -17,423 against the crossbar build with -111 FFs, i.e. the crossbar rewritten as a register file and nothing else. Slack: REFUTED -- routed WNS -27.665, the worst this design has produced, against a pre-registered prediction of roughly -12.9. The worst path is SHALLOWER than Stage 0's (94 vs 121 logic levels, less logic delay) but carries +15.45 ns of ROUTE delay: congestion, absent from every earlier build. Traced at pin level by the synthesis lane: D-cache read-port grant (fan-out 88) -> the rev-node's memory-channel write-request selector, a deep combinational Anvil mux (fan-out 70) -> the cache's WRITE decode. The fix hung a 256-entry write decode off that selector. The read mux is not on the path. Next: register the fill taps, so the decode sees flops. It
keeps the positive-validity-cache design below but rebuilds its access structure as an ordinary
register file — no `_d` array, narrow per-set write enables, a lookup that reads the registered array,
and the same-cycle invalidation as one 16-bit comparator — which removes the crossbar that made
`079dc720a` unimplementable. Geometry is deliberately unchanged at 4 ways × 64 sets. Acceptance fixture:
**exactly 7 traps**, the three revoked accesses trapping 25 and both live-alias controls still returning
`0x005b5b5b5b5b5b5b`. Lint at the committed baseline.

**A worked example of a fixture blind to what it certifies.** The first register-file rebuild,
`6ee277cc3`, introduced an authority escape of **this issue's own class**: its invalidate matched the
array *before* the same cycle's install, so a read-tap install for an index being invalidated that cycle
created a **live** entry for a node just written dead — and every later access through that id would
have been allowed. `f83fe9342` closes it. **This fixture returned exactly 7 traps before and after that
fix.** It cannot see it: all three cause-25 arms emit the same cause and `tval`, it never evicts, and it
never presents a same-cycle install/invalidate coincidence. The escape was found by an adversarial audit
that built a microtest from verbatim extracts of the committed logic, with mutants as positive controls.
A passing run of this fixture is evidence about the rotation defect it was built for, not about
coincidence paths.

The paragraphs below record the two hashes that were synthesized and rejected, in order. Neither is a
reflash candidate.

**FIX VALIDATED IN SIMULATION (2026-09-22), NOT YET SYNTHESISED AND NOT ON SILICON.** *[superseded by the
status above]*
`capstone-ariane` **`079dc720a`** (branch `r35-m1-revnode-cache`) replaces the single core-wide tracker
with a tagged 4-way x 64-set positive validity cache filled only by observing the rev-node unit's own
node-memory traffic, so an access is allowed only if its exact 30-bit `(generation, index)` is resident
and was last seen live. On the acceptance fixture the pair **inverts**: the three accesses that
previously succeeded through the revoked capability now trap 25, both live-alias controls still return
data, and the surviving read proves the store never reached memory. A matched-pair 95-test sweep shows
**identical per-test trap counts on all 92 comparable tests**, with the build proved to track source by
re-running the fixture on the reverted tree (4 traps against 7). Lint at exactly the committed baseline,
`UNOPTFLAT` unchanged at 40. Full readings and the limitations — chiefly that **this revision denies on
a cache miss**, which is safe but produces false denies the existing suite is too small to bound — are
in `results/sim-m1-cache-validation.result-lines.txt`.

**SYNTHESIS HAS NOW RUN AND THIS HASH DOES NOT FIT — 2026-09-23. NOT REFLASHABLE.**
`079dc720a` post-synth is **195,092 LUTs / 103,030 FFs = 95.73 % of the device**; the cache alone costs
**+22,968 LUTs and +8,356 FFs**. No build on this design has ever routed above 84.98 %, and this is
21,755 LUTs beyond the only build that ever failed to route. The flop cost was as predicted (within
2 %); the LUT cost is a **defect in the RTL, not the price of a cache** — the lookup reads the
combinational `_d` array through a dynamic 64-way index while the update rebuilds all 8,192 bits every
cycle with four dynamically-indexed write sites, so Vivado built a crossbar rather than a register
file. The remedy was a rebuild of the cache's access structure — per-set write enables in `always_ff`
and a lookup against `_q` — now done at `f83fe9342`. *(Corrected 2026-09-23: this sentence also proposed
"a far smaller array". That is withdrawn. Under deny-on-miss a smaller cache converts misses into false
denies, so what is size-independent is **authority escape**, not correctness; the zero-miss sweep result
is evidence about 256 entries only.)* The post-synth occupancy also overstated the routing risk:
`079dc720a` **did route**, at 192,642 LUTs = 94.53 %, so this design's routable ceiling is about nine
points higher than the 84.98 % it had been calibrated on. **The fix is correct in simulation and not
implementable as written.**

**AND THE UNDERLYING STAGE 0 HASH IS A TIMING REGRESSION.** `247b76896` routed at **WNS -12.900 /
TNS -654,920 / 99,635 failing endpoints** against its parent `054cea69b`'s **-8.307 / -312,530 /
90,379** — i.e. **-4.593 ns and roughly double the TNS from a 32-line change with zero new signal
declarations**. So neither hash in this folder is a reflash candidate.

*(Corrected 2026-09-23 — this paragraph used to end "and the `_q` -> `_d` retarget needs rework at the
CPMP site too, not just the LSU". **That is wrong, and acting on it would reintroduce an escape.**)* An
audit localized the 4.593 ns to Stage 0's **LSU** half, whose post-adopt value fed `cap_exception`
combinationally. Every CPMP consumer reads the **registered** value, so the CPMP half is flop-to-flop into
16 endpoints and cannot account for +9,256 failing endpoints. *That is the audit's structural argument,
not a measurement — Stage 0 changed both halves in one commit, and the one build that would separate them
(`247b76896` with `pmp_data_if.sv` reverted) has not been run.* And reverting the CPMP half to compare `_q`
**admits a persistent false ALLOW**: an entry holding X adopts Y while the same cycle's broadcast names Y,
the compare against the stale X misses, and the entry vouches for a dead Y indefinitely. Leave the CPMP
half as it is.

**Every one of these costs was invisible to our pre-synthesis checks**: nine lint counters at exactly
the committed baseline and `UNOPTFLAT` unmoved at 40, on both hashes.

Sibling issues, so a reader who arrived with the wrong symptom is redirected now:
`../R34-lsu-exception-lost-on-immediate-grant/` is the LSU dropping exceptions it generates, and its fix
(`c77c65324`) **is an ancestor of this bitstream's RTL `054cea69b`** — verified by `git merge-base
--is-ancestor`, so R-34 does not explain this. `../RTL-cap-mcause-off-by-one/` (R-24) is the capability
causes aliasing the core's `DEBUG_REQUEST = 24`, which would route a trap into debug mode instead; the
same commit says it "move[s] the debug sentinel off 24, and put[s] the capability causes on the spec's
base", so R-24 does not explain this either. **This folder is one issue: a capability check that does
not fire on silicon.**

## What was observed

The harness retains one alias per allocation, runs the allocator to its buffer limit, and then probes
the aliases it kept. Board transcript, `results/board-full-probe.r1-lines.txt`, image
`35fb3fec3196841b`, bitstream `caplifive_m1_054cea69b.bit`:

    R1 m1 end arm=pressure stop=buffer alloc=43297 minted=43328 revoked=43297 retained=43296 released=0
    R1 m1 stale-take go
    R1 m1 stale-deref ok k=0     reuses_since=2706 byte=16 live_byte=16 is_live_data=1
    R1 m1 stale-deref ok k=21648 reuses_since=1353 byte=16 live_byte=16 is_live_data=1
    R1 m1 stale-deref ok k=43295 reuses_since=0    byte=31 live_byte=31 is_live_data=1
    R1 m1 stale-write ok
    R1 m1 stale-write readback via_live_alias=165 wrote=165

Read in order:

- **The reference is revoked.** `revoked=43297` — every alias probed belongs to an object whose handle
  was given back. `k=0` was retained at the first allocation and its storage has since been handed to a
  new object **2,706 times**.
- **It still reads, at every age.** Oldest, middle and newest all return a byte.
- **What it reads is the CURRENT OCCUPANT'S LIVE DATA, not residue.** `live_byte` is recorded by the
  harness as the arm runs — the value the most recent allocation wrote into that storage — and
  `is_live_data=1` says the stale read returned exactly that. So this is not a stale-bytes leak; it is a
  working pointer into somebody else's object.
- **It also WRITES.** `0xA5` (165) stored through the revoked reference is then read back **through a
  live alias to the same storage** and is there. The revoked reference can corrupt the current occupant.
- **Nothing trapped.** No wedge; `k800 retval=4` on both the opening and closing control.

## The controls, because a clean result has to be distinguishable from an instrument that never fired

- **The emulator REFUSES the identical instruction, in the identical image.** `results/emulator-refuses.txt`:
  `domain halted by capability fault: cause = 24`, at `pc` offset `0x4354`, which disassembles to the
  `lbu` of the first probe. Cause 24 is defined in `capstone-qemu/target/riscv/op_helper.c` from the
  spec as *"`x[rs1]` is not a capability"*. So the architectural intent is a trap, and the probe
  demonstrably fires.
- **A LIVE alias through the same code path succeeds on the same silicon** (boot of image
  `aed492ab985653f3`, `mepc` landing on the later mint, not on the dereference). So the path is not
  broken in a way that would make everything succeed.
- **An earlier probe using `mrev` proved nothing and is recorded so nobody repeats it:** `mrev` requires
  `CAP_TYPE_LIN` and an alias is `NONLIN`, so it is a type error on *any* alias, live or stale. Both
  faulted; the detector separated neither.

## SETTLED 2026-09-20: the flashed bitstream IS post-fix, so this is a new defect

The provenance question below was answered on the board, by comparison rather than inference.

**The method.** Raise the *same* deliberate capability fault in the *same* image on both the emulator
and the board. The emulator is base 23 by construction, so board == emulator means post-fix and
board == emulator + 1 means pre-fix. **Neither reading needs to know which enum fired**, which is what
makes it immune to the aliasing that blocks every value already on file.

**The fault.** `cincoffset` with a scalar operand — an **execute-path** check (`helper_cscincoffset`,
`op_helper.c:747`), chosen because the execute path demonstrably still checks on this bitstream (`mrev`
through a NONLIN alias faulted with `mcause` 26). Image `c01e454f8652493c`.

| | instruction | cause |
|---|---|---|
| emulator | offset **0x42b4** | **24** (`UNEXP_OP_TYPE`, *"`x[rs1]` is not a capability"*) |
| board | `mepc` 0x81a042b4 − `DBAS` 0x81A00000 = offset **0x42b4** | `sw=255` 0x98 → seen=1, **24** |

**Same instruction, same cause. The flashed image carries `c77c65324`.**

**A first attempt was discarded before it cost a boot**, and is recorded so nobody repeats it: an
out-of-bounds store, which the emulator answers with cause **7** (`STORE_AMO_ACCESS_FAULT`) because
capstone-qemu maps memory-path bounds violations onto the standard code deliberately
(`op_helper.c:1539`). A standard code carries no capability enum and discriminates nothing.

**So hypothesis (a) below is dead and (b) stands: the untagged-base check does not fire on the LSU
path, while the execute path checks correctly on the same silicon, in the same boot, in the same
image.** That is the defect this folder reports.

## The two hypotheses as they stood before 2026-09-20 — (a) is now excluded

Both R-34's and R-24's fixes are ancestors of `054cea69b`. Therefore either:

**(a) The flashed bitstream is not built from `054cea69b`.** The board exposes **no digest** for a
resident image — `GET` returns 405, there is no download route, and `flash_state` carries only
state/name/epoch — so the image is known by filename alone. This project has been bitten by a label
naming different content before (R-29). If the flashed build predates `c77c65324`, R-34 and R-24 come
back as the explanation and nothing here is new.

**(b) The untagged-base check does not fire on the LSU path.** `UNEXPECTED_OPERAND` is raised in
`core/capstone_flu_unit.anvil.sv`; a plain `lbu`/`sb` through a capability base is handled by the
load/store unit, and whether that path checks the tag at all is a separate question from the two fixed
issues. If so this is a new, and severe, gap: revocation does not deny access.

**That question — was the flashed `caplifive_m1_054cea69b.bit` built from `054cea69b`, i.e. with
`c77c65324` in? — is answered YES by the probe above.** An independent answer from the build record
would be a useful cross-check on the provenance trail, but it is no longer what this folder waits on.

## SCOPE, settled 2026-09-21: the LSU enforces BOUNDS but not REVOCATION

The obvious next question was whether this path enforces anything. It does. A store one byte past the
end of a **live, perfectly valid** alias — nothing revoked, nothing stale, the only thing under test
being the bounds check on a good capability — **faults on the board**:

    mcause  sw=255 = 0x9c -> seen=1, cause 28 (OUT_OF_BOUNDS)
    mepc    0x81a042f4 - DBAS 0x81A00000 = offset 0x42f4   <- the store
    tval    0xac100040 = arena base + 64                   <- exactly the byte past the 64-byte leaf

**So the defect is specific, not general.** On the same path, in the same image: a **bounds** violation
on a valid capability is caught and reported precisely, while an access through a revoked reference is
not caught at all.

**CORRECTED 2026-09-21: the revoked reference was NOT untagged, and the tag check is NOT what failed.**
An earlier version of this section said it was. The tag reaches the check — `cap_rmetadata` →
`operand_a_cap_regfile` → `cap_metadata_a` (`cva6.sv:249`, "S-06 fix: {tag, metadata}") →
`decompress_cap_tagged` (`ariane_pkg.sv:762-781`), which returns `NOT_CAP` for a clear tag — and an
untagged `rs1` raises **cause 24** at `load_store_unit.sv:973-975`. Cause 24 was never observed. The alias
was still **tagged** on silicon, and the clause that was defeated is the **revocation** one. The
emulator's "x[rs1] is not a capability" is a model divergence, not a description of the silicon.

Cause 28 does more than narrow the scope: it **identifies the module**. See "ROOT CAUSE" below.

The cause also **corroborates** the post-fix finding independently: `OUT_OF_BOUNDS` is enum 5, and
5 + 23 = 28 on the spec base, where a pre-fix base 24 would have given 29. (It does not *prove* it on
its own — 28 also aliases pre-fix `INSUFFICIENT_PERMISSION` — which is why the execute-path comparison
above is the proof and this is support for it.)

### A model/RTL divergence in cause CLASS, recorded separately because it is not this defect

For the **same instruction at the same offset on the same address**, the two sides classify differently:

| | cause |
|---|---|
| emulator | **7** — `STORE_AMO_ACCESS_FAULT`, a *standard* RISC-V code (`op_helper.c:1539`, deliberate) |
| board | **28** — `CAP_OOB`, the *capability* enum |

Both refuse the access, so neither is a safety gap, and this folder's defect does not depend on it. But
software that classifies faults by `mcause` will classify this one differently under the emulator and on
silicon, which is worth knowing before a corpus run is read either way.

## ROOT CAUSE (2026-09-21) — `load_store_unit.sv:966-971`, a single core-wide revnode tracker that re-adopts

**Which module, settled by the cause number rather than by argument.** Two blocks can check a data
access and they are gated on **complementary values of one signal**:

| block | gate | can emit |
|---|---|---|
| `load_store_unit.sv:948-996` `cap_violation_detection` | `capmode_i && ld_st_priv_lvl_i == PRIV_LVL_M` | 24, 25, 26, 27, **28** |
| `pmp_data_if.sv:292-306` CPMP data check | `capmode_i && ld_st_priv_lvl_i != PRIV_LVL_M` | **5 / 7 only** |

The bounds probe above returned **28**, latched in hardware (`cva6.sv:1116-1124`,
`recent_nontrivial_mcause_log_q <= ex_commit.cause`). CPMP cannot emit 28. **So the M-gated LSU block is
the live path for these accesses**, and CPMP is not involved — independently confirmed by the harness
never touching CPMP at all (`grep -ci 'cpmp\|ccsr' sublet/r1/r1_slots_pools.c` → 0; CPMP entries are
written only by the monitor, `csr_regfile.sv:1931` and `:2418-2559`).

This also means the harness's accesses run at **M-mode**: entering a domain does not change privilege.
`priv_lvl_d` has six writers in `csr_regfile.sv` — `:1048` hold, `:2155` trap entry, `:2318` MRET,
`:2341` SRET, `:2362` VS-RET, `:2376` DRET — **none on a capability or domain-switch path**, and
`capstone_dom_switcher.anvil` has zero `mstatus`/`priv`/`mpp` references.

**The mechanism**, quoted from the board's own RTL:

```systemverilog
// load_store_unit.sv:966-971 @ 054cea69b
// update revnode tracking when a new instruction arrives with a different revnode
if (lsu_cap_type != NOT_CAP
    && lsu_cap_a.metadata.revnode_id != lsu_revnode_id_d) begin
  lsu_revnode_id_d    = lsu_cap_a.metadata.revnode_id;
  lsu_revnode_valid_d = 1'b1;        // <-- an UNTRACKED revnode is assumed VALID
end
```

`lsu_revnode_id_q` is a **single** 30-bit register for the whole core (`:209-210`), and the invalidation
broadcast clears it only on an exact match (`:945-946`). The LSU's only revnode ports are those two
broadcast inputs (`:198-199`) — **there is no query path**, so on a miss the block cannot ask and guesses
VALID. The broadcast is applied *before* the adopt, so an adopt always wins its cycle.

**Why this harness is the worst case by construction.** M1 keeps 16 slots rotating, so consecutive
accesses almost always carry different revnode ids, the one-entry tracker is thrashed, and nearly every
access re-validates itself. That is why the board saw it at **every** age — `k=0` (2,706 storage reuses
later), `k=21648` and `k=43295` alike. The approved M1 specification predicted exactly this residual:
*"the same stale capability installed into a different CPMP entry is re-adopted until the next broadcast
of that index — a property of the tracker, not the reclaimer."*

### The fix is architectural, and the obvious one-liner does not work

Flipping `1'b1` to `1'b0` fails **closed**: every access whose revnode is not the tracked one would then
raise cause 25, which under a 16-slot rotation is essentially every access. Correctness needs the LSU to
**ask** whether an untracked revnode is valid. The rev-node unit already answers that question —
`capstone_rev_node.anvil`'s `IDLE_STAGE` serves `ep.query_req` and replies `ep.query_res(node_in.valid)`
— but **the LSU has no port to it**, and adding one puts a rev-node read latency stall on the access
path in the common case. The alternative is a wider tracker with a miss path.

Either way this is a structural change, not an edit. CLAUDE.md's rule about feeding a new signal into a
cone that already carries a combinational loop applies directly, **only synthesis proves
synthesizability**, and a bitstream is ~90 minutes plus a reflash. The change and any respin are the RTL
lane's and the project lead's calls. **No RTL has been changed on the strength of this folder.**

### Acceptance criteria for a fix — two traps, both of which make a CORRECT fix look broken

Raised by the RTL lane on review, and both are properties of the code rather than opinions.

**1. Cause 28 pre-empts cause 25.** The revnode test is the **last** arm of the priority chain —
`:973` NOT_CAP → `:976` type → `:979`/`:982` permissions → `:985` bounds → `:989` revnode. So an access
that is both revoked *and* out of bounds reports **28**, even with the tracker repaired. Pre-register
cause 25 only for an access that is **in bounds and permitted**, or a working fix reads as a failure.

**2. A single-capability test cannot see this bug at all.** With one capability no access ever presents
a differing revnode id, so the adopt at `:967-971` never runs, the tracker stays correctly invalid, and
**cause 25 fires** — the test passes and reports the defect absent. The ids must **rotate**. This is the
"the synthetic test must CREATE the triggering condition, not merely contain the shape" trap that cost
S-12 a day. `CAPCREATE` cannot supply the rotation — it hardcodes `revnode_id = 2`
(`capstone_flu_unit.anvil:385` @ `054cea69b`) — so distinct ids need `SPLIT`.

A two-sided simulation test is therefore: cause 25 **fires** on repaired RTL and **does not fire** on
`054cea69b`, for an in-bounds permitted access through a revoked capability, with the ids rotating.

### Two analytical notes for anyone reasoning about the trackers, added 2026-09-22 on the RTL lane's request

**1. A DATA-DEPENDENCY argument is strong; a CYCLE-COUNT argument is not. Do not mix them up.**
Several same-cycle races appear around these trackers, and they are not all the same kind of thing:

* An **independent** coincidence — two causally unrelated events that happen to land in one cycle,
  e.g. an invalidation broadcast for index *i* arriving while an access adopts index *i*. Nothing
  orders them, so the only honest statement is that it *will* happen eventually. These are the races
  worth fixing in RTL.
* A **sequenced** coincidence — two events on one causal chain with a `>>` between them, e.g. the
  reclaim pop's generation bump and the handing out of `(g+1, i)`. The Anvil `>>` across a `send` is
  **ack-gated** (the successor event is conditioned on the write grant, visible in the generated SV),
  so these cannot be reordered at all and the "race" is not a race.

The practical rule: *"these probably will not coincide"* is a cycle-count argument and is worth
little — it degrades silently when frequencies, latencies or a `S12_MEM_DELAY` change. *"these cannot
be reordered, because B's enable is derived from A's acknowledgement"* is a data-dependency argument
and holds under any timing. **When deferring a race, say which of the two you have.**

**2. The two broadcast compares have OPPOSITE widths, and each has a worked example of getting it
wrong.** The invalidation compare must be **16-bit** (a bare index) and any future validation compare
must be **30-bit** (`{generation, index}`). Anyone "harmonising" them breaks one or the other, so both
are commented at both ends. The two traps, both real and both found the hard way:

* **A field that is 30 bits WIDE but carries a bare INDEX.** `node_wr_req[29:0]` looks like a
  composed id and is not: `send_revnode_update` masks the generation off at source, and its own
  comment says *"the generation lives in the node, never in a link or an address."* A 30-bit compare
  fed from that field would test `(0, i)` against a tracker holding `(g+1, i)`, **never match at any
  generation ≥ 1**, and so silently stop working after the first reclaim. The width being right is
  exactly what makes it look safe. This caught a peer lane mid-review, on this wire, an hour after
  they had written the opposite-widths rule themselves.
* **A payload whose generation already reads `g+1` on the write that means DEAD.** The reclaim pop
  stores `{… generation = g+1; free = 1'd0}` while `valid` is still 0 — and *that write is the
  invalidation broadcast*. So an unconditional compose would put `(g+1, i)` on the **invalidation**
  path, and a 30-bit compare there would fail to clear a tracker holding the stale `(g, i)` — the one
  tracker you most want cleared. The compose must therefore be **gated on the `valid` bit**, the same
  bit that already decides whether to fire at all.

**A note on the enumeration, because the tempting version of it is wrong.** It is natural to justify
the mask by counting broadcast call sites — but the count is easy to get wrong (13 syntactic, 16 after
inlining, and an earlier version of this folder said 14). The property worth relying on is structural
instead: there is exactly **one** `send mem_ch.write_req` in the whole rev-node unit, and it masks the
id, so **every** broadcast is a bare index by construction rather than by enumeration.

### Related instances, none of which is this defect

- `commit_stage.sv:239` — the same optimistic re-adopt for the **PC** capability.
- `pmp_data_if.sv:82-102` — the same shape per CPMP entry; a real latent defect, already on file as
  ISSUES.md:4079 under R-12 A5, but **not** what was measured here.
- Found while auditing, both worth their own items. **Both were stated too loosely in the first
  version of this paragraph and are corrected here (2026-09-22); the second was simply wrong.**
  - `pmp_data_if.sv:88-101` runs the adopt **first** and the invalidate **last**, comparing the
    **pre-adopt `_q`**. That makes the same-cycle race fail in *opposite directions* depending on
    which id the broadcast names, which is why one sentence could not describe it:
    - broadcast names the **newly adopted** id: `_q` still holds the old id, no match, validity is
      **not** cleared -> a live-looking entry for an id that was just invalidated (**false ALLOW**,
      the CPMP instance of finding 4);
    - broadcast names the **displaced** id: `_q` matches, validity **is** cleared -- but the entry
      now tracks the *new* id, which is left marked invalid with no further broadcast for it ever
      arriving (**false DENY**, permanent; finding 5).
  - **The "reinstall into a fault loop" claim is RETRACTED.** `swap_cpmp` skips a region that is
    already loaded -- `if(region_cpmp[region_id] != -1) continue;`
    (`sbi_capstone.c:1911`) -- so a second fault for the same region does **not** reinstall. It falls
    through to `region_id >= region_n` and takes the one-shot terminal path: `print_regions()`,
    `CAPSTONE_TAG_CPMX`/`CAPSTONE_NO_CPMP_REGION` over UART, then `fault_return_from_domain`, which does
    not return (`:1925-1941`). So the symptom is **a single domain kill with a diagnostic report**,
    not a loop.
  - And **re-adoption does happen**, so "can never re-adopt" is too strong. The round-robin eject
    sets `region_cpmp[ejected_region_id] = -1` (`:1958`), so a later install lands in an entry whose
    tracked id differs and the adopt fires normally. The narrow true statement is: *an invalidated
    entry cannot re-adopt **the same** capability into **the same** entry with **no intervening
    occupant**.*
  - Recorded because this paragraph was cited as independent corroboration for a scope decision
    while being itself unverified -- a circular citation. Verify a citation in its own tree.

### REPRODUCED IN SIMULATION, 2026-09-22 — two-sided, by the RTL lane

**Produced by the RTL lane; VERIFIED here against the transcript they left in `results/`, reading
`sim-rotate-stale.result-lines.txt` line by line rather than taking the summary.** On
`054cea69b` RTL at `S12_MEM_DELAY=12`, every pre-registered value matched:

| arm | rotation between revoke and access | reading |
|---|---|---|
| **A** — positive control | none | value 0, **cause 25 — the check FIRES** |
| **B** — the defect | an access through a different revnode intervenes | sentinel returned, **cause 0** |
| **C** — stale store | as B | **cause 0, the store LANDS** |

Checked against the transcript rather than the summary: `SPLIT` yields genuinely distinct ids
(`Reg[10]` revnode **2**, `Reg[16]` revnode **3**, and the unit logs `mrev_req on parent 2` then
`parent 3`, so the rotation is real); Arm A's cause reads `Reg[20] = 0x19` = **25**; Arm B returns
`0x00a5a5a5a5a5a5a5`, sentinel **A** — the revoked region's own data — through a reference whose node
`LCC` has just reported invalid; Arm C reads back `0xa5`. Sources and transcript are in `src/` and
`results/`.

Arms A and B differ in **exactly one thing**: whether an access carrying a different revnode id
intervenes. That is the matched pair this folder's earlier attempts could not build.

The revoke is **witnessed in-run** rather than assumed: `LCC` reads alias A as valid `1` before the
revoke and `0` after, while its sibling B stays `1` — the sibling surviving is the depth arithmetic
confirming itself, and it independently confirms that `REVOKE` on an `MREV` handle does invalidate the
original capability's own node.

**Why this is stronger than the board evidence above.** Simulation carries no timing artefact, which is
precisely what the bitstream caveat at the end of this folder says the board measurements cannot shed.
The board shows the defect on silicon; the simulation shows it on a clean instrument with a control
that fires.

The rotation came from `SPLIT`, not two `CAPCREATE`s — `CAPCREATE` hardcodes `revnode_id = 2`
(`capstone_flu_unit.anvil:385` @ `054cea69b`), so two created regions share one node and cannot
displace each other. That is the same trap recorded below.

### Directed reproducer

`capstone-ariane/verif/tests/custom/capstone/r35-stale-deref.S` (`board/r35-directed-repro`) does **not**
yet reproduce this, and its header records exactly why: it never executes CAPENTER, so `capmode_i` is 0
and the block is inert (exit 16), and `CAPCREATE` hardcodes `revnode_id = 2`
(`capstone_flu_unit.anvil:385` @ `054cea69b`), so its two regions shared one revnode and neither could displace the
other. A working version needs CAPENTER, two genuinely distinct revnodes via SPLIT, and must sit on the
**m1-reclaimer** line — `054cea69b` is not an ancestor of that branch.

## Reproduce

    # the probe image: off-by-default guards, so no measured image moves
    OUT_DIR=<dir> DOMAIN_BASE_VA=0x410000 \
      R1_EXTRA_DEFS="-DM1_MAXRET=43296 -DM1_STALE_DEREF=1 -DM1_STALE_MINT=0 -DM1_STALE_WRITE=1" \
      bash capstone/sublet/r1/build-r1-silicon.sh        # -> 35fb3fec3196841b

    # emulator: must halt at cause 24, offset 0x4354  (the control that proves the probe fires)
    R1_DOM=<dir>/r1_slots_pools.dom R1_HOST=<sqlite_host_rr.user> OUT=<out> \
      bash capstone/sublet/r1/run-r1-qemu.sh \
      "--arm pressure --series m1 --pattern shared --cap 64 --budget 60000 --stale-take" 2097152

    # board: one invocation, drivers/lists/m1-staletake.txt, via board-r1e4.sh

`M1_STALE_MINT=0` is what makes the domain **return** instead of wedging; with the mint present the
fault destroys the output buffer and the only evidence is a latched `mepc`.

## What this is NOT evidence of

The bitstream is an integration branch whose timing does not close (WNS −8.307, 90,379 failing
endpoints). Nothing here separates a design property from an artefact of this build, and no
measurement in this folder was taken on a timing-clean image.
