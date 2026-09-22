# R-35 — on silicon a REVOKED capability still reads AND writes the storage its object has given up, at every age, and the access does not trap

**Status (2026-09-21): ROOT-CAUSED, in the board's own RTL, and the cause is NOT a missing tag check.**
REPRODUCED ON THE BOARD four times with controls; provenance SETTLED (the flashed bitstream carries the
R-34/R-24 fix, verified by a cause comparison on the board rather than by the build record). The defect
is in `core/load_store_unit.sv:966-971` of RTL `054cea69b`: the revocation check is real, but it depends
on a **single core-wide tracked revnode id**, and an access presenting a different id **re-adopts itself
as valid**. See "ROOT CAUSE" below. Bounds and permissions survive because they are read from the
capability's own metadata, which is why this path enforces bounds and not revocation.

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

### Related instances, none of which is this defect

- `commit_stage.sv:239` — the same optimistic re-adopt for the **PC** capability.
- `pmp_data_if.sv:82-102` — the same shape per CPMP entry; a real latent defect, already on file as
  ISSUES.md:4079 under R-12 A5, but **not** what was measured here.
- Found while auditing, both worth their own items: CPMP's invalidation compares against the *old*
  tracked id, so a broadcast arriving in the same cycle as an adopt cannot clear it; and an invalidated
  CPMP entry can never re-adopt the same capability, so `swap_cpmp` can reinstall it into a fault loop.

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
