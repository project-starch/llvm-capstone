# R-35's root cause is the LSU's single core-wide revnode tracker, optimistically re-adopting — and my own CPMP correction is retracted

**2026-09-21.** This file was published earlier today claiming R-35's root cause was CPMP's per-entry
revnode tracker. **That is wrong and is retracted below.** The original attribution —
`load_store_unit.sv` — was right, and my retraction of it was the error. An adversarial audit caught
it; every load-bearing point below was then re-verified against the board RTL directly.

The filename is now wrong too. It is kept rather than renamed so the earlier commit's link still
resolves.

## The answer

On this silicon the harness's plain loads and stores are checked by `cap_violation_detection` in
`core/load_store_unit.sv` (board RTL `054cea69b`, `:948-996`). Its revocation term depends on a
**single core-wide tracked revnode id**, and an access presenting a different id re-adopts itself as
valid (`:966-971`):

```systemverilog
// update revnode tracking when a new instruction arrives with a different revnode
if (lsu_cap_type != NOT_CAP
    && lsu_cap_a.metadata.revnode_id != lsu_revnode_id_d) begin
  lsu_revnode_id_d    = lsu_cap_a.metadata.revnode_id;
  lsu_revnode_valid_d = 1'b1;        // <-- an UNTRACKED revnode is assumed VALID
end
```

With the M1 harness's 16 rotating slots, essentially every access presents a different id, so
`lsu_revnode_valid_d` is 1 whenever the check is reached and the `!lsu_revnode_valid_d` clause
(cause 25) can never fire. **Bounds (28) and permissions (27) survive because they are read from the
capability's own metadata**, which is exactly the folder's observed scope: the same path enforces
bounds and not tags.

Worse than the CPMP block in two specific ways: the invalidation broadcast is applied at `:945-946`,
*before* the adopt at `:966-971`, so an adopt always wins its cycle; and there is **one** tracked id
for the whole core, not sixteen.

## RETRACTED: "the root cause is CPMP (pmp_data_if.sv:82-102)"

Refuted on four independent grounds, each re-verified here against `054cea69b` rather than taken on
report:

1. **The two blocks are gated on complementary values of the same signal.** `cap_violation_detection`
   requires `ld_st_priv_lvl_i == riscv::PRIV_LVL_M` (`load_store_unit.sv:949`); the CPMP data check
   requires `ld_st_priv_lvl_i != riscv::PRIV_LVL_M` (`pmp_data_if.sv:293`). Exactly one runs.
2. **CPMP cannot emit the cause the board returned.** Its only exception is
   `ST_ACCESS_FAULT`/`LD_ACCESS_FAULT` — 7 and 5 (`pmp_data_if.sv:297-298`). The board's bounds probe
   returned **28**: `results/board-bounds-probe.wedge.txt`, `sw=255 TRAP LOG {seen,mcause[6:0]} 0x9c`
   → `0x9c = 1001_1100` → seen=1, `mcause[6:0] = 28`. Latched in hardware
   (`cva6.sv:1116-1124`, `recent_nontrivial_mcause_log_q <= ex_commit.cause`), not derived by the
   monitor. Cause 28 therefore proves the M-gated block evaluated.
3. **The harness never installs a CPMP entry.** `grep -ci 'cpmp\|ccsr' sublet/r1/r1_slots_pools.c`
   returns **0**. CPMP entries are written only by the monitor, via the domain-switch register restore
   (`csr_regfile.sv:1931`) and the `CCSR_CPMP0..15` writes (`:2418-2559`). CPMP holds whole-region
   monitor capabilities, so it could never fault at leaf+64 — and the board's `tval` is `0xac100040`,
   the byte past the 64-byte leaf. Only a check reading `rs1`'s own metadata can produce that.
4. **The premise "domains run in S-mode" is false**, and that is the root of my error — see below.

## The prior-art note that misled me is itself wrong on its central premise

`docs/history/15-09-2026_lsu-capmode-gate-why-domains-cannot-satisfy-it.md` concludes that a domain
cannot satisfy the gate's privilege half because domains run in S-mode. **Entering a domain does not
change privilege at all.** Verified twice in the board RTL:

- `priv_lvl_d` has exactly six writers in `csr_regfile.sv` — `:1048` (hold), `:2155` (trap entry),
  `:2318` (MRET), `:2341` (SRET), `:2362` (VS-RET), `:2376` (DRET). **None is on a capability or
  domain-switch path.**
- `core/anvil_build/capstone_dom_switcher.anvil` contains **zero** occurrences of `mstatus`, `priv` or
  `mpp`.

So a domain entered from M-mode monitor code runs at M-mode, the gate is satisfied, and the block is
live — which is what cause 28 independently shows. That note's derivation rests on `sbi_capstone.S`'s
`mret`s, which are labelled `call_into_smode` / `resume_smode`: the **S-mode host**, not a domain. A
domain is entered by `__domcallsaves` from monitor code.

**This is owed to the RTL lane**, since it changes that note's conclusion and the note is committed.
I have not edited their note; this is the correction, and it needs their sign-off.

**Still unresolved, flagged rather than assumed:** whether *every* domain access is at M, or only the
class measured. It is established for the accesses that produced cause 28 — same instruction class,
same alias array, same image family as the stale probe — and not beyond that.

## Also corrected: the folder's "untagged reference" wording

The tag *does* reach this check. `cap_rmetadata` → `operand_a_cap_regfile` (65-bit,
`issue_read_operands.sv:206-208`) → `cap_metadata_a` (`cva6.sv:249`, "S-06 fix: {tag, metadata}") →
`lsu_ctrl.cap_metadata_a` → `decompress_cap_tagged` (`ariane_pkg.sv:762-781`), which returns `NOT_CAP`
when the tag bit is 0. An untagged `rs1` would have raised **cause 24** at `:973`. It did not — so the
stale alias was **still tagged** on silicon, and the defect is the revocation clause, not the tag.
QEMU's "x[rs1] is not a capability" is a model divergence and is its own item.

## The fix is architectural, and that difficulty is the real one

The LSU's only revnode ports are the two broadcast inputs (`load_store_unit.sv:198-199`): **there is no
query path**, so on a miss the block cannot ask and can only guess. Flipping `1'b1` to `1'b0` fails
closed and would raise on every rotation, i.e. constantly. A real fix needs either a lookup into the
revocation node table — `capstone_rev_node.anvil`'s `IDLE_STAGE` already serves `ep.query_req` and
answers `ep.query_res(node_in.valid)`, but the LSU has no port to it and adding one puts a stall on the
access path in the common case — or a wider tracker with a miss path.

**CLAUDE.md's rule about adding a signal into a cone that already carries a combinational loop applies
directly, and only synthesis proves synthesizability.** No RTL is changed by this note; the fix and any
respin are the RTL lane's and the lead's calls.

`commit_stage.sv:239` carries the same optimistic-adopt shape for the PC capability and should be
fixed with it. `pmp_data_if.sv:82-102` carries it too and is a genuine latent defect — it is simply not
what the board measured, and ISSUES.md:4079 already records it under R-12 A5.

## Two further defects found while auditing, neither of them R-35

- **A same-cycle race in CPMP.** `pmp_data_if.sv:95-101` compares the broadcast against
  `cpmp_tracked_revnode_id_q` — the *old* tracked id — so a broadcast arriving in the same cycle as an
  adopt cannot clear the freshly adopted id. `commit_stage.sv:239-244` documents that it alone compares
  against `_d`.
- **A CPMP re-install livelock, fail-safe but not benign.** Once an entry is invalidated, re-installing
  the *same* capability never re-adopts (the `!=` guard fails), so the access faults, `swap_cpmp`
  reinstalls the identical region, and it faults again.

The 16-bit invalidation compare against a 30-bit adopt guard over-invalidates rather than
under-invalidates, so it is fail-safe; everything currently runs at generation 0 (ISSUES.md:3887).
Robustness item, not a safety gap.

## What the directed reproducer established

`capstone-ariane/verif/tests/custom/capstone/r35-stale-deref.S` on `board/r35-directed-repro` did not
reproduce R-35, and its own header records why. Measured, not assumed: it exits 16 because it never
executes CAPENTER, so `capmode_i` is 0 and the block is inert; a 64-nop barrier after `REVOKE` did not
change the reading, ruling out walk timing; and `CAPCREATE` hardcodes `revnode_id = 2`
(`capstone_flu_unit.anvil:385` @ `054cea69b` (`:337` on the repro branch)), so both its regions shared one revnode and neither could displace the
other's tracker entry.

**With the privilege question settled the test is now fixable rather than misconceived** — it needs
CAPENTER (capmode is sticky thereafter) and two genuinely distinct revnodes, which means SPLIT rather
than two CAPCREATEs. It must also move to the **m1-reclaimer** line: `054cea69b` is not an ancestor of
`board/r35-directed-repro`, and `load_store_unit.sv` differs by 22 lines and `capstone_rev_node.anvil`
by 278 between them.

`MREV` semantics, confirmed against the simulation log rather than assumed: it inserts the new node
immediately before its parent in DFS order and pushes the parent one level deeper, so `REVOKE(handle)`
does reach the parent capability.

## Process note, since this is the second retraction on one issue in one session

Both errors have the same shape: I trusted a written conclusion over the RTL. The first time it was my
own reading of the wrong branch; the second time it was a committed note whose premise was wrong. The
check that caught both was the same one — read the gate, then ask which side of it the measured cause
could have come from. **Cause 28 was in the folder from the start and settles the module by itself.**
