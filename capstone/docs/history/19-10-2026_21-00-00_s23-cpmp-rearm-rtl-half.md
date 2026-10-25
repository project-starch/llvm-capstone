# S-23's data face, the RTL half: what re-arms a CPMP validity bit, and what cannot

RTL lane, 2026-10-19 evening. This note is the RTL-side trail for S-23's data face: the source facts, the arm, its
readings, and three corrections to things this lane said earlier the same day. The semantics owner's rulings and
the monitor's revision are recorded in the vp2 corpus plan; this is the half that lives in the hardware.

Arm: `capstone-ariane`, branch `virtual-profile-2-c`, `s23-cpmp-rearm.S` — pre-registered at `b860d4e2a`, run,
corrected and re-run at `d6d7f48c0`. Its independent twin is `r12-recl-cpmp.S`, which until today was registered
in no testlist and had never been run in either of this lane's trees.

## The source facts, enumerated rather than inferred

**Exactly two setters of the per-entry validity bit**, in `core/pmp/src/pmp_data_if.sv`:

- the ADOPT, `:88-93`, whose third term requires the entry's revnode id to **differ** from the tracked one;
- the NAMESPACE FLUSH, `:109-114`.

One clearer, `:96-108`, a broadcast whose 16-bit index matches. Reset leaves all sixteen INVALID, `:120`. Every
assignment of 1 to `cpmp_revnode_valid_d` was enumerated, not sampled.

**The six allow terms**, `:141-151`: tag, `cap_type != NOT_CAP`, `paddr >= bound_start`, `paddr < bound_end`,
`cpmp_perm_ok[i]`, and the validity bit. `cpmp_perm_ok` selects the bit by access type — `perm[1]` for a store,
`perm[2]` for a load (`:138-139`), `perm[0]` for a fetch (`:171`, `:182`). Nothing else filters, so **every
capability type except NOT_CAP passes**, CAP_TYPE_REVOKE included: a REV-typed capability loaded in a CPMP entry
grants S-mode access subject only to bounds, permission and validity.

**MREV changes the type and nothing else.** `capstone_dyn_unit.anvil:114-141`: the handle is
`modify_cap_type(rs1, CAP_TYPE_REVOKE)`, so bounds, permissions, cursor and revnode id are all carried over.
Taken with the line above, a region's MREV handle left in a CPMP slot still grants whatever the region granted.

**So an id-preserving rewrite cannot re-arm a cleared bit.** The monitor's fetch-retry handler rewrites every slot
with the same capability and therefore the same id, so the adopt's third term is false and the bit stays clear.
The handler is not the trigger. It is also not what hands execution back to the DATA access that panics, which is
how this note first put it: `fetch_retry_covered` returns before its sweep on anything that is not a fetch fault
(`sbi_capstone.c:3505-3506`), so on a cause-5 refusal the sweep does not run at all. What is true is narrower --
the sweep ran in the same STEP, on the fetch fault it retried -- and it has a consequence for the workaround:
a strobe appended to that function is unreachable from the data path and would have to sit in `swap_cpmp`'s
no-region path instead.

**The strobe is the only software-reachable setter, and it is M-mode.** `csr_regfile.sv:2019` sets
`ns_flush_strobe_d = csr_wdata[2]` on a csupctl write, registered at `:3380` and OR'd into `revnode_ns_flush_o`
(`ex_stage.sv:1420`) with no gating. `CSUPCTL` is `12'h7C4`, so bits [9:8] = M and an S write traps
(`:2929-2932`). csupctl bit 2 **reads back as zero** (`:1077`) and `ns_flush_strobe_d` defaults to 0 every cycle
(`:1183`), so `csrsi 0x7C4, 4` is a one-cycle strobe that preserves `csupctl_resume`, `csupctl_virt` and
`csupctl_table`. The S-reachable alternative is an srevroot change, whose edge detector samples the live root only
while the rev-node unit is idle (`ex_stage.sv:1403-1420`, gated on `rev_busy`), and whose failure to fire is
invisible to software because a read of srevroot returns the CSR, not the held copy (`:1078`).

## The readings

Run `s23c` on `bf3db3175`: harness SUCCESS, 15,824 cycles, 806 retired, 6 denials, 21 readings, every
pre-registered value holding.

| probe | read | |
|---|---|---|
| Q1 slot 0 installed | ALLOWED | instrument control |
| Q2 the handler's sweep over all sixteen slots, live entry | ALLOWED | the handler alone is harmless |
| Q3 REVOKE the handle above the child | DENIED | the broadcast clears the bit |
| Q4 the handler's sweep, cleared entry | DENIED | an id-preserving rewrite cannot re-arm |
| Q5 `csrsi csupctl, 4` | ALLOWED | the strobe re-arms it |
| Q6 a broadcast naming a different index | ALLOWED | the clear is index-scoped |
| Q7 the matching index | DENIED | |
| Q8 srevroot changed and restored, spaced | ALLOWED | the S-mode path works |
| Q9 a fresh clear | DENIED | |
| Q10 the same, back to back | ALLOWED | adjacency alone does not collapse the sample |
| q11a–q11d two slots installed then both revoked | ALLOWED, ALLOWED, DENIED, DENIED | |
| q12a, q12b after ONE strobe | ALLOWED, ALLOWED | **the strobe's reach is global** |

The denial surfaces as **cause 5**, a load access fault, not as a capability refusal. The arm pre-registered
"expect 24" for that reading and this note first called the 5 a point of contact with the field panics, which is
recasting a missed prediction as a finding: cause 5 is what ANY data access fault reports. What carries the
field's information is that its badaddr was PHYSICAL, which only the CPMP data check produces, and this arm
cannot test that -- it runs bare with no translation and its handler reads `mcause` only, never `mtval`. `r12-recl-cpmp` read exactly its seven pre-registered deltas, 0 1 1 0 1 0 1 with the total 4,
including its P3, which is Q4 by another route.

## Three corrections this lane made the same day

1. **Every deny in the arm is a TRUE deny, Q5's included**, and it was reported as the workaround working. Each
   clear is a REVOKE of the handle above that slot's own child, so the capability left in the slot is dead; the
   bit Q5 re-armed was protecting revoked memory. Q5 and "a true denial masked" are the same reading. The arm
   contains no false deny, so it cannot grade the workaround's safety in either direction. A false deny needs two
   LIVE capabilities sharing an index, which cannot happen inside one namespace and can happen across them,
   because under the virtual flag ids carry no generation.
2. **Q8 and Q10's first readings were void by construction.** The arm saved the live srevroot to restore it and
   wrote `x0` as the intermediate, but the live root was 0, so both writes wrote 0 over 0 and no edge could have
   been produced. Fixed with a nonzero aligned intermediate; the re-run reads ALLOWED for both.
3. **The `make_hole` live-path finding was wrong, and the board lane corrected it.** This lane classified
   `make_hole`'s body, which never writes the slot, and concluded the slot stays loaded. But `C_READ_CCSR` is
   `ccsrrw(rd, cpmp_n, x0)`, so every `read_cpmp` swaps x0 into the slot, and both `make_hole` call paths read the
   slot first — it is already empty when the tables are cleared. The slip is specific: this lane had already
   observed that `managed_unmap`'s `discard = read_cpmp(...)` empties the slot by reading it, and then failed to
   apply that to another function's callers. A body is not a path.

## What the flag's definition implies, which the arm does not test

`vctx_table_o` (`csr_regfile.sv:2885-2887`) is `(sup_active && (csupctl_virt || sup_runv)) || (!sup_active && S &&
srevroot != 0) || (!sup_active && M && csupctl_table)`. In `capstone_vm.c` the only writers of `0x5c1` are
`select_root` and `restore_root`, in four windows (`:420/423`, `:507/524`, `:602/610`, `:1081/1083`), **none of
which contains a CSRUNV** — the run path is `vm_ioctl`'s `run(t->frame_pa, action)` at `:985`, and the context's
table comes from `frame[1]` at `:983`. The monitor never writes `0x5c1` at all. So srevroot is 0 at S across every
entry and escape, the flag falls at each escape, and the edge-triggered flush fires there. Consequences:

- a cross-namespace clear created INSIDE a context self-heals at the escape;
- every escape therefore re-arms every tracker as assumed-live, including any stale entry, so the strobe's
  "assumed live" cost is already paid on the module's own path and adds no new class;
- a deny that persists must be created by a broadcast after the last flush and observed before the next, which
  fits the observed rarity better than "a cleared bit nothing re-arms".

Every `.insn r 0x5b` site in `capstone_vm.c` was enumerated against those windows (dev's content, last touched by
`2ff451359fdf`): all seven are inside a window except the CSRUNV itself, and no other file under
`capstone/runtime/virtual/` issues one. So the module runs no lifetime instruction at S in the physical namespace.
That enumeration is of dev's source, not of the board's module build, which is the gap it leaves.

## What an auditor took apart, the same evening

Run over this note's own claim before it could reach the registry. The simulation readings survived — it
reproduced every one by its own invocation and confirmed from the reflog that the pre-registration was pushed
before the run — and the FIELD attribution built on them did not:

- **"of the six allow terms only the validity bit is left" is REFUTED, and the error is mine.** This lane read the
  board's `CPHT:00000007` as a tag. It is a **slot index**: `capstone_report(CAPSTONE_TAG_CPHT, i)` for a loaded
  entry whose bounds cover badaddr (`sbi_capstone.c:3586`), where "loaded" means `cpmp_region[i] != -1`, the
  monitor's own bookkeeping. The dump prints CPID, CPBA, CPEN, CPPM and CPHT — **no tag, no cap_type, no revnode
  id** — so the field deny is localised to no single allow term.
- **A surviving alternative the strobe cannot repair:** `csr_regfile.sv:2100`, the switcher's RESTORE walk writes
  each entry's tag from the saved granule's MEMORY tag (`data[128]`). A restore with intact metadata and tag 0
  gives the captured signature exactly, with a dump that reads healthy, and both the adopt (`:89`) and the allow
  (`:144`) require the tag.
- **The field deny did not persist**, which this project's own records already said: `ISSUES.md:544-546` has the
  same PA read again through slot 7 by `die()`'s `spin_lock_irqsave` before the Oops, with no second refusal, and
  `S23-data-face/README.md:69-70` states this lane's candidate as "re-armed by the next ADOPT". A latched cleared
  bit does not fit that; a transient does.
- **A strictly narrower repair exists:** the adopt fires on any id CHANGE in a slot (`:91`), so two writes to that
  slot — any other capability, then the real one — re-arm that one entry and touch neither the other fifteen nor
  the LSU's revocation cache nor the PC tracker. r12's P4 is the evidence. The global flush was called "the
  repair" without that comparison.
- **Q5 lacks a control:** `csrsi 0x7C4, 4` sets the strobe AND `flush_o`, so it does not separate the re-arm from
  a pipeline flush clearing a stale deny. A `csrsi 0x7C4, 1` arm separates them and is owed.

So what this note supports is the scoped simulation result and nothing above it: in simulation on bf3db3175 an
id-preserving rewrite cannot re-arm a cleared validity bit, a csupctl bit-2 write does, and that re-arm is global.
The field mechanism is open, and the cheapest instrument for it is one monitor line printing slot 7's tag,
cap_type and revnode id at the refusal — not another arm.

## The walk's broadcast timing, answered on tree A 8522d72fe

Asked by the semantics owner against a candidate in which a retire's revocation walk is still broadcasting after
its CSRETIRE retires. It is not, and the answer relocates the candidate rather than killing it.

**The walk's broadcasts all precede the instruction's result.** `capstone_rev_node.anvil:46-52` sends
`ep.retire_res(2'd0)` AFTER `call send_revnode_update(...)` and sets `stage := 3'd1` with it, so the response marks
the walk's end, not its first step (the loop continues at `:86`). `send_revnode_update` (`:1-11`) is
fire-and-forget — `send mem_ch.write_req(msg_pack) >> cycle 1 >> set mem_write_flag := 1'b0`, with
`recv mem_ch.write_res` deliberately commented out — and the broadcast taps the write REQUEST, as that file's own
comment says (`ex_stage.sv:1615` in this tree). `capstone_dyn_unit.anvil:613-639` has CSRETIRE await `retire_res`
before its own result. So: every broadcast, then the response, then the result, then commit.

**The clear still happens, inside the select_root window.** Those broadcasts carry the CONTEXT's indices, and the
invalidation compare has no namespace term — `revnode_invalidation_id_i[15:0] == cpmp_tracked_revnode_id_d[i][15:0]`
— so a context index equal to a physical index held in a CPMP slot clears that slot there and then.

**Whether it persists turns on one thing: whether the rev-node unit was idle when the window opened.** The held
namespace copies update only on a `!rev_busy` cycle, so:

- unit IDLE at `select_root` -> the held copy sampled the context's root, `restore_root(0)` is a change, the flush
  fires there, and the deny is repaired before any later kernel access. This is why a re-read of the same address
  a few instructions later passes.
- unit BUSY across the whole window -> the held copy never sampled the context's root, `restore_root(0)` is NOT a
  change, no edge, no flush, and the deny persists.

**That second regime is the same phenomenon as the srevroot workaround's silent collapse** — a held copy skipping
an intermediate value because the unit was busy across both writes. One fact, not two: what makes the workaround
fail silently is what makes a deny persist.

**No software-visible drain exists.** `CSR_CSNODEFREE` reads `rev_node_head_i`/`rev_node_free_len_i`
combinationally (`csr_regfile.sv:1078`), registered outputs that cannot stall, and no CSR exposes `rev_busy`. What
does work is a dependency drain: any lifetime instruction needing a unit response cannot complete until the unit
services it. A CSRETIRE with an out-of-range id is not one — it returns from the dyn unit's early branch without
touching the unit (`capstone_dyn_unit.anvil:618-624`).

## Owed

The cross-namespace collision arm: a CSRUNV context whose lifetime table hands out an index equal to a physical
index held in a loaded CPMP entry, with the context reporting the index its allocator issued and the physical
capability probed live first, as a matched pair on srevroot at S across the escape — 0 predicting ALLOWED (the
escape's own flush repairing it, the field shape) and nonzero predicting DENIED (the false deny shown to exist).
The caller must be at S in both halves, since at M the flag's surviving term is `csupctl_table` and it falls
either way.
