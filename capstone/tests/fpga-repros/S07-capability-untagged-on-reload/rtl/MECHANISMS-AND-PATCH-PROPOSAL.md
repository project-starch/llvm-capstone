# S-07: candidate mechanisms, and what we propose changing

Written for whoever owns the RTL. Everything here is quoted from the tree; where we could not settle
something we say UNRESOLVED rather than guess. **We are not asking you to accept any of this — we are
asking you to check the two we could not, and we say which experiment kills each one.**

Read `../00-README.md` first for the symptom and the measurements. The short version: a capability
read back from memory is NOT_CAP, so the next consumer raises mcause 25. It reproduced at roughly
3 wedges in 12 executions during one window on 2026-08-14 and **has not reproduced since**, across
18 further executions including a boot of the byte-identical firmware. Reproduction status is in
`../00-README.md`; please do not plan a long experiment assuming it fires on demand.

---

## The constraint every candidate must satisfy

`cap_metadata_t` is `{revnode_id, perm, cap_type, bounds}` (`core/include/ariane_pkg.sv:632-637`)
and `NOT_CAP = 3'b000` (`:650`). **An all-zero metadata word decodes to NOT_CAP.** So any mechanism
that zeroes or fails to deliver metadata produces exactly this symptom, and mechanisms that corrupt
metadata to a *random* value mostly would not (they would give wrong bounds or a wrong type, not
cleanly NOT_CAP).

Cause codes, `core/include/riscv_pkg.sv:349-353`: `UNEXPECTED_OPERAND_TYPE = 25`,
`INVALID_CAPABLITY = 26`, `CAPABLITY_OUT_OF_BOUND = 29`. We see **25**, which excludes the whole
revocation-validity family arithmetically.

**The guard that fires has two arms** — `core/anvil_build/capstone_flu_unit.anvil:29-31`:

```
func CINCOFFSET(data){
    if((data.cap_rs1.metadata.cap_type==cap_type_t::NOT_CAP)||(data.cap_rs2.metadata.cap_type!=cap_type_t::NOT_CAP)){
        call raise_exception(data.trans_id,ex_code::UNEXPECTED_OPERAND)
```

so mcause 25 at a `cincoffset` means **rs1 lost its tag OR rs2 gained one**, and our measurements do
not separate those. Only the third observed instance (which faults *at* an `ldc`, whose guard is
rs1-only, `capstone_dyn_unit.anvil:327-330`) is unambiguous. Everything below is organised around
that split: **A-family = rs1 lost its tag. B-family = rs2 gained one.**

---

## A-1. The capability load syncer tracks ONE outstanding request (A-family)

**Status: the strongest structural concern we found. NOT confirmed.**

**Correction to something you may hear from elsewhere: the transaction ID is NOT under-width.**
`core/anvil_build/capstone_dyn_unit.anvil:550-551`:

```
reg cap_trans_id : logic[3];
reg req_set : logic[1];
```

3 bits is exactly right — `build_config_pkg.sv:81` sets
`TRANS_ID_BITS = $clog2(NrScoreboardEntries)` and this core has
`CVA6ConfigNrScoreboardEntries = 8` (`core/include/capstone_cv64a6_imafdc_sv39_config_pkg.sv:60`),
so 3 bits uniquely names a scoreboard entry. **There is no ID aliasing from a narrow field, and we
withdraw that framing.**

The real concern is **depth**. The tracker is a single register plus a single valid bit, and every
arming path overwrites it unconditionally (`:564`, `:573`, `:577` all `set cap_trans_id := trans_id`).
So if a second capability access is issued while one is still pending, the first is displaced. The
matcher then discriminates purely on that one id (`:583`):

```
if(*req_set == 1'd1 && msg.trans_id == *cap_trans_id){ ... }
else { send lsu_ep.normal_res(msg) >> cycle 1 }        // :594 -- the bypass
```

**Why that could produce this symptom.** The bypass at `:594` routes the response to `LOAD_WB`
instead of `CAP_WB`. `LOAD_WB` carries no capability — `core/scoreboard.sv:320-324` ties
`wb[1..3].cap_data` to `'0` — and the scoreboard then erases the entry's capability result
(`:242-246`). Commit writes metadata `'0` (`core/commit_stage.sv:279`) into the metadata regfile
under the **plain GPR write enable** (`core/issue_read_operands.sv:1578` `we_pack[i] = we_gpr_i[i]`).
Net effect: the `ldc` retires with a correct cursor and **NOT_CAP metadata**, having never touched
memory.

There is a second, worse variant. `check_load_data` (`core/anvil_build/capstone_unit.anvilh:583-609`)
decides from the loaded capability's type whether the LDC **zeroes its source memory slot** (the
linear-clear), and the clear itself fires in `core/load_unit.sv:214-217`. We checked: the type sets
in those two places **match**, so ordinary list drift is excluded. But if a displaced transaction fed
the wrong `cap_type` into that decision, a clear would fire for a capability that must never be
cleared. The capability at our faulting site is reloaded every loop iteration and is therefore
NONLIN — *not* in the clear set — so a spurious clear is the only way its backing memory becomes
zero.

**Fits:** produces mcause 25 and only 25; sporadic, since it needs two capability accesses in flight
at once; leaves memory-oriented rungs clean in the first variant (nothing in memory is corrupted),
which is exactly what we observe — four rungs return `0xFFFF` with firing controls.

**UNRESOLVED, and this is the one thing we most want checked:** does `loop dyn_ep.flush`
(`capstone_dyn_unit.anvil:554`) reset `req_set` / `cap_trans_id` on a pipeline flush, and can a
response actually arrive between the arming `send cap_load_ri.init` (`:326`) and the matcher? We
could not settle either from the `.anvil` source; the generated
`core/capstone_dyn_unit.anvil.sv` would settle it.

**What kills it:** a simulation assertion that **no `LOAD_WB` writeback ever carries an `ldc`'s
`trans_id`**, with a negative control that forces the bypass and makes the assertion fail. If that
assertion cannot be made to fire in a directed test with several capability accesses in flight, A-1
is dead and we stop pointing at it.

### Proposed change, if A-1 is confirmed

Smallest correct fix, in preference order:

1. **Make displacement impossible rather than silent.** Replace the single `cap_trans_id`/`req_set`
   pair with an 8-entry valid vector indexed by `trans_id` (`logic [7:0] cap_req_pending`), set on
   arm and cleared on match. Same storage order as the scoreboard, no new ordering assumptions, and
   it removes the concept of "the pending request" entirely.
2. If a one-deep tracker must stay for timing reasons, **stall the second capability access** rather
   than overwrite: refuse to arm while `req_set` is high, so the issue stage backpressures. Slower,
   but no silent loss.
3. **Do not** simply route unmatched responses somewhere else — the bug is the displacement, not the
   bypass; the bypass is what makes the displacement invisible.

**Acceptance criterion, and it must FAIL before the fix:** a directed test that issues N capability
loads with overlapping lifetimes and asserts every one retires through `CAP_WB` with non-zero
metadata. Please confirm it fails on the current RTL before believing it passes on the fixed RTL.

---

## A-2. Shadow-tag DRAM round trip (A-family)

**Status: plausible, unquantified, and it has a documented unfixed sibling.**

Tags on a cache miss are not in the data path; they live in a separate DRAM region driven by an FSM
(`core/cache_subsystem/wt_axi_adapter.sv:109-114`, `TAG_IDLE/TAG_WAIT/TAG_WR/TAG_RD`). That FSM
already carries a known, deliberately unfixed ordering defect for atomics (`:143-152`): ATOMIC_REQ is
excluded from `needs_tag` because including it wedged the core, leaving a real tag-resurrection path.
That proves the *class* — a genuine refill reading the wrong tag state — is live in this block.

We could not confirm or refute an analogous window for a plain store→load. The return path already
needed one mis-association fix (`:129-132` notes `tag_addr_q` may be overwritten before the R-beat
returns), so a residual one is not far-fetched.

**Relevant to our workload:** the domain contains **zero** `amo*`/`lr.*`/`sc.*` (measured by
disassembling all 327 860 instructions), so the *known* AMO defect is not our fault. Any proposal
here must explain a non-atomic path.

**What kills it, board-free:** `verif/tests/custom/capstone/cap-tag-cache.S` already has the
evict-and-refill scenario, but inserts quiescing `nop`s before the eviction loop (`:97-98`), so it
tests the functional path and not the race. A variant with those NOPs removed, started immediately
after the `STC`, is the cheapest stress. **We have not run it** — it needs the Verilator model and we
did not want to disturb an in-flight build in that submodule.

**No patch proposed.** We have no evidence that would tell you what to change, and guessing at an FSM
with a documented wedge-on-modification history would be irresponsible.

---

## B-1. Capability metadata forwarding (B-family) — the half our measurements cannot see

**Status: UNRESOLVED and untested. This is the other half of the symptom.**

`core/issue_read_operands.sv:694` and `:769` build a wide, register-index-matched select network over
`NrWbPorts` plus scoreboard-depth sources feeding `rs1_cap_metadata_res` / `rs2_cap_metadata_res`
(`:923-952`). This is the natural place for a plain integer to spuriously *acquire* capability
metadata — which is the rs2 arm of the `CINCOFFSET` guard, and would raise mcause 25 with the
capability perfectly intact.

Note the asymmetry that makes this attractive: the metadata regfile's main write port is enabled by
the **plain GPR** write enable (`:1578`), so every integer writeback writes metadata too. It normally
writes `'0`; a path that let it write a stale non-zero value would tag an integer.

**We have a ready discriminator and could not run it** — see `board/G6P-DISCRIMINATOR.md`. It is a
4-byte binary patch that removes the rs2 arm at the faulting instruction. One wedge on the patched
build proves A-family; a long clean run proves B-family. It is built, verified, and waiting for the
defect to reproduce.

**Proposed change: none yet.** Confirm the direction first — patching a forwarding network on
suspicion is how correctness bugs get introduced.

---

## Ruled out — please do not spend time re-deriving these

| ruled out | why |
|---|---|
| the whole revocation-validity family | those raise mcause **26**, we see **25** — arithmetic, no experiment needed |
| the AMO tag-resurrection path | the domain contains **zero** atomics (disassembled, 327 860 instructions) |
| anything needing hardware multiply/divide | **zero** `mul`/`div`/`rem`; soft routines only |
| DRAM soft error / refresh bit-flip | off by orders of magnitude from a ~23% per-execution rate |
| state persisting across a power cycle | `cap_tag_q` (`wt_dcache_mem.sv:408-418`), the tag FSM (`wt_axi_adapter.sv:900-919`) and the rev-node pool (`capstone_rev_node.anvil:179`) all reset on `rst_ni` |
| marginal timing / thermal | the core runs at **25 MHz** (`corev_apu/fpga/xilinx/xlnx_clk_gen/tcl/run.tcl:18,32`; the `// 50 MHz` comment at `ariane_xilinx.sv:1208` is stale). A hot-fails path also predicts the opposite of what we saw. **UNRESOLVED**: no post-route timing report for this bitstream is in the tree, so routed slack on the wide capability compares is unverified |
| rev-node pool exhaustion | pool holds 65536; heads at the wedges were ~250-600 |
| the write-buffer `.user` clobber | needs a coalescing plain store to the **same word**; at the faulting site the scalar is a different word and a different 16-byte granule (`wt_dcache_mem.sv:276`, `wt_dcache_wbuffer.sv:444`) |
| an R-20 analogue on another register | `f623c48a1` is an ancestor of every candidate synthesis tree and was never reverted; the only register-literal special cases in the core are CAPENTER/x10-x11 (`issue_read_operands.sv:573`, `scoreboard.sv:236-238`, `decoder.sv:1287`) and we fault on x11/x12 |

---

## If you can run only one thing

Run the **A-1 assertion** in simulation: no `LOAD_WB` writeback may ever carry an `ldc`'s
`trans_id`, negative-tested so you know it can fail. It is board-free, it needs no reproduction of
the sporadic fault, and it either kills our leading candidate or hands you the defect.
