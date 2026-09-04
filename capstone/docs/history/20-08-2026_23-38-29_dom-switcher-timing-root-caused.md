# The domain switcher's timing failure, root-caused — and two corrections to my own numbers

**Date:** 2026-08-20
**Artifacts:** `synth-39b21639d-exit0.tar.gz` (control), `synth-80843404c-exit0.tar.gz` (S-10).
Both carry `core/capstone_dom_switcher.anvil.sv` **byte-identical** to the worktree
(`sha256 82eeea1a4b1f5390da20c26eeffbf61d27644386bd34ae0c5cf558468f769fb6`), so no
stale-artifact problem: the file analysed below is the file that built both bitstreams.

## The headline, and it is a single register bit

Complete enumeration from each routed checkpoint —
`get_timing_paths -max_paths 1000000 -nworst 1 -slack_lesser_than 0` (one worst path per
failing endpoint, not a worst-N sample):

| build | WNS | failing endpoints | startpoints |
|---|---|---|---|
| `39b21639d` (control) | −10.629 | 96,727 | **96,727 = 100.0%** `dom_switcher/cur_idx_q_reg[1]` |
| `80843404c` (S-10) | −16.400 | 102,774 | **102,769 = 100.0%** `dom_switcher/cur_idx_q_reg[3]`; 5 DDR |

Where they land (S-10 / control):

```
i_tracer            65,442  63.7%   /  65,562  67.8%     debug-only instrumentation
issue_stage_i       21,728  21.1%   /  19,297  19.9%
ex_stage_i           8,529   8.3%   /   8,052   8.3%
cache_subsystem      3,931   3.8%   /   1,625   1.7%
csr_regfile_i          978   1.0%   /     499   0.5%
dom_switcher (own)     497           /     234            the mechanism itself
```

**One 7-bit FSM counter is the startpoint of essentially every failing path in the design.**

## Why — two structural facts, both quoted from source

**1. `cur_idx` is combinationally muxed onto the architectural register file's address port.**

```systemverilog
// core/issue_read_operands.sv:1537-1540
if (dom_switch_sel && dom_switch_reg_req_valid_i && !dom_switch_reg_req_i.is_set) begin
  raddr_pack[i*OPERANDS_PER_INSTR+0] = dom_switch_reg_id;      // = cur_idx - 25
end else begin
  raddr_pack[i*OPERANDS_PER_INSTR+0] = ... issue_instr_i[i].rs1[4:0];
end
```

Same on the write side (`:1566` `waddr_pack`, `:1609` `cap_waddr_pack`). The regfile read is
**asynchronous** — `core/ariane_regfile.sv:57`: `assign rdata_o[i] = mem[raddr_i[i]]` — and
`rdata[0]` is operand_a of the main pipeline. So `cur_idx` is not switcher-local state; it is an
address into the datapath.

**2. The register channel closes its round trip combinationally, in one cycle.** All three
responders, and the source says so itself:

```
core/issue_read_operands.sv:1526   dom_switch_reg_req_ack_o    = dom_switch_sel && dom_switch_reg_req_valid_i;
core/issue_read_operands.sv:1527   dom_switch_reg_resp_valid_o = dom_switch_reg_req_ack_o;  // same cycle response
core/csr_regfile.sv:389            // FIXME: a hack
core/csr_regfile.sv:395-396        dom_switch_reg_req_ack_o = dom_switch_req_en;
                                   dom_switch_reg_resp_valid_o = dom_switch_req_en;
core/frontend/frontend.sv:452-453  dom_switch_reg_req_ack_o = 1'b1;  dom_switch_reg_resp_valid_o = 1'b1;
core/cva6.sv:868-869               the three are OR-ed back into one bus
```

The **data** channel, by contrast, already has a registered response — `core/store_unit.sv:288`
gates `dom_switch_data_resp_valid_o` on `sel_dom_switch_last`, registered at `:558`. That is an
existence proof that the Anvil handshake tolerates response latency.

## Why a 2-cycle exception on `-from cur_idx_q_reg*` is unsound

The FSM consumes a newly written `cur_idx` exactly **one** cycle later:

```
core/capstone_dom_switcher.anvil.sv:618   EVENTS0[101] = [100]||[93]||[76]||[69]||[51]||[44]||[27]||[20]||[1]
                                  :617   _thread_0_event_counter_102_1_n = EVENTS0[101]
                                  :616   EVENTS0[102]  = _thread_0_event_counter_102_1_q
                                  :779   EVENTS0[0]    = _init_0 || EVENTS0[102]
                                  :775   EVENTS0[4]    = EVENTS0[3] && thread_0_wire$19   ($19 = f(cur_idx_q))
```

`cur_idx_q` is written by events 20/44/69/93, which are themselves terms of `EVENTS0[101]`.
Exactly one registered stage on every writer's return path.

The argument that admits no handshake escape: the FSM state registers clock
**unconditionally**, no clock enable —

```
core/capstone_dom_switcher.anvil.sv:1056   _thread_0_event_syncstate_5_q <= _thread_0_event_syncstate_5_n;
                                    :774   _thread_0_event_syncstate_5_n = (EVENTS0[4] || syncstate_5_q) && !_data_ch_req_ack;
```

Whatever `ack` does, a combinational function of the new `cur_idx` is captured at the edge
ending T+1. No value of `ack` makes that capture independent of `cur_idx`.

## TWO CORRECTIONS TO WHAT I PREVIOUSLY WROTE

**Correction 1 — the count in `fb228796e` is mislabelled and understated.** I wrote "129 on
`data_read_q_reg`, 25 on `_thread_0_event_counter_*`, 1 on the log register = 155". The
`_thread_0_event_counter_*` glob matches **9**, not 25; 25 was 9 counters + 16
`_thread_0_event_syncstate_*`. And the list omitted `cur_base_q_reg` (64) and `cur_idx_q_reg`'s
own self-loop (7), which are just as much single-cycle mechanism registers. Exact by-endpoint
counts, identical in both builds except `commit_req_q_reg`:

| endpoint group | control | S-10 |
|---|---|---|
| `data_read_q_reg[*]` | 129 | 129 |
| `_thread_0_event_syncstate_*_q_reg` | 16 | 16 |
| `_thread_0_event_counter_*_q_reg` | 9 | 9 |
| `cur_base_q_reg[*]` | 64 | 64 |
| `_thread_0_event_reg_*_q_reg[*]` | 8 | 8 |
| `cur_idx_q_reg[*]` | 7 | 7 |
| `req_en_q_reg` | 1 | 1 |
| `dom_switch_last_data_metadata_en_log_q_reg` | 1 | 1 |
| `dom_switch_pc_loaded_value_seen_log_q_reg` | 1 | 1 |
| `commit_req_q_reg[*]` | 0 | 263 |
| **total under `dom_switcher/`** | **234** | **497** |

**The honest figure is 234 / 497, not 155.** And the broad exception would not have masked
those alone — it applies to *every* failing path in the design, since all of them launch from
`cur_idx_q_reg`.

**Correction 2 — "the reg-channel round trip is the critical path" was an inference, and the
measured evidence points at the DATA channel.** Not one of the 500 worst paths in either build
touches the reg channel:

```
grep -c '_reg_ch_req_valid_selector'  worst_500_paths.rpt   ->  0 / 0
grep -c '_data_ch_req_valid_selector' worst_500_paths.rpt   ->  1500 / 1000
```

The one fully-detailed failing path available runs `cur_idx_q_reg[1]/Q -> _busy_ch_idx_0 ->
_thread_0_event_syncstate_78_q_i_4 -> _data_ch_req_valid_selector_q[0]_i_1 ->
dom_switch_data_req[write_en] -> lsu_bypass_i/sel_dom_switch -> dtlb -> ... ->
issue_read_operands/fu_data_q_reg[..]/D` — **123 logic levels, 50.536 ns, 82% route delay**.

Both structural facts above are still true and still worth fixing. But *which* channel dominates
the critical path is **UNRESOLVED**, and the settling measurement is
`report_timing -to [get_cells */dom_switcher/data_read_q_reg*] -max_paths 20` against
`work-fpga/ariane_xilinx_routed.dcp` (present in both tarballs). No detailed path report to
`data_read_q_reg` exists in either archive — it appears only as a one-line summary at ≈ −3.9 to
−4.1 ns.

## Instrument notes, so the next reader does not repeat a mistake

* `methodology.rpt` caps TIMING-16 at 1000 rows. In the S-10 build those rows are filled by worse
  `csr_regfile_i` entries, so `data_read_q_reg` appears **zero** times there. **A grep of
  `methodology.rpt` alone yields a false zero.** The section-3 by-ENDPOINT enumeration in
  `TIMING-FORENSICS.txt` is the authoritative list.
* `timing-forensics.tcl:44,47` truncates both grouping keys to path depth 4. `dom_switcher` sits
  at depth 3 so its per-register names survive; anything deeper collapses to a module name. Do
  not compute a per-register statistic outside `dom_switcher` from that list.

## What is actually wrong, and what is a question for the hardware side

**Not a constraints problem.** ≥234 endpoints in a cone that requires single-cycle capture are
failing setup, and no timing exception fixes that. The fix is RTL: register the channel response
so the round trip is not combinational. The Anvil `syncstate` pattern tolerates the extra latency
by construction (`.anvil.sv:647-649` — level-held wait-until-valid), and the data channel already
runs that way.

### CORRECTION, same day — the cost figure above was wrong by 4-8x, and both "designer's questions" turned out to be answerable

**A real domain switch iterates 8 indices, not 67.** `is_full` is hardwired `1'b0` at both of its
only call sites, so `capstone_dom_switcher.anvil:114` always takes the `else` arm, which passes
`val_n = 7'd7`; `cur_idx` runs 0..7. The 67-index path is the same unreachable branch documented
in the sibling Anvil-misparse note. `CAPENTER` does not drive the switcher at all (RVFI-verified,
`history/14-08-2026_18-30-00_s06-rtl-fix-p0-p6.md:118`) - only `cscall` and `csreturn` do.

So the delta is **+17 to +33 cycles per switch** (+2/index if the two channel round trips launch
from a shared event, +4 if serial; the generated FSM proves the shared-launch case for one phase
per `process()` inline and leaves the other UNRESOLVED without a directed run), not +134.

**As a fraction of any published number the cost is exactly zero.** Every silicon measurement
brackets the compute only - `ref/fpga-silicon-measurements-for-paper.md:77`: *"Both counter reads
sit inside `domain_main`, around the kernel - so domain entry/exit is excluded from both halves"* -
and the SQLite overhead model has no switch term to perturb.

**Two denominators are MISSING and neither is expensive to get.** (1) The absolute cost of a
domain switch has never been measured, on silicon or in sim - count cycles from
`dom_switch_valid_o` (`commit_stage.sv:355`) to `req_en` clearing; the switcher already broadcasts
`busy_ch.busy`/`busy_ch.idx` every cycle. (2) Boundary-crossing frequency measured as actual
`cscall`/`csreturn` events does not exist either: the ~1-per-19,000 figure counts capability
**borrows**, in harnesses that perform no domain switch at all (Tier-1 is native x86 with a
renaming shim; Tier-2 runs SQLite inside ONE domain and emulates the boundary through a shared
hostcall region). Under the explicit and unsupported assumption that one borrow implies one
call/return pair, the delta would be +0.09%..+0.17% on speedtest1 at CPI 2 and +0.6%..+1.2% in
the boundary-densest scan phase.

**The request-side register already exists in this repository**, on `origin/capt-implementation`
and `origin/capt-verilator`: `42ff49cf6` (2026-06-24) registered the dom-switch request signals
for Verilator convergence, and `c09469628` (2026-07-10) upgraded it to a real skid buffer after
the blind delay corrupted the register walk. On that branch all three responders receive the
registered copies (`cva6.sv:1314/1551/1972` -> `dom_switch_reg_valid_q`/`dom_switch_reg_req_q`);
on our line they receive the raw combinational nets (`cva6.sv:1403/1623/2042`). That register sits
on exactly the path that launches 100% of our failing endpoints. It has never been synthesised,
it leaves the response side combinational, and its own comment flags the data channel as still a
blind delay awaiting the same treatment. Merge-base with our line is `6205f6dbb` (2026-05-25);
47 `core/` files differ.

**The mux was an expedient, and the history says so.** `capstone_dom_switcher.anvil:85`'s
`// TODO: this is a low-performance implementation` was written in `1db856802` (2025-04-14),
**one day before** `2e268c771` (2025-04-15) wired the `raddr_pack` mux - and that commit left the
response unassigned behind `// TODO: obtain the dom switch req response`, filled in two days later
by `f69e403c5`. Six bring-up commits, all one-line subjects, none describing an interface or a
trade-off. No response-side register has ever existed on any branch
(`git log --all -S'dom_switch_reg_resp_q'` returns zero). **And the spec is silent on latency** -
`capstone-spec` specifies only the byte-exact context layout and the swap semantics, and the Anvil
channel declaration is a plain latency-agnostic handshake. Registering is architecturally free.

**A correction inside the correction:** the original text below cited `csr_regfile.sv:389`'s
`// FIXME: a hack` as evidence the same-cycle response was unintended. It is not - that comment
was added three weeks later (`12b7d49d3`, 2025-05-06) and annotates a widening of the reg_id
range. The same-cycle assigns two lines below carry no FIXME and never have. The
`low-performance` TODO still supports the point; the FIXME does not.

### The original text, retained so the correction above has something to correct

The remaining questions are genuinely the designer's, not answerable from RTL:

* is **+1 cycle per index** of switch latency acceptable (67 indices x 2 channel ops)?
* was the combinational regfile-address mux intended? `core/csr_regfile.sv:389` says
  `// FIXME: a hack` and `core/anvil_build/capstone_dom_switcher.anvil:85` says
  `// TODO: this is a low-performance implementation`, which suggests not.
* `core/anvil_build/capstone_dom_switcher.anvil:73` has an explicit `cycle 1` before the `ra`
  writeback. It reads as a scheduling bubble rather than a latency assumption, but that deserves
  a directed simulation test before a synthesis run is spent on the change.

## Separate incidental defect found in the same trace — comment contradicts code

`core/controller.sv:220-225`:

```systemverilog
      // Do not flush EX during a domain switch: the dom_switch_busy signal is
      // a level (not a pulse), so flush_commit_i stays high for the entire
      // domain switch operation.  Asserting flush_ex_o here would continuously
      // kill the d-cache transaction that the domain switcher is waiting for,
      // causing a deadlock where req_en_q never clears.
      flush_ex_o             = 1'b1;
```

A five-line comment explaining precisely why the signal must not be asserted, immediately
followed by asserting it. Two lines up, `set_pc_commit_o = ~dom_switch_busy_i;` **is** qualified
by `dom_switch_busy_i`, so the qualification idiom was applied there and not here. Either the
comment is stale or the line should read `~dom_switch_busy_i`. **UNRESOLVED** — not changed,
because the design demonstrably does not deadlock today and the reason for that is not
established. Whoever resolves it should explain why the described deadlock does not occur.
