# S-07: candidate mechanisms, and what we propose changing

Written for whoever owns the RTL. **We are not asking you to accept any of this — we are asking you
to check what we could not, and we say which experiment kills each candidate.**

Read `../00-README.md` first for the symptom and the measurements.

## Ground rules for the citations, because they are not all resolvable at one commit

Line numbers below are given against **`capstone-ariane` HEAD = `25035c4c0`** unless a line is marked
otherwise. Two warnings a reader must have:

* **The bitstream RTL is not in the repo.** `caplifive_12august.bit` is, per the driver's own guard
  comment (`tests/rtl-smoke/fpga_driver/run_sqlite_fpga.py:56-58`), `capstone-ariane 7aac52f93` plus
  the latched-mepc debug mux and the total LCC field-1 type query — deltas that were never published.
  It is **not reconstructible from this tree**. So every citation here describes RTL that is *close
  to*, not identical with, the silicon that produced the measurements.
* **Where a line has moved, the QUOTED TEXT is authoritative.** `core/` changed substantially between
  `7aac52f93`, `013e162fd` (2026-08-12) and HEAD. If a line number does not match your checkout,
  grep for the quoted string rather than assuming the claim is wrong.

One consequence worth stating separately, because it changes an exclusion in the README: the
write-buffer `.user` write **is unconditional at 2026-08-12 and earlier**, but at HEAD it is guarded
(`core/cache_subsystem/wt_dcache_wbuffer.sv:619-626`, `if (req_port_i.data_is_cap || !(...is_cap))`).
The README's exclusion of that path reasons from the *unconditional* form, which is what the board
most likely has. If you are reading HEAD, the guard you find there is **not** what ran.

---

## The constraint every candidate must satisfy

`cap_metadata_t` is `{revnode_id, perm, cap_type, bounds}`
(`core/include/ariane_pkg.sv:633-638`) and `NOT_CAP = 3'b000` (`:651`). **An all-zero metadata word
decodes to NOT_CAP** — so any mechanism that zeroes or fails to deliver metadata gives exactly this
symptom, while one that corrupts metadata to a random value mostly would not.

Cause codes (`core/include/riscv_pkg.sv:349-353`, unchanged since 2026-08-02):
`UNEXPECTED_OPERAND_TYPE = 25`, `INVALID_CAPABLITY = 26`, `CAPABLITY_OUT_OF_BOUND = 29`. We see 25,
which excludes the revocation-validity family arithmetically.

**The guard has two arms** (`core/anvil_build/capstone_flu_unit.anvil:29-31`):

```
func CINCOFFSET(data){
    if((data.cap_rs1.metadata.cap_type==cap_type_t::NOT_CAP)||(data.cap_rs2.metadata.cap_type!=cap_type_t::NOT_CAP)){
        call raise_exception(data.trans_id,ex_code::UNEXPECTED_OPERAND)
```

so 25 at a `cincoffset` means **rs1 lost its tag OR rs2 gained one**. Below, **A-family = rs1 lost
it; B-family = rs2 gained one.**

**Two observations are rs1-unambiguous**, which is the strongest single fact in this folder:

1. an instance faulting *at* an `ldc`, whose guard is rs1-only (`capstone_dyn_unit.anvil:327-330`);
2. `sqlite3_strnicmp+0x134`, faulting at `cincoffsetimm a0, a0, 1` — the **immediate** form, whose
   guard checks rs1 only and has no rs2 arm at all (`capstone_flu_unit.anvil:57-61`). Recorded in
   `../src/s06spill_kernel.h:9-16`.

So **A-family is established as real**; B-family remains possible as an *additional* mechanism at the
`cincoffset` sites, not as a replacement.

---

## A-1. The capability load syncer tracks ONE outstanding request

> ### RETRACTED 2026-08-15: we said this was downgraded. It is NOT. It is UNMEASURED.
>
> We built two rungs to test A-1 and both returned 0, and we wrote that up as evidence against the
> mechanism. **Both rungs are void.** Disassembling the binaries that actually ran shows neither ever
> put two capability loads in flight:
>
> * `s07chase` chases a **dependent** chain, and a dependent `ldc` cannot issue while its predecessor
>   is outstanding — it needs that result as its base address. Structurally incapable.
> * `s07indep` was written to fix that with eight **independent** loads. But it is built at `-O0`
>   (`build-ladder-domain.sh:73`), so every loaded value is spilled immediately: **18 of its 43
>   `ldc`s have their result consumed one instruction later**, 10 more at a distance of 2. The
>   "back-to-back independent loads" exist only in the C source.
>
> Both rungs had firing positive controls — but a control proves the **detector** works, not that the
> **trigger** was ever created. That distinction is what we got wrong, twice in the same rung family.
>
> **So A-1 is neither confirmed nor weakened. No valid instrument has yet been pointed at it.** The
> zeros stand as measurements; the interpretation we attached to them does not.
>
> **If you want to test it, do not reuse our rungs.** They need an inline-asm burst of independent
> `ldc`s with nothing between them, disassembly-verified before a boot is spent. The board-free
> assertion at the end of this document is unaffected and remains the cheapest route.

**Status: leading structural candidate, and UNMEASURED — see the retraction above. The bypass chain
below is real and well-sourced; what we have never managed is a rung that creates the condition.**

**A framing you may hear elsewhere, which we withdraw: the transaction ID is NOT under-width.**
`core/anvil_build/capstone_dyn_unit.anvil:550-551`:

```
reg cap_trans_id : logic[3];
reg req_set : logic[1];
```

3 bits is correct: `core/include/build_config_pkg.sv:81` sets
`cfg.TRANS_ID_BITS = $clog2(CVA6Cfg.NrScoreboardEntries)` and this core has
`CVA6ConfigNrScoreboardEntries = 8`
(`core/include/capstone_cv64a6_imafdc_sv39_config_pkg.sv:60`). Corroborated three ways:
`core/anvil_build/capstone_defs.anvilh:10` (`type trans_id_t = logic[3]; // for 8-entry scoreboard`),
the generated `core/capstone_dyn_unit.anvil.sv:9457` (`logic[2:0] cap_trans_id_q;` — which also
fixes the `logic[N]` convention as N bits, not N-1:0), and `req_set : logic[1]` being assigned
`1'd0`/`1'd1`. **There is no ID aliasing from a narrow field.**

The concern is **depth**. The tracker is one register plus one valid bit, and all three arming paths
overwrite unconditionally (`:564`, `:573`, `:577`, each `set cap_trans_id := trans_id`). The matcher
(`:583`) discriminates on that single id:

```
if(*req_set == 1'd1 && msg.trans_id == *cap_trans_id){ ... }
else { send lsu_ep.normal_res(msg) >> cycle 1 }        // :593-595 -- the bypass
```

**Why that would produce this symptom.** The bypass routes the response to `LOAD_WB` instead of
`CAP_WB` (`LOAD_WB = 2`, `CAP_WB = 4`, `core/include/ariane_pkg.sv:237-243`). `LOAD_WB` carries no
capability — `core/scoreboard.sv:322-324` ties `wb[1..3].cap_data` to `'0` — and the scoreboard
erases the entry's capability result for every port but 0 and 4 (`:241-247`). Commit then writes
metadata `'0` (`core/commit_stage.sv:279`) into the metadata regfile under the **plain GPR** write
enable (`core/issue_read_operands.sv:1572`,`:1578` `we_pack[i] = we_gpr_i[i]`, wired to the metadata
regfile's main `we_i` at `:1658`,`:1679`). Net: the `ldc` retires with a **correct cursor and NOT_CAP
metadata, having never touched memory** — which is also why every memory-oriented rung in this folder
returns `0xFFFF`: they measure a subsystem this mechanism never enters.

**Answered, so you do not have to:** `loop dyn_ep.flush` **does** reset both fields —
`core/capstone_dyn_unit.anvil.sv:9985-9991` clears `cap_trans_id_q` and `req_set_q` on
`_dyn_ep_flush_0`. An earlier draft listed this as our top open question; it was answerable from our
own repo and we should have looked.

**The genuine open question, and a tension we cannot resolve:** can a response arrive in the window
between the arming `send cap_load_ri.init` (`:326`) and the matcher, and can the LSU return
out of order into it? **And if displacement were this easy, why is the fault not far more common?** (The old
"~23% and site-fixed" form of this question is withdrawn: that rate came from a bitstream that no
longer exists, and the post-fix wedges are at a fourth function in two different binaries.) The dyn unit dispatches LDC without waiting (`:505-529`), `NrLoadBufEntries = 2`, and
`waiting_for_load_syncer` is a debug-LED signal only (`core/cva6.sv:1126`,
`core/ex_stage.sv:964`) — not backpressure. Back-to-back `ldc` is common in the very code that
faults. **A-1 must explain the rarity and the site-fixedness, and we cannot make it do so.** We are
flagging that against our own candidate rather than leaving you to find it.

**What kills it:** an assertion that **no `LOAD_WB` writeback ever carries an `ldc`'s `trans_id`**,
negative-tested by forcing the bypass so you know the assertion can fail. Board-free, needs no
reproduction of the sporadic fault.

### Proposed change, only if A-1 is confirmed

1. **Replace the one-deep tracker with an 8-entry valid vector** indexed by `trans_id`
   (`logic [7:0] cap_req_pending`), set on arm, cleared on match. Same indexing as the scoreboard, no
   new ordering assumptions, and it deletes the concept of "the pending request".
2. If one-deep must stay for timing, **backpressure** instead of overwriting: refuse to arm while
   `req_set` is high.
3. **Do not** merely re-route unmatched responses — the defect would be the displacement; the bypass
   is only what makes it silent.

**Acceptance criterion, and it must FAIL first:** a directed test issuing N capability loads with
overlapping lifetimes, asserting each retires through `CAP_WB` with non-zero metadata. Confirm it
fails on current RTL before trusting a pass on fixed RTL.

---

## A-2. Shadow-tag DRAM round trip

**Status: plausible, unquantified. No patch proposed.**

Tags on a cache miss live in a separate DRAM region driven by an FSM
(`core/cache_subsystem/wt_axi_adapter.sv:109-114`). That FSM carries a **documented, deliberately
unfixed** ordering defect for atomics (`:143-152`): ATOMIC_REQ is excluded from `needs_tag` because
including it wedged the core, leaving a real tag-resurrection path. That proves the class — a genuine
refill reading the wrong tag state — is live in this block. The return path also already needed one
mis-association fix (`:129-132`).

**Our workload contains zero `amo*`/`lr.*`/`sc.*`** (disassembled, all 327 860 instructions), so the
known AMO defect is not our fault; any proposal here must explain a non-atomic path.

**What stresses it, board-free:** `verif/tests/custom/capstone/cap-tag-cache.S` has the
evict-and-refill scenario but inserts quiescing `nop`s before the eviction loop (`:97-98`), so it
tests the functional path and not the race. A variant with those NOPs removed, started immediately
after the `STC`. **We have not run it.**

**No patch proposed** — we have no evidence that would tell you what to change, and this FSM has a
history of wedging when modified.

---

## B-1. Capability metadata forwarding

**Status: UNRESOLVED and untested.**

`core/issue_read_operands.sv:694` and `:769` build a wide, register-index-matched select network over
`NrWbPorts` plus scoreboard-depth sources feeding `rs1_cap_metadata_res`/`rs2_cap_metadata_res`
(arbiter instances from `:918`). This is the natural place for a plain integer to *acquire*
capability metadata — the rs2 arm — which would raise 25 with the capability intact. The asymmetry
that makes it plausible: the metadata regfile's main write port is enabled by the plain GPR write
enable, so every integer writeback writes metadata; it normally writes `'0`.

Given the two rs1-unambiguous observations above, B-1 cannot be the *whole* story. It could still be
a second mechanism at the `cincoffset` sites.

**Discriminator, built and waiting:** `../board/G6P-DISCRIMINATOR.md`.

**No patch proposed** — confirm the direction first.

---

## Ruled out — please do not re-derive

| ruled out | why |
|---|---|
| the revocation-validity family | raises mcause **26**; we see **25**. Arithmetic |
| the AMO tag-resurrection path | domain contains **zero** atomics (disassembled) |
| hardware multiply/divide paths | **zero** `mul`/`div`/`rem`; soft routines only |
| DRAM soft error / refresh flip | orders of magnitude below a ~25% per-execution rate |
| state persisting across a power cycle | `cap_tag_q` (`wt_dcache_mem.sv`, reset branch of the cap-tag always_ff), the tag FSM (`wt_axi_adapter.sv:900-919`) and the rev-node pool all reset on `rst_ni` |
| marginal timing / thermal | core runs at **25 MHz** — `corev_apu/fpga/xilinx/xlnx_clk_gen/tcl/run.tcl:18` **and** `:32`, i.e. both branches of the target `if`; the `// 50 MHz` comment at `ariane_xilinx.sv:1209` is stale. A hot-fails path also predicts the opposite of the observed direction. **UNRESOLVED:** no post-route timing report for this bitstream is in the tree, so routed slack is unverified |
| rev-node pool exhaustion | heads at the wedges were ~250-600, **below the sentinel under either pool size** — the pool is 1024 entries at `7aac52f93` and 65536 at 2026-08-12/HEAD (`capstone_rev_node.anvil:187`), and we cannot tell which is in the bitstream |
| the write-buffer `.user` clobber | needs a coalescing plain store to the **same word**; at the faulting site the scalar is a different word *and* a different 16-byte granule. See the HEAD-vs-board caveat at the top |
| an R-20 analogue on another register | `f623c48a1` is an ancestor of every candidate synthesis tree and was never reverted. A grep for 5-bit register literals across `core/*.sv` finds only CAPENTER/x10-x11 sites (`issue_read_operands.sv:576`, `scoreboard.sv:236-238`, `decoder.sv:1287` — the last being a *destination* `rd = 5'd11`), plus RVZCMP-gated `macro_decoder.sv` sites (`CVA6ConfigZcmpExtEn = 0` in this config, so gated out) and `compressed_decoder.sv:853`'s architectural `c.jalr` x1. We fault on x11/x12 as **source** operands |

---

## If you can run only one thing

The **A-1 assertion**: no `LOAD_WB` writeback may carry an `ldc`'s `trans_id`, negative-tested so you
know it can fail. Board-free, needs no reproduction, and it either kills our leading candidate or
hands you the defect.
