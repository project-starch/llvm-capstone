# The M1 baseline's release cost is an unpruned revocation-chain walk, and the variable is per-domain minting — not table occupancy

**Read at the desk on 2026-09-15** from the board lane's raw captures
(`~/capstone-artifacts/unify/board-r1f6-b{1,2,3}/boot.txt`, image `079b1f3a2801205a`), with the RTL
read by the oracle, **every load-bearing line re-read here**, and the whole localisation put through
`claim-auditor`, which SUPPORTED the mechanism and **refuted one piece of my own evidence** (§2).
Instrument: `capstone/sublet/r1/m1-cost-variable.py`.

## RETRACTION, first, because it is mine

I argued that the cost is independent of live nodes **because the four retention arms (retaining 0,
16, 2048, 2048) agree at matched minting**. That evidence is **vacuous**. `retained` is a plain C
pointer copy of a NONLIN alias — `r1_slots_pools.c:503`, `m1_ret_alias[nret++] = old;` — and copying
a NONLIN capability mints nothing, while `delin` only clears `linear` on an existing node. So at
matched `minted` **all four arms have identical node populations and an identical valid-node count of
16**. Their agreement is guaranteed by construction and could never have discriminated anything.

The conclusion survives on simpler evidence: the valid-node count is **constant at 16 throughout**
while release cost rises about twelvefold. What the arms do establish is narrower and still worth
recording — **holding stale references to dead objects costs nothing**.

Second correction to my own numbers: the arm figures I quoted (925/910/965/933) were **individual
draws, not means**, and the dispersion swamps the difference. At per-domain `minted == 2079` exactly,
nine invocations per arm across the three boots:

| arm | n | mean | min | max | within-arm spread |
|---|---|---|---|---|---|
| drop | 9 | 919.4 | 896.8 | 951.6 | 6.0 % |
| pressure | 9 | 919.0 | 856.7 | 964.5 | 11.7 % |
| release | 9 | 920.6 | 894.9 | 944.5 | 5.4 % |
| ring | 9 | 957.9 | 905.6 | 1003.1 | 10.2 % |

Between-arm spread of the four means: **4.2 %** — *smaller* than the within-arm spread. The correct
statement is "indistinguishable at the run-to-run scatter", and it must be written with the
dispersion beside it.

## What the measurement says

The rise is real and reproducible, and it **refutes `chain-m1.sh`'s own pre-registration**, which
predicted `give_cyc/n` FLAT ("release = a childless revoke", "no walk") and named a rise as "the
alternative that would matter". A pre-registered prediction failing is a stronger result than an
unregistered curve, and the §7 entry should say so.

**The independent variable is not table occupancy.** A boot is **twelve separate domain invocations**
(twelve each of `A/dom-ok`, `G/enter`, `H/return`), and `minted` on a snapshot line is per-invocation
(`minted() - m0`). The pool behind it is not: **R-12** (`docs/ref/ISSUES.md:3443`) records the
65,536-node pool with a monotonic head that is **not reclaimed between runs in a boot** — two Sublet
workloads in one boot cross the 65,535 sentinel, observed at a wedge on two separate firmwares. So
the head at invocation *k* is the sum of every earlier invocation's minting:

```
  #       arm  first give/n  last give/n  pool head BEFORE
  1      drop         160.0       2225.7                 0
  9      drop         159.9       2248.7             18684
 12   release         164.4        907.5             25946
  pool head after the boot: 28026 nodes
```

Invocation 9 traces invocation 1's curve with 18,684 nodes already consumed. **If cost tracked the
table's fill, invocation 9 would start where invocation 1 ended.** It starts 0.06 % away. (Honesty
note: the pool head is **never measured** in this capture — the column is a cumulative sum the script
computes, and R-12 withdraws the debug-aperture head reading on healthy boots. The refutation rests
on RTL + R-12, with the invocation overlay as the consistency check.)

**The strongest single piece of evidence isolates the growth to REVOKE.** `take_cyc` brackets a
*constant* number of accesses to the same node memory (MREV = 2 reads + 3 writes, DELIN = 1 read + 1
write, plus the slot's LDC/STC). Arm `drop`, pooled over nine invocations:

| per-domain minted | take/n | give/n |
|---|---|---|
| ~47 | 68.7 | 176.4 |
| ~1024 | 69.0 | 327.8 |
| ~2048 | 81.7 | 883.0 |
| ~2560 | 99.8 | 2114.7 |

A constant-access-count operation on the same memory stays inside ×1.5 while the growing operation
goes ×12. That kills "general memory-system pressure" as an alternative. (Do **not** convert take's
+31 cycles into a per-access miss cost: its nodes were touched moments earlier and are hot.)

Window slopes, cycles per 1000 per-domain minted, arm `drop`, reproducing across three boots to ~5 %:

| boot | w0 | w1 | w2 | w3 | w4 |
|---|---|---|---|---|---|
| b1 | 203 | 117 | 341 | 815 | 2565 |
| b2 | 201 | 115 | 343 | 724 | 2547 |
| b3 | 204 | 108 | 328 | 817 | 2571 |

Minting is **not** "near-flat" as first reported: `take_cyc/n` holds 66-69 and then rises to ~100.

## The mechanism, verified against primary source

1. **`REVOKE` is a multi-cycle walk.** `capstone-ariane/core/anvil_build/capstone_rev_node.anvil:13-34`,
   `REVOKE_NODE`, re-enters its own FSM state (`set stage := 2'd2`) once per visited node and
   terminates on exactly one condition, `node_in.depth <= *depth_bound`, with `depth_bound` set once
   per request from the revoked node's own depth (`:150-158`). The loop carries a `// TODO: optimise`.
2. **A revoked node is never spliced out.** The `valid == 1'd1` branch writes the node being
   *visited* (`send_revnode_update(*revoke_index, *temp_revnode)`); the invalid branch only follows
   `.next`; `change_rev_node_validity` (`capstone_unit.anvilh:552-554`) preserves `prev`, `next` and
   `depth` and clears only `valid`. Nothing writes the revoked node's own `.next`.
3. **The slot's chain anchor persists across a give — established from RTL, without the emulator.**
   `capstone_dyn_unit.anvil:63` has REVOKE's result changing only the *type*
   (`modify_cap_type(rs1, CAP_TYPE_LINEAR)`), carrying `rs1.metadata.revnode_id` through untouched,
   while `:97` shows MREV's result is the only thing that receives a new id. With `sublet_take`
   (`sublet.h:66-76`: the slot receives the mrev result) and `sublet_give_to` (`:129-153`), the next
   take's parent is **always the previous handle**.
4. **The depth arithmetic makes the walk grow, and cannot make it constant.** The child inherits the
   parent's **pre-bump** depth (`:123`) while the parent is bumped by one (`:137`), so `depth_bound`
   is pinned at the carve depth `d0` for the life of the slot and every dead handle sits at exactly
   `d0+1`. Round *r*'s walk therefore visits *r−1* dead handles plus the dead leaf plus one
   terminating read: **r + 2 reads**. It also cannot run away — the chain is bracketed by the next
   slot's live handle, still at `d0`, where the termination test fires.
5. **It is live on the FLASHED bitstream.**
   `git diff --quiet 1bfff7776 f6ec6c198 -- core/anvil_build/capstone_rev_node.anvil` returns 0, and
   the instrument was positive-controlled (the same command on a file that *did* change returns 1).
   `1bfff7776` is the flashed **bitstream** of record (`caplifive_r30r31_1bfff7776.bit`,
   hash-verified, `docs/state/current-state.md:127-128`). Note this is a different label from the
   **sweep** baseline, which the RTL lane records as `4cc068572` — byte-identical RTL, but the two
   labels answer different questions and should not be interchanged. **Caveat:** the generated
   `core/capstone_rev_node.anvil.sv` that synthesis actually consumes is **untracked**, so a `git
   diff` on it would be vacuous; this rests on anvil's determinism, which was not verified.
   **Line numbers above are head's** — at the flashed commit the corresponding sites are
   `ex_stage.sv:1160-1161` and `cva6.sv:2191-2193`/`1716-1719`, and every load-bearing fact was
   re-verified there.
6. **QEMU is flat for a structural reason, not an accounting one.** `cap_rev_tree_revoke`
   (`capstone-qemu/target/riscv/cap_rev_tree.c:118`) splices the invalidated run out at `:136-140`
   (`_CAP_REV_NODE(tree, node_id).next = cur;` plus the `prev` fix-up), and `cap_rev_tree_release`
   (`:145`) returns ids to a free list. The RTL does neither. **A cycle-accurate emulator built on
   this model would still read flat**, so the emulator's flatness is not evidence about hardware —
   and "icount slope 0.00 means no algorithmic walk" does not follow. icount rules out extra
   *instructions*; a hardware FSM walk inside one instruction is invisible to it by construction.

`sublet.h:22` states the behaviour in the project's own words — *"revoke walks the junior run of
nodes"* — and `:15-17` that "the next sublet_take hands it out under a new handle". This is a
documented behaviour meeting an undocumented cost.

## What is NOT established

The walk accounts for growth that is **linear** in accumulated mints. The measured curve is convex,
and the D-cache account for the tail **does not fit**:

* Consecutive handles of one slot are **16 ids = 256 bytes** apart (round-robin over `M1_LIVE` = 16,
  `r1_slots_pools.c:511`). With 16-byte lines and 256 sets, each slot's chain occupies **16 sets ×
  8 ways = 128 lines**, and the sixteen slots use disjoint set groups. A knee at r = 128 would be
  alloc 2048 — numerically the same as "the array outgrows 2048 entries", but by a different route,
  and sensitive to whatever else lives in those sets.
* **The measured transition starts at r ≈ 66 (minted ≈ 1050), not at r = 128.** Marginal cycles per
  added hop run ~2.0-3.9 for r ≲ 60, then 5-8 by r ≈ 90-100, 16-21 by r ≈ 122-130, and 38-45 by
  r ≈ 138-154. A floor of ~2.4 and a ceiling of ~38-45 cyc/hop is a hit cost and a miss cost with a
  **gradual** transition — qualitatively what LFSR-random replacement gives
  (`wt_dcache_missunit.sv:213`, `repl_way = all_ways_valid ? rnd_way : inv_way`), not a sharp
  capacity crossing. A hard knee at 128 under-predicts the excess at minted 2079 (~390 cycles where
  a capacity model predicts ~0); one at 80 over-predicts it.
* **The non-monotone opening is real and unexplained.** It is not a first-snapshot artifact —
  dropping early points leaves window 0 at 202.7/206.3/215.6/203.0 cyc per 1000 as the cutoff moves,
  against 113.5 for window 1. The marginal hop cost genuinely **falls** from ~3.3 cyc/hop over
  r ≈ 13-30 to ~1.8 over r ≈ 30-63 before rising. Nothing in walk-growth plus a cache crossing
  predicts a falling marginal cost. Carried as unexplained rather than fitted.

**Ruled out structurally:** the dTLB (the revnode port is physically addressed from
`CAP_REVNODE_MEM_BASE + id*16` and bypasses the MMU), and any distinct node cache or region-lookup
table — none exists on the path. The hops *are* cacheable (`miss_nc_o`, `wt_dcache_ctrl.sv:115`;
`0xBFF0_0000` lies inside the single cacheable window `[0x8000_0000, 0xC000_0000)`), which the
~2.4 cyc/hop floor independently confirms.

**No software-readable hop counter exists on the deployed bitstream.** `rev_node_debug_ex` reaches
only `debug_led_o` (`core/cva6.sv:1259-1266`), an 8-bit port wired to physical LEDs
(`corev_apu/fpga/src/ariane_xilinx.sv:797`) — not a CSR, not memory-mapped. R-12's "debug aperture"
is the JTAG path, already recorded as unreliable. Hop counts come from simulation or a new
instrumented bitstream.

## The two experiments that close this, neither needing a bitstream

**S1 — off-board, decisive for the walk.** A directed `.S` doing N rounds of mrev/revoke on **one**
slot for N ∈ {1, 8, 64, 128, 160}, `mcycle` read around each REVOKE. Prediction if the mechanism
holds: the rev-node read count is exactly **N + 2** and cycles are linear in N at the hit cost.
**The memory latency must be non-zero** (`S12_MEM_DELAY`): at the testbench's default of zero the
cache tail cannot appear, and the run would read a flat per-hop cost and be misread as refuting the
cache story when it had only tested the walk. This is the S-12 lesson applied before the fact.

**S2 — one board boot, on the deployed bitstream.** `M1_LIVE` at 4 / 16 / 64 as three arms of one
boot. The prediction sharpens: cost is a function of **minted / M1_LIVE**, so at matched `minted` the
excess over the base scales as 1/(slot count) — the 64-slot arm at minted 2591 should read like
today's minted-640 point, and the 4-slot arm should saturate about four times earlier. A pure
address-spread account predicts no dependence on the slot count. `M1_LIVE` is a compile-time
`#define`, so this needs an image rebuild — **which does run on apollo**: this host has clang
18.1.3 and ninja, and the board lane builds these images here daily (including the boots' own image
`079b1f3a2801205a`). An earlier version of this note said otherwise, from a stale memory of the
host, and it was wrong; S2 has no cross-host scheduling dependency.

## Bring-up: this host runs RTL simulation now

Recorded here rather than in `ONBOARDING.md` because that file **already carried another lane's
uncommitted edits** when this session started, and `-o` scopes a commit by path and not by
authorship. **The ONBOARDING correction below is owed and is deliberately not made here.**

Positive control: the R-34 folder's own directed test now runs end to end on this host —
`*** SUCCESS *** (tohost = 0) after 1290 cycles`, with capability readings printing (type 2 perm 6
and perm 2, the NONLIN RW and write-only capabilities the folder describes).

`docs/ONBOARDING.md:277-290` records the `cva6-build-rv` Dockerfile as three layers on
`docker.io/corank/anvil:cva6`. Pulled on **2026-09-15** that base needs five additions — consistent
with ONBOARDING's own warning that a rebuild re-pulls the base and can drift, and its advice to
prefer `docker save`/`docker load` from a host that already has the image:

* **`USER root`** — the base runs as `opam`, so the apt layer dies on `/var/lib/apt/lists/lock`;
* **`apt-get update &&`** before the install, or nothing resolves;
* **`cmake`** — Spike's vendored `yaml-cpp` needs it (`Error 127`);
* **the Python deps** — `cva6.py` imports `yaml` and the image has no `pip3`; `python3-yaml`
  `python3-bitstring` `python3-tabulate` `python3-pandas` suffice for a directed test;
* **`ccache`** — this host's Verilator has `OBJCACHE=ccache` baked in, so the C++ link stage dies
  `Error 127` without it.

Four host facts, none of them in any document:

* **`sg docker -c '<cmd>'` picks up the docker group without a re-login** (the group is in
  `/etc/group` but a session predating it has no access; the consolidated queue assumed a re-login).
  Side effect: under `sg`, `id -g` returns the *docker* gid, so a harness computing
  `--user $(id -u):$(id -g)` runs the container in the wrong group.
* **`make -C core/anvil_build` must run before any simulation** — `anvil.Flist` and the `.anvil.sv`
  files are gitignored, so a fresh tree has none and the failure reads as a broken checkout
  (`Cannot open -f command file`).
* **This host's `tools/verilator-v5.008` was built natively.** It runs fine inside the container but
  has the HOST path baked in as its root, in an *installed* layout. Two consequences:
  `verif/sim/setup-env.sh` puts the image's own Verilator **5.024** first on `PATH` while
  `cva6.py:1033` hard-gates 5.008, so `PATH` must be re-pointed *after* sourcing it; and
  `VERILATOR_ROOT` expects `$ROOT/include/` while the install put headers in
  `$ROOT/share/verilator/include/` (fixed locally with a symlink, `tools/` being gitignored).
  Mounting the tree at its host path *as well as* at `/workdir` makes the baked path resolve.

## A process slip, because the rule already existed

Twice here a gate's exit status was read through a trailing command —
`sg docker -c '<build>; echo DONE'` — so a **failed** image build was reported as exit 0. That is
CLAUDE.md's "never filter between a gate and its exit status", in the shape the rule names, caught
only by reading the build log. Redirect, never chain, and read the gate's own status.
