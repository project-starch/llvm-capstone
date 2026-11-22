# B5b-1 step 1b (the Sublet memory format in the core): synthesis-only gate, board lane copy of the pre-registration

Status: pre-registered 2026-11-22, before the launch. This is Line B option C, as the lead ruled. The keys are the rtl
lane's (land/rtl-b5b1 365f5caeee95, `docs/plans/2026-11-22-b5b1-format.md`, "Step 1b's synthesis-only gate"), copied
unchanged.

**Tree, read at source.** capstone-ariane b5b1-format 4c576521d.
- Its base 1c5d31474 (the R-75 (d) tree) is an ancestor.
- `core/` and `corev_apu/` are identical to bd686c6a4's (`git diff --stat` is empty).
- Against the base, nine files under `core/` change, +186 / -20:
  - the dom_switcher .anvil and .anvilh;
  - decoder.sv, ex_stage.sv, ariane_pkg.sv, the config package, load_store_unit.sv, load_unit.sv and store_unit.sv.
- Outside `core/`, only `verif/` changes (two directed tests, a testlist and the lint REF). No submodule pin moves.
- The config package sets `define CAPSTONE_SUBLET_FMT` unless `CAPSTONE_IDS55_FMT` is defined. No file under
  `corev_apu/fpga` and no Makefile names `IDS55_FMT`, so the FPGA flow builds the Sublet format.
- `*.anvil.sv` is git-ignored and generated in the run. The R-75 (d) gate's switcher SV was written at 08:13:31Z, after
  its 08:08:19Z launch. This gate also records the generated switcher's hash against the base's, and its whole-word
  `snap` count (the base reads 0).
- Lint on bd686c6a4 is PASS by the rtl lane's numbers. UNOPTFLAT is 39, with the names identical. UNUSEDSIGNAL was
  re-baselined to 1661; all of that change is in the generated switcher.

**Gate** (`twc/b5b1/post-b5b1.sh`, graded by `twc/b5b1/grade-b5b1.py` against the R-75 (d) gate,
`<mnt-hdd>/synth/r75d-1c5d31474-gate-out`). The grader was dry-run twice:
- the base against itself: PASS 4/4, every delta +0;
- B5a's gate as trial: FAIL on mapping (+1,814 MB) and core LUT (+2,804).

| key | prediction | if missed |
|---|---|---|
| mapping peak | within 10 % of 5,051.6 MB, inside the 40 GB guard | guard kill: partition read, no retry |
| loops | 0 by every probe, each control firing | named |
| core LUT / FF | 162,982 ± 2,500 / 54,345 ± 2,000 | per-module census |
| dom_switcher LUT | reported against the base. The 11 inline range tests are the only new logic there. Above +1,000, the rtl lane moves the snap computation to ONE SV-side test before any full build | |
| load_unit / store_unit / ex_stage LUT | reported. The decoder is reported in id_stage_i | |

No full build follows: the genesis change is owed first.

## Reading (2026-11-22, gate graded 10:52:44Z): PASS 4/4

The run launched at 09:40:42Z and wrote gate.done at 10:48:50Z. The guard ended it as base gates end (exit=143 after
gate.done; the base's own gate read the same). Guard peak 31.44 GB (base 28.29 GB). Host MemAvailable never went below
68 GB, so there was no memory incident. srcwatch OK (vendored=0 fix=1 all=1).

| key | read | verdict |
|---|---|---|
| mapping peak | TM 5,053.0 MB vs 5,051.6 (+1.4, limit ±505.2); growth across TM +11.9 (base +11.9); whole-run 5,121.2 MB | PASS |
| loops | 'Found timing loop' 0 (base 0); Synth 8-326 0; check_timing loops 0 (toy 1); LUTLP-1 0 (toy 1); in-flight loops-only gate PASS | PASS |
| core LUT | 162,040 (−942) | PASS |
| core FF | 53,987 (−358) | PASS |

Reported, against the base (Total LUTs; FFs):
- dom_switcher 11,545 (+897); 1,291 (−1). This is **under the rtl lane's +1,000 line**, so the snap computation stays
  inline.
- i_load_unit 454 (−451); i_store_unit 5,490 (+155); lsu_i 24,936 (−355).
- ex_stage_i 57,205 (+2,759); id_stage_i (the decoder) 1,679 (+302).
- Elsewhere: dyn_unit +3,123, rev_node +624, issue_stage_i −5,885, i_wt_dcache +272. The core total moves
  −942. Two of these moves exceed the ±2,500 per-module redistribution floor: dyn_unit +3,123 and issue_stage_i −5,885.
  Opposite moves across module boundaries with a near-flat core total have the shape of redistribution, but this gate
  does not attribute them, and they are not keys. RAMB36 113 (+0), LUTRAM +0, and no BRAM or DRAM mapping row names rev_node or ex_stage on either side.

**The generated switcher, and an instrument defect in this gate's own record.**
- The generated `capstone_dom_switcher.anvil.sv` differs from the base's: 8fbec9d2a5143cd5 against 51ae47d750095c7a.
  It was written at 09:46:28Z, after the launch.
- The pre-registered whole-word `snap` count reads 0, as the base's does. That count **cannot fire**. Anvil flattens a
  struct into a packed bus, so a field name never reaches the generated SV. Even a case-insensitive substring count
  reads 0.
- Read instead by what the field generates:
  - the `dom_switch_data_req` bus is `[220:0]` at all 60 sites (base `[219:0]` at 60): one bit wider, which is `snap : logic`;
  - the constant 1976, the snapshot range's end, appears 61 times (base 2);
  - 1024 appears 65 times (base 6).
- So the 11 range tests are generated. The +897 dom_switcher LUTs say synthesis kept them.

**Genesis.** "No full build follows: the genesis change is owed first", above, is superseded by the rtl lane's
823bea0679bf on land/rtl-b5b1: no RTL genesis change is owed.
- The rtl lane's reading is that CAPENTER is the only genesis in this RTL (capstone_flu_unit.anvil). That reading is
  theirs; this lane has not traced flu_unit.
- Step 1b makes CAPENTER illegal under Sublet. Read at source in the gate's own tree (decoder.sv:1417-1421):
  - `7'b000_1101: begin`;
  - `instruction_o.op = ariane_pkg::CAPENTER;`;
  - then, under `ifdef CAPSTONE_SUBLET_FMT`, the illegal-instruction line, inside CAPENTER's own case.
- The rtl lane reports the simulation's ILLEGAL_INSTR trap at 0x8000002c on the Sublet format; that result is theirs.
- A full build of 1b alone is not requested. It is batched with the rtl lane's next hash.
