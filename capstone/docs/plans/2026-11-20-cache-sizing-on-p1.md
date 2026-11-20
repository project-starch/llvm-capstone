# Cache-sizing variants rebuilt on P1: L1D 64 KiB and I-cache 64-byte lines (board lane, pre-registration)

Status: pre-registered 2026-11-20, before either launch. Brief: post-submission programme, section 7.2 item 4
(spec-silicon 7b0e8cd4b52b). Each tree differs from P1 (capstone-ariane d83d3a633) in one config-package change
only, checked with `git diff d83d3a633 <hash>`.

| Variant | capstone-ariane | The one change | Launch |
|---|---|---|---|
| L1D 64 KiB | perf/p1-l1d64k 317b4fab118d | `DcacheByteSize` 32768 → 65536, `DcacheSetAssoc` 8 → 16 (4 KiB per way, 16-byte lines) | deferred (below) |
| I-cache 64-byte lines | perf/p1-ic64b 3ce15b95741a | `IcacheLineWidth` 128 → 512 (P1's 32 KiB and 8 ways kept: 64 sets, 4 KiB per way) | synthesis-only gate, then full build on GO |

## What the brief's statements read at source

- **L1D.** a7f95f371 sits on 5f41c486e, which is 65 commits behind P1. The same two lines were re-applied as 100bb9ad5,
  and built as b0f02bac7 on eeb654104 (F2), a P1 ancestor 30 commits back. `git diff eeb654104 b0f02bac7 -- core
  corev_apu` is those 4 config lines and nothing else. Both builds are sealed and have never been flashed:

  | Build | WNS | Placed LUT | FF | Slices |
  |---|---|---|---|---|
  | eeb654104 (F2) | -12.229 ns | 188,095 | 75,224 | 50,770 |
  | b0f02bac7 (F2 + L1D 64 KiB) | -17.714 ns | 195,310 | 77,259 | 50,905 (99.91 %) |
  | **Δ, the lever** | **-5.485 ns** | **+7,215** | **+2,035** | **+135** |

  b0f02bac7's own full grade FAILs the WNS null-band key: -8.06 against F, whose re-gate was its reference. The pair
  is a one-variable silicon measurement of the lever that is already available.
- **I-cache.** 105c464bc sits on f285ffc36 (merge-base with P1 38ae2d775). Its message reads "16 KiB, 4 ways, 64 sets"
  because its base had a 16 KiB, 4-way I-cache. P1 has 32 KiB and 8 ways (ic32k), so on P1 the same one-line change
  gives 8 ways × 64 sets × 512 bits. No synthesis of any 512-bit-line I-cache exists in the store.
- **Line-width consumers** (stock CVA6, parameterised): `wt_axi_adapter` (`AxiNumWords` becomes 8 and
  `AxiRdBlenIcache` 7; the capability write burst at :601 stays at 2 beats), `cva6_icache_axi_wrapper`, `cva6.sv`'s
  refill struct, and the HPDcache arbiter (not instantiated: DcacheType is WT). No capstone addition in the adapter reads
  the I-cache burst length. Function is the 95-test sweep's question; see "Before hand-off".

## Lint (rtl-lint-gate.sh against P1's committed baseline `verif/sim/rtl-lint.REF.txt`)

- **perf/p1-ic64b 3ce15b95741a: PASS, every counter at baseline.** LATCH 44, MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0,
  UNOPTFLAT 39, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 1480, CASEOVERLAP 2, SELRANGE 100, ANVIL_UNOPTFLAT 0. 9,562 lint lines,
  0 `%Error`. Anvil was regenerated first, and the negative control (an unwritable OUT) reads BLOCKED (rc 1). Run in
  cva6-build-rv with verilator 5.008 (`.run-lint-host.sh`).
- **perf/p1-l1d64k 317b4fab118d: PASS, every counter at baseline** (the same eleven values; 9,574 lint lines, 0 `%Error`,
  negative control BLOCKED). Read after the I-cache gate's launch; it is not needed until the L1D's deferred launch. A first run on both branches
  ERRORED and is void, not a reading: the worktree lacked core/cvfpu's nested submodule `fpu_div_sqrt_mvp`, so verilator
  stopped with 8 errors before analysis.

## Predictions (written before launch)

### L1D 64 KiB, full build (tag p1l1d-317b4fab1): DEFERRED

Not launched. The runtime lane's slot decision of 2026-11-20 reads: the sealed pair eeb654104 / b0f02bac7 runs on board 2 first, after
7180a83b6's Addendum 197, and a P1 build of the lever is decided from that reading. It would probably need area freed first,
e.g. by line B's deletions. The second slot is held for the rtl lane's bitstream 1. The predictions below stand for
whenever it is launched.

From the pair's deltas applied to P1's seal (187,368 LUT, 76,543 FF, 50,800 slices, WNS -13.173):
- placed LUT about 194,600 (95.5 %) ± 2,500; FF about 78,600 ± 1,000;
- slices about 50,935, against about 50,950 on the device (P1 is 50,800 = 99.71 %);
- WNS about -18.7 ± 3.7 ns. That is outside P1's null band, so grade_full.py's WNS key is predicted to FAIL;
- wt_dcache_mem: `cap_tag_q` and `vld_mirror` are flop arrays of 256 sets × ways, so 16 ways adds about +2,000 FF there.

**Failure branch (pre-registered; the likely one).** If placement or routing fails (a slice-limited placement error,
or nets with routing errors), the reading is **"L1D 64 KiB does not fit beside P1"**. There is no retry, no flow or
strategy change, and no partial seal. The lever's silicon reading then comes from the eeb654104/b0f02bac7 pair. A
synthesis blow-up past the guard is recorded with its phase, and also gets no retry.

**If it routes:** it is sealed as
`caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim-twc16k-merge-f2-hpmtag-lrscfix-tagoverlap6c-pagedtbl-l1d64k_317b4fab1.bit`.
It is graded with grade_full.py against P1's seal, and handed to the runtime lane after the sweep. A WNS of about
-18.7 would be the worst of any bitstream handed to the board, so its own Tier 1 boot is the first check.

### I-cache 64-byte lines, synthesis-only gate (tag p1ic-3ce15b957)

Chain: `twc/ic/post-ic.sh`, graded by `twc/ic/grade-ic.py` against P1's own gate (pw-d83d3a633-gate-out). Every
census comes from the `.vds`. The grader was dry-run on B5a's gate (it read core +3,383 LUT and TM +8 MB, as
recorded). It FAILs on an injected LUTLP-1 count and exits 2 on a missing log.
- KEY1 memory: growth across Technology Mapping at most P1's + 500 MB; whole-run peak inside the 40 GB guard.
- KEY2 loops: no "Found timing loop" in the `.vds`; check_timing loops 0 and LUTLP-1 0 on the synth checkpoint, each
  with its toy control at ≥ 1; in-flight loops-only PASS.
- KEY3 fit, the go/no-go for the full build: core (i_cva6) post-synth LUT at most +7,500 (b0f02bac7's routed delta is
  +7,215, the largest yet routed at this utilisation), and top RAMB36 inside the device's 445.

Readings predicted for the gate (reported, not graded). P1: i_cva6_icache 1,034 LUT, 142 FF, 24 RAMB36 (8 tag and 16
data cuts, 256 × 64 each); top RAMB36 113.
- The 64-bit cuts become 64 deep. One of two readings:
  - (a) BRAM kept: 8 tag and 64 data cuts, so i_cva6_icache about 72 RAMB36 (+48) and LUT +500 to +2,000;
  - (b) LUTRAM: i_cva6_icache RAMB36 at most 8, and LUTRAM +3,000 to +6,000.
  Either can pass KEY3. The gate reports which one Vivado chose.
- Fetch-side mux: an 8:1 select of 64 bits per way, in place of 2:1. Together with the adapter's 512-bit refill shift
  register, the prediction is core LUT +500 to +2,500 beyond the RAM reading, and FF +500 to +1,500.
- Memory: TM growth within ±100 MB of P1's +8 MB.

**On GO:** full build `p1ic-3ce15b957` with the same full-build failure branch as L1D (no retry, "does not fit
beside P1"). Its name is `caplifive_…-pagedtbl-ic64b_3ce15b957.bit`, with the same prefix as above. Predictions: WNS
within P1's null band (the fetch path is not on P1's critical path, cpmp_q → scoreboard); placed LUT P1 + the gate's
core delta ± 2,500.

**On NO-GO:** no full build. The reading is "64-byte I-cache lines do not fit beside P1" with the gate's numbers,
returned to the runtime lane.

## Before hand-off of either seal

- The 95-test sweep with the RVFI trace compare on the hash, against P1's sweep: values and traces identical, cycles
  reported. The sweep is the rtl lane's (lanes table); it is requested through the runtime lane. No seal is handed over
  for a flash before it has run. Prior art for L1D: a 95-test sweep of L1D 64 KiB on the F3 line ran in the rtl
  lane's vp8 worktree (`.sup/swf3l1d64`, "SWEEP-RAN 95 TESTS", lint gate PASS). Its compare against a base is that
  lane's record and is not read here, so it is not cited as a pass.
- The simulator has no NI rule (the FPGA's [0, 0x8000_0000) rule is off in sim). An 8-beat I-cache refill under that
  rule is a Tier 1 boot question, not a sweep question.
- Slots: the I-cache gate runs in one slot; the other is held for the rtl lane's bitstream 1. If that hash is not
  ready within about an hour of the gate finishing, the slot goes to the spatial-only core (runtime lane, 2026-11-20).
  A full build of the I-cache variant waits for both the gate's GO and the sweep.

## Reading: the I-cache synthesis-only gate is GO, 3 of 3 keys (apollo, tag p1ic-3ce15b957, 2026-11-20)

Record: `<mnt-hdd>/synth/p1ic-3ce15b957-gate-out/grade-ic.txt`; chain log `<mnt-hdd>/synth/twc/post-p1ic-3ce15b957.log`;
kept checkpoint `twc/gate-synth-p1ic-3ce15b957.dcp` (sha256 0b8571a8cab6a589…). The guard stopped the job after
synth_design, as designed (rc 143 from the gate copier), at a peak RSS of 27.54 GB (P1's gate: 27.35).

| Key | Reading | |
|---|---|---|
| 1 memory | growth across Technology Mapping +11.9 MB (P1 +8.0); TM peak 5,109.5 MB (P1 5,041.7); whole-run peak 5,177.6 MB | PASS |
| 2 loops | no "Found timing loop" in the `.vds` (P1 0); Synth 8-326 0; check_timing loops 0 (toy 1); LUTLP-1 0 (toy 1); in-flight loops-only PASS; source-list watch OK | PASS |
| 3 fit | core LUT 163,458, +1,055 against P1 (limit +7,500); top RAMB36 161 of 445 | PASS |

Per module, post-synth, against P1's gate (reported):

| Instance | LUT | FF | RAMB36 |
|---|---|---|---|
| top | 190,750 (+1,057) | 78,580 (+447) | 161 (+48) |
| i_cva6 | 163,458 (+1,055) | 54,509 (+414) | 146 (+48) |
| i_cache_subsystem | 20,500 (+1,032) | 5,800 (+414) | 144 (+48) |
| i_cva6_icache | 1,903 (+869) | 170 (+28) | 72 (+48) |
| i_adapter | 5,694 (+163) | 1,320 (+386) | 48 (+0) |
| i_wt_dcache | 12,903 (+0) | 4,310 (+0) | 24 (+0) |
| i_frontend | 2,175 (+18) | 774 (+0) | 2 (+0) |

Against the predictions:
- Reading (a) held: BRAM was kept. 72 I-cache rows (8 tag and 64 data cuts, each 64 x 64) replace P1's 24 (256 x 64),
  so +48 RAMB36 exactly. The Distributed RAM table is unchanged.
- LUT: +1,055 at core, inside the combined prediction (+1,000 to +4,500 for reading (a) plus the fetch-side mux). TM
  growth is within ±100 MB of P1's.
- **FF +414 at core is a miss, below the predicted +500 to +1,500.** It is almost all the adapter's refill shift
  register (+386).

**Next:** the full build (tag p1ic-3ce15b957, name `caplifive_…-pagedtbl-ic64b_3ce15b957.bit`) waits only for the rtl
lane's 95-test sweep on 3ce15b95741a, per the runtime lane's slot decision. Placed-LUT prediction: P1's 187,368 + about
1,060 = about 188,400 ± 2,500. The slice prediction follows the LUT; P1's 99.71 % leaves about 150 slices, so the
route-failure branch registered above still applies.
