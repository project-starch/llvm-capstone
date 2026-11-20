# Line B step (a): the Sublet codec's synthesis-only gate (board lane copy of the pre-registration)

Status: pre-registered 2026-11-20, before the launch. The run is the rtl lane's request; the estimates are its own, as
relayed to this lane, and are copied here unchanged. Brief: post-submission programme, section 7.2 item 3.

**Tree.** capstone-ariane lineb-codec 2fcbebf64, parent P1 d83d3a633. `git diff d83d3a633 2fcbebf64` adds
`core/sublet_codec.sv` (46 lines: `sublet_cap_compress` and `sublet_cap_decompress`, combinational) and seven files
under `verif/tests/sublet-codec/`. Nothing in `core/` or `corev_apu/` instantiates the codec, and no Flist lists it.

**Run.** The rtl lane's `verif/tests/sublet-codec/synth-codec.tcl`: synth_design out of context on xc7k325tffg900-2,
`-flatten_hierarchy none`, top `sublet_codec_gate`. Both modules sit between registers and are NOT chained, because a
round trip once took technology mapping from 5 GB to 94 GB. The clock is 20 ns. It runs under vivado-guard at 40 GB in a
fresh detached worktree (`twc/codec/run-codec.sh`). The board flow is not touched.

**Hard checks** (`twc/codec/grade-codec.py`; it exits 2 on missing data):
- loops: check_timing loops 0 on the codec checkpoint, with the toy checkpoint as the control (must read ≥ 1);
- latches: 0 `[Synth 8-327]` in the run's log, with P1's synthesis log as the counter's control (it reads 2), and 0 LD*
  cells;
- hierarchy trigger: either module above 2,500 LUT sends the hierarchy rows to the rtl lane before anything is connected.

**Estimates** (the rtl lane's; reported, not graded):
- mapping peak in the low single GB;
- compress roughly 400-900 LUT;
- decompress roughly 500-1,200 LUT;
- register-to-register WNS positive at 20 ns.

**Pre-synthesis evidence** (the rtl lane's, not re-run here):
- 1,006,498 vectors from the reference model's own encoder (capstone-qemu c1985bb529), with 0 compress and 0
  decompress mismatches;
- both mutant controls fire.
- The lint gate does not see the codec (it is in no Flist), so it was not run.

## Reading: PASS, 3 of 3 hard checks (apollo, 2026-11-20, 12:07-12:09Z)

Record: `<mnt-hdd>/synth/codec-2fcbebf64-gate-out/grade-codec.txt`, with `gate/util_hier.rpt`, `gate/timing.rpt` and
`gate/codec.dcp`. The guard exited 0 after 50 s at a peak RSS of 2.19 GB. The log's peak is 2,735 MB.

| Check | Reading | |
|---|---|---|
| loops | check_timing loops 0 (toy control 1) | PASS |
| latches | `[Synth 8-327]` 0 (P1's log 2, the control); LD* cells 0 | PASS |
| > 2,500 LUT per module | none | PASS |

| Module | LUT (util_hier) | Estimate |
|---|---|---|
| sublet_cap_compress | **1,110** | 400-900: above |
| sublet_cap_decompress | 779 | 500-1,200: in |
| gate top | 1,889 (714 FF, all the gate's own registers) | |

- Register-to-register worst slack at 20 ns: +5.859 ns (MET).
- The probe's primitive census reads 1,290 and 820 LUT* cells. That is a different unit from report_utilization's
  rows; the rows are the reading.
- **Reported to the rtl lane, not graded:** synthesis reports `cursor_i[63:38]` of sublet_cap_compress as having no
  load (`[Synth 8-7129]`, 26 bits). The encoder reads only the cursor's low 38 bits. Whether that is the format's
  intent is the rtl lane's to confirm. The reference model's vectors would not show it if the reference truncates the
  same way.
