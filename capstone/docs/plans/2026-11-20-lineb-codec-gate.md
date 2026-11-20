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
