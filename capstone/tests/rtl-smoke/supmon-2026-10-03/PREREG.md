# FPGA monitor checkpoints C3 and C5 on silicon -- pre-registered 2026-10-03 20:05:04, committed and pushed BEFORE the boots

Bitstream caplifive_supcall_36a641e0b.bit (identified by csnodefree 0xFFCD). Both boots use the same payload Image (sha256
33b63f59113e) and the same stages, and differ only in the OpenSBI build (firmware/SHA256SUMS):
- monitor capstone-sbi monitor/supcall-fpga 1fd1bbe: the board line 472990f, merged with the context-slots line 4674ab6,
  plus the silicon CSR-event form;
- wrapper caplifive-opensbi wrapper/supcall-fpga 849c8e1: carves the 1 KiB save area under the define.
Stages, in this order: k800 (stock b2d60e52, the control); then P1 cell 5 -O0 speedtest (e6ee5255c896aa21, its
recorded oracle) with `--speedtest1 --testset main --size 1 --verify` through sqlite_host_rr.user 2c9e82d1; then k800.

C3 = fw-c3 b317668bc7f8: the MERGED monitor with supervision compiled out (no defines beyond TARGET_FPGA).
  Predicted: k800 retval=4 twice; speedtest Verification Hash 112006 38bb59fd, HEAP 2097152 DROPPED 0 RC 0,
  SPEEDTEST1-CYCLES within 1 % of the record 2,551,483,818; one boot banner; no stall.
  This shows the merge itself is sound on silicon.

C5 = fw-c5 49864994aa64: + CAPSTONE_SUPERVISED_CALL + CAPSTONE_SUPERVISOR_CSR_EVENTS + CAPSTONE_SUPERVISE_CLASSIC_TEST.
  Every classic call runs under supervision, with quantum 2,000,000 cycles, and the monitor resumes each preemption.
  Predicted:
  - k800 retval=4 twice, each with SUPN 0 and SUPK 0. k800 is ~4,500 cycles, well under one quantum.
  - speedtest Verification Hash 112006 38bb59fd, HEAP 2097152 DROPPED 0 RC 0, exactly as C3. The workload's
    result is unchanged by ~1,275 preempt/resume cycles.
  - SUPN about 1,275 (2,551M / 2M). This is a timing quantity, reported. Its invariants are SUPN > 0 and SUPK 0 at
    the end.
  - Every SUPA line 0 (armed).
  - SPEEDTEST1-CYCLES is REPORTED, not predicted. The domain's mcycle bracket includes the monitor's resume loop and
    its two UART trace lines per resume.
  REFUTED if:
  - the hash or HEAP differs;
  - the call ends in SUPK 2 (a fault event);
  - SUPN is 0 on the speedtest (supervision never preempted);
  - any SUPA is non-zero;
  - the boot wedges (classified by ENT1/ENT2 and the SUP tags, not as an R-16 stall).

## ADDENDUM 2026-10-03 20:08:30, before either boot: the image was re-baked, and the control relinked
The first attempt never booted. The runner's preflight refused it before any upload, on two grounds:
- C15: speedtest1.dom and the stock k800 both enter at 0x10000, which is R-3;
- the image carried unused files.
The image was re-baked with the k800 RELINKED at 0x20000 (589ceee3853c6092; its silicon record is 4 at both ends
of both P1 boots on R-42) in place of the stock one, and with speedtest1_baseline and sqlite_host.user retired.
The new payload Image is 9f66fb53af98. Firmware rebuilt from the same sources on it (firmware/SHA256SUMS):
- C3 96884501d098;
- C5 eec16b773423.
Preflight now reads GO.
Predictions are unchanged, except that the control is the relinked k800 (retval 4).
