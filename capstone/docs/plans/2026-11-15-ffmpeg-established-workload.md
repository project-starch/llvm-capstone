# FFmpeg on a fixed subset of FATE decode tests: reuse distance at the pools, pre-registration, 2026-11-15

Committed before any measured run of this workload. A reading that differs from what is written
here is a finding; this file is not edited afterwards. Definition, observer, hooks, arms and report
format are those of `2026-10-11-nested-reuse-distance.md` (16d2171a3edc). The workload changes from
the run of record's 30-frame clip to FATE, FFmpeg's own regression suite, and each arm runs twice.

## The question

Does the lifetime patch (buffer-pool port patch 0003: every AVBufferPool and AVRefStructPool entry
handed out as a Sublet child lifetime) change when FFmpeg's pools reuse an entry's address, on
FFmpeg's own decode tests instead of a clip of our choosing?

## Boundaries

Unchanged (`experiments/study/reuse-hooks/ffmpeg/pools-reuse-gap-{stock,protected}.patch`):
`buffer-pool` (av_buffer_pool_get hand-out, pool_release_buffer return, keyed by the entry's data
pointer) and `refstruct-pool` (refstruct_pool_get_ext hand-out, pool_return_entry return, keyed by
the entry's RefCount header), 16,384-slot table each, one REUSE-GAP line per process and boundary.
The decoders below draw on both: frame buffers from avcodec_default_get_buffer2's AVBufferPools,
and per-picture tables from RefStruct pools (H.264 qscale, mb_type, motion_val and ref_index; HEVC
tab_mvf and rpl_tab; VP8 and VP9 progress frames; VP9 frame extradata).

## Workload

- **Program:** FFmpeg 9.0.1, https://ffmpeg.org/releases/ffmpeg-9.0.1.tar.xz, sha256
  `cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635`. The image is the run of
  record's (`reuse-hooks/ffmpeg/build.sh stock|protected`, main of
  `experiments/applications/workloads/decode.c`, the port's `ffapp_decode.c`) with two additions:
  `FFREUSE_CONFIGURE="--enable-demuxer=h264,hevc,ivf --enable-decoder=h264,hevc,vp8,vp9"` (configure
  selects the parsers and DSP code these need), and an optional third argument `framecrc` that
  prints FATE framecrc's checksum (adler32 of the packed frame) instead of framemd5's MD5. Built
  from trap/eval ed8cc375c2 (libc and lifetime patches of native-kernel-new-qemu 7ba3f427f9).
- **Command per test:** `ffmpeg.dom SAMPLE 1 framecrc|framemd5`, the hash FATE's command uses; one
  process per test, as FATE runs it; all tests of one arm and repetition in one boot, in the order
  of the subset file. The process's stderr goes to its own file and is printed after its stdout,
  so log lines and the exit-time REUSE-GAP lines never split a frame line.
- **Images:** stock `ffmpeg.dom` sha256 `8a6e542336b46a545f4b7107e7f2c56a45a4802ab8f32a3a71a0e44926136847`,
  protected `a0d0716196929dc313e8d6fb93b473824b1908ed243ddd9ae3c0cacb18caa076` (built on p13 for
  this plan; the calibration used the same images).
- **Samples:** the official fate-suite, `rsync://fate-suite.ffmpeg.org/fate-suite/` (directories
  `h264-conformance`, `hevc-conformance`, `vp8-test-vectors-r1`, `vp9-test-vectors`, fetched
  2026-11-15); every staged file's sha256 is in the subset's hash list and in each repetition's
  provenance. Subset paths are relative to a root with `suite/` (the rsync copy) and `mkv/` (the
  Matroska copies of the Annex B samples, below).
- **Output check:** each test's frame lines against FATE's reference file of the same source tree
  (`tests/ref/fate/<test>`): the frame count, and per frame the size and checksum columns. The
  timestamp columns are the harness's frame index and are not compared.

### The subset (`experiments/study/reuse-hooks/ffmpeg/fate-subset.tsv`, 146 tests)

The candidates are the FATE tests of the four suites whose command is a plain
`framecrc|framemd5 -i SAMPLE` in `tests/fate/{h264,hevc,vpx}.mak` (for HEVC 8-bit, FATE's
`-pix_fmt yuv420p` is the decoder's own format; `-framerate 19` of ba1_ft_c changes timestamps
only). From them, by family:

| family | rule | tests | frames |
|---|---|---|---|
| H.264 conformance (`FATE_H264`) | 176x144 references with at most 150 frames | 37 | 1,722 |
| HEVC conformance (`HEVC_SAMPLES_8BIT`) | 416x240 references with at most 17 frames, minus 2 that fault | 28 | 184 |
| VP8 test vectors (`VP8_SUITE`) | all | 17 | 844 |
| VP9 test vectors (`FATE_VP9_SUITE` 02-size) | all 64 frame-size vectors, 8x8 to 66x66 | 64 | 640 |

The frame and resolution limits keep one boot inside the hour: the stock calibration of exactly
these 146 tests took 1,832 guest seconds (seven QEMU instances on the host), and handed out 14,922
refstruct-pool and 11,175 buffer-pool entries (H.264 11,922 and 5,961; HEVC 736 and 552; VP8 844
and 2,532; VP9 1,420 and 2,130). The 16 QCIF H.264 tests of 300 frames and LS_SVA_D (1,700 frames)
would have added about 40 minutes. Every one of the 146 matched its FATE reference in the stock
calibration (exit 0, all frames equal); the protected calibration ran the 29 first H.264 tests,
all matching, before it was stopped to free the host. The two HEVC tests dropped, MVCLIP_A_qualcomm_3
and MVEDGE_A_qualcomm_3, end in a capability fault in the stock arm: their motion vectors point
far outside the picture, and the memcpy under emulated_edge_mc_8 receives an untagged source
address. Staged files and their sha256: `fate-subset.sha256`.

### A deviation that calibration forced: Annex B inputs remuxed into Matroska

The FATE H.264 and HEVC conformance samples are raw Annex B bitstreams. On this platform the
stock image decodes them wrongly from the raw demuxer: in the stock calibration of 2026-11-15
(`h264-conformance` BA2_Sony_F, CABA3_TOSHIBA_E, CAMA1_TOSHIBA_B, CVFI1_Sony_D; four
`hevc-conformance` samples) the decoder logs "No start code is found" and corrupt macroblocks, no
test matches its reference, and BA2_Sony_F ends in a capability fault (an untagged pointer in
memcpy under emulated_edge_mc_8). The same H.264 decoder matches FATE exactly on a sample that
reaches it in Matroska (fate-h264-direct-bff, 11 of 11 frames). The fault therefore lies in the
Annex B path (raw demuxer, parser or start-code split) of the FFmpeg port on this toolchain; it
was not diagnosed further, and it is outside the pools. As the October adapted-FATE qualification did
(`experiments/study/fate-mpeg4-inputs.json`), each H.264 and HEVC sample is copied losslessly into
Matroska by the host's FFmpeg 8.0.1 (Ubuntu 8.0.1-3ubuntu2):

    ffmpeg -nostdin -v error -r 25 -f h264|hevc -i SAMPLE -map 0:v:0 -c:v copy -bsf:v setts=pts=DTS -f matroska -y SAMPLE.mkv

(`setts=pts=DTS` because Matroska needs a timestamp on every packet; the decoder orders frames by
picture order count, not by packet timestamps). The host decoder reproduces the FATE reference
exactly from 310 of the 312 copies (`-fps_mode passthrough`, size and checksum columns); the two
exceptions (NUT_A_ericsson_5, RPS_D_ericsson_6) are not in the subset. `remux-annexb.sh` repeats
the copy and the check for the subset. The pools see the decoder's allocations, which the container does
not change. The VP8 and VP9 samples run unchanged (IVF and WebM, as FATE runs them).

## Platform

The run-3 trap-route kit `~/asr-work/eval/kit-run-3` on p13: QEMU capstone-qemu address-space-root
9289114e06 (`qemu-1b`), kernel `Image-0133b9e21cd`, firmware `f-keep/fw_jump.elf`, run-3 initramfs
and capstone-vexec, through `runtime/virtual/trap_platform.py run` (the run of record's runner),
wrapped by `reuse-hooks/ffmpeg/run-ffmpeg-fate.py`. At most 8 QEMU instances on the host.

## Arms and repetitions

stock and protected, two repetitions each (stock-1, stock-2, protected-1, protected-2), one boot
per arm and repetition. Results in `~/asr-work/eval/run-5/reuse/ffmpeg-<arm>-<rep>/`. Each
boundary's counters and bins are summed over the processes; a reuse distance never crosses a
process.

## Acceptance

A repetition is rejected if its boot did not complete, if any test's process does not exit 0, if
any test's frames differ from its FATE reference, if any observer `error` is nonzero, or if a
test's frame list differs between the arms.

## Predictions

1. **Noise floor:** the decoders are single-threaded and deterministic on a file, and the pools
   take entries in free-list order. stock-1 and stock-2 give identical counters and bins at both
   boundaries; so do protected-1 and protected-2 (reusediff 0.0 within each arm).
2. **Equal arms:** stock and protected give the same histogram bucket for bucket at both
   boundaries (reusediff 0.0), with equal issues, releases, reuses and distinct starts, as on the
   run of record's clip (run-3).
3. **Output:** every test matches its FATE reference in all four repetitions.

The report gives, per arm and repetition, issues, releases, reuses, distinct starts and error per
boundary, the output check, and the three reusediff values of `collect-reuse.py` (stock-1 vs
stock-2, protected-1 vs protected-2, stock vs protected).
