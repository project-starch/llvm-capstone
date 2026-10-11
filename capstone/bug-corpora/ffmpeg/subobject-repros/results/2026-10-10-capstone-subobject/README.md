# ffmpeg/subobject-repros on the Capstone arm `capstone-subobject` -- 2026-10-10 (R2c)

Pre-registered at `49e8e23f4122` (docs/history/10-10-2026_23-30-00_whole-corpus-audit-cross-program.md, R2).
The runner no longer accepts a fault anywhere on this arm: a catch is a fault AT the labelled probe, or in
a function the case declared before the run (`fault_sites` in case.json, with `fault_sites_why` quoting
case.c). Every image is byte-identical to the 2026-10-09 run's (last column), so this re-run changes the
judgement, not the measurement: the same faults at the same offsets now pass the stricter rule.

Platform: Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43, rootfs 9903242c37a7; SDK heap=level0 heap_log=22 runtime afa624ec6387; compiler 7d01722aab88; tool 749c8afb4399.

Two earlier attempts the same evening read nothing and are not records: R2 built nothing (the chain
omitted the buffer-pool sources), and R2b linked a different SDK (runtime 3c72f36acd5b, not the
recorded afa624ec6387), whose heap could not hold the payload (CONTROL-FAILED 604, fixed and buggy).

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/ffmpeg/subobject-repros --arm capstone-subobject --sdk <SDK, heap=level0, runtime afa624ec6387> \
      --state <capstone-vm state> --out <fresh dir> \
      --cc-arg=-Xclang \
      --cc-arg=-fcapstone-subobject-bounds \
      --cc-arg=-std=c11 \
      --cc-arg=-Daligned_alloc(a,n)=malloc(n) \
      --cc-arg=-isystem \
      --cc-arg=~/dev/llvm-capstone-cc/build-release/lib/clang/22/include \
      --cc-arg=-I<scratch>/wt-merge/capstone/ports/ffmpeg/buffer-pool/src/shared \
      --cc-arg=-I/tmp/capstone/ffmpeg-buffer-pool/build/native/sources/ffmpeg-ported \
      --cc-arg=-I<scratch>/wt-merge/capstone/ports/ffmpeg/buffer-pool/cmake/replay-config \
      --cc-arg=-I<scratch>/wt-merge/capstone/runtime/include \
      --cc-arg=/tmp/capstone/ffmpeg-buffer-pool/build/native/sources/ffmpeg-ported/libavutil/buffer.c \
      --cc-arg=/tmp/capstone/ffmpeg-buffer-pool/build/native/sources/ffmpeg-ported/libavutil/refstruct.c \
      --cc-arg=<scratch>/wt-merge/capstone/ports/ffmpeg/buffer-pool/src/shared/observe-pool-events.c \
      --cc-arg=<scratch>/wt-merge/capstone/ports/ffmpeg/buffer-pool/src/shared/pool-allocator.c \
      --cc-arg=<scratch>/wt-merge/capstone/ports/ffmpeg/buffer-pool/src/shared/metadata-allocator.c \
      --cc-arg=<scratch>/wt-merge/capstone/ports/ffmpeg/buffer-pool/src/native/replay/payload-pointers.c

## Controls, from the same boot

    clean    RETURNED  required RETURNED  CONTROL clean RETURNED 1
    oob      CAUGHT    required CAUGHT    cause=7 pc=0xc020e5b8 address=0xc0251958 in ctl_write_probe+0x58 (the labelled probe)
    uaf      RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    subobj   CAUGHT    required CAUGHT    cause=7 pc=0xc020e5b8 address=0xc0251948 in ctl_write_probe+0x58 (the labelled probe)

## Cases: {'CAUGHT': 7, 'DEFECT-REPRODUCED': 2}; 9 of 9 as pre-registered

    case  outcome  pre-registered (R2)  fixed  detail  image(sha256/16)  same image on 2026-10-09
    00_8864fd0aec_cbs_h265_pic_timing_member
        CAUGHT  predicted CAUGHT in ff2_case_run  held  fixed: FIXED exit=0
        cause=7 pc=0xc020e7b0 address=0xc125e2f0 in ff2_case_run+0x250 (NOT the labelled probe) -- a fault site the case declared before the run (case.json fault_sites)  9e9fb2a2c1243de4  yes
    01_68845e26f7_vulkan_hevc_refpicset_member
        CAUGHT  predicted CAUGHT in ff2_case_run  held  fixed: FIXED exit=0
        cause=7 pc=0xc020e820 address=0xc125e048 in ff2_case_run+0x2c0 (NOT the labelled probe) -- a fault site the case declared before the run (case.json fault_sites)  0192c4457dbf0dde  yes
    02_e058af88ab_vulkan_hevc_dpb_member
        CAUGHT  predicted CAUGHT in ff2_case_run  held  fixed: FIXED exit=0
        cause=7 pc=0xc020e8cc address=0xc125e008 in ff2_case_run+0x36c (NOT the labelled probe) -- a fault site the case declared before the run (case.json fault_sites)  752de7aefac5bd19  yes
    03_89de2f0de1_aac_arith_last_member
        CAUGHT  predicted CAUGHT in ff2_case_run  held  fixed: FIXED exit=0
        cause=5 pc=0xc020e888 address=0xc125e180 in ff2_case_run+0x328 (NOT the labelled probe) -- a fault site the case declared before the run (case.json fault_sites)  c469c6bf18f78e53  yes
    04_1a00ea51cb_rtsp_control_url_underflow
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED strlen(control_url)-1 wrapped to SIZE_MAX and the read landed one byte before control_url, on interleaved_max, inside one allocation  ee9215a334358e80  yes
    05_d29ff88422_vulkan_av1_tile_sizes
        CAUGHT  predicted CAUGHT in write_probe  held  fixed: FIXED exit=0
        cause=7 pc=0xc020e5b8 address=0xc125e600 in write_probe+0x58 (the labelled probe)  bd3659d49e79ebb4  yes
    06_a809a784ec_vvc_entry_point_start_ctu
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED j was bounded by nothing, so the writes walked 8 words off the end of entry_point_start_ctu and overwrote the next member, inside one allocation  9b97adc96d7bd151  yes
    07_275e217b10_hlsenc_key_uri_strlcpy_size
        CAUGHT  predicted CAUGHT in memcpy  held  fixed: FIXED exit=0
        cause=7 pc=0xc020211c address=0xc125e540 in memcpy+0xdc (NOT the labelled probe) -- a fault site the case declared before the run (case.json fault_sites)  d7152600c3d9bbe9  yes
    08_fb862976df_cbs_h266_col_width_val
        CAUGHT  predicted CAUGHT in write_probe  held  fixed: FIXED exit=0
        cause=7 pc=0xc020e5b8 address=0xc125e2a8 in write_probe+0x58 (the labelled probe)  edd73c688a672c7b  yes

The runner's own `predicted` field (record.json) is its corpus-wide default for an interior crossing,
DEFECT-REPRODUCED; the per-case predictions above are the pre-registered ones.
Case 07's fault is in `memcpy`, the weaker form of attribution: this domain records no return address.
The supervised CheriBSD run of the same source (results/2026-10-10-cheribsd-subobject-supervised/, R3c)
names the caller, `ff2_strlcpy`, the case's own copy.
