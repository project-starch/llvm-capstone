# ffmpeg/subobject-repros on the Capstone arm `capstone-subobject` -- 2026-10-09

Predictions committed before the run: 49498f856e87.
Platform: Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=level0 heap_log=22 runtime afa624ec6387; compiler 7d01722aab88.

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/ffmpeg/subobject-repros --arm capstone-subobject --sdk <SDK, heap=level0> \
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

    clean  RETURNED  required RETURNED  CONTROL clean RETURNED 1
    oob    CAUGHT    required CAUGHT    cause=7 pc=0xce70e5b8 address=0xce751958 in ctl_write_probe+0x58 (the labelled probe)
    uaf    RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    subobj CAUGHT    required CAUGHT    cause=7 pc=0xce70e5b8 address=0xce751948 in ctl_write_probe+0x58 (the labelled probe)

## Cases: {'CAUGHT': 7, 'DEFECT-REPRODUCED': 3}

    case  outcome  as-predicted  fixed  detail  image(sha256/16)
    00_8864fd0aec_cbs_h265_pic_timing_member
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xce70e7b0 address=0xcf75e2f0 in ff2_case_run+0x250 (NOT the labelled probe)  9e9fb2a2c1243de4
    01_68845e26f7_vulkan_hevc_refpicset_member
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xce70e820 address=0xcf75e048 in ff2_case_run+0x2c0 (NOT the labelled probe)  0192c4457dbf0dde
    02_e058af88ab_vulkan_hevc_dpb_member
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xce70e8cc address=0xcf75e008 in ff2_case_run+0x36c (NOT the labelled probe)  752de7aefac5bd19
    03_89de2f0de1_aac_arith_last_member
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xce70e888 address=0xcf75e180 in ff2_case_run+0x328 (NOT the labelled probe)  c469c6bf18f78e53
    04_1a00ea51cb_rtsp_control_url_underflow
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED strlen(control_url)-1 wrapped to SIZE_MAX and the read landed one byte before control_url, on interleaved_max, inside one allocation  ee9215a334358e80
    05_d29ff88422_vulkan_av1_tile_sizes
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xce70e5b8 address=0xcf75e600 in write_probe+0x58 (NOT the labelled probe)  bd3659d49e79ebb4
    06_a809a784ec_vvc_entry_point_start_ctu
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED j was bounded by nothing, so the writes walked 8 words off the end of entry_point_start_ctu and overwrote the next member, inside one allocation  9b97adc96d7bd151
    07_275e217b10_hlsenc_key_uri_strlcpy_size
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xce70211c address=0xcf75e540 in memcpy+0xdc (NOT the labelled probe)  d7152600c3d9bbe9
    08_fb862976df_cbs_h266_col_width_val
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xce70e5b8 address=0xcf75e2a8 in write_probe+0x58 (NOT the labelled probe)  edd73c688a672c7b
    09_ac59fc542f_thumbnail_carved_histogram_slice
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED a 10-bit sample was used unshifted as an index, so the bump landed 768 entries past its 256-entry carved sub-slice, inside one av_calloc  ca77f632dc998178
