# ffmpeg/plane-repros on the Capstone arm `capstone-carve-bounds` -- 2026-10-09

Predictions committed before the run: 83dcfa032845.
Platform: Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=level0 heap_log=22 runtime 3c72f36acd5b; compiler 7d01722aab88.

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/ffmpeg/plane-repros --arm capstone-carve-bounds --sdk <SDK, heap=level0> \
      --state <capstone-vm state> --out <fresh dir> \
      --cc-arg=-DFFP_CARVE_BOUNDS \
      --cc-arg=-isystem \
      --cc-arg=~/dev/llvm-capstone-cc/build-release/lib/clang/22/include \
      --cc-arg=-I/tmp/capstone/ffmpeg-app/src-9.0.1-f7e8b586f7b4 \
      --cc-arg=-I/tmp/capstone/ffmpeg-app/domain/ffmpeg-build \
      --cc-arg=/tmp/capstone/ffmpeg-app/domain/ffmpeg-build/libavutil/libavutil.a

## Controls, from the same boot

    clean  RETURNED  required RETURNED  CONTROL clean RETURNED 1
    oob    CAUGHT    required CAUGHT    cause=7 pc=0xc020e5b8 address=0xc0251948 in ctl_write_probe+0x58 (the labelled probe)
    uaf    RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    subobj RETURNED  required RETURNED  CONTROL subobj RETURNED b0=0x5a

## Cases: {'CAUGHT': 1}

    case  outcome  as-predicted  fixed  detail  image(sha256/16)
    00_b7946098b1_alphablend_row_past_plane
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc020fba0 address=0xc02788a0 in ffp_read_probe+0x38 (the labelled probe)  f10ad50364ba5d9d
