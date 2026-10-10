# ffmpeg/plane-repros on the Capstone arm `capstone-subobject` -- 2026-10-09

Predictions committed before the run: 49498f856e87.
Platform: Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=level0 heap_log=22 runtime 3c72f36acd5b; compiler 7d01722aab88.

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/ffmpeg/plane-repros --arm capstone-subobject --sdk <SDK, heap=level0> \
      --state <capstone-vm state> --out <fresh dir> \
      --cc-arg=-Xclang \
      --cc-arg=-fcapstone-subobject-bounds \
      --cc-arg=-isystem \
      --cc-arg=~/dev/llvm-capstone-cc/build-release/lib/clang/22/include \
      --cc-arg=-I/tmp/capstone/ffmpeg-app/src-9.0.1-f7e8b586f7b4 \
      --cc-arg=-I/tmp/capstone/ffmpeg-app/domain/ffmpeg-build \
      --cc-arg=/tmp/capstone/ffmpeg-app/domain/ffmpeg-build/libavutil/libavutil.a

## Controls, from the same boot

    clean  RETURNED  required RETURNED  CONTROL clean RETURNED 1
    oob    CAUGHT    required CAUGHT    cause=7 pc=0xc670e5b8 address=0xc6751958 in ctl_write_probe+0x58 (the labelled probe)
    uaf    RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    subobj CAUGHT    required CAUGHT    cause=7 pc=0xc670e5b8 address=0xc6751948 in ctl_write_probe+0x58 (the labelled probe)

## Cases: {'DEFECT-REPRODUCED': 1}

    case  outcome  as-predicted  fixed  detail  image(sha256/16)
    00_b7946098b1_alphablend_row_past_plane
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED the last row's vertical average read the alpha row below the plane's last, inside the frame's one AVBuffer  2af6bd8d96b8a1ad
