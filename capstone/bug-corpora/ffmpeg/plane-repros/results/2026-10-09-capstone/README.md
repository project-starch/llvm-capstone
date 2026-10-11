# ffmpeg/plane-repros on the Capstone application domain arms -- 2026-10-09

Predictions committed before the run: 60f1c5be25a5 (tools/run-capstone-domain.py, predicted()).

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/ffmpeg/plane-repros --arm spatial|sublet \
      --sdk <application SDK built with CAPSTONE_APPLICATION_HEAP=level0|sublet> \
      --state <capstone-vm state> --out <fresh dir> \
      --cc-arg=-isystem \
      --cc-arg=<compiler build>/lib/clang/22/include \
      --cc-arg=-I/tmp/capstone/ffmpeg-app/src-9.0.1-f7e8b586f7b4 \
      --cc-arg=-I/tmp/capstone/ffmpeg-app/domain/ffmpeg-build \
      --cc-arg=/tmp/capstone/ffmpeg-app/domain/ffmpeg-build/libavutil/libavutil.a

libavutil.a and its config come from the FFmpeg app port's domain build
(ports/ffmpeg/app/host/build-domain.sh), compiled by the same compiler (7d01722aab88) against
musl public headers byte-identical to this SDK's. It was configured with HAVE_POSIX_MEMALIGN 0,
so av_malloc reaches the SDK's malloc and the frame's buffer is bounded to its size.

Result LINES only, from each arm's record.json.

## spatial

    platform  Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=level0 runtime 3c72f36acd5b; compiler 7d01722aab88
    env       {'CAPSTONE_GP_NONLIN': '1', 'CAPSTONE_REV_NODES': '65536'}
    tool      376cc8f440aae03b   predicted buggy outcome: DEFECT-REPRODUCED
    control clean RETURNED  required RETURNED  CONTROL clean RETURNED 1
    control oob   CAUGHT    required CAUGHT    cause=7 pc=0xc020e5b8 address=0xc0251768 in ctl_write_probe+0x58 (the labelled probe)
    control uaf   RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    00_b7946098b1_alphablend_row_past_plane                            fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the last row's vertical average read the alpha row below the plane's last, inside the frame's one AVBuffer  image 9708638436b7117b

## sublet

    platform  Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=sublet runtime b73508e5005b; compiler 7d01722aab88
    env       {'CAPSTONE_GP_NONLIN': '1', 'CAPSTONE_REV_NODES': '65536'}
    tool      376cc8f440aae03b   predicted buggy outcome: DEFECT-REPRODUCED
    control clean RETURNED  required RETURNED  CONTROL clean RETURNED 1
    control oob   CAUGHT    required CAUGHT    cause=7 pc=0xc020f70c address=0xc8802018 in ctl_write_probe+0x58 (the labelled probe)
    control uaf   CAUGHT    required CAUGHT    cause=24 pc=0xc020f758 address=0x0 in ctl_read_probe+0x38 (the labelled probe)
    00_b7946098b1_alphablend_row_past_plane                            fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the last row's vertical average read the alpha row below the plane's last, inside the frame's one AVBuffer  image de9a3e0e6d8c189c

