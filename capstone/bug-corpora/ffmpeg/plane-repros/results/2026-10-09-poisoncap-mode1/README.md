# ffmpeg/plane-repros, arm `poisoncap-protected` -- 2026-10-09

Platform: PoisonCap mode 1: the same platform with runtime revocation ON -- poison on free, then revocation; libavutil built FFPURECAP_MALLOC_ONLY=1, because this platform's posix_memalign returns a non-zero status under quarantine (booted on the second host).
Predictions committed before the run: the corpus's committed prediction. Runtime revocation reported by the ABI control: 1.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds: CHERI_BOUNDARY_READY

## Cases: {'NOT CAUGHT': 1}

    00_b7946098b1_alphablend_row_past_plane
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the last row's vertical average read the alpha row below the plane's last, inside the frame's one AVBuffer

The first mode-0 run, with the posix_memalign libavutil, read the same (completes); mode 1 with that
libavutil could not run (av_frame_alloc returned NULL, exit 75 -- the platform defect in
cpython/pymalloc-repros/platform/README.md). Both modes are recorded from the malloc-only build, so
they differ in the platform mode alone.
