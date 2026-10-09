# ffmpeg/plane-repros, arm `poisoncap-protected` -- 2026-10-09

Platform: PoisonCap mode 1: the same platform with runtime revocation ON -- poison on free, then revocation (booted on the second host).
Predictions committed before the run: the corpus's committed prediction. Runtime revocation reported by the ABI control: 1.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds: CHERI_BOUNDARY_READY

## Cases: {'FIXED-ARM-FAILED': 1}

    00_b7946098b1_alphablend_row_past_plane
        fixed: exit 75 NOT FIXED
        buggy: exit 75 -> FIXED-ARM-FAILED
