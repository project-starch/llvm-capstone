# ffmpeg/plane-repros, arm `cheribsd-carve-bounds` -- 2026-10-09

Platform: stock CheriBSD purecap, kernel CHERI-PURECAP-QEMU, revocation on, the case built with -DFFP_CARVE_BOUNDS (booted on the second host).
Predictions committed before the run: 83dcfa032845. Runtime revocation reported by the ABI control: 1.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds: CHERI_BOUNDARY_READY

## Cases: {'CAUGHT': 1}

    00_b7946098b1_alphablend_row_past_plane
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT
