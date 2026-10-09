# ffmpeg/plane-repros, arm `poisoncap-spatial` -- 2026-10-09

Platform: PoisonCap mode 0: the rebuilt published PoisonCap platform (CHERI-PURECAP-QEMU-POISON kernel, libc with the mrs poison-retire fix), runtime revocation OFF (booted on the second host).
Predictions committed before the run: the corpus's committed prediction. Runtime revocation reported by the ABI control: 0.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=0
    cheribsd-bounds: CHERI_BOUNDARY_READY

## Cases: {'NOT CAUGHT': 1}

    00_b7946098b1_alphablend_row_past_plane
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the last row's vertical average read the alpha row below the plane's last, inside the frame's one AVBuffer
