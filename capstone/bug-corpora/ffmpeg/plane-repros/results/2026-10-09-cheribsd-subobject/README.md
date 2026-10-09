# ffmpeg/plane-repros, arm `cheribsd-subobject` -- 2026-10-09

Platform: stock CheriBSD purecap, kernel CHERI-PURECAP-QEMU, revocation on, every case built with -Xclang -cheri-bounds=subobject-safe (booted on the second host).
Predictions committed before the run: 886abf4cc336. Runtime revocation reported by the ABI control: 1.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds: CHERI_BOUNDARY_READY
    subobj-control: exit 162

## Cases: {'NOT CAUGHT': 1}

    00_b7946098b1_alphablend_row_past_plane
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the last row's vertical average read the alpha row below the plane's last, inside the frame's one AVBuffer
