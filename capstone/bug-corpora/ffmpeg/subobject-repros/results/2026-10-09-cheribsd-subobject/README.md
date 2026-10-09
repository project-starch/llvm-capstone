# ffmpeg/subobject-repros, arm `cheribsd-subobject` -- 2026-10-09

Platform: stock CheriBSD purecap, kernel CHERI-PURECAP-QEMU, revocation on, every case built with -Xclang -cheri-bounds=subobject-safe (booted on the second host).
Predictions committed before the run: 886abf4cc336. Runtime revocation reported by the ABI control: 1.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds: CHERI_BOUNDARY_READY
    subobj-control: exit 162

## Cases: {'CAUGHT': 8, 'NOT CAUGHT': 2}

    00_8864fd0aec_cbs_h265_pic_timing_member
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT
    01_68845e26f7_vulkan_hevc_refpicset_member
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT
    02_e058af88ab_vulkan_hevc_dpb_member
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT
    03_89de2f0de1_aac_arith_last_member
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT
    04_1a00ea51cb_rtsp_control_url_underflow
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED strlen(control_url)-1 wrapped to SIZE_MAX and the read landed one byte before control_url, on interleaved_max, inside one allocation
    05_d29ff88422_vulkan_av1_tile_sizes
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT
    06_a809a784ec_vvc_entry_point_start_ctu
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT
    07_275e217b10_hlsenc_key_uri_strlcpy_size
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT
    08_fb862976df_cbs_h266_col_width_val
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT
    09_ac59fc542f_thumbnail_carved_histogram_slice
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED a 10-bit sample was used unshifted as an index, so the bump landed 768 entries past its 256-entry carved sub-slice, inside one av_call

## Against the prediction

886abf4cc336 predicted CAUGHT for 00-08 and NOT for 09. 04 was NOT caught, refuting its row; the
other nine held. Why CHERI's field bound did not apply to 04's read (one byte before control_url) is
not established by this run.
