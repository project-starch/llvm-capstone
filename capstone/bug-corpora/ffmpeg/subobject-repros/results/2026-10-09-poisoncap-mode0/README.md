# ffmpeg/subobject-repros, arm `poisoncap-spatial` -- 2026-10-09

Platform: PoisonCap mode 0: the rebuilt published PoisonCap platform (CHERI-PURECAP-QEMU-POISON kernel, libc with the mrs poison-retire fix), runtime revocation OFF (booted on the second host).
Predictions committed before the run: the corpus's committed predictions. Runtime revocation reported by the ABI control: 0.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=0
    cheribsd-bounds: CHERI_BOUNDARY_READY

## Cases: {'NOT CAUGHT': 10}

    00_8864fd0aec_cbs_h265_pic_timing_member
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED index 600 of num_nalus_in_du_minus1[600] overwrote the low half of du_cpb_removal_delay_increment_minus1[0], inside one refstruct allo
    01_68845e26f7_vulkan_hevc_refpicset_member
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED nb_refs = 16 wrote 8 bytes past RefPicSetStCurrBefore[8] into RefPicSetStCurrAfter, inside one refstruct allocation
    02_e058af88ab_vulkan_hevc_dpb_member
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the DPB walk wrote ref_src[16], which is h265_refs[0], inside one refstruct allocation
    03_89de2f0de1_aac_arith_last_member
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED index 512 of last[512] read the low byte of last_len, inside one allocation, and fed it into the arithmetic-coding context
    04_1a00ea51cb_rtsp_control_url_underflow
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED strlen(control_url)-1 wrapped to SIZE_MAX and the read landed one byte before control_url, on interleaved_max, inside one allocation
    05_d29ff88422_vulkan_av1_tile_sizes
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the pre-loop `> MAX_TILES` check let tileCount reach 256 and the write landed on std_ref's first four bytes, inside one allocation
    06_a809a784ec_vvc_entry_point_start_ctu
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED j was bounded by nothing, so the writes walked 8 words off the end of entry_point_start_ctu and overwrote the next member, inside one 
    07_275e217b10_hlsenc_key_uri_strlcpy_size
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the source length was passed as av_strlcpy's destination size, so the copy ran 16 bytes past key_uri into key_string (15 'A' plus the 
    08_fb862976df_cbs_h266_col_width_val
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the loop's guard was keyed to num_tile_columns, which is assigned only after the loop, so 6 writes ran past col_width_val into row_hei
    09_ac59fc542f_thumbnail_carved_histogram_slice
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED a 10-bit sample was used unshifted as an index, so the bump landed 768 entries past its 256-entry carved sub-slice, inside one av_call
