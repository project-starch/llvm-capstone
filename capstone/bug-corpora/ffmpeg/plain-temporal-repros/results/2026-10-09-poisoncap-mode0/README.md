# ffmpeg/plain-temporal-repros, arm `poisoncap-spatial` -- 2026-10-09

Platform: PoisonCap mode 0: the rebuilt published PoisonCap platform (CHERI-PURECAP-QEMU-POISON kernel, libc with the mrs poison-retire fix), runtime revocation OFF (booted on the second host).
Predictions committed before the run: 2852edcdd747. Runtime revocation reported by the ABI control: 0.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=0
    cheribsd-bounds: CHERI_BOUNDARY_READY
    revocation-control: exit 0

## Cases: {'NOT CAUGHT': 13}

    00_716d2a47c565_ops_dispatch_interior_alias_after_free
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED comp is an interior pointer into the freed pass struct, so reading comp->backend->flags reads storage that now belongs to another obje
    01_c98810ab47fa_hw_base_encode_list_walk_reads_freed_link
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the loop's increment read pic->next out of the node it had just freed, so the walk continues through storage that now belongs to anoth
    02_43de8b328b62_lzf_write_cursor_stale_after_realloc
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the write cursor was not rebased after av_reallocp moved the buffer, so the copy writes into storage that now belongs to another objec
    03_dc87758775e2_aviobuf_checksum_ptr_stale_after_realloc
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED checksum_ptr still pointed into the IO buffer that the seekback growth freed, so the checksum update reads storage that now belongs to
    04_4b2248594c7f_jpeg2000_packed_headers_stream_stale_base
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the packed-headers reader kept the base of the block av_realloc freed, so it reads storage that now belongs to another object while pa
    05_d6458f6a8bf1_aacdec_tag_che_map_keeps_freed_element
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED tag_che_map still named the channel element after av_freep cleared only the owning pointer, so the tag lookup returns storage that now
    06_e8714f6f93d1_h264_delayed_pic_holds_interior_pointers
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED delayed_pic held interior pointers into the DPB block that av_freep released, so setting ->reference writes into storage that now belo
    07_a43e9cdd442b_isom_extradata_freed_before_callee_frees_it
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED av_free left st->codecpar->extradata pointing at the released buffer, so ff_get_extradata releases the same allocation a second time
    08_8a4ea9644833_diracdec_realloc_on_the_wrong_field
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the realloc consumed s->thread_buf while publishing the result as s->slice_params_buf, so thread_buf is left naming storage that now b
    09_265731f201f1_tx_subcontext_field_left_dangling_on_failure
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the failure path freed the subcontext through a local copy, so s->sub kept the released address and teardown reaches storage that now 
    10_e7a65142b972_aacpsy_clears_the_local_not_the_field
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED av_freep cleared the local pctx while ctx->model_priv_data kept the released address, so the teardown reaches storage that now belongs
    11_ba28222a14ab_ratecontrol_expr_field_not_cleared
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED av_expr_free does not clear its argument, so a second run of the same uninit reaches storage that now belongs to another object
    12_2e04d35c69e6_vlc_buf_freed_by_callee_then_read
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED vlc_common_end freed the scratch table because it differed from the fallback it was handed, and the next call read it
