# ffmpeg/plain-temporal-repros on the Capstone application domain arms -- 2026-10-09

Predictions committed before the run: 60f1c5be25a5 (tools/run-capstone-domain.py, predicted()).

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/ffmpeg/plain-temporal-repros --arm spatial|sublet \
      --sdk <application SDK built with CAPSTONE_APPLICATION_HEAP=level0|sublet> \
      --state <capstone-vm state> --out <fresh dir>

Result LINES only, from each arm's record.json.

## spatial

    platform  Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=level0 runtime 3c72f36acd5b; compiler 7d01722aab88
    env       {'CAPSTONE_GP_NONLIN': '1', 'CAPSTONE_REV_NODES': '65536'}
    tool      0383d3c2caa89fb6   predicted buggy outcome: DEFECT-REPRODUCED
    control clean RETURNED  required RETURNED  CONTROL clean RETURNED 1
    control oob   CAUGHT    required CAUGHT    cause=7 pc=0xc020e5b8 address=0xc0251768 in ctl_write_probe+0x58 (the labelled probe)
    control uaf   RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    00_716d2a47c565_ops_dispatch_interior_alias_after_free             fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED comp is an interior pointer into the freed pass struct, so reading comp->backend->flags reads storage that now belongs to another object  image c7c990e1815924f2
    01_c98810ab47fa_hw_base_encode_list_walk_reads_freed_link          fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the loop's increment read pic->next out of the node it had just freed, so the walk continues through storage that now belongs to another object  image 31398cd38d6625a8
    02_43de8b328b62_lzf_write_cursor_stale_after_realloc               fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the write cursor was not rebased after av_reallocp moved the buffer, so the copy writes into storage that now belongs to another object  image 3d696c5feea31063
    03_dc87758775e2_aviobuf_checksum_ptr_stale_after_realloc           fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED checksum_ptr still pointed into the IO buffer that the seekback growth freed, so the checksum update reads storage that now belongs to another object  image e0e02f21c46a6e16
    04_4b2248594c7f_jpeg2000_packed_headers_stream_stale_base          fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the packed-headers reader kept the base of the block av_realloc freed, so it reads storage that now belongs to another object while passing its own bounds checks  image e08890772929ad20
    05_d6458f6a8bf1_aacdec_tag_che_map_keeps_freed_element             fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED tag_che_map still named the channel element after av_freep cleared only the owning pointer, so the tag lookup returns storage that now belongs to another object  image e35193328333e579
    06_e8714f6f93d1_h264_delayed_pic_holds_interior_pointers           fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED delayed_pic held interior pointers into the DPB block that av_freep released, so setting ->reference writes into storage that now belongs to another object  image a62a73a7451aa120
    07_a43e9cdd442b_isom_extradata_freed_before_callee_frees_it        fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED av_free left st->codecpar->extradata pointing at the released buffer, so ff_get_extradata releases the same allocation a second time  image d55aaf097a7f6932
    08_8a4ea9644833_diracdec_realloc_on_the_wrong_field                fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the realloc consumed s->thread_buf while publishing the result as s->slice_params_buf, so thread_buf is left naming storage that now belongs to another object  image 40607e42509de299
    09_265731f201f1_tx_subcontext_field_left_dangling_on_failure       fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the failure path freed the subcontext through a local copy, so s->sub kept the released address and teardown reaches storage that now belongs to another object  image 918a02d05c8b46ee
    10_e7a65142b972_aacpsy_clears_the_local_not_the_field              fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED av_freep cleared the local pctx while ctx->model_priv_data kept the released address, so the teardown reaches storage that now belongs to another object  image 138fb510dcf7cf85
    11_ba28222a14ab_ratecontrol_expr_field_not_cleared                 fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED av_expr_free does not clear its argument, so a second run of the same uninit reaches storage that now belongs to another object  image 80c7b3d7e2dd6619
    12_2e04d35c69e6_vlc_buf_freed_by_callee_then_read                  fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED vlc_common_end freed the scratch table because it differed from the fallback it was handed, and the next call read it  image 6fa9c236120d77ac

## sublet

    platform  Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=sublet runtime b73508e5005b; compiler 7d01722aab88
    env       {'CAPSTONE_GP_NONLIN': '1', 'CAPSTONE_REV_NODES': '65536'}
    tool      0383d3c2caa89fb6   predicted buggy outcome: CAUGHT
    control clean RETURNED  required RETURNED  CONTROL clean RETURNED 1
    control oob   CAUGHT    required CAUGHT    cause=7 pc=0xc020f70c address=0xc8802018 in ctl_write_probe+0x58 (the labelled probe)
    control uaf   CAUGHT    required CAUGHT    cause=24 pc=0xc020f758 address=0x0 in ctl_read_probe+0x38 (the labelled probe)
    00_716d2a47c565_ops_dispatch_interior_alias_after_free             fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fc24 address=0x0 in fft_read_probe+0x38 (the labelled probe)  image df079e6666110982
    01_c98810ab47fa_hw_base_encode_list_walk_reads_freed_link          fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fc14 address=0x0 in fft_read_probe+0x38 (the labelled probe)  image de9a80fc62906e7c
    02_43de8b328b62_lzf_write_cursor_stale_after_realloc               fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc0210440 address=0x0 in fft_write_probe+0x58 (the labelled probe)  image 5ea896b3363da447
    03_dc87758775e2_aviobuf_checksum_ptr_stale_after_realloc           fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fc90 address=0x0 in fft_read_probe+0x38 (the labelled probe)  image 33e8d3837a25f120
    04_4b2248594c7f_jpeg2000_packed_headers_stream_stale_base          fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc0210488 address=0x0 in fft_read_probe+0x38 (the labelled probe)  image ef1f0291621d434e
    05_d6458f6a8bf1_aacdec_tag_che_map_keeps_freed_element             fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fc64 address=0x0 in fft_read_probe+0x38 (the labelled probe)  image 7bcbf162a3a5278f
    06_e8714f6f93d1_h264_delayed_pic_holds_interior_pointers           fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fcb4 address=0x0 in fft_write_probe+0x58 (the labelled probe)  image 11f9b82ecc88329f
    07_a43e9cdd442b_isom_extradata_freed_before_callee_frees_it        fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fbb8 address=0x0 in fft_read_probe+0x38 (the labelled probe)  image 669db20571081a76
    08_8a4ea9644833_diracdec_realloc_on_the_wrong_field                fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc0210588 address=0x0 in fft_read_probe+0x38 (the labelled probe)  image 4804dccd3e6f02ff
    09_265731f201f1_tx_subcontext_field_left_dangling_on_failure       fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fc18 address=0x0 in fft_read_probe+0x38 (the labelled probe)  image 23dcc054654f7797
    10_e7a65142b972_aacpsy_clears_the_local_not_the_field              fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fc28 address=0x0 in fft_read_probe+0x38 (the labelled probe)  image df86df13bca84e10
    11_ba28222a14ab_ratecontrol_expr_field_not_cleared                 fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fbfc address=0x0 in fft_read_probe+0x38 (the labelled probe)  image cb08e200cf8bc3ab
    12_2e04d35c69e6_vlc_buf_freed_by_callee_then_read                  fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fcd4 address=0x0 in fft_read_probe+0x38 (the labelled probe)  image 0545b61a6ef05e43

