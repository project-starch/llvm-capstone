# ffmpeg/subobject-repros on the Capstone application domain arms -- 2026-10-09

Predictions committed before the run: 60f1c5be25a5 (tools/run-capstone-domain.py, predicted()).

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/ffmpeg/subobject-repros --arm spatial|sublet \
      --sdk <application SDK built with CAPSTONE_APPLICATION_HEAP=level0|sublet> \
      --state <capstone-vm state> --out <fresh dir> \
      --cc-arg=-std=c11 \
      '--cc-arg=-Daligned_alloc(a,n)=malloc(n)' \
      --cc-arg=-isystem \
      --cc-arg=<compiler build>/lib/clang/22/include \
      --cc-arg=-Icapstone/ports/ffmpeg/buffer-pool/src/shared \
      --cc-arg=-I/tmp/capstone/ffmpeg-buffer-pool/build/native/sources/ffmpeg-ported \
      --cc-arg=-Icapstone/ports/ffmpeg/buffer-pool/cmake/replay-config \
      --cc-arg=-Icapstone/runtime/include \
      --cc-arg=/tmp/capstone/ffmpeg-buffer-pool/build/native/sources/ffmpeg-ported/libavutil/buffer.c \
      --cc-arg=/tmp/capstone/ffmpeg-buffer-pool/build/native/sources/ffmpeg-ported/libavutil/refstruct.c \
      --cc-arg=capstone/ports/ffmpeg/buffer-pool/src/shared/observe-pool-events.c \
      --cc-arg=capstone/ports/ffmpeg/buffer-pool/src/shared/pool-allocator.c \
      --cc-arg=capstone/ports/ffmpeg/buffer-pool/src/shared/metadata-allocator.c \
      --cc-arg=capstone/ports/ffmpeg/buffer-pool/src/native/replay/payload-pointers.c

The SDK pair for this corpus differs from the plain corpora's ONLY in heap capacity, because
shared/driver.c takes 80 MiB of arenas (FF2_PAYLOAD_BYTES + FF2_META_BYTES): build-sdk.sh with
`-DCAPSTONE_APPLICATION_ARENA_BYTES=134217728` (level0) and `-DCAPSTONE_APPLICATION_HEAP_LOG=27
-DCAPSTONE_APPLICATION_GRANT_BYTES=268435456` (sublet). On the default 64 MiB heap every case
refused at its own allocation check (CONTROL-FAILED 604) on both arms -- recorded, not scored.
The ported FFmpeg sources are the buffer-pool port's `native` preset output
(`ports/ffmpeg/buffer-pool`, `<build>/sources/ffmpeg-ported`); the port's patches are unchanged
since 2026-09-20. `aligned_alloc` is compiled as malloc: the application SDK has none.

Result LINES only, from each arm's record.json.

## spatial

    platform  Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=level0 runtime afa624ec6387; compiler 7d01722aab88
    env       {'CAPSTONE_GP_NONLIN': '1', 'CAPSTONE_REV_NODES': '65536'}
    tool      376cc8f440aae03b   predicted buggy outcome: DEFECT-REPRODUCED
    control clean RETURNED  required RETURNED  CONTROL clean RETURNED 1
    control oob   CAUGHT    required CAUGHT    cause=7 pc=0xc8f0e5b8 address=0xc8f51768 in ctl_write_probe+0x58 (the labelled probe)
    control uaf   RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    00_8864fd0aec_cbs_h265_pic_timing_member                           fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED index 600 of num_nalus_in_du_minus1[600] overwrote the low half of du_cpb_removal_delay_increment_minus1[0], inside one refstruct allocation  image f9c13c636723f8ab
    01_68845e26f7_vulkan_hevc_refpicset_member                         fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED nb_refs = 16 wrote 8 bytes past RefPicSetStCurrBefore[8] into RefPicSetStCurrAfter, inside one refstruct allocation  image bc091e2db4f37df6
    02_e058af88ab_vulkan_hevc_dpb_member                               fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the DPB walk wrote ref_src[16], which is h265_refs[0], inside one refstruct allocation  image 74c072afdae7c682
    03_89de2f0de1_aac_arith_last_member                                fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED index 512 of last[512] read the low byte of last_len, inside one allocation, and fed it into the arithmetic-coding context  image ab9892e180e35f34
    04_1a00ea51cb_rtsp_control_url_underflow                           fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED strlen(control_url)-1 wrapped to SIZE_MAX and the read landed one byte before control_url, on interleaved_max, inside one allocation  image ee9215a334358e80
    05_d29ff88422_vulkan_av1_tile_sizes                                fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the pre-loop `> MAX_TILES` check let tileCount reach 256 and the write landed on std_ref's first four bytes, inside one allocation  image e5c55fb5dda33054
    06_a809a784ec_vvc_entry_point_start_ctu                            fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED j was bounded by nothing, so the writes walked 8 words off the end of entry_point_start_ctu and overwrote the next member, inside one allocation  image 9b97adc96d7bd151
    07_275e217b10_hlsenc_key_uri_strlcpy_size                          fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the source length was passed as av_strlcpy's destination size, so the copy ran 16 bytes past key_uri into key_string (15 'A' plus the terminator), inside one allocation  image b3d200b39acc9435
    08_fb862976df_cbs_h266_col_width_val                               fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the loop's guard was keyed to num_tile_columns, which is assigned only after the loop, so 6 writes ran past col_width_val into row_height_val  image bac4d0e72798ef6b
    09_ac59fc542f_thumbnail_carved_histogram_slice                     fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED a 10-bit sample was used unshifted as an index, so the bump landed 768 entries past its 256-entry carved sub-slice, inside one av_calloc  image ca77f632dc998178

## sublet

    platform  Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=sublet runtime e5427101c0a7; compiler 7d01722aab88
    env       {'CAPSTONE_GP_NONLIN': '1', 'CAPSTONE_REV_NODES': '65536'}
    tool      376cc8f440aae03b   predicted buggy outcome: DEFECT-REPRODUCED
    control clean RETURNED  required RETURNED  CONTROL clean RETURNED 1
    control oob   CAUGHT    required CAUGHT    cause=7 pc=0xc020f70c address=0xe0002018 in ctl_write_probe+0x58 (the labelled probe)
    control uaf   CAUGHT    required CAUGHT    cause=24 pc=0xc020f758 address=0x0 in ctl_read_probe+0x38 (the labelled probe)
    00_8864fd0aec_cbs_h265_pic_timing_member                           fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED index 600 of num_nalus_in_du_minus1[600] overwrote the low half of du_cpb_removal_delay_increment_minus1[0], inside one refstruct allocation  image a6b71052fd594c93
    01_68845e26f7_vulkan_hevc_refpicset_member                         fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED nb_refs = 16 wrote 8 bytes past RefPicSetStCurrBefore[8] into RefPicSetStCurrAfter, inside one refstruct allocation  image 3c167c6ab83d40db
    02_e058af88ab_vulkan_hevc_dpb_member                               fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the DPB walk wrote ref_src[16], which is h265_refs[0], inside one refstruct allocation  image 1e569e21ea03e1c8
    03_89de2f0de1_aac_arith_last_member                                fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED index 512 of last[512] read the low byte of last_len, inside one allocation, and fed it into the arithmetic-coding context  image 43da394dedf905bb
    04_1a00ea51cb_rtsp_control_url_underflow                           fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED strlen(control_url)-1 wrapped to SIZE_MAX and the read landed one byte before control_url, on interleaved_max, inside one allocation  image 7cb9961c85360877
    05_d29ff88422_vulkan_av1_tile_sizes                                fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the pre-loop `> MAX_TILES` check let tileCount reach 256 and the write landed on std_ref's first four bytes, inside one allocation  image d5556ed5e81f9006
    06_a809a784ec_vvc_entry_point_start_ctu                            fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED j was bounded by nothing, so the writes walked 8 words off the end of entry_point_start_ctu and overwrote the next member, inside one allocation  image 5bc3afe928214b8d
    07_275e217b10_hlsenc_key_uri_strlcpy_size                          fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the source length was passed as av_strlcpy's destination size, so the copy ran 16 bytes past key_uri into key_string (15 'A' plus the terminator), inside one allocation  image ae3bb05b6097efc7
    08_fb862976df_cbs_h266_col_width_val                               fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the loop's guard was keyed to num_tile_columns, which is assigned only after the loop, so 6 writes ran past col_width_val into row_height_val  image c2e3e7961963d0c1
    09_ac59fc542f_thumbnail_carved_histogram_slice                     fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED a 10-bit sample was used unshifted as an index, so the bump landed 768 entries past its 256-entry carved sub-slice, inside one av_calloc  image 00516aa0e9930483

