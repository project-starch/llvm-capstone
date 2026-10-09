# ffmpeg/plain-temporal-repros, arm `poisoncap-protected` -- 2026-10-09

Platform: PoisonCap mode 1: the same platform with runtime revocation ON -- poison on free, then revocation (booted on the second host).
Predictions committed before the run: 2852edcdd747. Runtime revocation reported by the ABI control: 1.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds: CHERI_BOUNDARY_READY
    revocation-control: exit 162; SUPERVISE fault signal=34 code=2 addr=0x101dd6 pc=0x101dd6

## Cases: {'CAUGHT': 12, 'NOT-REISSUED': 1}

    00_716d2a47c565_ops_dispatch_interior_alias_after_free
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1022a0 INSIDE fft_read_probe [0x102290, 0x1022ac)
        SUPERVISE fault signal=34 code=3 addr=0x1022a0 pc=0x1022a0
    01_c98810ab47fa_hw_base_encode_list_walk_reads_freed_link
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10228e INSIDE fft_read_probe [0x10227e, 0x10229a)
        SUPERVISE fault signal=34 code=3 addr=0x10228e pc=0x10228e
    02_43de8b328b62_lzf_write_cursor_stale_after_realloc
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102392 INSIDE fft_write_probe [0x10237a, 0x10239e)
        SUPERVISE fault signal=34 code=3 addr=0x102392 pc=0x102392
    03_dc87758775e2_aviobuf_checksum_ptr_stale_after_realloc
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1022aa INSIDE fft_read_probe [0x10229a, 0x1022b6)
        SUPERVISE fault signal=34 code=3 addr=0x1022aa pc=0x1022aa
    04_4b2248594c7f_jpeg2000_packed_headers_stream_stale_base
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1023e0 INSIDE fft_read_probe [0x1023d0, 0x1023ec)
        SUPERVISE fault signal=34 code=3 addr=0x1023e0 pc=0x1023e0
    05_d6458f6a8bf1_aacdec_tag_che_map_keeps_freed_element
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1022ba INSIDE fft_read_probe [0x1022aa, 0x1022c6)
        SUPERVISE fault signal=34 code=3 addr=0x1022ba pc=0x1022ba
    06_e8714f6f93d1_h264_delayed_pic_holds_interior_pointers
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1022ce INSIDE fft_write_probe [0x1022b6, 0x1022da)
        SUPERVISE fault signal=34 code=3 addr=0x1022ce pc=0x1022ce
    07_a43e9cdd442b_isom_extradata_freed_before_callee_frees_it
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102262 INSIDE fft_read_probe [0x102252, 0x10226e)
        SUPERVISE fault signal=34 code=3 addr=0x102262 pc=0x102262
    08_8a4ea9644833_diracdec_realloc_on_the_wrong_field
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10240a INSIDE fft_read_probe [0x1023fa, 0x102416)
        SUPERVISE fault signal=34 code=3 addr=0x10240a pc=0x10240a
    09_265731f201f1_tx_subcontext_field_left_dangling_on_failure
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1022a6 INSIDE fft_read_probe [0x102296, 0x1022b2)
        SUPERVISE fault signal=34 code=3 addr=0x1022a6 pc=0x1022a6
    10_e7a65142b972_aacpsy_clears_the_local_not_the_field
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1022ae INSIDE fft_read_probe [0x10229e, 0x1022ba)
        SUPERVISE fault signal=34 code=3 addr=0x1022ae pc=0x1022ae
    11_ba28222a14ab_ratecontrol_expr_field_not_cleared
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10226e INSIDE fft_read_probe [0x10225e, 0x10227a)
        SUPERVISE fault signal=34 code=3 addr=0x10226e pc=0x10226e
    12_2e04d35c69e6_vlc_buf_freed_by_callee_then_read
        fixed: exit 0 VERDICT FIXED
        buggy: exit 1 -> NOT-REISSUED
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x30)

## Against the prediction

Case 12 did NOT fault, refuting its pre-registered SIGPROT: the stale read completed and returned 0x30,
neither the freed object's 0x11 nor the new allocation's 0xAA. N = 1; the mechanism is not established.
The other twelve faulted at the labelled probe with si_code 3.
