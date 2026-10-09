# memcached/plain-temporal-repros, arm `poisoncap-protected` -- 2026-10-09

Platform: PoisonCap mode 1: the same platform with runtime revocation ON -- poison on free, then revocation (booted on the second host).
Predictions committed before the run: 2852edcdd747. Runtime revocation reported by the ABI control: 1.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds: CHERI_BOUNDARY_READY
    revocation-control: exit 162; SUPERVISE fault signal=34 code=2 addr=0x101dd6 pc=0x101dd6

## Cases: {'CAUGHT': 3}

    00_0d4901071c74_restart_line_freed_at_entry_but_not_cleared
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1022ae INSIDE mct_read_probe [0x10229e, 0x1022ba)
        SUPERVISE fault signal=34 code=3 addr=0x1022ae pc=0x1022ae
    01_e7793811f8c8_logger_write_to_watcher_freed_by_the_poll
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102290 INSIDE mct_write_probe [0x102278, 0x10229c)
        SUPERVISE fault signal=34 code=3 addr=0x102290 pc=0x102290
    02_3bc58f6ea55a_logger_loop_condition_still_reads_the_freed_watcher
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1022b6 INSIDE mct_read_probe [0x1022a6, 0x1022c2)
        SUPERVISE fault signal=34 code=3 addr=0x1022b6 pc=0x1022b6
