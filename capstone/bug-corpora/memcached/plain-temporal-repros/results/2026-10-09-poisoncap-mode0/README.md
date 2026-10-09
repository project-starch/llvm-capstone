# memcached/plain-temporal-repros, arm `poisoncap-spatial` -- 2026-10-09

Platform: PoisonCap mode 0: the rebuilt published PoisonCap platform (CHERI-PURECAP-QEMU-POISON kernel, libc with the mrs poison-retire fix), runtime revocation OFF (booted on the second host).
Predictions committed before the run: 2852edcdd747. Runtime revocation reported by the ABI control: 0.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=0
    cheribsd-bounds: CHERI_BOUNDARY_READY
    revocation-control: exit 0

## Cases: {'NOT CAUGHT': 3}

    00_0d4901071c74_restart_line_freed_at_entry_but_not_cleared
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the entry free did not clear c->line, and three of the four return paths never republish it, so the next call releases storage that no
    01_e7793811f8c8_logger_write_to_watcher_freed_by_the_poll
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the poll the function called may close and free the watcher, and the flag store that follows writes into storage that now belongs to a
    02_3bc58f6ea55a_logger_loop_condition_still_reads_the_freed_watcher
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED guarding only the write left the loop condition re-reading w->failed_flush and w->buf out of the freed watcher, so the loop cannot ter
