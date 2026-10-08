# memcached/plain-temporal-repros on the Capstone application domain arms -- 2026-10-09

Predictions committed before the run: 60f1c5be25a5 (tools/run-capstone-domain.py, predicted()).

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/memcached/plain-temporal-repros --arm spatial|sublet \
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
    00_0d4901071c74_restart_line_freed_at_entry_but_not_cleared        fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the entry free did not clear c->line, and three of the four return paths never republish it, so the next call releases storage that now belongs to another object  image 3922371fed5993c1
    01_e7793811f8c8_logger_write_to_watcher_freed_by_the_poll          fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the poll the function called may close and free the watcher, and the flag store that follows writes into storage that now belongs to another object  image 268497d91ef7c3a0
    02_3bc58f6ea55a_logger_loop_condition_still_reads_the_freed_watcher fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED guarding only the write left the loop condition re-reading w->failed_flush and w->buf out of the freed watcher, so the loop cannot terminate  image 08915597a762a1a5

## sublet

    platform  Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=sublet runtime b73508e5005b; compiler 7d01722aab88
    env       {'CAPSTONE_GP_NONLIN': '1', 'CAPSTONE_REV_NODES': '65536'}
    tool      0383d3c2caa89fb6   predicted buggy outcome: CAUGHT
    control clean RETURNED  required RETURNED  CONTROL clean RETURNED 1
    control oob   CAUGHT    required CAUGHT    cause=7 pc=0xc020f70c address=0xc8802018 in ctl_write_probe+0x58 (the labelled probe)
    control uaf   CAUGHT    required CAUGHT    cause=24 pc=0xc020f758 address=0x0 in ctl_read_probe+0x38 (the labelled probe)
    00_0d4901071c74_restart_line_freed_at_entry_but_not_cleared        fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fc58 address=0x0 in mct_read_probe+0x38 (the labelled probe)  image 3f11820f93a93e1a
    01_e7793811f8c8_logger_write_to_watcher_freed_by_the_poll          fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fc54 address=0x0 in mct_write_probe+0x58 (the labelled probe)  image 13a3c4b6f0b68ac3
    02_3bc58f6ea55a_logger_loop_condition_still_reads_the_freed_watcher fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fcbc address=0x0 in mct_read_probe+0x38 (the labelled probe)  image 6d1210282e45b121

