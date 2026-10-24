# run-exec.py: image text only from the image file, 26 of 26

[`result.json`](result.json) is `run-exec.py`'s own record of one run, 2026-10-21, on QEMU's virtual
profile, one hart: status `PASS`, 26 of 26 checks (the 25 of the write-denial gate and
`text_from_other_file_refused`). `lines` holds the guest's `@@` lines each check reads and `sha256`
the inputs; `source_sha256` hashes the gate's own sources, which are the files of this commit.

The new check: `memfd-launch text-source` registers the image with `CV_IMAGE` and asks `CV_ADD` for a
64 KiB image grant (reg 3) whose first page is a private read+execute mapping of a file, on three
fresh device handles. The guest line reads `TEXT_SOURCE image=ok other=EACCES noimage=EACCES`: with
the text taken from the image the request is admitted; with the text taken from another file of
identical bytes (a copy, another inode) it is refused; with no image registered it is refused.

## The control: the same gate against a module without the test

[`result-without-cv-add-inode-test.json`](result-without-cv-add-inode-test.json) is the same run
with a module whose `CV_ADD` text test no longer compares the inode (the first of the two
`file_inode(v->vm_file) != file_inode(c->image)` conditions in `capstone_vm.c`, replaced by `0`; a
build copy, not committed). Status `FAIL`, one check false, `text_from_other_file_refused`; the guest
line reads `TEXT_SOURCE image=ok other=ok noimage=EACCES`. Only the other-file answer changes, which
shows the check is the inode test and not the grant being invalid for some other reason (the
`image=ok` answer says the grant is otherwise admissible, and `noimage=EACCES` is refused by the
separate "no image held" condition).

What it was run on, both records: QEMU `bd6cd022b077` (`svadu=false`,
`x-capstone-supervisor-slots=32`), the Image, `fw_jump.elf` and `rootfs.ext2` hashed in the files,
launcher `c513e8895ed2`, the SPEC toolchain v4 clang (an integration build, not dev alone), the helper
(`memfd-launch.c`, static, built with the QEMU buildroot's riscv64 cross compiler, `2d0ca15397e7`).
The module is this tree's `capstone_vm.c` (`8426446326fb`, flat table, built against kernel 6.1.0 with
the supervisor-slot default changed from 1 to 32 in the build copy only); the control's is
`c634a8d6efc2`.

## Not covered

The same inode test in `resolve()`'s text path is not exercised by this check. Reaching it takes a
registered image grant whose text mapping is replaced by a mapping of another file while a domain
runs, and only the launcher starts domains, so a test would need a hook in product code. That path is
covered by reading it, not by a run.
