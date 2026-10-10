# memcached allocator-repros on stock CheriBSD, fixed and buggy in one boot -- 2026-10-10 (R1)

Pre-registered at `dba29812b295` (docs/history/10-10-2026_23-30-00_whole-corpus-audit-cross-program.md, R1),
pushed before the run. Every prediction held: 18 of 18 arms.

    python3 capstone/bug-corpora/memcached/allocator-repros/runners/cheribsd/run-defects.py <build> <out> \
      --sdk ~/cheri/output/sdk --rootfs ~/cheri/rootfs-purecap --image ~/cheri/output/cheribsd-riscv64-purecap.img \
      --runtime-revocation on

Built with `shared/build-cases.sh cheribsd` (event value 1 selects the fixed sequence). `verdicts.json` beside
this file is the runner's own record; no raw log is committed.

Platform: qemu 16135483052dfdd6, firmware f0e1fe57b0f85075, kernel 8ab453f46dc76cf2, libc fdce2289224bb519, image 0cb16209c16c5edf; runtime revocation on, guest default preserved.

Revocation control: passed=True, SIGPROT si_code 2, addr = pc = the resolved mc_defect_read (0x101e42).

| case | fixed | buggy | buggy fault | attributed to | program sha256 |
|---|---|---|---|---|---|
| 00 | complete | as predicted (complete) | none |  | 9c4e076cdb8b1c9a |
| 01 | complete | as predicted (complete) | none |  | 487440b18cf68885 |
| 02 | complete | as predicted (complete) | none |  | c840b52d112882c3 |
| 03 | complete | as predicted (complete) | none |  | 791d81eace089488 |
| 04 | complete | as predicted (complete) | none |  | 4a36197b7c25fdca |
| 05 | complete | as predicted (complete) | none |  | 989282a901dd3c34 |
| 06 | complete | as predicted (complete) | none |  | 5a046f1c2d995268 |
| 07 | complete | as predicted (complete) | none |  | 3c6b2571f7a56017 |
| 08 | complete | as predicted (fault) | SIGPROT si_code 1 pc=0x10553a | mc_case_body+0x1e8 | 65aafbe629c2cd79 |

Case 08's fault lies in `mc_case_body`, the site its case.json declared before the run (`fault_sites`):
the defect's own leading-space scan, faulting on the first byte past the 16 KiB cache.c object before
`mark(8)`. Its fixed arm, the same program, completes. Until this run the catch was a pc and an exit status
with no fixed arm and no attribution.
