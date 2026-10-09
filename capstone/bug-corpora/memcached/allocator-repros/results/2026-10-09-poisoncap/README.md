# memcached/allocator-repros on PoisonCap, all 9 cases, both modes -- 2026-10-09

The port's PoisonCap adapter (shared/build-cases.sh poisoncap), the rebuilt published platform, one boot:

    python3 runners/poisoncap/run-defects.py <build> <out> --sdk ... --rootfs ... --image ... --disable-default-revocation

Platform controls: cheribsd-abi PASS, cheribsd-bounds PASS.
Cases 0-4 re-measure the earlier reading; 5-8 (the spatial cases) are their first. The runner's oracles
were written for the temporal cases, so it scores the spatial rows FAIL; the readings are the fault lines.

    case mode exit  fault
       0    0    0  completes
       0    1  162  SIGPROT si_code 2 pc=0x10646c
       1    0    0  completes
       1    1  162  SIGPROT si_code 2 pc=0x1063f4
       2    0    0  completes
       2    1  162  SIGPROT si_code 2 pc=0x105c7a
       3    0    0  completes
       3    1  162  SIGPROT si_code 2 pc=0x105c5a
       4    0    0  completes
       4    1  162  SIGPROT si_code 2 pc=0x105c62
       5    0  162  SIGPROT (exit 162; supervise line: see below)
       5    1  162  SIGPROT si_code 1 pc=0x105c2e
       6    0    0  completes
       6    1    0  completes
       7    0    0  completes
       7    1    0  completes
       8    0  162  SIGPROT (exit 162; supervise line: see below)
       8    1  162  SIGPROT si_code 1 pc=0x106564

Mode-0 rows of 5 and 8 report exit 162 without a parsed fault; supervise's own lines in those boots read
signal=34 code=1 at pc 0x105c2e (5) and 0x106564 (8), the same as mode 1.
Addresses: case 5's 0x105c2e is 0x100000 + mc_defect_write; case 8's 0x106564 is mc_case_body+0x1e8 (llvm-nm on the binaries).
