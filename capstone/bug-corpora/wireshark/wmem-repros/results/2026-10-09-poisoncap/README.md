# wireshark/wmem-repros on PoisonCap, all 22 cases, both modes -- 2026-10-09

The wmem port's PoisonCap backend (ports/wireshark/wmem/host/cheribsd/poisoncap/build.sh --poisoncap --corpus),
the rebuilt published platform, one boot, guest revocation on as in the 2026-09-21 run:

    python3 ports/wireshark/wmem/host/cheribsd/poisoncap/run.py <build> <out> --modes 0,1 --runtime-revocation on --sdk ... --rootfs ... --image ...

Controls: cheribsd-abi (CHERI_ABI pointer_bytes=16 runtime_revocation=1), cheribsd-bounds, allocator-example -- all PASS.

Cases 0-12 re-measure the 2026-09-21 reading; 13-21 (the spatial cases) are their first.

    case mode  exit  reading                              counters
       0    0     0  completes                             mode=0 sweeps=0 poison_bytes=0 epochs=1 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
       0    1   162  SIGPROT si_code 2 pc=0x104164 AT wm_defect_probe mode=1 sweeps=1 poison_bytes=2097152 epochs=1 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
       1    0     0  completes                             mode=0 sweeps=0 poison_bytes=0 epochs=1 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
       1    1   162  SIGPROT si_code 2 pc=0x1041b0 AT wm_defect_probe mode=1 sweeps=1 poison_bytes=2097152 epochs=1 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
       2    0     0  completes                             mode=0 sweeps=0 poison_bytes=0 epochs=1 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
       2    1   162  SIGPROT si_code 2 pc=0x104100 AT wm_defect_probe mode=1 sweeps=1 poison_bytes=2097152 epochs=1 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
       3    0     0  completes                             mode=0 sweeps=0 poison_bytes=0 epochs=1 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
       3    1   162  SIGPROT si_code 2 pc=0x104100 AT wm_defect_probe mode=1 sweeps=1 poison_bytes=2097152 epochs=1 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
       4    0     0  completes                             mode=0 sweeps=0 poison_bytes=0 epochs=1 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
       4    1   162  SIGPROT si_code 2 pc=0x104154 AT wm_defect_probe mode=1 sweeps=1 poison_bytes=2097152 epochs=1 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
       5    0     0  completes                             mode=0 sweeps=0 poison_bytes=0 epochs=1 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
       5    1   162  SIGPROT si_code 2 pc=0x1040b6 AT wm_defect_probe mode=1 sweeps=1 poison_bytes=2097152 epochs=1 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
       6    0     0  completes                             mode=0 sweeps=0 poison_bytes=0 epochs=1 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
       6    1   162  SIGPROT si_code 2 pc=0x104110 AT wm_defect_probe mode=1 sweeps=1 poison_bytes=2097152 epochs=1 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
       7    0     0  completes                             mode=0 sweeps=0 poison_bytes=0 epochs=1 released_chunks=0 region_releases=0 regions=8 unrepresentable=0 pointer_bytes=16
       7    1   162  SIGPROT si_code 2 pc=0x104150 AT wm_defect_probe mode=1 sweeps=1 poison_bytes=2097152 epochs=1 released_chunks=0 region_releases=0 regions=8 unrepresentable=0 pointer_bytes=16
       8    0     0  completes                             mode=0 sweeps=0 poison_bytes=0 epochs=1 released_chunks=0 region_releases=0 regions=8 unrepresentable=0 pointer_bytes=16
       8    1   162  SIGPROT si_code 2 pc=0x104188 AT wm_defect_probe mode=1 sweeps=1 poison_bytes=2097152 epochs=1 released_chunks=0 region_releases=0 regions=8 unrepresentable=0 pointer_bytes=16
       9    0     0  completes                             mode=0 sweeps=0 poison_bytes=0 epochs=1 released_chunks=0 region_releases=0 regions=8 unrepresentable=0 pointer_bytes=16
       9    1   162  SIGPROT si_code 2 pc=0x104148 AT wm_defect_probe mode=1 sweeps=1 poison_bytes=2097152 epochs=1 released_chunks=0 region_releases=0 regions=8 unrepresentable=0 pointer_bytes=16
      10    0     0  completes                             mode=0 sweeps=0 poison_bytes=0 epochs=1 released_chunks=0 region_releases=0 regions=8 unrepresentable=0 pointer_bytes=16
      10    1   162  SIGPROT si_code 2 pc=0x1040fc AT wm_defect_probe mode=1 sweeps=1 poison_bytes=2097152 epochs=1 released_chunks=0 region_releases=0 regions=8 unrepresentable=0 pointer_bytes=16
      11    0     0  completes                             mode=0 sweeps=0 poison_bytes=0 epochs=1 released_chunks=0 region_releases=0 regions=8 unrepresentable=0 pointer_bytes=16
      11    1   162  SIGPROT si_code 2 pc=0x104170 AT wm_defect_probe mode=1 sweeps=1 poison_bytes=2097152 epochs=1 released_chunks=0 region_releases=0 regions=8 unrepresentable=0 pointer_bytes=16
      12    0     0  completes                             mode=0 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      12    1   162  SIGPROT si_code 2 pc=0x104194 AT wm_defect_probe mode=1 sweeps=1 poison_bytes=16 epochs=0 released_chunks=1 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      13    0   162  SIGPROT si_code 1 pc=0x10429c AT wm_defect_probe mode=0 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      13    1   162  SIGPROT si_code 1 pc=0x10429c AT wm_defect_probe mode=1 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      14    0   162  SIGPROT si_code 1 pc=0x104208 AT wm_defect_probe mode=0 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      14    1   162  SIGPROT si_code 1 pc=0x104208 AT wm_defect_probe mode=1 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      15    0   162  SIGPROT si_code 1 pc=0x10419e AT wm_defect_probe mode=0 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      15    1   162  SIGPROT si_code 1 pc=0x10419e AT wm_defect_probe mode=1 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      16    0   162  SIGPROT si_code 1 pc=0x1041de NOT AT wm_defect_probe mode=0 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      16    1   162  SIGPROT si_code 1 pc=0x1041de NOT AT wm_defect_probe mode=1 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      17    0   162  SIGPROT si_code 1 pc=0x1041a0 NOT AT wm_defect_probe mode=0 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      17    1   162  SIGPROT si_code 1 pc=0x1041a0 NOT AT wm_defect_probe mode=1 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      18    0   162  SIGPROT si_code 1 pc=0x104248 NOT AT wm_defect_probe mode=0 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      18    1   162  SIGPROT si_code 1 pc=0x104248 NOT AT wm_defect_probe mode=1 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      19    0   162  SIGPROT si_code 1 pc=0x104204 AT wm_defect_probe mode=0 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      19    1   162  SIGPROT si_code 1 pc=0x104204 AT wm_defect_probe mode=1 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      20    0   162  SIGPROT si_code 1 pc=0x104236 NOT AT wm_defect_probe mode=0 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      20    1   162  SIGPROT si_code 1 pc=0x104236 NOT AT wm_defect_probe mode=1 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      21    0   162  SIGPROT si_code 1 pc=0x1041aa NOT AT wm_defect_probe mode=0 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
      21    1   162  SIGPROT si_code 1 pc=0x1041aa NOT AT wm_defect_probe mode=1 sweeps=0 poison_bytes=0 epochs=0 released_chunks=0 region_releases=0 regions=7 unrepresentable=0 pointer_bytes=16
