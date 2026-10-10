# memcached/allocator-repros 06 and 07 under carve bounds (MC_CARVE_BOUNDS) -- 2026-10-09

Pre-registered at 83dcfa032845: CAUGHT at the labelled probe on Capstone and on CheriBSD.

## Capstone (application domain, level0 heap; runners/capstone-domain/run-defects.py --cases 6,7 --modes spatial)

The corpus runner's own oracle for this mode is 'complete' -- the arm it was written for -- so it scores
both rows FAIL; the reading is the fault line each boot printed AFTER the case marker, and the probe
addresses the same boot published through that marker:

    case 6: marker 0xcf1c000000000006; probes read=0x1018aa2fc write=0x1018aa388; fault cause=7 pc=0x1018aa388 tval=0xe1afff51 -> AT mc_defect_write; image d51225769219329c
    case 7: marker 0xcf1c000000000007; probes read=0x10186a2fc write=0x10186a388; fault cause=5 pc=0x10186a2fc tval=0xe1afff59 -> AT mc_defect_read; image 203dff92eaa9ae06

## CheriBSD (stock purecap, revocation on; runners/cheribsd/run-defects.py --cases 6,7)

Its oracle for mode 0 is 'completed', so it scores both rows FAIL; the reading is supervise's fault line, and
the probe addresses are read from each case binary (llvm-nm) at the load base supervise reported (0x100000):

    controls: cheribsd-abi exit 0 PASS; cheribsd-bounds exit 162 PASS; revocation-control exit 162 fault {'signal': 34, 'code': 2, 'addr': '0x101e42', 'pc': '0x101e42'}
    case 6: exit 162, SIGPROT si_code 1 (bounds), pc=0x104d5c = 0x100000 + mc_defect_write 0x4d5c -> AT mc_defect_write
    case 7: exit 162, SIGPROT si_code 1 (bounds), pc=0x104d18 = 0x100000 + mc_defect_read 0x4d18 -> AT mc_defect_read
