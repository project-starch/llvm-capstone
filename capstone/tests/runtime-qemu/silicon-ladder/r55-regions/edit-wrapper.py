#!/usr/bin/env python3
"""R-55 (regions): cap_env_init records the genesis regions' nominal bounds, turns exact_regions on when the encoder
rounds, ends Linux's region at CAPSTONE_DRAM_TOP and cuts every boot slot exact (FPGA only); the plain FPGA build's
dom_stack grows to 16 KiB for the larger table. argv[1]: the wrapper worktree."""
import sys
root = sys.argv[1]


def edit(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, (path, n, old[:80])
        s = s.replace(old, new)
    open(path, "w").write(s)


edit(root + "/lib/sbi/sbi_capstone_dom.c", [
    ("""void cap_env_init(__linear void *cap0, __linear void *cap1, __linear void *cap2) {
    int i;
""", """void cap_env_init(__linear void *cap0, __linear void *cap1, __linear void *cap2) {
    int i;
#ifdef CAPSTONE_TARGET_FPGA
    unsigned k;
    unsigned boot_n;
    __linear unsigned *cap_above;
    /* R-55 (regions): the genesis regions' NOMINAL bounds (sbi_capstone_init.S: CAPENTER's a0, the boot stack's two
       SPLITs) and every other slot's "not known", set before any carve reads them; never rely on zero-init. The
       full-scheme encoder rounds region 1's decoded base below _cap_text_end (it is the lower piece of a SPLIT of a1
       = [_cap_text_end, 2^40)); the cursorless one does not. That is what turns exact_regions on: on a cursorless
       bitstream (0568f93a9) the table behaves exactly as before. */
    exact_regions = 0;
    if(cap_base(cap1) != _cap_text_end) {
        exact_regions = 1;
    }
    for(k = 0; k < CAPSTONE_MAX_REGION_N; k += 1) {
        region_nbase[k] = 0;
        region_nend[k] = 0;
    }
    region_nbase[0] = 0;
    region_nend[0] = _cap_text_start;
    region_nbase[1] = _cap_text_end;
    region_nend[1] = dom_stack;
    region_nbase[2] = dom_stack_end;
    region_nend[2] = 0x10000000000;
#endif
"""),
    ("""    cap_int_code = __setcursor(cap_int_code, __int_handler_entry_entry);
""", """    cap_int_code = __setcursor(cap_int_code, __int_handler_entry_entry);
#ifdef CAPSTONE_TARGET_FPGA
    /* R-55 (regions): end Linux's region at the DRAM window. The part above it is kept as its own slot, so nothing
       S-mode could reach before is lost, and every remainder carved from Linux's region later is at most 1 GiB, whose
       granule is at most 1 MiB, CMA's alignment cap. Then cut every slot the boot left into exact pieces. Both only
       when the encoder rounds. */
    if(exact_regions != 0) {
        cap_above = split_out_cap(CAPSTONE_DRAM_TOP, 0x10000000000 - CAPSTONE_DRAM_TOP, 1);
        region_append(cap_above, CAPSTONE_DRAM_TOP, 0x10000000000);
        exact_regions = 2;   /* from here on every carve's remainders are cut exact */
        boot_n = region_n;
        for(k = 0; k < boot_n; k += 1) {
            region_make_exact(k);
        }
    }
#endif
"""),
])

S = open(root + "/lib/sbi/sbi_capstone_init.S").read()
BOARD_STACK = """#ifdef CAPSTONE_TARGET_FPGA
.zero 4096*2
#else"""
if S.count(BOARD_STACK) == 1:      # the board line: no supervision variant
    edit(root + "/lib/sbi/sbi_capstone_init.S", [(BOARD_STACK, """#ifdef CAPSTONE_TARGET_FPGA
/* 16 KiB since R-55 (regions, 2026-10-15): the 127-slot region table and its nominal bounds add about 3.5 KiB of
   globals to the 4.84 KiB that 8 KiB held. The firmware's RW still ends below 0x800A0000. */
.zero 4096*4
#else""")])
else:
  edit(root + "/lib/sbi/sbi_capstone_init.S", [
    ("""#ifdef CAPSTONE_SUPERVISED_CALL
.zero 4096*8
#else
.zero 4096*2
#endif""", """#ifdef CAPSTONE_SUPERVISED_CALL
.zero 4096*8
#else
/* 16 KiB since R-55 (regions, 2026-10-15): the 127-slot region table and its nominal bounds add about 3.5 KiB of
   globals to the 4.84 KiB that 8 KiB held. The firmware's RW still ends below 0x800A0000. */
.zero 4096*4
#endif"""),
  ])
