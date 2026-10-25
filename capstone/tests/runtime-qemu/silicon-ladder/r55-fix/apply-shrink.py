"""R-55, the carve's bounds: split_out_cap SHRINKs the piece it returns to exactly [base, base + len). Under a
full-scheme encoder SPLIT rounds a long piece outward, so a carve out of a long region came back covering memory far
below `base` (a REGION_CREATE in CMA decoded from 0x80000000, ~700 MiB, and the module refused to map it). SHRINK
(capstone_flu_unit.anvil func SHRINK) narrows rd in place, keeps its node, and clamps the cursor into the new bounds,
so the cursor also lands on `base`. Usage: apply-shrink.py <sbi_capstone.c>."""
import sys
p = sys.argv[1]; s = open(p).read()
if "R-55 (bounds)" in s:
    sys.exit("already patched")
anchor = "#endif\n\n    __linear void *region_linear;\n    unsigned ty = cap_type(region);\n"
assert s.count(anchor) == 1, "split_out_cap tail"
ins = ("#endif\n"
       "#ifdef CAPSTONE_TARGET_FPGA\n"
       "    /* R-55 (bounds): SPLIT rounds a long piece outward under a full-scheme encoder, so the piece carved above\n"
       "       can cover far more than [base, base + len) (a CMA REGION_CREATE decoded from 0x80000000, ~700 MiB, and\n"
       "       the module refused to map it). SHRINK narrows `region` in place to exactly the request, keeps its\n"
       "       node, and clamps the cursor to base. `region` is __linear, so capstone-c writes the register back to\n"
       "       its home slot after the template (the C_RECLAIM note); checked in the generated assembly. */\n"
       "    __asm__ volatile(\".insn r 0x5b, 0x1, 0x1, %0, %1, %2\" :: \"r\"(region), \"r\"(base), \"r\"(base + len));\n"
       "#endif\n")
s = s.replace(anchor, anchor.replace("#endif\n", ins, 1))
open(p, "w").write(s); print("R-55 (bounds) SHRINK applied:", p)
