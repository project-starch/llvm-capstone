import sys
p = sys.argv[1]; s = open(p).read()
old = "    unsigned *cap = split_out_cap(SBI_MTIME_ADDR, 8, 0);\n    mtime = cap;\n"
new = ("    unsigned *cap = split_out_cap(SBI_MTIME_ADDR, 8, 0);\n"
       "#ifdef CAPSTONE_TARGET_FPGA\n"
       "    /* FP-respin diagnostic (2026-10-15, private build, never landed): under the full bounds scheme the\n"
       "       first SPLIT's upper piece rounds its base down, and SPLIT gives the lower piece cursor = its decoded\n"
       "       start, so this capability's cursor is not SBI_MTIME_ADDR and handle_exception's *mtime reads another\n"
       "       CLINT register. Report base and cursor, then put the cursor on mtime. */\n"
       "    capstone_report(0x4d544d42 /* MTMB */, cap_base(cap));\n"
       "    capstone_report(0x4d544d43 /* MTMC */, cap_cursor(cap));\n"
       "    cap = __setcursor(cap, SBI_MTIME_ADDR);\n"
       "    capstone_report(0x4d544d46 /* MTMF */, cap_cursor(cap));\n"
       "#endif\n"
       "    mtime = cap;\n")
assert s.count(old) == 1, "anchor"
open(p, "w").write(s.replace(old, new))
print("patched")
