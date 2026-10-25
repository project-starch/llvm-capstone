"""R-55, runtime half: in sbi_capstone.c, every piece the FPGA monitor dereferences (or hands to a domain, whose
glue dereferences it) at its cursor gets that cursor set explicitly. SPLIT gives a lower piece its parent's DECODED
start, which a full-scheme encoder rounds below the carve point. Usage: apply-monitor.py <sbi_capstone.c>."""
import sys, re
p = sys.argv[1]; s = open(p).read()
if "R-55" in s:
    sys.exit("already patched")
G0, G1 = "#ifdef CAPSTONE_TARGET_FPGA\n", "#endif\n"
# 1. create_domain: the initialized-globals copy reads dom_code and writes dom_data, both cursor-relative.
old = "        for(ci = 0; ci < glob_c; ci += 1)\n            dom_data[ci] = dom_code[gpoff_c + ci];\n"
assert s.count(old) == 1, "copy loop"
new = (G0 +
       "        /* R-55: dom_code is the seal SPLIT's lower piece and dom_data may be the descriptor SPLIT's, so each\n"
       "           cursor is its piece's DECODED start; under a full-scheme encoder that is below the carve point. */\n"
       "        C_SET_CURSOR(dom_code, dom_code, base_addr);\n"
       "        C_SET_CURSOR(dom_data, dom_data, base_addr + split_size + data_off);\n" + G1 + old)
s = s.replace(old, new)
# 2. create_region: the region is stored, and shared_region_annotated later hands it to a domain at this cursor.
m = re.search(r"(static unsigned create_region\(unsigned base, unsigned len\) \{.*?    void \*region = split_out_cap\(base, len, 1\);\n)", s, re.S)
assert m, "create_region"
s = s[:m.end()] + G0 + "    C_SET_CURSOR(region, region, base);   /* R-55: handed to domains at this cursor */\n" + G1 + s[m.end():]
# 3. share_child_region: body is handed to the child at its cursor (the glue uses it as its pointer).
old3 = "        tail = __split(body, cend); /* body -> [cstart,cend); tail -> [cend,pend) */\n    }\n"
assert s.count(old3) == 1, "share_child_region tail split"
s = s.replace(old3, old3 + G0 + "    C_SET_CURSOR(body, body, cstart);   /* R-55: the child is handed body at this cursor */\n" + G1)
open(p, "w").write(s); print("R-55 runtime patch applied (3 sites):", p)
