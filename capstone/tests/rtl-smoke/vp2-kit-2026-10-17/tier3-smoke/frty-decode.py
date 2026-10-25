# frty-decode.py <ko> <binutils prefix> <page-off hex>...: instructions at each page offset in the module's .text, with
# the function+offset and the previous word; flags a previous word that is a CSRUNV (funct7 0x24) or CSRETIRE (0x53).
import subprocess, re, sys
ko, b = sys.argv[1], sys.argv[2]; offs = {int(x, 16) for x in sys.argv[3:]}
syms = sorted((int(l.split()[0], 16), l.split()[2]) for l in subprocess.run([b + "nm", "--defined-only", ko], capture_output=True, text=True).stdout.splitlines() if len(l.split()) == 3 and l.split()[1] in "tT")
out = subprocess.run([b + "objdump", "-d", "-j", ".text", ko], capture_output=True, text=True).stdout
rows = [(int(m.group(1), 16), m.group(2), m.group(3).strip()) for m in re.finditer(r"^\s+([0-9a-f]+):\s+([0-9a-f]{4,8})\s+(.*)$", out, re.M)]
def fn(a):
    c = [s for s in syms if s[0] <= a]; return f"{c[-1][1]}+{a - c[-1][0]:#x}" if c else "?"
for i, (a, w, t) in enumerate(rows):
    if (a & 0xfff) in offs:
        pw = int(rows[i - 1][1], 16); tag = ""
        if len(rows[i - 1][1]) == 8 and (pw & 0x7f) == 0x5b and (pw >> 25) in (0x24, 0x53): tag = {0x24: "  <== after CSRUNV", 0x53: "  <== after CSRETIRE"}[pw >> 25]
        print(f"{a:#06x} {fn(a):26s} prev {rows[i-1][1]:>8s}{tag}")
