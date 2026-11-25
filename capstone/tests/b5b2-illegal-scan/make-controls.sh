#!/usr/bin/env bash
# make-controls.sh OUT CLANG: the three controls for scan_illegal.py, written into OUT.
#   ctl.o       24 planted illegal words (one group 2 bytes off after a c.nop) and 14 legal ones;
#               expected ILLEGAL=24 and every legal word counted legal
#   empty.o     no executable bytes; expected NO-DATA, exit 2
#   gl-noimg.o / gl-img.o  a global access compiled by CLANG without and with -capstone-image-gp;
#               expected DELIN=2 and DELIN=0 (C-27: the image-gp ABI drops the delin)
# The scanner exits 1 on ctl.o and gl-noimg.o; that is the control firing, not a failure.
set -euo pipefail
out=${1:?output directory} clang=${2:?clang}
mkdir -p "$out"
python3 - "$out/ctl.S" <<'EOF'
import sys
def r(f7, rs2, rs1, f3, rd, op): return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op
bad = [r(f7, 6, 5, 1, 10, 0x5b) for f7 in (0x00, 0x02, 0x03, 0x06, 0x07, 0x08, 0x09, 0x0b, 0x0d, 0x20, 0x21, 0x22,
                                           0x24, 0x50, 0x53, 0x7f)]
bad += [r(0x52, 6, 5, 1, 0, 0x5b), r(0x54, 6, 5, 1, 10, 0x5b), r(0, 0, 5, 5, 1, 0x5b), r(0, 0, 5, 7, 10, 0x5b),
        r(0, 0, 0, 0, 0, 0x7b), r(0x7c, 0, 5, 0, 0, 0x0b), r(0, 0, 0, 0, 0, 0x2b),
        (0x7c4 << 20) | (5 << 15) | (1 << 12) | 0x73]
good = [r(0x01, 6, 5, 1, 10, 0x5b), r(0x04, 6, 5, 1, 10, 0x5b), r(0x05, 6, 5, 1, 10, 0x5b), r(0x0a, 0, 5, 1, 10, 0x5b),
        r(0x0c, 6, 5, 1, 10, 0x5b), r(0x51, 6, 5, 1, 10, 0x5b), r(0x52, 0, 5, 1, 0, 0x5b), r(0x54, 0, 5, 1, 10, 0x5b),
        r(0, 0, 5, 0, 10, 0x5b), r(0, 0, 5, 2, 10, 0x5b), r(0, 0, 5, 3, 10, 0x5b), r(0, 6, 5, 4, 0, 0x5b),
        r(0, 0, 5, 6, 10, 0x5b), (0x300 << 20) | (5 << 15) | (2 << 12) | (10 << 7) | 0x73]
with open(sys.argv[1], 'w') as f:
    f.write('.text\n')
    f.writelines('.4byte %#x\n' % w for w in bad[:-6])
    f.write('.2byte 0x0001\n')
    f.writelines('.4byte %#x\n' % w for w in bad[-6:] + good)
EOF
printf '.data\n.4byte 0\n' > "$out/empty.S"
riscv64-unknown-elf-gcc -c "$out/ctl.S" -o "$out/ctl.o"
riscv64-unknown-elf-gcc -c "$out/empty.S" -o "$out/empty.o"
printf 'int g[4];\nint f(int i){ return g[i]; }\nint main(void){ g[1]=3; return f(1); }\n' > "$out/gl.c"
flags=(-target capstone64-unknown-elf -O1 -c -mllvm -capstone-gp-free -Xclang -target-abi -Xclang lp64)
for feature in +f +d +movc-keeps-integer-source +c +cap-base-rvc; do flags+=(-Xclang -target-feature -Xclang "$feature"); done
"$clang" "${flags[@]}" "$out/gl.c" -o "$out/gl-noimg.o"
"$clang" "${flags[@]}" -mllvm -capstone-image-gp "$out/gl.c" -o "$out/gl-img.o"
