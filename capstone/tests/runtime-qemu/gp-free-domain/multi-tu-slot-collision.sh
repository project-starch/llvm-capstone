#!/usr/bin/env bash
# Reproduce, in seconds, the limit that blocks any multi-object domain under the
# gp-captable ABI: cap-table slot indices are TU-LOCAL and there is no relocation
# for them, so two translation units both address slots 0,1,2 and collide.
#
# Found while taking mruby to gp-captable, where it appears as 39 descriptor
# fragments totalling 2670 globals of which the glue initializes the first 5.
# That takes a full archive plus a ten-minute build to see; this takes two files.
#
# It then shows the REMEDY, which needs no compiler change: the descriptor is
# emitted per MODULE, so a full-LTO link presents one module, one descriptor, and
# globally unique slots. Measured here as reada using slots 0,1,2 and readb using
# 3,4,5 out of a single 6-record descriptor.
#
# THE GLOBALS ARE `volatile` AND THE FUNCTIONS `noinline`, and that is not
# decoration. Without it, LTO internalizes three never-written arrays, proves the
# reads are zero, folds both functions to `movc a0, zero`, and the LTO arm then
# reports no cap-table access at all -- which reads exactly like "the pass did not
# run" and cost a wrong conclusion once already.
#
# Exits 1 while the collision stands in the non-LTO arm. It is a REPRODUCER, not a
# regression test: the day slots carry a relocation, the first arm starts passing
# and this should be kept as the test that they do.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../capstone-test-env.sh"
CLANG=${CAPSTONE_CLANG:?}
LD_LLD=${CAPSTONE_LD_LLD:?}
OD="$CAPSTONE_LLVM_BIN/llvm-objdump"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/a.c" <<'EOF'
volatile int a0[4], a1[4], a2[4];
__attribute__((noinline)) int reada(int i) { return a0[i] + a1[i] + a2[i]; }
EOF
cat > "$WORK/b.c" <<'EOF'
volatile int b0[4], b1[4], b2[4];
__attribute__((noinline)) int readb(int i) { return b0[i] + b1[i] + b2[i]; }
EOF
cat > "$WORK/main.c" <<'EOF'
int reada(int), readb(int);
void domain_main(unsigned *res, unsigned func) { (void)func; *res = reada(0) + readb(0); }
EOF

for f in a b main; do
  "$CLANG" -target capstone64-unknown-elf -ffreestanding -fno-jump-tables \
    -mllvm -capstone-gp-captable -O1 -c "$WORK/$f.c" -o "$WORK/$f.o" || exit 2
done
"$CLANG" -target capstone64-unknown-elf -ffreestanding -DCAPSTONE_GLUE_YIELD=1 \
  -c "$SCRIPT_DIR/../silicon-ladder/start-gp-captable-interp.S" -o "$WORK/start.o" || exit 2
"$CLANG" -target capstone64-unknown-elf -ffreestanding \
  -c "$SCRIPT_DIR/../gct-section-end.S" -o "$WORK/gct.o" || exit 2
sed "s/0x10000 + 0x1000/0x10000 + 0x10000/" "$SCRIPT_DIR/link-gpfree.ld" > "$WORK/link.ld"
"$LD_LLD" -T "$WORK/link.ld" -o "$WORK/t.dom" \
  "$WORK/start.o" "$WORK/main.o" "$WORK/a.o" "$WORK/b.o" "$WORK/gct.o" || exit 2

# python3, not grep: this has to COUNT and to mean something when it counts zero.
CAPSTONE_OD="$OD" python3 - "$WORK/t.dom" <<'PY'
import io, os, re, struct, subprocess, sys
dom = sys.argv[1]
od = os.environ["CAPSTONE_OD"]

dis = subprocess.run([od, "-d", "--disassemble-symbols=reada,readb", dom],
                     capture_output=True, text=True).stdout
if not dis.strip():
    sys.exit("no disassembly for reada/readb -- the check learned nothing")

slots, cur = {}, None
for line in dis.splitlines():
    m = re.match(r"^[0-9a-f]+ <(\w+)>:", line.strip())
    if m:
        cur = m.group(1)
        continue
    m = re.search(r"ldc\s+\w+,\s*(0x[0-9a-f]+|\d+)\(gp\)", line)
    if m and cur:
        slots.setdefault(cur, []).append(int(m.group(1), 0) // 16)

for fn in ("reada", "readb"):
    if not slots.get(fn):
        sys.exit(f"{fn} makes no cap-table access -- the check learned nothing")
    print(f"  {fn:6s} reads cap-table slots {slots[fn]}")

shared = sorted(set(slots["reada"]) & set(slots["readb"]))
print(f"  descriptor fragments and their counts: ", end="")

blob = io.open(dom, "rb").read()
sec = subprocess.run([od.replace("objdump", "readelf"), "-SW", dom],
                     capture_output=True, text=True).stdout
off = size = None
for line in sec.splitlines():
    m = re.match(r"\s*\[\s*\d+\]\s+\.capstone_gp_initdesc\s+\S+\s+[0-9a-f]+"
                 r"\s+([0-9a-f]+)\s+([0-9a-f]+)", line)
    if m:
        off, size = int(m.group(1), 16), int(m.group(2), 16)
if off is None:
    sys.exit("no .capstone_gp_initdesc -- not a gp-captable image")
d, pos, frags = blob[off:off + size], 0, []
while pos + 32 <= len(d):
    built, count = struct.unpack_from("<QQ", d, pos)
    if count == 0 and built == 0:
        pos += 16
        continue
    frags.append(count)
    pos = (pos + 32 + 24 * count + 15) // 16 * 16
print(frags, f"-> the glue builds {frags[0]} slots from the FIRST header only")

if not shared:
    print("PASS: the two translation units use disjoint slots.")
    sys.exit(0)
print(f"REPRODUCED: slots {shared} are used by BOTH translation units for "
      f"different globals.")
print("  Cause: the slot index is an immediate baked in at compile time. The "
      "objects carry")
print("  NO relocation for it, so the linker cannot renumber, and a merged "
      "descriptor alone")
print("  would not help -- every TU would still address the same low slots.")
sys.exit(1)
PY
NONLTO=$?

echo
echo "== the remedy: one module, via full LTO"
for f in a b main; do
  "$CLANG" -target capstone64-unknown-elf -ffreestanding -fno-jump-tables \
    -flto -mllvm -capstone-gp-captable -O1 -c "$WORK/$f.c" -o "$WORK/$f.lto.o" || exit 2
done
# The pass is a cl::opt read at CODEGEN time, and under LTO codegen happens in the
# LINKER, so a -mllvm at compile time does not reach it and --plugin-opt does.
# Both are passed deliberately: the compile-time one still governs the IR-level
# passes that consult the same option.
"$LD_LLD" -T "$WORK/link.ld" -o "$WORK/t-lto.dom" \
  "$WORK/start.o" "$WORK/main.lto.o" "$WORK/a.lto.o" "$WORK/b.lto.o" "$WORK/gct.o" \
  --plugin-opt=-capstone-gp-captable || exit 2

CAPSTONE_OD="$OD" python3 - "$WORK/t-lto.dom" <<'LTOPY'
import os, re, subprocess, sys
dom, od = sys.argv[1], os.environ["CAPSTONE_OD"]
dis = subprocess.run([od, "-d", "--disassemble-symbols=reada,readb", dom],
                     capture_output=True, text=True).stdout
if not dis.strip():
    sys.exit("no disassembly for reada/readb -- the check learned nothing")
slots, cur = {}, None
for line in dis.splitlines():
    m = re.match(r"^[0-9a-f]+ <(\w+)>:", line.strip())
    if m:
        cur = m.group(1)
        continue
    m = re.search(r"ldc\s+\w+,\s*(0x[0-9a-f]+|\d+)\(gp\)", line)
    if m and cur:
        slots.setdefault(cur, []).append(int(m.group(1), 0) // 16)
for fn in ("reada", "readb"):
    if not slots.get(fn):
        sys.exit(f"{fn} makes no cap-table access under LTO -- either the pass did not "
                 f"run, or the globals were folded away; see the note on volatile and "
                 f"noinline at the top of this script")
    print(f"  {fn:6s} reads cap-table slots {slots[fn]}")
if set(slots["reada"]) & set(slots["readb"]):
    print("LTO DOES NOT FIX IT: the slots still overlap.")
    sys.exit(1)
print("FIXED BY LTO: the two translation units use disjoint slots, from one descriptor.")
LTOPY
LTO=$?

echo
if [[ $NONLTO -ne 0 && $LTO -eq 0 ]]; then
  echo "SUMMARY: separate objects collide; a full-LTO link does not, and that needs no"
  echo "         compiler change. What LTO does NOT buy is the per-file escape hatch:"
  echo "         instruction selection moves to the link, so a file the backend cannot"
  echo "         select for stops being one bad object and becomes a failed link."
fi
exit "$NONLTO"
