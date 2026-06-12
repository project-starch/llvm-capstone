#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_ndes_capstone.dom}

NDES_SRC=$BEEBS_SRC_DIR/src/ndes/libndes.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_NDES_SRC=$OUT_DIR/libndes_capstone.c

if [[ ! -f "$NDES_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS ndes source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Avoid by-value aggregate loads in the current backend by passing the small
# benchmark structs through pointers in the generated source.
awk '
  { print }
  $0 == "unsigned long bit[33];" {
    print ""
    print "#define CAPSTONE_DELIN(rd) \\"
    print "  __asm__ volatile (\".insn r 0x5b, 0x1, 0x3, %0, x0, x0\" : \"+r\"(rd))"
    print ""
    print "#define BEEBS_NDES_BIT_PTR() \\"
    print "  ({ unsigned long *bp = bit; CAPSTONE_DELIN(bp); bp; })"
  }
' "$NDES_SRC" > "$PATCHED_NDES_SRC"
perl -0pi \
  -e 's#void des\(immense inp, immense key, int \* newkey, int isw, immense \* out\);#void des(const immense \*inp, const immense \*key, int \*newkey, int isw, immense \*out);#g;' \
  -e 's#unsigned long getbit\(immense source, int bitno, int nbits\);#unsigned long getbit(const immense \*source, int bitno, int nbits);#g;' \
  -e 's#void cyfun\(unsigned long ir, great k, unsigned long \* iout\);#void cyfun(unsigned long ir, const great \*k, unsigned long \*iout);#g;' \
  -e 's#void des\(immense inp, immense key, int \* newkey, int isw, immense \* out\) \{#void des(const immense \*inp, const immense \*key, int \*newkey, int isw, immense \*out) {#g;' \
  -e 's#unsigned long ic,shifter,getbit\(\);#unsigned long ic, shifter;#g;' \
  -e 's#   int ii,i,j,k;#   long ii, i, j, k;#g;' \
  -e 's#   great pg;\n##g;' \
  -e 's#getbit\(itmp,#getbit(\&itmp,#g;' \
  -e 's#for\(i=1;i<=16;i\+\+\) \{pg = kns\[i\]; ks\(/\* key,\*/ i, \&pg\); kns\[i\] = pg;\}#for (i = 1; i <= 16; i++) ks(/* key,*/ i, \&kns[i]);#g;' \
  -e 's#cyfun\(itmp\.l, kns\[ii\], \&ic\);#cyfun(itmp.l, \&kns[ii], \&ic);#g;' \
  -e 's#unsigned long getbit\(immense source, int bitno, int nbits\) \{#unsigned long getbit(const immense \*source, int bitno, int nbits) {#g;' \
  -e 's#source\.r#source->r#g;' \
  -e 's#source\.l#source->l#g;' \
  -e 's#getbit\(icd,#getbit(\&icd,#g;' \
  -e 's#   int i,j,k,l;#   long i, j, k, l;#g;' \
  -e 's#void cyfun\(unsigned long ir, great k, unsigned long \* iout\) \{#void cyfun(unsigned long ir, const great \*k, unsigned long \*iout) {#g;' \
  -e 's#k\.r#k->r#g;' \
  -e 's#k\.c#k->c#g;' \
  -e 's#k\.l#k->l#g;' \
  -e 's#   int jj,irow,icol,iss,j,l,m;#   long jj, irow, icol, iss, j, l, m;#g;' \
  -e 's#des\(inp, key, \&newkey, isw, \&out\);#des(\&inp, \&key, \&newkey, isw, \&out);#g;' \
  "$PATCHED_NDES_SRC"

perl -0pi \
  -e 's#bit\[1\]=shifter=1L;#BEEBS_NDES_BIT_PTR()[1] = shifter = 1L;#g;' \
  -e 's#bit\[j\] = \(shifter <<= 1\);#BEEBS_NDES_BIT_PTR()[j] = (shifter <<= 1);#g;' \
  -e 's#return bit\[bitno\] & source->r \? 1L : 0L;#return BEEBS_NDES_BIT_PTR()[bitno] \& source->r ? 1L : 0L;#g;' \
  -e 's#return bit\[bitno-nbits\] & source->l \? 1L : 0L;#return BEEBS_NDES_BIT_PTR()[bitno - nbits] \& source->l ? 1L : 0L;#g;' \
  -e 's#   p = bit;#   p = BEEBS_NDES_BIT_PTR();#g;' \
  "$PATCHED_NDES_SRC"

perl -0pi \
  -e 's@(const static char ipc2\[49\]=\{0,14,17,11,24,1,5,3,28,15,6,21,\n   10,23,19,12,4,26,8,16,7,27,20,13,2,41,52,31,\n   37,47,55,30,40,51,45,33,48,44,49,39,56,34,\n   53,46,42,50,36,29,32\};)@$1\n\n#define BEEBS_NDES_IPC1(i) ({ const char *p = ipc1; CAPSTONE_DELIN(p); p[(i)]; })\n#define BEEBS_NDES_IPC2(i) ({ const char *p = ipc2; CAPSTONE_DELIN(p); p[(i)]; })@g;' \
  -e 's@(static great kns\[17\];)@$1\n#define BEEBS_NDES_IP(i) ({ const char *p = ip; CAPSTONE_DELIN(p); p[(i)]; })\n#define BEEBS_NDES_IPM(i) ({ const char *p = ipm; CAPSTONE_DELIN(p); p[(i)]; })\n#define BEEBS_NDES_KNS_PTR(i) ({ great *p = kns; CAPSTONE_DELIN(p); &p[(i)]; })@g;' \
  -e 's@(static char ibin\[16\]=\{0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15\};)@$1\n#define BEEBS_NDES_IET(i) ({ const int *p = iet; CAPSTONE_DELIN(p); p[(i)]; })\n#define BEEBS_NDES_IPP(i) ({ const int *p = ipp; CAPSTONE_DELIN(p); p[(i)]; })\n#define BEEBS_NDES_IS(a, b, c) ({ const char (*p)[4][9] = is; CAPSTONE_DELIN(p); p[(a)][(b)][(c)]; })\n#define BEEBS_NDES_IBIN(i) ({ char *p = ibin; CAPSTONE_DELIN(p); p[(i)]; })@g;' \
  -e 's#ipc1\[j\]#BEEBS_NDES_IPC1(j)#g;' \
  -e 's#ipc1\[k\]#BEEBS_NDES_IPC1(k)#g;' \
  -e 's#ipc2\[j\]#BEEBS_NDES_IPC2(j)#g;' \
  -e 's#ipc2\[k\]#BEEBS_NDES_IPC2(k)#g;' \
  -e 's#ipc2\[l\]#BEEBS_NDES_IPC2(l)#g;' \
  -e 's#ip\[j\]#BEEBS_NDES_IP(j)#g;' \
  -e 's#ip\[k\]#BEEBS_NDES_IP(k)#g;' \
  -e 's#ipm\[j\]#BEEBS_NDES_IPM(j)#g;' \
  -e 's#ipm\[k\]#BEEBS_NDES_IPM(k)#g;' \
  -e 's#\&kns\[i\]#BEEBS_NDES_KNS_PTR(i)#g;' \
  -e 's#\&kns\[ii\]#BEEBS_NDES_KNS_PTR(ii)#g;' \
  -e 's#iet\[j\]#BEEBS_NDES_IET(j)#g;' \
  -e 's#iet\[l\]#BEEBS_NDES_IET(l)#g;' \
  -e 's#iet\[m\]#BEEBS_NDES_IET(m)#g;' \
  -e 's#ipp\[j\]#BEEBS_NDES_IPP(j)#g;' \
  -e 's#is\[icol\]\[irow\]\[jj\]#BEEBS_NDES_IS(icol, irow, jj)#g;' \
  -e 's#ibin\[iss\]#BEEBS_NDES_IBIN(iss)#g;' \
  "$PATCHED_NDES_SRC"

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$PATCHED_NDES_SRC" \
  -o "$OBJ_DIR/beebs_ndes.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_ndes_domain.c" \
  -o "$OBJ_DIR/beebs_ndes_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_ndes.o" \
  "$OBJ_DIR/beebs_ndes_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
