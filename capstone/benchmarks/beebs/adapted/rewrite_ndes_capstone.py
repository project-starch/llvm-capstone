#!/usr/bin/env python3
"""Generate the Capstone-adapted ndes source.

The upstream benchmark passes small aggregate structs by value and indexes
several static tables through pointers that need explicit delinearization in
the current Capstone runtime/backend combination. This helper keeps those
source-level adaptations out of the shell build script.
"""

import re
import sys


CAPSTONE_DELIN_MACROS = r'''
#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

#define BEEBS_NDES_BIT_PTR() \
  ({ unsigned long *bp = bit; CAPSTONE_DELIN(bp); bp; })
'''

IPC_MACROS = r'''
#define BEEBS_NDES_IPC1(i) ({ const char *p = ipc1; CAPSTONE_DELIN(p); p[(i)]; })
#define BEEBS_NDES_IPC2(i) ({ const char *p = ipc2; CAPSTONE_DELIN(p); p[(i)]; })
'''

IP_KNS_MACROS = r'''
#define BEEBS_NDES_IP(i) ({ const char *p = ip; CAPSTONE_DELIN(p); p[(i)]; })
#define BEEBS_NDES_IPM(i) ({ const char *p = ipm; CAPSTONE_DELIN(p); p[(i)]; })
#define BEEBS_NDES_KNS_PTR(i) ({ great *p = kns; CAPSTONE_DELIN(p); &p[(i)]; })
'''

CYFUN_TABLE_MACROS = r'''
#define BEEBS_NDES_IET(i) ({ const int *p = iet; CAPSTONE_DELIN(p); p[(i)]; })
#define BEEBS_NDES_IPP(i) ({ const int *p = ipp; CAPSTONE_DELIN(p); p[(i)]; })
#define BEEBS_NDES_IS(a, b, c) ({ const char (*p)[4][9] = is; CAPSTONE_DELIN(p); p[(a)][(b)][(c)]; })
#define BEEBS_NDES_IBIN(i) ({ char *p = ibin; CAPSTONE_DELIN(p); p[(i)]; })
'''


def replace_exact(text, old, new):
    if old not in text:
        raise RuntimeError(f"expected source fragment not found: {old!r}")
    return text.replace(old, new)


def main():
    if len(sys.argv) != 3:
        print("usage: rewrite_ndes_capstone.py <src> <dst>", file=sys.stderr)
        return 2

    src_path, dst_path = sys.argv[1], sys.argv[2]
    with open(src_path, encoding="utf-8") as src_file:
        text = src_file.read()

    text = replace_exact(
        text,
        "unsigned long bit[33];",
        "unsigned long bit[33];\n" + CAPSTONE_DELIN_MACROS,
    )

    replacements = [
        (
            "void des(immense inp, immense key, int * newkey, int isw, immense * out);",
            "void des(const immense *inp, const immense *key, int *newkey, int isw, immense *out);",
        ),
        (
            "unsigned long getbit(immense source, int bitno, int nbits);",
            "unsigned long getbit(const immense *source, int bitno, int nbits);",
        ),
        (
            "void cyfun(unsigned long ir, great k, unsigned long * iout);",
            "void cyfun(unsigned long ir, const great *k, unsigned long *iout);",
        ),
        (
            "void des(immense inp, immense key, int * newkey, int isw, immense * out) {",
            "void des(const immense *inp, const immense *key, int *newkey, int isw, immense *out) {",
        ),
        ("unsigned long ic,shifter,getbit();", "unsigned long ic, shifter;"),
        ("   int ii,i,j,k;", "   long ii, i, j, k;"),
        ("   great pg;\n", ""),
        ("getbit(itmp,", "getbit(&itmp,"),
        (
            "for(i=1;i<=16;i++) {pg = kns[i]; ks(/* key,*/ i, &pg); kns[i] = pg;}",
            "for (i = 1; i <= 16; i++) ks(/* key,*/ i, &kns[i]);",
        ),
        ("cyfun(itmp.l, kns[ii], &ic);", "cyfun(itmp.l, &kns[ii], &ic);"),
        (
            "unsigned long getbit(immense source, int bitno, int nbits) {",
            "unsigned long getbit(const immense *source, int bitno, int nbits) {",
        ),
        ("source.r", "source->r"),
        ("source.l", "source->l"),
        ("getbit(icd,", "getbit(&icd,"),
        ("   int i,j,k,l;", "   long i, j, k, l;"),
        (
            "void cyfun(unsigned long ir, great k, unsigned long * iout) {",
            "void cyfun(unsigned long ir, const great *k, unsigned long *iout) {",
        ),
        ("k.r", "k->r"),
        ("k.c", "k->c"),
        ("k.l", "k->l"),
        ("   int jj,irow,icol,iss,j,l,m;", "   long jj, irow, icol, iss, j, l, m;"),
        ("des(inp, key, &newkey, isw, &out);", "des(&inp, &key, &newkey, isw, &out);"),
        ("bit[1]=shifter=1L;", "BEEBS_NDES_BIT_PTR()[1] = shifter = 1L;"),
        ("bit[j] = (shifter <<= 1);", "BEEBS_NDES_BIT_PTR()[j] = (shifter <<= 1);"),
        (
            "return bit[bitno] & source->r ? 1L : 0L;",
            "return BEEBS_NDES_BIT_PTR()[bitno] & source->r ? 1L : 0L;",
        ),
        (
            "return bit[bitno-nbits] & source->l ? 1L : 0L;",
            "return BEEBS_NDES_BIT_PTR()[bitno - nbits] & source->l ? 1L : 0L;",
        ),
        ("   p = bit;", "   p = BEEBS_NDES_BIT_PTR();"),
        ("ipc1[j]", "BEEBS_NDES_IPC1(j)"),
        ("ipc1[k]", "BEEBS_NDES_IPC1(k)"),
        ("ipc2[j]", "BEEBS_NDES_IPC2(j)"),
        ("ipc2[k]", "BEEBS_NDES_IPC2(k)"),
        ("ipc2[l]", "BEEBS_NDES_IPC2(l)"),
        ("ip[j]", "BEEBS_NDES_IP(j)"),
        ("ip[k]", "BEEBS_NDES_IP(k)"),
        ("ipm[j]", "BEEBS_NDES_IPM(j)"),
        ("ipm[k]", "BEEBS_NDES_IPM(k)"),
        ("&kns[i]", "BEEBS_NDES_KNS_PTR(i)"),
        ("&kns[ii]", "BEEBS_NDES_KNS_PTR(ii)"),
        ("iet[j]", "BEEBS_NDES_IET(j)"),
        ("iet[l]", "BEEBS_NDES_IET(l)"),
        ("iet[m]", "BEEBS_NDES_IET(m)"),
        ("ipp[j]", "BEEBS_NDES_IPP(j)"),
        ("is[icol][irow][jj]", "BEEBS_NDES_IS(icol, irow, jj)"),
        ("ibin[iss]", "BEEBS_NDES_IBIN(iss)"),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    text = re.sub(
        r"(const static char ipc2\[49\]=\{0,14,17,11,24,1,5,3,28,15,6,21,\n"
        r"   10,23,19,12,4,26,8,16,7,27,20,13,2,41,52,31,\n"
        r"   37,47,55,30,40,51,45,33,48,44,49,39,56,34,\n"
        r"   53,46,42,50,36,29,32\};)",
        r"\1\n" + IPC_MACROS,
        text,
    )
    text = re.sub(
        r"(static great kns\[17\];)",
        r"\1\n" + IP_KNS_MACROS,
        text,
    )
    text = re.sub(
        r"(static char ibin\[16\]=\{0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15\};)",
        r"\1\n" + CYFUN_TABLE_MACROS,
        text,
    )

    with open(dst_path, "w", encoding="utf-8") as dst_file:
        dst_file.write(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

