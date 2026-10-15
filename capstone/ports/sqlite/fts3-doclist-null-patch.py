#!/usr/bin/env python3
# FTS3 Capstone source adaptation, applied to the staged amalgamation before compile.
#
# fts3EvalDlPhraseNext computes the end of a phrase's doclist up front:
#     char *pEnd = &pDL->aAll[pDL->nAll];
# A phrase with an empty doclist has aAll == NULL and nAll == 0, so that is
# cincoffset on a NULL capability. Pointer arithmetic on a null pointer is
# undefined behaviour in C: a flat-address machine computes NULL, Capstone
# faults (cause 24, address 0). With aAll NULL the function then starts at
# pIter = aAll, finds pIter >= pEnd and reports EOF, so keep the end NULL too.
# Case 10 of the engine corpus faulted here on every virtual arm, stock memsys5
# included, before reaching its defect.
#
# Exactly one match is asserted: a source change that stops matching fails the
# build instead of silently skipping the fix.
import sys
fp = sys.argv[1]
s = open(fp).read()

n = "  char *pEnd = &pDL->aAll[pDL->nAll];     /* 1 byte past end of aAll */"
r = "  char *pEnd = pDL->aAll ? &pDL->aAll[pDL->nAll] : 0; /* 1 byte past end of aAll */"
c = s.count(n)
if c != 1:
    sys.stderr.write("FTS3 doclist NULL-guard patch matched %d times (expected 1)\n" % c)
    sys.exit(3)
open(fp, "w").write(s.replace(n, r, 1))
print("== applied FTS3 doclist NULL-guard patch")
