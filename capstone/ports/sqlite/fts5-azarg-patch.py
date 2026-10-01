#!/usr/bin/env python3
# FTS5 freestanding-Capstone source adaptation, applied to the adapted
# amalgamation before compile.
#
# The default-tokenizer path calls
#     sqlite3Fts5GetTokenizer(pGlobal, /*azArg*/0, /*nArg*/0, ...)
# (fts5ConfigDefaultTokenizer), and GetTokenizer then evaluates &azArg[1]
# unconditionally to pass as arg2:
#     rc = pMod->x.xCreate(pMod->pUserData, &azArg[1], (nArg?nArg-1:0), ppTok);
# With azArg == NULL that is cincoffset on a NULL capability. Pointer arithmetic
# on a null pointer is undefined behaviour in C: CHERI tolerates it, Capstone
# faults (cause 24), which killed every fts5 domain during
# CREATE VIRTUAL TABLE ... USING fts5, before any INSERT or MATCH. nArg==0 so
# xCreate never reads the array, so keep NULL as NULL.
#
# Exactly one match is asserted: a source change that stops matching fails the
# build instead of silently skipping the fix.
import sys
fp = sys.argv[1]
s = open(fp).read()

n = "    rc = pMod->x.xCreate(pMod->pUserData, &azArg[1], (nArg?nArg-1:0), ppTok);"
r = "    rc = pMod->x.xCreate(pMod->pUserData, (azArg ? &azArg[1] : 0), (nArg?nArg-1:0), ppTok);"
c = s.count(n)
if c != 1:
    sys.stderr.write("FTS5 azArg NULL-guard patch matched %d times (expected 1)\n" % c)
    sys.exit(3)
open(fp, "w").write(s.replace(n, r, 1))
print("== applied FTS5 azArg NULL-guard patch")
