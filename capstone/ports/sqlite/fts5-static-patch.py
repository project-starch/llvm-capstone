#!/usr/bin/env python3
# FTS5 freestanding-Capstone source adaptations (2026-09-30). Applied to the
# adapted amalgamation before compile. Two independent CREATE-time fixes:
#
# (1) aBuiltin[] static: the built-in tokenizer table in sqlite3Fts5TokenizerInit
#     is an AUTOMATIC (stack) array; at -O0 clang materialises its function-pointer
#     initialisers as gp-relative data caps (cincoffset a0, gp, a0) with no code
#     bounds/execute permission. Making it "static" routes them through __cap_relocs
#     (proper sealed code caps), like fts3s static-const sqlite3_module.
#
# (2) azArg NULL-guard: the default-tokenizer path calls
#       sqlite3Fts5GetTokenizer(pGlobal, 0, 0, ...)   (azArg == NULL, nArg == 0)
#     and GetTokenizer evaluates &azArg[1] unconditionally to pass as arg2. On
#     Capstone that is cincoffset on a NULL/untagged capability -> cause-24 fault
#     (CHERI tolerates NULL pointer arithmetic; Capstone does not). nArg==0 so
#     xCreate never reads the array; keep NULL as NULL.
import sys
fp = sys.argv[1]
s = open(fp).read()

n1 = "  struct BuiltinTokenizer {\n    const char *zName;\n    fts5_tokenizer x;\n  } aBuiltin[] = {"
r1 = "  static struct BuiltinTokenizer {\n    const char *zName;\n    fts5_tokenizer x;\n  } aBuiltin[] = {"
c1 = s.count(n1)
if c1 != 1:
    sys.stderr.write("FTS5 aBuiltin static patch matched %d times (expected 1)\n" % c1)
    sys.exit(3)
s = s.replace(n1, r1, 1)
print("== applied FTS5 aBuiltin static patch")

n2 = "    rc = pMod->x.xCreate(pMod->pUserData, &azArg[1], (nArg?nArg-1:0), ppTok);"
r2 = "    rc = pMod->x.xCreate(pMod->pUserData, (azArg ? &azArg[1] : 0), (nArg?nArg-1:0), ppTok);"
c2 = s.count(n2)
if c2 != 1:
    sys.stderr.write("FTS5 azArg NULL-guard patch matched %d times (expected 1)\n" % c2)
    sys.exit(4)
s = s.replace(n2, r2, 1)
print("== applied FTS5 azArg NULL-guard patch")

open(fp, "w").write(s)
