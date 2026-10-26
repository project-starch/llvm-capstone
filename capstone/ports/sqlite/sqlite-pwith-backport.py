#!/usr/bin/env python3
# Backport of a real, released SQLite use-after-free into the 3.22.0 amalgamation, by reverting
# its one-line upstream fix.
#
# Upstream fix f932f7145e (2015-04-12, "Fix a problem with a stale Select.pWith pointer when a CTE
# is used together with a compound query and an ORDER BY clause") added exactly one line to
# convertCompoundSelectToSubquery() in src/select.c:
#
#       p->pNext = 0;
#   +   p->pWith = 0;
#       p->selFlags &= ~SF_Compound;
#
# That function rewrites a compound SELECT with an ORDER BY COLLATE into a subquery. It shallow-
# copies the Select (*pNew = *p), so pNew and p momentarily share every pointer field, then nulls
# each shared field on the original so the two do not both own it. Without the pWith line both
# Selects own the same With object, which is freed twice on parse teardown (sqlite3SelectDelete ->
# clearSelect -> sqlite3WithDelete). The bug shipped in released SQLite 3.8.4 through 3.8.8.
#
# 3.22.0's convertCompoundSelectToSubquery is the same function and carries the fix line, so stock
# 3.22.0 is NOT vulnerable: the defect is reintroduced here, and the demonstration is labelled a
# backport, never a native 3.22.0 bug. The stale With object is served from lookaside
# (sqlite3DbMallocZero with a live db, ~48 bytes for one CTE), i.e. a slot: this is the point of
# the case. Exactly one match is asserted, so a source change that stops matching fails the build.
import sys

fp = sys.argv[1]
s = open(fp).read()
block = ("  p->pNext = 0;\n"
         "  p->pWith = 0;\n"
         "  p->selFlags &= ~SF_Compound;\n")
repl = ("  p->pNext = 0;\n"
        "  /* BACKPORT f932f7145e (released SQLite 3.8.4-3.8.8): the stale Select.pWith fix line\n"
        "     p->pWith = 0; is removed here, so p and the shallow copy pNew both own the same With\n"
        "     object (a lookaside slot) and it is freed twice on parse teardown. */\n"
        "  p->selFlags &= ~SF_Compound;\n")
c = s.count(block)
if c != 1:
    sys.stderr.write("pWith backport: block matched %d times (expected 1)\n" % c)
    sys.exit(3)
open(fp, "w").write(s.replace(block, repl, 1))
print("== applied pWith stale-pointer backport (reverted f932f7145e)")
