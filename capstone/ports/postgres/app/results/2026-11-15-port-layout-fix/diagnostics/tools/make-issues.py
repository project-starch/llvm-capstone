#!/usr/bin/env python3
"""Issues-per-context diagnostic patches (not part of the port): the block census plus a count of
the chunks the reuse hooks record as issued, per context name.

  make-issues.py STOCK_SRC PROTECTED_SRC CENSUS_DIR OUT

Each hooked exp_gap_issue call in mcxt.c gets an exp_name_issue call before it with the same
context; at exit one line per context name: EXP-ISSUES name=N issues=K.
"""
import difflib
import sys
from pathlib import Path

DEF = r'''
#ifdef EXP_REUSE_GAP
/* Diagnostic (scratch): issued chunks per context name. */
#define EXP_NAME_SLOTS 1024
static struct { const char *name; unsigned long long issues; } exp_name[EXP_NAME_SLOTS];
static int exp_name_used, exp_name_overflow;
static void exp_name_report(void)
{
	for (int i = 0; i < exp_name_used; i++)
		fprintf(stderr, "EXP-ISSUES name=%s issues=%llu\n", exp_name[i].name, exp_name[i].issues);
	fprintf(stderr, "EXP-ISSUES-END slots=%d overflow=%d\n", exp_name_used, exp_name_overflow);
}
static void exp_name_issue(MemoryContext context)
{
	const char *name = context && context->name ? context->name : "?";
	int i;
	for (i = 0; i < exp_name_used; i++)
		if (exp_name[i].name == name || strcmp(exp_name[i].name, name) == 0) break;
	if (i == exp_name_used)
	{
		if (exp_name_used == EXP_NAME_SLOTS) { exp_name_overflow++; return; }
		if (exp_name_used == 0) atexit(exp_name_report);
		exp_name[i].name = name;
		exp_name_used++;
	}
	exp_name[i].issues++;
}
#endif
'''
INCLUDE = '#ifdef EXP_REUSE_GAP\n#include "exp-reuse-gap.h"\n#endif\n'

stock, protected, census, out = map(Path, sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
for arm, base in (("stock", stock), ("protected", protected)):
    rel = "src/backend/utils/mmgr/mcxt.c"
    old = (base / rel).read_text()
    assert old.count(INCLUDE) == 1
    new = old.replace(INCLUDE, INCLUDE + DEF)
    n1 = new.count("\texp_gap_issue((uintptr_t) context, ")
    n2 = new.count("\t\t\texp_gap_issue((uintptr_t) exp_gap_old_context, ")
    new = new.replace("\texp_gap_issue((uintptr_t) context, ", "\texp_name_issue(context);\n\texp_gap_issue((uintptr_t) context, ")
    new = new.replace("\t\t\texp_gap_issue((uintptr_t) exp_gap_old_context, ",
                      "\t\t\texp_name_issue(exp_gap_old_context);\n\t\t\texp_gap_issue((uintptr_t) exp_gap_old_context, ")
    assert (n1, n2) == (7, 2), (arm, n1, n2)
    patch = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}"))
    (out / f"issues-{arm}.patch").write_text((census / f"census-{arm}.patch").read_text() + patch)
print("ok")
