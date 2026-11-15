#!/usr/bin/env python3
"""Block-census patches for the layout fix (diagnostic, not part of the port).

  make-census.py STOCK_SRC PROTECTED_SRC OUT

The sites and the census text are run-5's (make-census-run5.py, read unchanged): every malloc
of a memory-context block counts one block for its manager path and the context name, printed at
exit as EXP-BLOCKS lines. The protected arm also counts, in MemoryChunkIssue, the chunks whose
bounds start below their payload and those whose bounds start more than one chunk header
(16 bytes) below it, with the largest such distance, printed as one EXP-SLACK line.
"""
import difflib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "make-census-run5.py").read_text()
src = src[:src.index("for arm, base in BASES.items():")]
run5 = {"__file__": str(HERE / "make-census-run5.py")}
exec(compile(src.replace("OUT = Path(sys.argv[1])", "OUT = None"), "make-census-run5.py", "exec"), run5)

SLACK_DEF = r"""
#ifdef EXP_REUSE_GAP
static unsigned long long exp_slack_issues, exp_slack_below, exp_slack_header, exp_slack_max;
static void exp_slack_report(void)
{
	fprintf(stderr, "EXP-SLACK issues=%llu below_payload=%llu below_header=%llu max=%llu\n",
			exp_slack_issues, exp_slack_below, exp_slack_header, exp_slack_max);
}
void exp_slack_count(void *child, void *payload)
{
	unsigned long long d;
	if (!child) return;
	if (exp_slack_issues++ == 0) atexit(exp_slack_report);
	d = (unsigned long long) ((uintptr_t) payload - __builtin_capstone_cap_get_base(child));
	if (d > 0) exp_slack_below++;
	if (d > 16) exp_slack_header++;
	if (d > exp_slack_max) exp_slack_max = d;
}
#endif
"""
ISSUE_OLD = """	return SubletDerive(SubletBlockAt((uintptr_t) payload), payload,
						SubletChunkLength(size));
}"""
ISSUE_NEW = """	void	   *child = SubletDerive(SubletBlockAt((uintptr_t) payload), payload,
									 SubletChunkLength(size));
#ifdef EXP_REUSE_GAP
	{
		extern void exp_slack_count(void *child, void *payload);

		exp_slack_count(child, payload);
	}
#endif
	return child;
}"""


def diff(rel, old, new):
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}"))


stock, protected, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
for arm, base in (("stock", stock), ("protected", protected)):
    parts = []
    for name, sites in run5["SITES"].items():
        rel = f"src/backend/utils/mmgr/{name}"
        old = (base / rel).read_text()
        new = run5["instrument"](old, sites, name == "aset.c")
        if arm == "protected" and name == "aset.c":
            k = new.index("#ifdef EXP_REUSE_GAP\n/* Reuse diagnostic (scratch)")
            k = new.index("#endif\n", k) + len("#endif\n")
            new = new[:k] + SLACK_DEF + new[k:]
        parts.append(diff(rel, old, new))
    if arm == "protected":
        rel = "src/include/utils/memutils_memorychunk.h"
        old = (base / rel).read_text()
        assert old.count(ISSUE_OLD) == 1
        parts.append(diff(rel, old, old.replace(ISSUE_OLD, ISSUE_NEW)))
    (out / f"census-{arm}.patch").write_text("".join(parts))
print("ok")
