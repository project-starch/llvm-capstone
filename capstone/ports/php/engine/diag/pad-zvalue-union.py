#!/usr/bin/env python3
"""Generate a patched copy of Zend/zend.h whose zvalue_value union keeps its
CAPABILITY-bearing members off the scalars' 16-byte granule.

WHY. The tag of a capability lives in a side table keyed by 16-byte granule, so an 8-byte
store to `lval` (or a 4-byte store to an aliasing `znode.u` member) clears the tag of a
`str.val` sharing that granule. Observed at zend_language_parser.c:2671, `sd` size 8, killing
the granule of a live string capability.

WHAT. `str.val` moves from union offset 0 to offset 16 by a leading pad INSIDE the existing
`str` struct. The member NAMES do not change, so all 333 `.value.str.val` sites compile
untouched -- this is option B's de-aliasing semantics at option A's diff cost.

The corpus tree is never modified; this writes a copy that precedes it on the include path.
"""
import sys, re
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
old = """typedef union _zvalue_value {
	long lval;					/* long value */
	double dval;				/* double value */
	struct {
		char *val;
		int len;
	} str;
	HashTable *ht;				/* hash table value */
	zend_object_value obj;
} zvalue_value;"""
new = """typedef union _zvalue_value {
	long lval;					/* long value */
	double dval;				/* double value */
	struct {
		/* CAPABILITY GRANULE PAD, injected by diag/pad-zvalue-union.py. Keeps `val` off
		 * the granule that `lval`/`dval` write, so an 8-byte scalar store cannot clear a
		 * live string capability's tag. Member names are unchanged. */
		long __cap_granule_pad;
		char *val;
		int len;
	} str;
	HashTable *ht;				/* hash table value */
	zend_object_value obj;
} zvalue_value;"""
if old not in t:
    sys.stderr.write("pad-zvalue-union: the union does not match the expected text; refusing\n")
    sys.exit(2)
open(dst, "w").write(t.replace(old, new, 1))
