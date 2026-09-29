/* Pointers computed through uintptr_t in a musl domain (CapstoneRecoverProvenance).
 *
 * Every check below takes a pointer out through an integer and back IN THE SAME
 * FUNCTION, then uses it. Built by run.sh three times:
 *   rt-O0, rt-O2  default compiler: every check must pass;
 *   rt-off        -mllvm -capstone-recover-provenance=false: the same program with
 *                 the round trips left untagged -- must halt at the first one.
 * The status is the number of failed checks, so LT-RESULT status=0 is the pass.
 *
 *   align-up     a pointer rounded up to 64 through uintptr_t, written through
 *   bounds       the rounded pointer's bounds are the buffer's: nothing widened
 *   global       a global's address moved by a constant, (char *)((uintptr_t)buf + 8):
 *                a constant expression at every -O level, never an instruction
 *   flag         a tag bit set in and cleared from a pointer, then dereferenced
 *                (at -O0 the tagged integer lives in a stack slot between the two)
 *   cursor       a uintptr_t cursor stepped through an array in a loop (at -O0,
 *                likewise, in a slot)
 *   callback     a known function address round-tripped within this function
 *   atexit       the runtime's capability-safe override runs its handler
 *   opaque-arg   a non-null argument has no NONLIN guarantee; its round trip
 *                stays untagged, and the original pointer remains usable
 *   opaque-load  a pointer loaded from global storage has no NONLIN guarantee
 *   null-source  a pointer that MAY BE NULL is left alone: `cincoffset` on a
 *                register holding no capability raises UNEXPECTED_OPERAND, so
 *                rewriting this would trap at the cast, where the untagged
 *                answer returned an address the caller simply did not use
 *   select-foreign  a select whose arms are not the same one pointer keeps the
 *                untagged answer: the other arm is the caller's own integer, and
 *                the result must not come back carrying buf's authority at it
 *   int-source   a pointer made from an integer, `(char *)(w | 1)`, rounded up:
 *                it holds no capability however non-zero it is, so it is left
 *                alone rather than moved -- moving it traps at the cast
 *   sentinel     a cursor that holds a static object on one path and `(char *)-1`
 *                on the other, rounded up: the sentinel holds no capability, so
 *                neither does the cursor (at -O0 it lives in a stack slot)
 *
 * Recovery requires a known NONLIN source as well as a valid capability.
 * Locals, global addresses and explicit DELIN results can qualify; opaque
 * arguments, loads and call returns cannot. The opaque tests use live NONLIN
 * inputs and must still decline: the callee cannot assume that runtime type.
 * recover-provenance-linear.ll also checks that live argument and MREV-load
 * shapes have identical assembly with recovery enabled and disabled.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int failures;

static void check(int ok, const char *name, long got, long want)
{
	printf("RT-TEST %s %s got=%ld want=%ld\n", ok ? "PASS" : "FAIL", name, got, want);
	fflush(stdout);
	failures += !ok;
}

static char buf[200];

struct node { long val; struct node *next; };

static int calls;
static void callback(void) { calls++; }

static void at_exit_handler(void) { printf("RT-TEST atexit handler ran\n"); }

/* A source the compiler cannot prove non-null: must come back untouched. */
__attribute__((noinline)) static char *align_maybe_null(char *p)
{
	return (char *)(((uintptr_t)p + 63) & ~(uintptr_t)63);
}

/* Two arms, and only one of them is p. */
__attribute__((noinline)) static char *pick(int c, char *p, uintptr_t x)
{
	return (char *)(c ? (uintptr_t)p : x);
}

/* An integer made into a pointer holds no capability, however non-zero. */
__attribute__((noinline)) static char *align_from_int(uintptr_t w)
{
	char *n = (char *)(w | 1);
	return (char *)(((uintptr_t)n + 7) & ~(uintptr_t)7);
}

/* A cursor that is an object on one path and a sentinel on the other. */
__attribute__((noinline)) static char *align_cursor(int full)
{
	static char arena[64];
	char *cur = arena;
	if (full)
		cur = (char *)-1;
	return (char *)(((uintptr_t)cur + 15) & ~(uintptr_t)15);
}

/* The one capability query that cannot trap: LCC with zimm==1, the TYPE query,
   made total as an S-06 enabler -- it answers 7 for a non-capability instead of
   raising (capstone_dyn_unit.anvil, the zimm!=1 guard). __builtin_capstone_cap_get_tag
   is selector 0, the validity query, which is NOT total and would trap on
   exactly the values these checks have to inspect. No builtin reaches selector
   1, hence the .insn. */
#define CAP_TYPE_NONE 7
static unsigned long cap_type(const void *p)
{
	unsigned long t;
	__asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(t) : "r"(p));
	return t;
}

/* These definitions must keep opaque origins even at -O2. The argument
   function is external and noinline, and the global load is volatile. */
__attribute__((noinline)) int opaque_argument(char *p)
{
	*p = 'a'; /* Valid and non-null, but that says nothing about linearity. */
	char *r = (char *)(((uintptr_t)p + 15) & ~(uintptr_t)15);
	int untagged = cap_type(r) == CAP_TYPE_NONE;
	*p = 'b';
	return untagged && *p == 'b';
}

static char *volatile saved_pointer;
__attribute__((noinline)) static int opaque_load(void)
{
	char *p = saved_pointer;
	*p = 'c';
	char *r = (char *)(((uintptr_t)p + 15) & ~(uintptr_t)15);
	int untagged = cap_type(r) == CAP_TYPE_NONE;
	*p = 'd';
	return untagged && *p == 'd';
}

int main(void)
{
	/* The first check touches memory through a round-tripped pointer: in rt-off
	   this is where the domain halts. */
	char *p = buf + 3;
	char *q = (char *)(((uintptr_t)p + 63) & ~(uintptr_t)63);
	*q = 'x';
	long idx = q - buf;
	check(idx >= 3 && idx < 3 + 64 && ((uintptr_t)q & 63) == 0 && buf[idx] == 'x',
	      "align-up", idx, 64 - ((uintptr_t)buf & 63) + 0);

	check(__builtin_capstone_cap_get_base(q) == __builtin_capstone_cap_get_base(buf) &&
	      __builtin_capstone_cap_get_end(q) == __builtin_capstone_cap_get_end(buf),
	      "bounds", (long)(__builtin_capstone_cap_get_end(q) - __builtin_capstone_cap_get_base(q)),
	      (long)sizeof buf);

	char *g = (char *)((uintptr_t)buf + 8);
	*g = 'g';
	check(buf[8] == 'g', "global", buf[8], 'g');

	struct node local = { 0, 0 };
	struct node *n = &local;
	n->val = 77;
	n->next = 0;
	uintptr_t tagged = (uintptr_t)n | 1;
	struct node *m = (struct node *)(tagged & ~(uintptr_t)1);
	check((tagged & 1) && m->val == 77, "flag", m->val, 77);

	long arr[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	long sum = 0;
	for (uintptr_t c = (uintptr_t)arr; c < (uintptr_t)(arr + 8); c += sizeof(long))
		sum += *(long *)c;
	check(sum == 36, "cursor", sum, 36);

	void *r = (void *)(uintptr_t)callback;
	((void (*)(void))(uintptr_t)r)();
	((void (*)(void))(uintptr_t)r)();
	check(calls == 2, "callback", calls, 2);

	/* Reaching this check at all is the result: if the pass moved a base it
	   could not prove non-null, the domain halted inside align_maybe_null. */
	char *nullaligned = align_maybe_null(0);
	check(nullaligned == 0, "null-source", (long)(nullaligned != 0), 0);

	long other[2] = { 0, 0 };
	char *picked = pick(0, buf, __builtin_capstone_cap_get_cursor(other));
	check(cap_type(picked) == CAP_TYPE_NONE, "select-foreign",
	      (long)cap_type(picked), CAP_TYPE_NONE);

	/* volatile, so that neither argument is propagated into its callee as a
	   constant and folded away there: both must reach the pass as values. */
	volatile uintptr_t seed = 0x1000;
	char *fromint = align_from_int(seed);
	check(cap_type(fromint) == CAP_TYPE_NONE, "int-source",
	      (long)cap_type(fromint), CAP_TYPE_NONE);

	volatile int full = 1;
	char *sentinel = align_cursor(full);
	check(cap_type(sentinel) == CAP_TYPE_NONE, "sentinel",
	      (long)cap_type(sentinel), CAP_TYPE_NONE);

	int arg_ok = opaque_argument(buf);
	check(arg_ok, "opaque-arg", arg_ok, 1);
	saved_pointer = buf;
	int load_ok = opaque_load();
	check(load_ok, "opaque-load", load_ok, 1);

	atexit(at_exit_handler);
	printf("RT-TEST-DONE failures=%d\n", failures);
	fflush(stdout);
	exit(failures);
}
