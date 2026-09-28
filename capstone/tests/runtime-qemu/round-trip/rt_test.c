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
 *   callback     musl's atexit idiom: a function pointer passed as
 *                (void *)(uintptr_t)f, called back as ((void (*)(void))(uintptr_t)p)()
 *   atexit       musl's own, unmodified atexit(): its handler runs at exit()
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

struct registration { void *arg; };
__attribute__((noinline)) static void register_cb(struct registration *r, void (*f)(void))
{
	r->arg = (void *)(uintptr_t)f;           /* musl atexit(): stores func this way */
}
__attribute__((noinline)) static void call_cb(void *p)
{
	((void (*)(void))(uintptr_t)p)();        /* musl call(): calls it back this way */
}

static void at_exit_handler(void) { printf("RT-TEST atexit handler ran\n"); }

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

	struct node *n = malloc(sizeof *n);
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

	struct registration r;
	register_cb(&r, callback);
	call_cb(r.arg);
	call_cb(r.arg);
	check(calls == 2, "callback", calls, 2);

	atexit(at_exit_handler);
	printf("RT-TEST-DONE failures=%d\n", failures);
	fflush(stdout);
	exit(failures);
}
