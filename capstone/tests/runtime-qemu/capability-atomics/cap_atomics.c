/* Atomics whose value is a pointer, run in a musl domain (C-54).
 *
 * A capability is 16 bytes and the largest lock-free atomic here is 8, so
 * each of these is a call to the generic __atomic_* functions in the domain
 * runtime (musl-capstone/runtime/atomic_libcalls.c). The value has to come out
 * TAGGED: every pointer that an atomic returned or stored is dereferenced
 * right after. An untagged one faults, the run ends before the DONE line, and
 * run.sh reports FAIL. A long atomic runs first as the control.
 */
#include <stdio.h>
#include <stdlib.h>

static int failures;
#define CHECK(cond, what)                                                       \
	do {                                                                    \
		if (cond) {                                                     \
			printf("ok   %s\n", what);                                  \
		} else {                                                        \
			printf("FAIL %s (line %d)\n", what, __LINE__);              \
			failures++;                                             \
		}                                                               \
	} while (0)

static long counter;
static int target[2] = { 11, 22 };
static int *slot;

int main(void)
{
	/* 0. Control: an 8-byte integer atomic, inline. */
	__atomic_store_n(&counter, 5, __ATOMIC_SEQ_CST);
	CHECK(__atomic_fetch_add(&counter, 2, __ATOMIC_SEQ_CST) == 5 && counter == 7,
	      "control: long store and fetch_add");

	/* 1. Store and load a pointer; the loaded one is dereferenced. */
	__atomic_store_n(&slot, &target[0], __ATOMIC_SEQ_CST);
	int *v = __atomic_load_n(&slot, __ATOMIC_SEQ_CST);
	CHECK(v == &target[0] && *v == 11, "pointer store, load, dereference");

	/* 2. Exchange: both the returned old pointer and the stored new one. */
	int *old = __atomic_exchange_n(&slot, &target[1], __ATOMIC_SEQ_CST);
	CHECK(*old == 11 && *slot == 22, "pointer exchange: old and new dereference");

	/* 3. Compare-exchange that succeeds. */
	int *expected = &target[1];
	CHECK(__atomic_compare_exchange_n(&slot, &expected, &target[0], 0,
	                                  __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) &&
	          *slot == 11,
	      "pointer compare-exchange succeeds, stored pointer dereferences");

	/* 4. Compare-exchange that fails writes the current value to expected. */
	expected = &target[1];
	CHECK(!__atomic_compare_exchange_n(&slot, &expected, &target[1], 0,
	                                   __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) &&
	          expected == &target[0] && *expected == 11,
	      "pointer compare-exchange fails, returned pointer dereferences");

	/* 5. A heap pointer through the same path. */
	int *heap = malloc(4 * sizeof(int));
	CHECK(heap != 0, "malloc");
	heap[3] = 33;
	__atomic_store_n(&slot, heap, __ATOMIC_SEQ_CST);
	int *h = __atomic_load_n(&slot, __ATOMIC_SEQ_CST);
	CHECK(h[3] == 33, "heap pointer through atomic store and load");

	printf("CAP-ATOMICS-DONE failures=%d\n", failures);
	return failures;
}
