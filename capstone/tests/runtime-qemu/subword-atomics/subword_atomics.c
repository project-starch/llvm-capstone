/* 8- and 16-bit atomics through capabilities, run in a musl domain (C-51).
 *
 * AtomicExpand turns every sub-word atomic into an LR/SC loop on the aligned
 * 32-bit word that holds it. This checks what a compile cannot: that the loop
 * runs on the capability, lands on the right lane of the word, and leaves the
 * other lanes alone. Each check prints one line; the exit status is the number
 * of failures, and "SUBWORD-ATOMICS-DONE failures=N" is printed last, so a run
 * that stops early (a capability fault) is told apart from one that failed.
 *
 * Order matters: the 32-bit control runs first. If it fails, capability
 * atomics do not work in this emulator at all and nothing after it says
 * anything about sub-word lowering.
 */
#include <stdint.h>
#include <stdio.h>

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

static uint32_t word32;
static _Alignas(16) uint8_t bytes[8];
static _Alignas(16) uint16_t halves[4];
/* CPython's PyMutex: one byte, locked by compare-exchange (Include/cpython/lock.h). */
typedef struct { uint8_t bits; } PyMutex;
static struct { uint32_t before; PyMutex m; uint8_t after[3]; } obj;
/* The risk case: one byte on its own. The aligned word around it reaches past
   the variable; if the capability for it is bounded to the variable, the LR/SC
   on that word is out of bounds. */
static uint8_t lone_byte;

int main(void)
{
	/* 0. Control: a 32-bit atomic through a capability (d5b5f11cae8f). */
	uint32_t e32 = 0;
	CHECK(__atomic_compare_exchange_n(&word32, &e32, 0x11223344u, 0,
	                                  __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) &&
	          word32 == 0x11223344u,
	      "control: 32-bit compare-exchange");

	/* 1. Compare-exchange on every byte lane; neighbours must not move. */
	for (int i = 0; i < 4; i++) {
		uint8_t e = 0;
		int ok = __atomic_compare_exchange_n(&bytes[i], &e, (uint8_t)(0xA0 + i), 0,
		                                     __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
		char what[48];
		snprintf(what, sizeof what, "8-bit compare-exchange, lane %d", i);
		CHECK(ok && bytes[i] == 0xA0 + i, what);
	}
	CHECK(bytes[0] == 0xA0 && bytes[1] == 0xA1 && bytes[2] == 0xA2 &&
	          bytes[3] == 0xA3 && bytes[4] == 0,
	      "8-bit lanes independent, next word untouched");

	/* 2. A failing compare-exchange reports the current value. */
	uint8_t e8 = 0x55;
	CHECK(!__atomic_compare_exchange_n(&bytes[2], &e8, 0x77, 0, __ATOMIC_SEQ_CST,
	                                   __ATOMIC_SEQ_CST) &&
	          e8 == 0xA2 && bytes[2] == 0xA2,
	      "8-bit compare-exchange fails and returns the old value");

	/* 3. Read-modify-write operations. */
	CHECK(__atomic_fetch_add(&bytes[1], 0x10, __ATOMIC_SEQ_CST) == 0xA1 &&
	          bytes[1] == 0xB1,
	      "8-bit fetch_add");
	CHECK(__atomic_exchange_n(&bytes[3], 0x33, __ATOMIC_SEQ_CST) == 0xA3 &&
	          bytes[3] == 0x33,
	      "8-bit exchange");
	CHECK(__atomic_fetch_nand(&bytes[0], 0x0F, __ATOMIC_SEQ_CST) == 0xA0 &&
	          bytes[0] == (uint8_t)~(0xA0 & 0x0F),
	      "8-bit fetch_nand");
	int8_t s8 = -5;
	__atomic_store_n((uint8_t *)&bytes[5], (uint8_t)s8, __ATOMIC_SEQ_CST);
	(void)__atomic_fetch_max((int8_t *)&bytes[5], (int8_t)3, __ATOMIC_SEQ_CST);
	CHECK((int8_t)bytes[5] == 3, "8-bit signed fetch_max");
	(void)__atomic_fetch_min((int8_t *)&bytes[5], (int8_t)-7, __ATOMIC_SEQ_CST);
	CHECK((int8_t)bytes[5] == -7, "8-bit signed fetch_min");

	/* 4. 16-bit, both halves of a word. */
	CHECK(__atomic_fetch_add(&halves[0], 0x1234, __ATOMIC_SEQ_CST) == 0 &&
	          __atomic_fetch_add(&halves[1], 0x4321, __ATOMIC_SEQ_CST) == 0 &&
	          halves[0] == 0x1234 && halves[1] == 0x4321 && halves[2] == 0,
	      "16-bit fetch_add, both halves");
	uint16_t e16 = 0x4321;
	CHECK(__atomic_compare_exchange_n(&halves[1], &e16, 0xBEEF, 0,
	                                  __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) &&
	          halves[1] == 0xBEEF && halves[0] == 0x1234,
	      "16-bit compare-exchange, upper half");
	(void)__atomic_fetch_max(&halves[0], (uint16_t)0x9000, __ATOMIC_SEQ_CST);
	CHECK(halves[0] == 0x9000, "16-bit unsigned fetch_max");

	/* 5. CPython's PyMutex_Lock / Unlock shape, inside a struct. */
	obj.before = 0xCAFEF00Du;
	uint8_t unlocked = 0;
	CHECK(__atomic_compare_exchange_n(&obj.m.bits, &unlocked, 1, 0,
	                                  __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) &&
	          obj.m.bits == 1,
	      "PyMutex lock");
	uint8_t locked = 1;
	CHECK(__atomic_compare_exchange_n(&obj.m.bits, &locked, 0, 0,
	                                  __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) &&
	          obj.m.bits == 0 && obj.before == 0xCAFEF00Du,
	      "PyMutex unlock, neighbour intact");

	/* 6. The risk case last: a fault here ends the run, and everything above
	   has already reported. */
	uint8_t e1 = 0;
	CHECK(__atomic_compare_exchange_n(&lone_byte, &e1, 0x5A, 0, __ATOMIC_SEQ_CST,
	                                  __ATOMIC_SEQ_CST) &&
	          lone_byte == 0x5A,
	      "8-bit compare-exchange on a lone one-byte global");

	printf("SUBWORD-ATOMICS-DONE failures=%d\n", failures);
	return failures;
}
