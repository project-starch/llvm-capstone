/* musl's src/exit/atexit.c, with atexit()'s function pointer kept a pointer.
 *
 * atexit() hands its function to __cxa_atexit as the argument of a trampoline,
 * and musl converts it through uintptr_t on the way in and on the way out:
 *
 *   return __cxa_atexit(call, (void *)(uintptr_t)func, 0);
 *   ((void (*)(void))(uintptr_t)p)();
 *
 * uintptr_t holds an address. A function pointer here is a capability, and the
 * round trip keeps the address and drops the tag, so every handler registered
 * with atexit() was called through an untagged value: cs.cjalr refuses it
 * (cause 24) in call(), at exit, in every musl domain. Found 2026-09-23 by
 * tests/runtime-qemu/return-flush: `call + 0x28` jumping to on_exit_handler's
 * address as an integer. The two conversions are now direct, function pointer
 * to void * and back, which keeps the capability; on a flat target they are the
 * same bits either way. Nothing else in this file differs from musl's.
 *
 * atexit.o also defines __funcs_on_exit, __cxa_finalize, __cxa_atexit and
 * __atexit_lockptr, so all of them are here: a program that references any of
 * them would otherwise pull musl's object back in and the definitions collide.
 */
#include <stdlib.h>
#include <stdint.h>
#include "libc.h"
#include "lock.h"
#include "fork_impl.h"

#define malloc __libc_malloc
#define calloc __libc_calloc
#define realloc undef
#define free undef

/* Ensure that at least 32 atexit handlers can be registered without malloc */
#define COUNT 32

static struct fl
{
	struct fl *next;
	void (*f[COUNT])(void *);
	void *a[COUNT];
} builtin, *head;

static int slot;
static volatile int lock[1];
volatile int *const __atexit_lockptr = lock;

void __funcs_on_exit()
{
	void (*func)(void *), *arg;
	LOCK(lock);
	for (; head; head=head->next, slot=COUNT) while(slot-->0) {
		func = head->f[slot];
		arg = head->a[slot];
		UNLOCK(lock);
		func(arg);
		LOCK(lock);
	}
}

void __cxa_finalize(void *dso)
{
}

int __cxa_atexit(void (*func)(void *), void *arg, void *dso)
{
	LOCK(lock);

	/* Defer initialization of head so it can be in BSS */
	if (!head) head = &builtin;

	/* If the current function list is full, add a new one */
	if (slot==COUNT) {
		struct fl *new_fl = calloc(sizeof(struct fl), 1);
		if (!new_fl) {
			UNLOCK(lock);
			return -1;
		}
		new_fl->next = head;
		head = new_fl;
		slot = 0;
	}

	/* Append function to the list. */
	head->f[slot] = func;
	head->a[slot] = arg;
	slot++;

	UNLOCK(lock);
	return 0;
}

static void call(void *p)
{
	((void (*)(void))p)();
}

int atexit(void (*func)(void))
{
	return __cxa_atexit(call, (void *)func, 0);
}
