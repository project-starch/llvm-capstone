/* The generic (unsized) atomic library calls, for a one-hart domain.
 *
 * C-54: a capability is 16 bytes and the largest lock-free atomic on this
 * target is 8, so an atomic on a pointer is a library call. The SIZED calls
 * (__atomic_load_16 and friends) pass the value in two integer registers,
 * which drops the tag before any implementation sees it; the compiler routes
 * capability-valued atomics to these generic ones instead, which pass every
 * value through memory, where a 16-byte capability store keeps it.
 *
 * Why no lock: a domain runs on one hart and has no clone, and nothing else
 * writes its memory while it runs -- the host runs only while the domain is
 * suspended in a hostcall. Each call here is therefore atomic with respect
 * to every observer that exists. This file is WRONG for any build that can
 * run two threads; such a build needs a lock or a capability CAS instruction.
 *
 * The names are the compiler's builtins, so the functions are declared under
 * other names and renamed at the symbol level, as compiler-rt's atomic.c does.
 */
#include <stdbool.h>
#include <stddef.h>
#include <string.h>


/* Copy one value of `size` bytes. A 16-byte value on 16-byte alignment is
   copied as a capability, which is the only copy that keeps a tag; anything
   else goes through the runtime's tag-preserving memcpy. */
static void copy_value(void *dst, const void *src, size_t size)
{
	if (size == sizeof(void *) &&
	    ((__UINTPTR_TYPE__)dst % sizeof(void *)) == 0 &&
	    ((__UINTPTR_TYPE__)src % sizeof(void *)) == 0)
		*(void **)dst = *(void *const *)src;
	else
		memcpy(dst, src, size);
}

#pragma redefine_extname capstone_atomic_load __atomic_load
#pragma redefine_extname capstone_atomic_store __atomic_store
#pragma redefine_extname capstone_atomic_exchange __atomic_exchange
#pragma redefine_extname capstone_atomic_compare_exchange __atomic_compare_exchange

void capstone_atomic_load(size_t size, void *src, void *dest, int model)
{
	(void)model;
	copy_value(dest, src, size);
}

void capstone_atomic_store(size_t size, void *dest, void *src, int model)
{
	(void)model;
	copy_value(dest, src, size);
}

void capstone_atomic_exchange(size_t size, void *ptr, void *val, void *old,
                              int model)
{
	(void)model;
	copy_value(old, ptr, size);
	copy_value(ptr, val, size);
}

bool capstone_atomic_compare_exchange(size_t size, void *ptr, void *expected,
                                      void *desired, int success, int failure)
{
	(void)success;
	(void)failure;
	/* By representation, as libatomic compares. Two capabilities with the
	   same bits compare equal whatever their tags; the stored value is the
	   desired one, copied whole, tag included. */
	if (memcmp(ptr, expected, size) == 0) {
		copy_value(ptr, desired, size);
		return true;
	}
	copy_value(expected, ptr, size);
	return false;
}
