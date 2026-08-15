/* __errno_location for a domain, without musl's TLS round-trip.
 *
 * THE DEFECT THIS AVOIDS. musl reaches errno through the thread pointer, and its
 * riscv64 arch layer round-trips that pointer through an INTEGER:
 *
 *     static inline uintptr_t __get_tp(void) {          // arch/riscv64/pthread_arch.h
 *         uintptr_t tp; __asm__("mv %0, tp" : "=r"(tp)); return tp;
 *     }
 *     #define __pthread_self() ((pthread_t)(__get_tp() - sizeof(struct __pthread)))
 *
 * `uintptr_t` is 8 bytes here and a pointer is a 128-bit capability, so the cast
 * back forges a capability from an integer: untagged, and the first dereference
 * faults. Our own tp setup is fine -- start-musl.S points it into the middle of
 * a 1024-byte block derived from gp, with room below for errno_val -- but no
 * setup can survive the round trip.
 *
 * WHY NOBODY HIT IT UNTIL 2026-08-15. errno is only written when a syscall
 * FAILS, and until now every syscall in every probe here had succeeded. It
 * surfaced on the first deliberate failure: lseek to a negative position. So the
 * whole libc error path was broken and no green run could have told us.
 *
 * A SINGLE STATIC int IS CORRECT HERE, not a shortcut: the domain is
 * single-threaded (see locks.c, same assumption, same condition to re-examine).
 * Defining this symbol also stops musl's src/errno/__errno_location.c being
 * pulled from the archive, so there is exactly one definition.
 */

static int capstone_errno;

int *__errno_location(void) { return &capstone_errno; }

/* musl's headers spell it both ways depending on the source file; both must
   resolve to the same storage or a failed call would set one and a caller read
   the other. */
extern __typeof(__errno_location) ___errno_location
    __attribute__((weak, alias("__errno_location")));
