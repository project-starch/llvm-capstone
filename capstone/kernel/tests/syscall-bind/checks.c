/* The binder's pointer-position table and core on the host (plan v5 step 5).
 *
 * Builds bind-core.c and the generated syscall-pointers.c unchanged, with the riscv64 call
 * numbers of a Linux tree and the host's fcntl, ioctl and prctl constants (the same values as
 * asm-generic's). Checks the table for known calls, then the core: references become handles
 * with their slot's number, plain addresses get number 0, a forged integer with number bits
 * gets EFAULT, calls and commands outside the allow-list are refused. Every check prints PASS
 * or FAIL; the exit status is the number of failures. */
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/prctl.h>
#include "bind.h"

static int failures;

static void check(int ok, const char *what)
{
	printf("%s %s\n", ok ? "PASS" : "FAIL", what);
	failures += !ok;
}

#define ADDR 0x3f80ef7000UL
#define NUMBER(h) ((h) >> 40)

/* Resolve a bound handle as the routines do: the slot of a0 to a5 by number, or none. */
static int selects(unsigned long handle, const struct sublet_bind_args *before, int *arg)
{
	unsigned long n = NUMBER(handle);

	if (handle & (3UL << 38) || n < 1 || n > 6)
		return 0;
	*arg = n - 1;
	return before->tagged & (1u << (n - 1));
}

int main(void)
{
	struct sublet_bind_args a, before;
	long rc;
	int arg;

	/* The generated table, from SYSCALL_DEFINE types. */
	check(sublet_syscall_pointers[__NR_read] == 0x02, "table: read(fd, *buf, count) is 0x02");
	check(sublet_syscall_pointers[__NR_write] == 0x02, "table: write(fd, *buf, count) is 0x02");
	check(sublet_syscall_pointers[__NR_openat] == 0x02, "table: openat(dfd, *filename, ...) is 0x02");
	check(sublet_syscall_pointers[__NR_mmap] == 0x00, "table: mmap's address is unsigned long");
	check(sublet_syscall_pointers[__NR_execve] == 0x07, "table: execve is 0x07");
	check(sublet_syscall_pointers[__NR_clone] == 0x0c, "table: clone is riscv's variant, 0x0c");
	check(sublet_syscall_pointers[__NR_getpid] == 0x00, "table: getpid has no pointer");

	/* write(1, reference, 12): the buffer becomes a handle of a1's slot (number 2). */
	memset(&a, 0, sizeof(a));
	a.arg[0] = 1; a.arg[1] = ADDR | (0x15UL << 40); a.arg[2] = 12; a.tagged = 0x2;
	rc = sublet_bind_core(__NR_write, &a);
	check(rc == 0 && a.arg[1] == (ADDR | (2UL << 40)) && a.arg[0] == 1 && a.arg[2] == 12,
	      "bind: write's reference becomes a handle with number 2, integers unchanged");
	/* A plain address gets number 0. */
	memset(&a, 0, sizeof(a));
	a.arg[0] = 1; a.arg[1] = ADDR; a.arg[2] = 12;
	rc = sublet_bind_core(__NR_write, &a);
	check(rc == 0 && a.arg[1] == ADDR, "bind: a plain address keeps number 0");
	/* An integer position keeps its upper bits. */
	memset(&a, 0, sizeof(a));
	a.arg[0] = 1; a.arg[1] = ADDR; a.arg[2] = 7UL << 50;
	rc = sublet_bind_core(__NR_write, &a);
	check(rc == 0 && a.arg[2] == 7UL << 50, "bind: an integer position passes unchanged");
	/* mmap, munmap and mremap: the hand-added address positions. */
	memset(&a, 0, sizeof(a));
	a.arg[0] = ADDR; a.arg[4] = ADDR + 0x10000; a.tagged = 0x11;
	rc = sublet_bind_core(__NR_mremap, &a);
	check(rc == 0 && NUMBER(a.arg[0]) == 1 && NUMBER(a.arg[4]) == 5,
	      "bind: mremap's addr and new_addr become handles 1 and 5");
	memset(&a, 0, sizeof(a));
	a.arg[0] = ADDR; a.tagged = 0x1;
	check(sublet_bind_core(__NR_munmap, &a) == 0 && NUMBER(a.arg[0]) == 1, "bind: munmap's addr is a handle");
	/* Command-dependent arguments. */
	memset(&a, 0, sizeof(a));
	a.arg[0] = 3; a.arg[1] = TIOCGWINSZ; a.arg[2] = ADDR; a.tagged = 0x4;
	check(sublet_bind_core(__NR_ioctl, &a) == 0 && NUMBER(a.arg[2]) == 3, "bind: ioctl TIOCGWINSZ binds arg");
	memset(&a, 0, sizeof(a));
	a.arg[0] = 3; a.arg[1] = 0x1234;
	check(sublet_bind_core(__NR_ioctl, &a) == -ENOTTY, "bind: an unknown ioctl command gets ENOTTY");
	memset(&a, 0, sizeof(a));
	a.arg[0] = 3; a.arg[1] = F_SETLK; a.arg[2] = ADDR; a.tagged = 0x4;
	check(sublet_bind_core(__NR_fcntl, &a) == 0 && NUMBER(a.arg[2]) == 3, "bind: fcntl F_SETLK binds arg");
	memset(&a, 0, sizeof(a));
	a.arg[0] = 3; a.arg[1] = F_SETFL; a.arg[2] = 1UL << 45;
	check(sublet_bind_core(__NR_fcntl, &a) == 0 && a.arg[2] == 1UL << 45, "bind: fcntl F_SETFL keeps an integer");
	memset(&a, 0, sizeof(a));
	a.arg[0] = 3; a.arg[1] = F_SETSIG;
	check(sublet_bind_core(__NR_fcntl, &a) == -EINVAL, "bind: fcntl F_SETSIG is refused");
	memset(&a, 0, sizeof(a));
	a.arg[0] = PR_SET_NAME; a.arg[1] = ADDR; a.tagged = 0x2;
	check(sublet_bind_core(__NR_prctl, &a) == 0 && NUMBER(a.arg[1]) == 2, "bind: prctl PR_SET_NAME binds arg2");
	memset(&a, 0, sizeof(a));
	a.arg[0] = PR_SET_MM;
	check(sublet_bind_core(__NR_prctl, &a) == -EINVAL, "bind: prctl PR_SET_MM is refused");
	/* Calls outside the allow-list. */
	memset(&a, 0, sizeof(a));
	check(sublet_bind_core(__NR_ptrace, &a) == -ENOSYS, "bind: ptrace gets ENOSYS");
	check(sublet_bind_core(__NR_process_vm_readv, &a) == -ENOSYS, "bind: process_vm_readv gets ENOSYS");
	check(sublet_bind_core(__NR_io_uring_setup, &a) == -ENOSYS, "bind: io_uring_setup gets ENOSYS");

	/* The forged integer: write(1, number 4 | address, 12) with a reference in a3, which
	 * write does not use. It must get EFAULT and never select a3's reference. */
	memset(&a, 0, sizeof(a));
	a.arg[0] = 1; a.arg[1] = ADDR | (4UL << 40); a.arg[2] = 12; a.arg[3] = ADDR; a.tagged = 0x8;
	before = a;
	rc = sublet_bind_core(__NR_write, &a);
	check(rc == -EFAULT, "forged: an integer with number bits in a pointer position gets EFAULT");
	check(!(rc == 0 && selects(a.arg[1], &before, &arg)),
	      "forged: the integer does not select another argument's reference");
	/* A carry into the guard bits never selects another reference either. */
	memset(&a, 0, sizeof(a));
	a.arg[0] = 1; a.arg[1] = (1UL << 38) | ADDR; a.arg[2] = 12;
	check(sublet_bind_core(__NR_write, &a) == -EFAULT, "forged: guard bits in an integer get EFAULT");

	printf("%d failures\n", failures);
	return failures;
}
