/* Thread-local storage for every context of a domain.
 *
 * musl reaches errno, the locale and the cancellation state through the thread
 * pointer, so a domain that never sets one faults at the first failing syscall
 * rather than at the first thread. Measured 2026-09-16: write() to a bad fd
 * returned -EBADF correctly, __syscall_ret recognised it correctly, and the
 * domain then halted in __errno_location with tp reading 0.
 *
 * ONE LAYOUT, musl's. __init_tls() cannot run here: it reads the TLS program
 * header out of auxv, and a domain has no auxv. So this file describes the
 * image's TLS segment to musl the way musl's static __init_tls would, from the
 * symbols the domain linker script defines around the PT_TLS segment, and then
 * every block is musl's own: __copy_tls lays out the first context's here,
 * musl's pthread_create lays out each thread's, and context.c a minted
 * context's. __init_tp then sets the first context's fields, as for any musl
 * program.
 *
 * Alignment. The compiler reaches a __thread variable as tp + %tprel(sym), with
 * tp a capability, RISC-V variant I: the block starts AT tp, struct pthread
 * just below it. link.ld aligns the segment's start to its own alignment, which
 * is not known here but is at most 4096 (link.ld asserts it), so the blocks
 * align tp to a page, which satisfies every alignment up to one. tp keeps the
 * allocation's capability, whose bounds cover the whole block.
 *
 * domain_main calls __capstone_init_tls on EVERY entry, so the first context's
 * block is made once and kept: thread-locals then live as long as the domain's
 * globals do, and a domain entered many times does not leak a block per entry.
 *
 * set_tid_address is answered, not refused (delegate.c): the unserved-syscall
 * instrument once reported it as the one thing a full stdio run asked for and
 * did not get, and a list with one permanent entry in it is a list people stop
 * reading.
 */
/* pthread_impl.h declares a locale_t member and does not pull the type in
   itself; musl's own users of it get locale_t from their preamble. Naming the
   header that supplies it is clearer than copying someone else's include list. */
#include <locale.h>
#include "pthread_impl.h"

int __capstone_init_tls(void);

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* The TLS template, placed by my_first_domain/link.ld. */
extern char __capstone_tls_image[], __capstone_tdata_end[], __capstone_tls_end[];

#define TLS_ALIGN 4096

static struct tls_module image_tls;

/* What musl's static __init_tls records for the main program's segment:
   __copy_tls reads it for every block. The size leaves room for the dtv
   (two words at the block's top) and for aligning tp. */
#ifdef CAPSTONE_GP_CAPTABLE_ABI
/* The silicon ABI (B0, docs/plans/b0-silicon-delegated-runtime.md): the template symbols above have no cap-table
   slot (naming them derives from gp and delins, which faults on silicon, C-13), so the extents come from the glue's
   accessors as integers. B0 cannot yet READ a non-empty .tdata (that needs a capability over the template), so one
   is refused; .tbss alone needs only its size. */
unsigned long __capstone_silicon_tls_tdata_bytes(void);
unsigned long __capstone_silicon_tls_mem_bytes(void);
#endif

static void describe_tls(void)
{
#ifdef CAPSTONE_GP_CAPTABLE_ABI
	size_t tdata = __capstone_silicon_tls_tdata_bytes();
	size_t memsz = __capstone_silicon_tls_mem_bytes();
	if (tdata != 0)
		abort();
	if (memsz) {
		image_tls.image = 0;
#else
	size_t tdata = __capstone_tdata_end - __capstone_tls_image;
	size_t memsz = __capstone_tls_end - __capstone_tls_image;
	if (memsz) {
		image_tls.image = __capstone_tls_image;
#endif
		image_tls.len = tdata;
		image_tls.size = memsz;
		image_tls.align = TLS_ALIGN;
		image_tls.offset = 0;
		libc.tls_head = &image_tls;
		libc.tls_cnt = 1;
	}
	libc.tls_align = TLS_ALIGN;
	libc.tls_size = (2 * sizeof(void *) + sizeof(struct pthread) + memsz + TLS_ALIGN + 15) & -16;
}

static char *__capstone_tp;

/* Set once the first context has its thread pointer; every context minted
   later has its TLS block from its first instruction (lock.c counts locks per
   context from here on). */
int __capstone_tls_ready;

int __capstone_init_tls(void)
{
	if (__capstone_tp)
		return __init_tp(__capstone_tp - sizeof(struct pthread));
	/* What musl's __init_libc takes from auxv's AT_PAGESZ, and a domain has
	   no auxv: PAGE_SIZE is libc.page_size on this port, so without it
	   sysconf(_SC_PAGESIZE) answered 0 and pthread_create's ROUND() sized
	   every thread's mapping 0. Linux's page, and mmap_shm_level0.c's. */
	libc.page_size = 4096;
	describe_tls();
	/* calloc: __init_tp expects every field it does not set to be zero, and
	   .tbss is the zeroed tail of the segment's copy. */
	unsigned char *mem = calloc(1, libc.tls_size);
	if (!mem)
		return -1;
	struct pthread *td = __copy_tls(mem);
	__capstone_tp = (char *)td + sizeof(struct pthread);
	if (__init_tp(td))
		return -1;
	__capstone_tls_ready = 1;
	return 0;
}

/* A minted context's TLS block (context.c): musl's layout, and the fields
 * __init_tp sets, except tid, which is the runtime's to assign. __init_tp is
 * not used: it installs tp for the CALLING context. */
size_t __capstone_tls_block_bytes(void)
{
	return libc.tls_size;
}

char *__capstone_tls_block_init(char *mem, size_t bytes)
{
	if (bytes < libc.tls_size)
		return 0;
	memset(mem, 0, bytes);
	struct pthread *td = __copy_tls((unsigned char *)mem);
	td->self = td;
	td->detach_state = DT_JOINABLE;
	td->locale = &libc.global_locale;
	td->robust_list.head = &td->robust_list.head;
	td->next = td->prev = td;
	return (char *)td + sizeof(struct pthread);
}

/* The calling context's thread identity: the pid for the first context once
 * the launch record is in (set below), before that 1, and for a minted
 * context the runtime identity context.c gave it. musl compares these in its
 * recursive locks, so no two live contexts share one. */
int __capstone_context_tid(void)
{
	int tid = __pthread_self()->tid;
	return tid ? tid : 1;
}

/* The thread pointer is installed before the launch record is applied, so
 * set_tid_address answered the placeholder 1. Once the record is in, the one
 * thread's tid is the task's pid: raise() and pthread_kill() send there. */
void __capstone_set_tid(int tid) { __pthread_self()->tid = tid; }
