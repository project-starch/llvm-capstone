/* The twelve entry points a WAMR platform must really implement, for a Capstone
 * domain. See platform_internal.h for what is deliberate here.
 *
 * The arena is one static array and os_mmap hands out slices of it in address
 * order, never reclaiming. That is enough for the runtime's own lifetime -- WAMR
 * mmaps its heap pool and each module's linear memory once at instantiation --
 * and it keeps the port from becoming an allocator in its own right, which would
 * put a second nesting level between the measurement and its subject.
 */
#include "bh_platform.h"

#ifndef CAPSTONE_WAMR_ARENA_BYTES
#define CAPSTONE_WAMR_ARENA_BYTES (1024u * 1024u)
#endif

static unsigned char wamr_arena[CAPSTONE_WAMR_ARENA_BYTES] __attribute__((aligned(16)));
static size_t wamr_arena_used;

int
bh_platform_init(void)
{
    wamr_arena_used = 0;
    return 0;
}

void
bh_platform_destroy(void)
{
}

/* os_malloc/os_free are NOT the runtime's allocator. WAMR routes every internal
   allocation through mem-alloc/ems, which it initialises over the buffer os_mmap
   returns; these three are only for the handful of allocations made before that
   exists. Backing them with the same bump arena keeps the count of allocators in
   this image at one. */
void *
os_malloc(unsigned size)
{
    size_t n = ((size_t)size + 15u) & ~(size_t)15u;
    if (n == 0 || n > CAPSTONE_WAMR_ARENA_BYTES - wamr_arena_used)
        return NULL;
    void *p = wamr_arena + wamr_arena_used;
    wamr_arena_used += n;
    return p;
}

void
os_free(void *ptr)
{
    /* ponytail: a bump arena does not reclaim. Ceiling: an image that mmaps and
       munmaps repeatedly exhausts CAPSTONE_WAMR_ARENA_BYTES. WAMR does not --
       it maps once per module instantiation -- and the upgrade path is a free
       list here, NOT routing to the domain libc, which would add the nesting
       level this port exists to avoid. */
    (void)ptr;
}

void *
os_realloc(void *ptr, unsigned size)
{
    void *p = os_malloc(size);
    if (p && ptr)
        memcpy(p, ptr, size);
    return p;
}

void *
os_mmap(void *hint, size_t size, int prot, int flags, os_file_handle file)
{
    (void)hint;
    (void)prot;
    (void)flags;
    (void)file;
    /* Page-align, because the vmcore assumes mmap returns page-aligned memory and
       sizes linear memory in pages. */
    size_t pad = (4096u - (wamr_arena_used & 4095u)) & 4095u;
    if (pad > CAPSTONE_WAMR_ARENA_BYTES - wamr_arena_used)
        return NULL;
    wamr_arena_used += pad;
    return os_malloc((unsigned)((size + 4095u) & ~(size_t)4095u));
}

void
os_munmap(void *addr, size_t size)
{
    (void)addr;
    (void)size;
}

int
os_mprotect(void *addr, size_t size, int prot)
{
    /* No MMU in a domain. Returning success is honest for this platform: there is
       nothing to protect and nothing that could enforce it. Capability bounds are
       the protection mechanism here and they are set elsewhere. */
    (void)addr;
    (void)size;
    (void)prot;
    return 0;
}

void
os_dcache_flush(void)
{
}

void
os_icache_flush(void *start, size_t len)
{
    (void)start;
    (void)len;
}

/* Single-threaded: a mutex cannot be contended, so these are not "unimplemented",
   they are complete. */
int
os_mutex_init(korp_mutex *mutex)
{
    *mutex = 0;
    return BHT_OK;
}

int
os_mutex_destroy(korp_mutex *mutex)
{
    (void)mutex;
    return BHT_OK;
}

int
os_mutex_lock(korp_mutex *mutex)
{
    (void)mutex;
    return BHT_OK;
}

int
os_mutex_unlock(korp_mutex *mutex)
{
    (void)mutex;
    return BHT_OK;
}

korp_tid
os_self_thread(void)
{
    return 1;
}

uint8 *
os_thread_get_stack_boundary(void)
{
    /* NULL means "unknown", which the vmcore handles by skipping its stack-overflow
       guard. Reporting a WRONG boundary would be worse than reporting none: the
       guard would then fire or fail to fire on the strength of a made-up number. */
    return NULL;
}

void
os_thread_jit_write_protect_np(bool enabled)
{
    (void)enabled;
}

uint64
os_time_get_boot_us(void)
{
    /* ponytail: no clock is wired up yet. Ceiling: anything WAMR times reads zero,
       so a profile taken through this is void rather than wrong-by-a-little. The
       upgrade path is the domain's cycle counter, and it should land before any
       timing number is quoted. */
    return 0;
}

uint64
os_time_thread_cputime_us(void)
{
    return 0;
}

int
os_getpagesize_impl(void)
{
    return 4096;
}
