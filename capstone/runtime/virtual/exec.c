#define _GNU_SOURCE
#include "wire.h"
#include "capstone/linux-domain-fault.h"
#include "capstone/spawn.h"
#include <linux/binfmts.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <sys/auxv.h>
#include "../linux/application-image.h"
#include "../linux/delegate-service.h"
#include <elf.h>
#include <errno.h>
#include <stdbool.h>
#include <limits.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <capstone/representable.h>
extern char **environ;
struct mapping {
    void *address;
    size_t bytes, visible;
    uint64_t id;
    int heap, kind;
};
/* Registered mappings sorted by address, highest first, grown on demand.
 * Registrations never overlap, so the highest mapping starting at or below an
 * address is the only one that can contain it. Linux places new mappings below
 * the existing ones (top-down mmap), so a new registration is appended and a
 * recent one is removed from near the end, without moving the rest. A pointer
 * into maps does not survive map_insert or map_remove. */
static struct mapping *maps;
static size_t map_count, map_capacity;
static void *reuse_address;
static size_t reuse_bytes;
static int device = -1, image_fd = -1;
static const char *image_path;
static struct capstone_spawner spawner = {.socket = -1};
static struct capstone_delegate_host host;
static __thread struct capstone_delegate_host *worker_host;
#define CV_MAX_THREADS 32
static uint64_t thread_ids[CV_MAX_THREADS] = {0};
static unsigned thread_done[CV_MAX_THREADS];
static unsigned thread_count = 1;
static pthread_mutex_t service_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t thread_state_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t thread_state_cond = PTHREAD_COND_INITIALIZER;
static unsigned long long launch_ns, start_ns;
static int trace;
static void *meta;
static void *virtual_thread_worker(void *opaque);

static int thread_add(uint64_t id)
{
    int result = 0;
    if (!id) return -EINVAL;
    pthread_mutex_lock(&thread_state_lock);
    if (thread_count >= CV_MAX_THREADS) result = -EAGAIN;
    else {
        for (unsigned i = 1; i < CV_MAX_THREADS; ++i)
            if (!thread_ids[i]) {
                thread_ids[i] = id;
                thread_done[i] = 0;
                ++thread_count;
                break;
            }
    }
    pthread_mutex_unlock(&thread_state_lock);
    return result;
}
static void thread_remove(uint64_t id)
{
    pthread_mutex_lock(&thread_state_lock);
    for (unsigned i = 1; i < CV_MAX_THREADS; ++i) {
        if (thread_ids[i] != id) continue;
        thread_ids[i] = 0;
        thread_done[i] = 0;
        --thread_count;
        pthread_mutex_unlock(&thread_state_lock);
        return;
    }
    pthread_mutex_unlock(&thread_state_lock);
}
static void thread_mark_done(uint64_t id)
{
    pthread_mutex_lock(&thread_state_lock);
    for (unsigned i = 1; i < CV_MAX_THREADS; ++i)
        if (thread_ids[i] == id) {
            thread_done[i] = 1;
            pthread_cond_broadcast(&thread_state_cond);
            break;
        }
    pthread_mutex_unlock(&thread_state_lock);
}
static int thread_join(uint64_t id)
{
    int result = 0;
    pthread_mutex_lock(&thread_state_lock);
    for (unsigned i = 1; i < CV_MAX_THREADS; ++i) {
        if (thread_ids[i] != id) continue;
        while (!thread_done[i])
            pthread_cond_wait(&thread_state_cond, &thread_state_lock);
        thread_ids[i] = 0;
        thread_done[i] = 0;
        --thread_count;
        pthread_mutex_unlock(&thread_state_lock);
        return 0;
    }
    result = -ESRCH;
    pthread_mutex_unlock(&thread_state_lock);
    return result;
}
static unsigned long long now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (unsigned long long)t.tv_sec * 1000000000 + t.tv_nsec;
}

static size_t rounded(size_t n)
{
    size_t size = 4096;
    if (!n || n > CV_MAX_REGION_BYTES) return 0;
    while (size < n) size <<= 1;
    return size;
}
static void *private_pages(void *p, size_t bytes)
{
    /* Demand resolution accounts for one base page at a time. The module
     * requires this VMA policy on kernels that can instantiate huge pages.
     * A kernel built without them answers EINVAL; it does every time, so it
     * is asked only once. */
    static int no_huge_pages;
    if (no_huge_pages) return p;
    if (madvise(p, bytes, MADV_NOHUGEPAGE)) {
        if (errno == EINVAL) { no_huge_pages = 1; return p; }
        int error = errno; munmap(p, bytes); errno = error; return NULL;
    }
    return p;
}
static void *reserve(size_t bytes)
{
    /* Let Linux reuse the most recently retired range if it is still free.
     * A fresh grant always creates new identities, even at this same VA. */
    if (reuse_address && bytes == reuse_bytes) {
        void *p = mmap(reuse_address, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        reuse_address = NULL;
        if (p != MAP_FAILED) return private_pages(p, bytes);
    }
    char *raw = mmap(NULL, bytes * 2, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return NULL;
    uintptr_t base = ((uintptr_t)raw + bytes - 1) & ~(bytes - 1);
    if (base != (uintptr_t)raw) munmap(raw, base - (uintptr_t)raw);
    size_t tail = (uintptr_t)raw + bytes * 2 - base - bytes;
    if (tail) munmap((void *)(base + bytes), tail);
    return private_pages((void *)base, bytes);
}
static void *reserve_aligned(size_t bytes, size_t alignment)
{
    /* mmap() promises page alignment only. Over-map by the worst-case distance to the next
     * alignment boundary -- the raw base is already page-aligned, so that is alignment - 4096
     * -- then trim both ends, leaving EXACTLY `bytes` mapped. Mapping exactly `bytes` is the
     * point: remap_mapping() below matches a heap on m->bytes == m->visible, so a padded heap
     * can never grow. At alignment == 4096 the slack is zero and this reduces to a plain
     * mmap() of `bytes`, which is what the heap path used to do. */
    size_t slack = alignment - 4096;
    char *raw = mmap(NULL, bytes + slack, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return NULL;
    uintptr_t base = ((uintptr_t)raw + alignment - 1) & ~(uintptr_t)(alignment - 1);
    if (base != (uintptr_t)raw) munmap(raw, base - (uintptr_t)raw);
    size_t tail = (uintptr_t)raw + bytes + slack - (base + bytes);
    if (tail) munmap((void *)(base + bytes), tail);
    return private_pages((void *)base, bytes);
}
/* The highest mapping starting at or below p, or NULL. */
static struct mapping *map_floor(uintptr_t p)
{
    size_t low = 0, high = map_count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        if ((uintptr_t)maps[middle].address > p) low = middle + 1;
        else high = middle;
    }
    return low < map_count ? &maps[low] : NULL;
}
static struct mapping *map_find(uintptr_t base)
{
    struct mapping *m = map_floor(base);
    return m && (uintptr_t)m->address == base ? m : NULL;
}
/* Room for one more mapping, made before the kernel mints its grant. */
static int map_reserve(void)
{
    if (map_count < map_capacity) return 0;
    size_t capacity = map_capacity ? map_capacity * 2 : 64;
    struct mapping *grown = realloc(maps, capacity * sizeof(*maps));
    if (!grown) { errno = ENOMEM; return -1; }
    maps = grown; map_capacity = capacity;
    return 0;
}
static void map_insert(struct mapping m)
{
    struct mapping *below = map_floor((uintptr_t)m.address);
    size_t at = below ? (size_t)(below - maps) : map_count;
    memmove(&maps[at + 1], &maps[at], (map_count - at) * sizeof(*maps));
    maps[at] = m; ++map_count;
}
static void map_remove(struct mapping *m)
{
    size_t at = (size_t)(m - maps);
    memmove(&maps[at], &maps[at + 1], (map_count - at - 1) * sizeof(*maps));
    --map_count;
}
static int grant(void *base, size_t bytes, unsigned reg, unsigned perms,
                 uintptr_t cursor, int linear, uint64_t thread, size_t rx_bytes)
{
    if (map_reserve()) return -1;
    /* Option (iii): refuse to mint an arena whose bounds the encoder cannot represent, rather
     * than hand the domain a capability silently widened below its base or above its end.
     *
     * CORRECTION (2026-10-17): an earlier version of this comment claimed the check "cannot
     * fire today", on the grounds that every size reaching a grant is a power of two. That was
     * wrong in the case that matters most. The local-malloc heap grants `visible` bytes, which
     * is page-rounded and granule-rounded but NOT a power of two -- a 6 MiB heap grants
     * 6291456 -- so this check is load-bearing for exactly the path option (iii) changed, and
     * it is COUPLED to the alignment heap.c requests: it passes only because acquire_mapping()
     * now aligns that mapping to the granule. The two must land together.
     *
     * It is also not true that every arena passes through this function: remap_mapping() grows
     * a heap through CV_REMAP_BEGIN/END and never calls grant(), which is why that function
     * carries its own alignment guarantee. Negative-tested in
     * tests/exact-bounds-iii/grant-repr-test.c. */
    if (!capstone_repr_ok((uintptr_t)base, bytes)) { errno = ERANGE; return -1; }
    struct cv_map r = {thread, (uintptr_t)base, bytes, perms, reg, cursor, linear, 0, rx_bytes};
    if (ioctl(device, CV_ADD, &r)) {
        /* Evaluate add()'s two disjuncts on the REAL values the module saw. The domain cannot do
         * this: cap_vm_acquire clears its slot on failure, so the base never reaches it, and a
         * power-of-two size refused with EINVAL is then indistinguishable from a misaligned one. */
        /* NOT gated on `trace`. A refusal is rare, so this cannot flood anything, and gating
         * it is what made its absence unprovable: four of this suite's own invocations
         * (population, churn, remap-align, the denial loop) run without CAPSTONE_VM_TRACE, so an
         * "ADD_REFUSED absent" check over them was an unproven zero. Turning the trace on
         * globally is not the fix -- churn alone performs 70,000 allocations and would bury the
         * serial in boundary lines. A diagnostic on an error path does not need a verbosity flag. */
        int err = errno;
        fprintf(stderr, "CAPSTONE_VM_ADD_REFUSED base=0x%lx bytes=%zu errno=%d"
                        " pow2=%d naturally_aligned=%d\n",
                (unsigned long)r.address, bytes, err,
                (bytes & (bytes - 1)) == 0, (r.address & (bytes - 1)) == 0);
        errno = err;
        return -1;
    }
    /* Every ADMITTED grant, in order, so two modules' grant sequences can be diffed. The
     * refusal side is already printed; without the success side a diff cannot say WHICH grant
     * first differs, only that something did. */
    /* slot= is the number of mappings registered before this one: the old array slot until the
     * first unmap, so the image's grant still reads slot=0 for the board lane's parsers. */
    fprintf(stderr, "CAPSTONE_VM_ADD_OK slot=%zu base=0x%lx bytes=%zu reg=%u perms=%u linear=%d\n",
            map_count, (unsigned long)(uintptr_t)base, bytes, reg, perms, linear);
    map_insert((struct mapping){.address = base, .bytes = bytes, .visible = bytes,
                                .id = r.id, .heap = linear,
                                .kind = linear ? CV_MAP_HEAP : -1});
    return 0;
}
static long unmap_arena(uintptr_t base, size_t bytes)
{
    struct mapping *m = map_find(base);
    if (!m || m->visible != bytes || !m->heap || m->kind == CV_MAP_METADATA)
        return -EINVAL;
    if (ioctl(device, CV_RETIRE, &m->id)) return -errno;
    if (munmap(m->address, m->bytes)) return -errno;
    reuse_address = m->address; reuse_bytes = m->bytes;
    map_remove(m);
    return 0;
}
static long protect_range(uintptr_t base, size_t bytes, int prot)
{
    if (!bytes || (base & 4095) || (bytes & 4095) ||
        (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC))) return -EINVAL;
    struct mapping *m = map_floor(base);
    uintptr_t at = m ? (uintptr_t)m->address : 0;
    if (!m || !m->heap || m->kind == CV_MAP_METADATA ||
        base - at > m->visible || bytes > m->visible - (base - at)) return -EINVAL;
    return mprotect((void *)base, bytes, prot) ? -errno : 0;
}

static int registered_span(uintptr_t p, size_t bytes)
{
    struct mapping *m = map_floor(p);
    uintptr_t base = m ? (uintptr_t)m->address : 0;
    return m && p - base <= m->visible && m->visible - (p - base) >= bytes;
}
static int registered_word(uintptr_t p)
{ return !(p & 3) && registered_span(p, sizeof(int)); }
static long acquire_mapping(uint64_t tid, struct cv_step *step)
{
    size_t visible = step->args[0], alignment = step->args[1];
    unsigned prot = step->args[2], rights = step->args[3], kind = step->args[4];
    size_t bytes = rounded(visible);
    if (!bytes || (visible & 4095) || alignment < 4096 ||
        (alignment & (alignment - 1)) || alignment > CV_MAX_REGION_BYTES ||
        (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC)) || rights > 7 ||
        kind > CV_MAP_METADATA ||
        (kind != CV_MAP_APPLICATION && (prot != (PROT_READ | PROT_WRITE) || rights != 6)) ||
        (kind == CV_MAP_APPLICATION && rights != 7)) return -EINVAL;
    if (bytes < alignment) bytes = alignment;
    void *p;
    if (kind == CV_MAP_HEAP) {
        /* Keyed on the HEAP, not on the alignment value. What requires bytes == visible is the
         * heap's growth contract -- remap_mapping() matches on m->bytes == m->visible == old --
         * and the old `alignment == 4096` spelling conflated that with the alignment mmap()
         * happens to provide for free. Option (iii) asks for the bounds granule, which exceeds
         * a page above 4 MiB, so every heap of 4 MiB or more silently left this path, got
         * padded to a power of two by reserve(), and could no longer grow at all. */
        bytes = visible;
        p = reserve_aligned(bytes, alignment);
    } else p = reserve(bytes);
    if (!p) return -errno;
    /* Padding remains inaccessible, including after later mprotect calls.
     * No capability reply exists until registration has succeeded. The range
     * is mapped readable and writable, so that protection needs no call. */
    if ((visible < bytes && mprotect((char *)p + visible, bytes - visible, PROT_NONE)) ||
        (prot != (PROT_READ | PROT_WRITE) && mprotect(p, visible, prot)) || grant(p, bytes, 10, rights, (uintptr_t)p, 1, tid, 0)) {
        int error = errno;
        munmap(p, bytes);
        return -error;
    }
    /* grant() registered p under this address; a miss is a registry defect,
     * and the launcher has nothing to answer with. */
    struct mapping *m = map_find((uintptr_t)p);
    if (!m) abort();
    m->visible = visible; m->kind = kind;
    return 0;
}
static long remap_mapping(uint64_t tid, struct cv_step *step)
{
    uintptr_t base = step->args[0];
    size_t old = step->args[1], bytes = step->args[2];
    /* Unconditional, for the same reason the CV_ADD refusal is: heap growth has never been
     * observed to COMPLETE in any arm of any profile -- CAPSTONE_VM_REMAP_OK reads 0 everywhere,
     * including with exact bounds on -- and remap_begin's own gate is not shown to fire. So the
     * question is which of these refusals happens, and a trace-gated print cannot answer it in the
     * stages that run without the trace. */
    if (!bytes || bytes > CV_MAX_REGION_BYTES ||
        ((base | old | bytes) & 4095)) {
        fprintf(stderr, "CAPSTONE_VM_REMAP_REFUSED stage=args base=0x%lx"
                        " old=%zu bytes=%zu page_ok=%d\n",
                (unsigned long)base, old, bytes,
                ((base | old | bytes) & 4095) == 0);
        return -EINVAL;
    }
    struct mapping *m = map_find(base);
    if (m && m->visible == old && m->bytes == old && m->heap && m->kind == CV_MAP_HEAP) {
        /* Option (iii): the GROWN mapping's base must still be a granule multiple, and mremap()
         * promises only page alignment. Above 4 MiB the granule exceeds a page, so even an
         * in-place growth can leave the base unrepresentable (2 MiB -> 4 MiB keeps the base and
         * moves the granule from 4096 to 8192).
         *
         * This has to be guaranteed HERE and not refused in the domain. The domain's
         * __capstone_malloc_group_remap() checks representability only after this ioctl pair
         * has already moved the memory and re-registered it, so refusing there is a
         * use-after-move, not a fallback -- realloc's malloc-plus-copy path would read from an
         * address that no longer maps. A nonzero __capstone_malloc_remap_unaligned therefore
         * means a bug in this function, not a measurement.
         *
         * Everything that can fail is done BEFORE the source is touched: mremap() fails
         * atomically, so an error leaves the old mapping intact and only the reservation to
         * release. */
        size_t align = capstone_repr_granule(bytes);
        if (align < 4096) align = 4096;
        if (bytes % align) {
            fprintf(stderr, "CAPSTONE_VM_REMAP_REFUSED stage=granule bytes=%zu align=%zu\n",
                    bytes, align);
            return -ERANGE;
        }
        void *dest = NULL;
        if (align > 4096) {
            dest = reserve_aligned(bytes, align);
            if (!dest) return -errno;
        }
        struct cv_remap r = {tid, m->id, bytes, 0};
        if (ioctl(device, CV_REMAP_BEGIN, &r)) {
            int e = errno;
            if (dest) munmap(dest, bytes);
            return -e;
        }
        void *p = dest ? mremap(m->address, old, bytes, MREMAP_MAYMOVE | MREMAP_FIXED, dest)
                       : mremap(m->address, old, bytes, MREMAP_MAYMOVE);
        if (p == MAP_FAILED) {
            int e = errno;
            if (ioctl(device, CV_REMAP_END, &r)) abort();
            if (dest) munmap(dest, bytes);
            return -e;
        }
        r.address = (uintptr_t)p;
        if (ioctl(device, CV_REMAP_END, &r)) abort();
        /* A moved mapping is re-sorted; removing it first leaves room. */
        struct mapping moved = *m;
        moved.address = p; moved.bytes = moved.visible = bytes;
        map_remove(m); map_insert(moved);
        if (trace) fprintf(stderr, "CAPSTONE_VM_REMAP_OK old=%zu bytes=%zu\n", old, bytes);
        return 0;
    }
    /* No arena matched. Report what the candidates looked like, because the match is on four
     * fields at once and "no match" alone cannot say which one differed. */
    fprintf(stderr, "CAPSTONE_VM_REMAP_REFUSED stage=nomatch base=0x%lx old=%zu bytes=%zu\n",
            (unsigned long)base, old, bytes);
    if (m) {   /* only the arena at this base matters */
        fprintf(stderr, "CAPSTONE_VM_REMAP_CAND base=0x%lx visible=%zu bytes=%zu heap=%d kind=%d\n",
                (unsigned long)(uintptr_t)m->address, m->visible, m->bytes, m->heap, m->kind);
    }
    return -EINVAL;
}
static void cleanup(void)
{
    if (device >= 0) { close(device); device = -1; }
    for (size_t i = 0; i < map_count; ++i)
        munmap(maps[i].address, maps[i].bytes);
    if (image_fd >= 0) { close(image_fd); image_fd = -1; }
    capstone_spawner_stop(&spawner);
    capstone_delegate_host_free(&host);
}
static int die(const char *what)
{
    fprintf(stderr, "capstone-vexec: %s: %s\n", what, strerror(errno));
    cleanup(); return 125;
}
/* Whether the NUL-terminated section name at `at` in a string table of `len` bytes is `want`,
 * without reading past the table. */
static int section_named(const char *names, uint64_t len, uint64_t at, const char *want)
{
    size_t n = strlen(want) + 1;
    return at <= len && n <= len - at && !memcmp(names + at, want, n);
}
/* The image as virtual/link.ld lays it out, checked against the sealed file:
 * exactly two load segments, read+execute text from the image's first page
 * and read+write data starting on the page right after it. The text segment
 * is whole pages of the file (p_filesz == p_memsz), so mapping it maps nothing
 * but image bytes; the data segment's file part may end inside a page. */
struct image_layout {
    uint64_t low, text_offset, text_bytes, data_offset, data_file, data_bytes, entry;
    size_t stack;
};
static int image_layout(const unsigned char *raw, size_t size, struct image_layout *l)
{
    /* image_fd_layout validated this mapping before asking. A change below the VFS in between
     * (the residual named there) may still make these bytes garbage; it must not make them an
     * out-of-bounds read in this process. So each header field is read ONCE into a local,
     * checked against the file size before it is used as an offset, count or length, and the
     * checked local is what every later step uses. */
    const Elf64_Ehdr *h = (const void *)raw;
    const uint64_t phoff = h->e_phoff, shoff = h->e_shoff, entry_va = h->e_entry;
    const unsigned phnum = h->e_phnum, shnum = h->e_shnum, shstr = h->e_shstrndx;
    if (phoff > size || phnum > (size - phoff) / sizeof(Elf64_Phdr) ||
        shoff > size || shnum > (size - shoff) / sizeof(Elf64_Shdr) || shstr >= shnum)
        return -1;
    const Elf64_Phdr *ph = (const void *)(raw + phoff);
    const Elf64_Shdr *sh = (const void *)(raw + shoff);
    const uint64_t names_at = sh[shstr].sh_offset, names_len = sh[shstr].sh_size;
    if (names_at > size || names_len > size - names_at) return -1;
    const char *names = (const void *)(raw + names_at);
    Elf64_Phdr seg[2];
    unsigned loads = 0;
    int virtual = 0;
    for (unsigned i = 0; i < phnum; ++i) {
        const Elf64_Phdr p = ph[i];
        if (p.p_type != PT_LOAD) continue;
        if (loads == 2) return -1;
        seg[loads++] = p;
    }
    if (loads != 2) return -1;
    const Elf64_Phdr *t = &seg[0], *d = &seg[1];
    if (t->p_flags != (PF_R | PF_X) || d->p_flags != (PF_R | PF_W) ||
        (t->p_vaddr | t->p_offset | t->p_filesz | d->p_vaddr | d->p_offset) & 4095 ||
        !t->p_filesz || t->p_filesz != t->p_memsz ||
        d->p_vaddr != t->p_vaddr + t->p_memsz || d->p_filesz > d->p_memsz || !d->p_memsz ||
        entry_va < t->p_vaddr || entry_va - t->p_vaddr >= t->p_filesz)
        return -1;
    /* Each segment's sizes are bounded, so no sum below can wrap, and its file part lies in
     * the file. */
    if (t->p_memsz > CV_MAX_REGION_BYTES || d->p_memsz > CV_MAX_REGION_BYTES ||
        t->p_offset > size || t->p_filesz > size - t->p_offset ||
        d->p_offset > size || d->p_filesz > size - d->p_offset)
        return -1;
    l->stack = 1u << 20;
    for (unsigned i = 0; i < shnum; ++i) {
        uint64_t v[3];
        const Elf64_Shdr s = sh[i];
        if (s.sh_type != SHT_PROGBITS) continue;
        if (section_named(names, names_len, s.sh_name, ".capstone_virtual") && s.sh_size == 8 &&
            s.sh_offset <= size && 8 <= size - s.sh_offset) {
            memcpy(v, raw + s.sh_offset, 8);
            virtual = v[0] == CV_IMAGE_MAGIC;
        }
        if (section_named(names, names_len, s.sh_name, ".capstone_domreq") && s.sh_size == 24 &&
            s.sh_offset <= size && 24 <= size - s.sh_offset) {
            memcpy(v, raw + s.sh_offset, 24);
            l->stack = rounded(v[2]);
        }
    }
    if (!virtual || !l->stack) return -1;
    l->low = t->p_vaddr;
    l->text_offset = t->p_offset; l->text_bytes = t->p_memsz;
    l->data_offset = d->p_offset; l->data_file = d->p_filesz;
    l->data_bytes = (d->p_memsz + 4095) & ~(uint64_t)4095;
    l->entry = entry_va;
    return 0;
}
/* Map the image, validate the mapping and read the layout from it. One mapping, so the
 * headers and spans checked are the ones indexed; a second mapping would trust a check
 * made on other pages. The denial (CV_IMAGE) keeps every writer Linux counts out from
 * before this until the process ends; a writer below the VFS (a raw block device, a loop
 * device's backing file) is not stopped. Both the loader and exec in place (before it
 * replaces this process) ask. */
static int image_fd_layout(int fd, struct capstone_application_descriptor_v2 *desc,
                           struct image_layout *l)
{
    struct stat st;
    if (fstat(fd, &st)) return -1;
    if (!S_ISREG(st.st_mode) || st.st_size < (off_t)sizeof(Elf64_Ehdr) ||
        st.st_size > 512 * 1024 * 1024) { errno = ENOEXEC; return -1; }
    unsigned char *raw = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (raw == MAP_FAILED) return -1;
    if (capstone_application_validate(raw, st.st_size, desc)) {
        int error = errno;
        munmap(raw, st.st_size); errno = error; return -1;
    }
    int rc = image_layout(raw, st.st_size, l);
    munmap(raw, st.st_size);
    if (rc) errno = ENOEXEC;
    return rc;
}
/* Map the image instead of copying it. Text is a private, read+execute
 * mapping of the sealed file: no store reaches it, so it holds no tag and
 * the module neither pins nor sweeps it, and Linux faults it in from the page
 * cache as it is executed. Data is a private writable mapping of the file
 * (the module's pins break copy-on-write, so the domain owns its pages), the
 * page where the file part ends is built in anonymous memory, and the rest of
 * .bss stays anonymous. Linux synchronises the instruction cache when it maps
 * an executable page; the launcher writes none. */
/* As private_pages: kernels without transparent huge pages refuse the advice. */
static int no_huge_pages(void *p, size_t bytes)
{
    return madvise(p, bytes, MADV_NOHUGEPAGE) && errno != EINVAL ? -1 : 0;
}
static int load(int fd, struct capstone_application_descriptor_v2 *desc, void **image,
                size_t *bytes, size_t *rx_bytes, uintptr_t *entry, size_t *stack)
{
    struct image_layout l;
    if (image_fd_layout(fd, desc, &l)) return -1;
    /* The grant's size, rounded as every grant is. The tail past the data's own pages stays
     * anonymous and zero, like bss. */
    *bytes = rounded(l.text_bytes + l.data_bytes);
    *rx_bytes = l.text_bytes;
    *stack = l.stack;
    if (!*bytes || *bytes > CV_MAX_EAGER_BYTES) { errno = ENOEXEC; return -1; }
    char *base = reserve(*bytes);
    if (!base) return -1;
    char *data = base + l.text_bytes;
    size_t whole = l.data_file & ~(uint64_t)4095, part = l.data_file - whole;
    if (mmap(base, l.text_bytes, PROT_READ | PROT_EXEC, MAP_PRIVATE | MAP_FIXED, fd,
             l.text_offset) == MAP_FAILED ||
        no_huge_pages(base, l.text_bytes) ||
        (whole && (mmap(data, whole, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, fd,
                        l.data_offset) == MAP_FAILED ||
                   no_huge_pages(data, whole))) ||
        (part && pread(fd, data + whole, part, l.data_offset + whole) != (ssize_t)part)) {
        int error = errno ? errno : EIO;
        munmap(base, *bytes); errno = error; return -1;
    }
    *image = base;
    *entry = (uintptr_t)base + l.entry - l.low;
    return 0;
}
struct exec_state {
  uint64_t magic;
  struct capstone_spawner spawner;
  unsigned child_count;
  int image;
  pid_t children[CAPSTONE_DELEGATE_CHILDREN];
};
#define EXEC_STATE_MAGIC UINT64_C(0x4350455845433031)

/* Deny writers to the image held by *fd (CV_IMAGE). Where the module cannot
 * (EOPNOTSUPP: a filesystem that does not keep one inode per file, such as
 * 9p), *fd becomes a sealed copy read from that same open file, reopened
 * read-only, and writers to the copy are denied instead. */
static int image_denied(int *fd, struct capstone_application_descriptor_v2 *desc)
{
    char path[64];
    if (capstone_application_fixed(*fd))
        return -1;
    if (!ioctl(device, CV_IMAGE, fd))
        return 0;
    if (errno == ENOTTY || errno == EINVAL) {
        /* An ioctl the module does not know: a module from before CV_IMAGE.
         * Module and launcher are one revision; there is no older path. */
        fprintf(stderr, "capstone-vexec: the capstone_vm module does not implement CV_IMAGE: "
                        "it is from another revision than this launcher\n");
        errno = ENOSYS;
        return -1;
    }
    if (errno != EOPNOTSUPP)
        return -1;
    int copy = capstone_application_copy(*fd, desc);
    if (copy < 0)
        return -1;
    snprintf(path, sizeof path, "/proc/self/fd/%d", copy);
    int sealed = capstone_application_open(path);
    int error = errno;
    close(copy);
    if (sealed < 0) {
        errno = error;
        return -1;
    }
    close(*fd);
    *fd = image_fd = sealed;
    return capstone_application_fixed(sealed) || ioctl(device, CV_IMAGE, fd) ? -1 : 0;
}

static int above_stdio(int fd) {
  if (fd < 0 || fd >= 3) return fd;
  int parked = fcntl(fd, F_DUPFD_CLOEXEC, 3);
  int error = errno;
  close(fd);
  errno = error;
  return parked;
}

static long exec_in_place(void) {
  static char *argv[CAPSTONE_SPAWN_STRINGS + 7], *envp[CAPSTONE_SPAWN_STRINGS + 1];
  static const char *paths[CAPSTONE_SPAWN_ACTIONS];
  struct capstone_spawn_view view;
  struct capstone_application_descriptor_v2 descriptor;
  host.exec_requested = 0;
  if (capstone_spawn_unpack(host.exec_block, host.exec_bytes, argv + 6,
                            CAPSTONE_SPAWN_STRINGS + 1, envp, CAPSTONE_SPAWN_STRINGS + 1, paths,
                            CAPSTONE_SPAWN_ACTIONS, &view))
    return -EINVAL;
  /* The new image is copied into a sealed memfd and checked here, while the
   * caller still exists to receive any error. The copy, not the file, is what
   * crosses the exec: nothing can change it between this check and the new
   * launcher's, and the module's denial of writers ends with this process. */
  int checked = above_stdio(capstone_application_image(view.path, &descriptor));
  if (checked < 0) return -errno;
  if (!view.argc) { close(checked); return -EINVAL; }
  /* An image this launcher cannot load fails here, while the caller still
   * exists to receive the error, not after the exec. */
  struct image_layout layout;
  if (image_fd_layout(checked, &descriptor, &layout)) { close(checked); return -ENOEXEC; }
  struct exec_state state = {.magic = EXEC_STATE_MAGIC, .spawner = spawner,
                            .child_count = host.child_count, .image = checked};
  memcpy(state.children, host.children, sizeof state.children);
  int fd = above_stdio(memfd_create("capstone-exec-state", MFD_CLOEXEC));
  if (fd < 0) { int error = errno; close(checked); return -error; }
  int error = 0;
  if (write(fd, &state, sizeof state) != sizeof state || lseek(fd, 0, SEEK_SET) < 0 ||
      fcntl(fd, F_SETFD, 0) < 0 || fcntl(checked, F_SETFD, 0) < 0 ||
      (spawner.socket >= 0 && fcntl(spawner.socket, F_SETFD, 0) < 0)) {
    error = errno ? errno : EIO;
  } else {
    char fd_string[32];
    snprintf(fd_string, sizeof fd_string, "%d", fd);
    argv[0] = spawner.self;
    argv[1] = "--resume";
    argv[2] = fd_string;
    argv[3] = "--application-argv";
    argv[4] = (char *)view.path;
    argv[5] = "--";
    execve(spawner.self, argv, envp);
    error = errno;
  }
  close(fd);
  close(checked);
  if (spawner.socket >= 0) fcntl(spawner.socket, F_SETFD, FD_CLOEXEC);
  return -error;
}

/* The domain's main stack, as reserved and granted in main(): the bounds the frame walk below
 * checks every read against. */
static void *domain_stack;
static size_t domain_stack_bytes;

/* THE FRAME CHAIN: the return addresses up the domain's stack from the faulting frame, read by
 * the launcher (the stack is this process's own mapping, granted to the domain, and the domain
 * is stopped on a terminal step), each frame validated before it is read, bounded. The
 * compiler's prologue stores ra as a scalar at fp-16 and the caller's fp as a capability at
 * fp-32 (`sd ra, 0x40(sp); stc s0, 0x30(sp); s0 = sp + 0x50`), and a capability slot's low word
 * is its cursor, which is how the module reads the register slots (capstone_vm.c `r.fp =
 * t->frame[24]`). So a frame is {fp-32: the caller's fp, fp-16: the return address}. The chain
 * stops at a frame outside the stack or misaligned, a zero return address, a caller's fp not
 * above this one, or the 16th frame. A thread's stack is not this mapping, so a worker's fault
 * gets no chain rather than a wrong one. Diagnostic: nothing here decides on it. */
static unsigned frame_chain(uint64_t fp, uint64_t *out, unsigned max)
{
    uint64_t lo = (uint64_t)(uintptr_t)domain_stack;
    if (!domain_stack) return 0;
    /* the walk itself lives in the delegate service, where a native test can hand it a
     * buffer (runtime/tests/application/delegate-service-test.c) */
    return capstone_delegate_frame_chain(fp, lo, lo + domain_stack_bytes, out, max);
}

static void fault(const struct cv_step *step)
{
    struct capstone_delegate_host *h = worker_host ? worker_host : &host;
    uint64_t chain[16];
    unsigned chain_len = step->fp ? frame_chain(step->fp, chain, 16) : 0;
    if (!h->image_sha256[0] && image_fd >= 0)
        capstone_application_hash(image_fd, h->image_sha256);
    const char *record = getenv("CAPSTONE_FAULT_RECORD");
    int fd = record && *record ? open(record, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600) : -1;
    if (fd >= 0) {
        capstone_delegate_fault_record(fd, h, image_path, step->cause, step->pc, step->address,
                                       step->ra, step->fp, chain, chain_len);
        close(fd);
    }
    if (isatty(2) || getenv("CAPSTONE_EXEC_DIAGNOSTICS"))
        capstone_delegate_fault_record(2, h, image_path, step->cause, step->pc, step->address,
                                       step->ra, step->fp, chain, chain_len);
    /* Signal death stops the process before Linux releases shared mappings. */
    capstone_domain_exit_on_fault(CAPSTONE_DOMAIN_FAULT_RETVAL, NULL, NULL);
}
static int reserve_stdio(unsigned *mask) {
  *mask = 0;
  for (int i = 0; i < 3; ++i) {
    if (fcntl(i, F_GETFD) >= 0) {
      *mask |= 1u << i;
      continue;
    }
    if (errno != EBADF)
      return -1;
    int fd = open("/dev/null", O_RDWR);
    if (fd < 0)
      return -1;
    if (fd != i) {
      int rc = dup2(fd, i);
      close(fd);
      if (rc < 0)
        return -1;
    }
  }
  return 0;
}

static int virtual_service(uint64_t tid, struct cv_step *step)
{
    if (trace) fprintf(stderr, "CAPSTONE_VM_BOUNDARY kind=%llu\n",
                       (unsigned long long)step->args[7]);
    step->reply = 1;
    step->result = 0;
    if (step->args[7] == CV_SERVICE_THREAD_SELF) {
        step->result = tid + 1; /* zero remains the wire lock's vacant value */
    } else if (step->args[7] == CV_SERVICE_THREAD_CREATE) {
        struct cv_thread_create request = {.frame = step->args[0]};
        if (ioctl(device, CV_THREAD_CREATE, &request)) {
            step->result = -errno;
        } else if (thread_add(request.thread)) {
            struct cv_thread_control rollback = {.thread = request.thread};
            ioctl(device, CV_THREAD_EXIT, &rollback);
            step->result = -EAGAIN;
        } else {
            uint64_t *child = malloc(sizeof(*child));
            pthread_t worker;
            if (!child || pthread_create(&worker, NULL, virtual_thread_worker,
                                         (child ? (*child = request.thread), child : NULL))) {
                free(child);
                thread_remove(request.thread);
                struct cv_thread_control rollback = {.thread = request.thread};
                ioctl(device, CV_THREAD_EXIT, &rollback);
                step->result = -EAGAIN;
            } else {
                pthread_detach(worker);
                step->result = request.thread;
            }
        }
    } else if (step->args[7] == CV_SERVICE_THREAD_EXIT || step->args[7] == CV_SERVICE_PTHREAD_EXIT) {
        struct cv_thread_control request = {.thread = tid};
        int posix = step->args[7] == CV_SERVICE_PTHREAD_EXIT;
        uintptr_t clear = posix ? step->args[0] : 0;
        if (clear && !registered_word(clear)) return -1;
        if (tid == 0 || ioctl(device, CV_THREAD_EXIT, &request))
            return -1;
        if (request.frame) {
            long cleanup_rc = unmap_arena(request.frame, 4096);
            if (cleanup_rc && cleanup_rc != -EINVAL) return -1;
        }
        thread_mark_done(tid);
        /* musl join may retire the child's stack/TLS after this store.
         * The virtual CPU has stopped and its tagged frame is gone first. */
        if (posix && step->args[1] && unmap_arena(step->args[1], step->args[2])) return -1;
        if (clear) {
            __atomic_store_n((int *)clear, 0, __ATOMIC_RELEASE);
            syscall(SYS_futex, (int *)clear, FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0);
            thread_remove(tid);
        }
        step->reply = 0;
        return 1;
    } else if (step->args[7] == CV_SERVICE_FUTEX) {
        int op = (int)step->args[1];
        int command = op & FUTEX_CMD_MASK;
        if (!registered_word(step->args[0])) step->result = -EFAULT;
        else if (command != FUTEX_WAIT && command != FUTEX_WAKE &&
                 command != FUTEX_WAIT_BITSET && command != FUTEX_WAKE_BITSET)
            step->result = -ENOSYS;
        else {
            struct timespec timeout = {(long)step->args[3], (long)step->args[4]};
            pthread_mutex_unlock(&service_lock);
            long result = syscall(SYS_futex, (int *)(uintptr_t)step->args[0],
                                  op | FUTEX_PRIVATE_FLAG, (int)step->args[2],
                                  timeout.tv_sec < 0 ? NULL : &timeout, NULL, (int)step->args[5]);
            int error = errno;
            pthread_mutex_lock(&service_lock);
            step->result = result < 0 ? -error : result;
        }
    } else if (step->args[7] == CV_SERVICE_THREAD_JOIN) {
        /* The child must acquire the same transport lock to publish its
         * exit.  Do not wait while holding it, or join would deadlock. */
        pthread_mutex_unlock(&service_lock);
        step->result = thread_join(step->args[0]);
        pthread_mutex_lock(&service_lock);
    } else if (step->args[7] == CV_SERVICE_DELEGATE ||
               step->args[7] == CV_SERVICE_THREAD_DELEGATE) {
        int separate = step->args[7] == CV_SERVICE_THREAD_DELEGATE;
        struct capstone_delegate_entry *request = separate ? (void *)(uintptr_t)step->args[0] : meta;
        struct capstone_delegate_host *h = &host;
        if (separate) {
            if (!registered_span(step->args[0], CAPSTONE_DELEGATE_META_BYTES) ||
                !step->args[2] || step->args[2] > (64UL << 20) ||
                !registered_span(step->args[1], step->args[2])) return -1;
            if (request == meta) {
                if (h->exchange != (void *)(uintptr_t)step->args[1] ||
                    h->exchange_bytes != step->args[2]) return -1;
                if (tid) capstone_signals_attach(&h->signals);
            } else if (tid) {
                if (!worker_host) {
                    worker_host = calloc(1, sizeof *worker_host);
                    if (!worker_host) return -1;
                    worker_host->exchange = (void *)(uintptr_t)step->args[1];
                    worker_host->exchange_bytes = step->args[2];
                    worker_host->private_count = host.private_count;
                    memcpy(worker_host->private_fds, host.private_fds, sizeof host.private_fds);
                    worker_host->entry_address = host.entry_address;
                    worker_host->code_base = host.code_base;
                    worker_host->code_end = host.code_end;
                    memcpy(worker_host->image_sha256, host.image_sha256, sizeof host.image_sha256);
                    struct capstone_signal_block *block = (void *)((char *)request + CAPSTONE_SIGNAL_OFFSET);
                    uint64_t inherited_mask = block->initial_mask;
                    capstone_signals_init_context(&worker_host->signals, block, &host.signals);
                    worker_host->signals.logical = inherited_mask;
                    block->initial_mask = inherited_mask;
                    capstone_signals_attach(&worker_host->signals);
                }
                h = worker_host;
                if (h->exchange != (void *)(uintptr_t)step->args[1] ||
                    h->exchange_bytes != step->args[2] ||
                    h->signals.block != (void *)((char *)request + CAPSTONE_SIGNAL_OFFSET)) return -1;
            } else return -1;
        }
        if (request->version != CAPSTONE_DELEGATE_VERSION) return -1;
        /* Each native worker owns its bounce buffer and signal ring. A
         * blocked epoll/read must not stop another virtual context. */
        if (separate) pthread_mutex_unlock(&service_lock);
        capstone_delegate_serve(h, request);
        if (separate) pthread_mutex_lock(&service_lock);
        if (trace) fprintf(stderr, "CAPSTONE_VM_SERVICE tid=%llu nr=%llu result=%lld\n",
                           (unsigned long long)tid, (unsigned long long)request->nr,
                           (long long)request->result);
        if (h->exec_requested) request->result = tid ? -ENOSYS : exec_in_place();
        if (h->exiting) {
            struct cv_stats stats;
            if (tid == 0 && (getenv("CAPSTONE_VM_STATS") || getenv("CAPSTONE_DELEGATE_STATS")) &&
                !ioctl(device, CV_STATS, &stats))
                fprintf(stderr, "CAPSTONE_VM_STATS arenas=%llu pages=%llu peak=%llu nodes=%llu steps=%llu faults=%llu collections=%llu reclaimed=%llu rounds=%llu bytes_in=%llu bytes_out=%llu node_bytes=%llu node_capacity=%llu node_growths=%llu launch_ns=%llu elapsed_ns=%llu\n",
                    stats.arenas, stats.pinned_pages, stats.peak_pages, stats.nodes, stats.steps,
                    stats.faults, stats.collections, stats.reclaimed, (unsigned long long)host.rounds,
                    (unsigned long long)host.bytes_in, (unsigned long long)host.bytes_out,
                    stats.node_bytes, stats.node_capacity, stats.node_growths,
                    launch_ns, now_ns() - start_ns);
            /* The image's identity on a run that ENDS, not only on one that faults: a
             * transcript that carries no digest cannot be tied to the image it claims to be
             * from, and a judge that reads a probe's successful run (the putenv probe) has
             * nothing else to check it against. Same gate as the fault record. */
            if (tid == 0 && (isatty(2) || getenv("CAPSTONE_EXEC_DIAGNOSTICS"))) {
                if (!host.image_sha256[0] && image_fd >= 0)
                    capstone_application_hash(image_fd, host.image_sha256);
                fprintf(stderr, "capstone-exec: exit image=%s sha256=%s\n",
                        image_path ? image_path : "?", host.image_sha256);
            }
            /* exit_group stops all native workers before Linux releases
             * mappings and the adapter fd. Unmapping here would race a
             * worker completing its blocked delegated read. */
            _exit(h->exit_status);
        }
    } else if (step->args[7] == CV_SERVICE_MAP) {
        step->result = acquire_mapping(tid, step);
    } else if (step->args[7] == CV_SERVICE_REMAP) {
        step->result = remap_mapping(tid, step);
    } else if (step->args[7] == CV_SERVICE_UNMAP) {
        step->result = unmap_arena(step->args[0], step->args[1]);
    } else if (step->args[7] == CV_SERVICE_PROTECT) {
        step->result = protect_range(step->args[0], step->args[1], step->args[2]);
    } else if (step->args[7] == CV_SERVICE_WAIT) {
        /* The heap's contended atomic mutex must let its owner run, including
         * the owner's mapping service. Never yield with the wire lock held. */
        pthread_mutex_unlock(&service_lock);
        sched_yield();
        pthread_mutex_lock(&service_lock);
    } else if (step->args[7] == CV_SERVICE_NODES) {
        struct cv_nodes nodes = {.thread = tid, .available = step->args[0]};
        step->result = ioctl(device, CV_NODES, &nodes) ? -errno : 0;
    } else {
        step->result = -ENOSYS;
    }
    return 0;
}

static int virtual_service_loop(uint64_t tid)
{
    struct cv_step step = {.thread = tid};
    for (;;) {
        while (ioctl(device, CV_STEP, &step))
            if (errno != EINTR && errno != EAGAIN) return die("step");
        step.reply = 0;
        if (step.kind == 1) {
            /* The supervisor quantum is the preemption point. Yielding here
             * lets Linux schedule another launcher thread on this one hart. */
            sched_yield();
            continue;
        }
        if (step.kind == 4) {
            struct cv_thread_control control = {.thread = step.thread};
            if (!ioctl(device, CV_RESOLVE, &control)) continue;
        }
        if (step.kind != 3) {
            pthread_mutex_lock(&service_lock);
            if (worker_host)
                worker_host->preparing_nr = ((struct capstone_delegate_entry *)
                    ((char *)worker_host->signals.block - CAPSTONE_SIGNAL_OFFSET))->nr;
            else
                host.preparing_nr = ((struct capstone_delegate_entry *)meta)->nr;
            fault(&step);
            pthread_mutex_unlock(&service_lock);
            return 125;
        }
        pthread_mutex_lock(&service_lock);
        int result = virtual_service(tid, &step);
        pthread_mutex_unlock(&service_lock);
        if (result == 1) return 0;
        if (result < 0) return die("service");
        if (result & 2) return result >> 8;
    }
}

static void *virtual_thread_worker(void *opaque)
{
    uint64_t tid = *(uint64_t *)opaque;
    free(opaque);
    int result = virtual_service_loop(tid);
    /* A virtual thread is hosted by a Linux worker, so normal virtual-thread
     * exit must return from this worker rather than terminate the process. */
    if (result) _exit(result);
    if (worker_host) {
        capstone_signals_detach();
        capstone_delegate_host_free(worker_host);
        free(worker_host);
        worker_host = NULL;
    }
    return NULL;
}

int main(int argc, char **argv)
{
    int binfmt = (getauxval(AT_FLAGS) & AT_FLAGS_PRESERVE_ARGV0) != 0;
    if (binfmt && argc < 3) return 125;
    struct exec_state resumed = {0};
    if (!binfmt && argc > 3 && !strcmp(argv[1], "--resume")) {
        char *end; long fd = strtol(argv[2], &end, 10);
        if (*end || fd < 3 || fd > 65535 || read(fd, &resumed, sizeof resumed) != sizeof resumed ||
            resumed.magic != EXEC_STATE_MAGIC || resumed.child_count > CAPSTONE_DELEGATE_CHILDREN)
            return 125;
        close(fd);
        if (resumed.spawner.socket >= 0 && fcntl(resumed.spawner.socket, F_SETFD, FD_CLOEXEC) < 0)
            return 125;
        spawner = resumed.spawner; host.child_count = resumed.child_count;
        memcpy(host.children, resumed.children, sizeof resumed.children);
        argc -= 2; argv += 2;
    }
    int app_args = !binfmt && argc >= 5 && !strcmp(argv[1], "--application-argv") && !strcmp(argv[3], "--");
    int literal = binfmt || (argc > 1 && !strcmp(argv[1], "--"));
    if (literal && !binfmt) { --argc; ++argv; }
    if (argc < 2) { fprintf(stderr, "usage: capstone-vexec [--] PROGRAM.dom [ARG...]\n"); return 2; }
    if (!literal && argc == 2 && !strcmp(argv[1], "--stats")) {
        int fd = open("/dev/capstone-vm", O_RDWR | O_CLOEXEC);
        struct cv_global stats;
        if (fd < 0 || ioctl(fd, CV_GLOBAL, &stats)) return 125;
        close(fd);
        printf("{\"version\":1,\"live_domains\":%llu,\"live_regions\":%llu,\"live_bytes\":%llu,\"cached_bytes\":0,\"poisoned_blocks\":0,\"nodes_high_water\":%llu,\"nodes_live\":%llu,\"nodes_retired\":%llu,\"nodes_allocated_total\":%llu,\"tag_pages\":%llu,\"node_capacity\":%llu,\"node_bytes\":%llu,\"collections\":%llu,\"nodes_reclaimed\":%llu}\n",
            stats.contexts, stats.arenas, stats.pinned_pages * 4096,
            stats.nodes_high_water, stats.nodes_live, stats.nodes_retired, stats.nodes_allocated,
            stats.pinned_pages, stats.node_capacity, stats.node_bytes, stats.collections, stats.reclaimed);
        return 0;
    }
    image_path = app_args ? argv[2] : argv[1];
    unsigned stdio_mask;
    if (reserve_stdio(&stdio_mask)) return 125;
    start_ns = now_ns();
    trace = getenv("CAPSTONE_VM_TRACE") != NULL;
    struct capstone_application_descriptor_v2 desc;
    char resumed_path[64];
    if (resumed.magic) snprintf(resumed_path, sizeof resumed_path, "/proc/self/fd/%d", resumed.image);
    /* No copy of the image on a local filesystem: the module denies writers
     * to the file (CV_IMAGE) before it is validated, and holds that until
     * this process ends, as Linux does for a running executable. load()
     * validates the one mapping it loads from, so the bytes validated are
     * the bytes loaded, and the domain cannot write its own image while it
     * runs. After exec in place, and on 9p, NFS or FUSE, the image is a
     * sealed copy, which the denial accepts as fixed. */
    int fd = capstone_application_open(resumed.magic ? resumed_path : image_path);
    int image_error = errno;
    if (resumed.magic) close(resumed.image);
    errno = image_error;
    if (fd < 0) { perror("capstone-vexec: image"); return errno == ENOENT ? 127 : 126; }
    image_fd = fd;
    if (spawner.socket < 0) { int error = capstone_spawner_start(&spawner);
        if (error) { errno = error; return die("spawner"); } }
    host.spawner = &spawner;
    device = open("/dev/capstone-vm", O_RDWR | O_CLOEXEC);
    if (device < 0) return die("open");
    if (image_denied(&fd, &desc)) { perror("capstone-vexec: image"); cleanup(); return 126; }
    void *image; size_t image_bytes, image_rx, stack_bytes; uintptr_t entry;
    if (load(fd, &desc, &image, &image_bytes, &image_rx, &entry, &stack_bytes)) {
        perror("capstone-vexec: image"); cleanup(); return 126;
    }
    host.private_fds[host.private_count++] = fd;
    /* Option (iii): no exact-bounds profile bit is required. Bounds are representable by
     * construction -- the allocator's geometry keeps every rounded bound inside its own slot --
     * so the launcher runs on the default profile. The bit may still be set; it is not read. */
    if (grant(image, image_bytes, 3, 7, entry, 0, 0, image_rx)) { munmap(image, image_bytes); return die("image grant"); }
    void *stack = reserve(stack_bytes);
    domain_stack = stack; domain_stack_bytes = stack_bytes;
    meta = reserve(CAPSTONE_DELEGATE_META_BYTES);
    size_t exchange_bytes = rounded(desc.exchange_bytes);
    void *exchange = exchange_bytes ? reserve(exchange_bytes) : NULL;
    void *startup = reserve(CAPSTONE_LAUNCH_BYTES);
    if (!stack || !meta || !exchange || !startup) return die("allocate startup");
    char *cwd = getcwd(NULL, 0);
    struct timespec rt, mt;
    clock_gettime(CLOCK_REALTIME, &rt); clock_gettime(CLOCK_MONOTONIC, &mt);
    struct capstone_launch_task task = { .pid = getpid(), .ppid = getppid(),
        .uid = getuid(), .euid = geteuid(), .gid = getgid(), .egid = getegid(),
        .realtime_ns = (uint64_t)rt.tv_sec * 1000000000 + rt.tv_nsec,
        .monotonic_ns = (uint64_t)mt.tv_sec * 1000000000 + mt.tv_nsec };
    int error = cwd ? capstone_launch_pack(startup, CAPSTONE_LAUNCH_BYTES,
        argc - (app_args ? 4 : binfmt ? 2 : 1), argv + (app_args ? 4 : binfmt ? 2 : 1), environ, cwd, stdio_mask, &task) : ENOMEM;
    free(cwd);
    if (error) { errno = error; return die("startup"); }
    if (grant(stack, stack_bytes, 2, 6, (uintptr_t)stack + stack_bytes, 0, 0, 0) ||
        grant(meta, CAPSTONE_DELEGATE_META_BYTES, 10, 6, (uintptr_t)meta, 0, 0, 0) ||
        grant(exchange, exchange_bytes, 11, 6, (uintptr_t)exchange, 0, 0, 0) ||
        grant(startup, CAPSTONE_LAUNCH_BYTES, 12, 4, (uintptr_t)startup, 0, 0, 0)) return die("grant");
    if (desc.v1.heap_bytes) {
        size_t bytes = rounded(desc.v1.heap_bytes);
        void *region = bytes ? reserve(bytes) : NULL;
        if (!region || grant(region, bytes, 13, 6, (uintptr_t)region, 1, 0, 0))
            return die("nested allocator grant");
    }
    host.exchange = exchange; host.exchange_bytes = exchange_bytes;
    host.private_fds[host.private_count++] = device;
    capstone_signals_settle_libc();
    capstone_signals_init(&host.signals, (void *)((char *)meta + CAPSTONE_SIGNAL_OFFSET));
    if ((error = capstone_delegate_seccomp())) { errno = error; return die("seccomp"); }
    for (int fd = 0; fd < 3; ++fd) if (!(stdio_mask & (1u << fd))) close(fd);
    launch_ns = now_ns() - start_ns;
    return virtual_service_loop(0);
}
