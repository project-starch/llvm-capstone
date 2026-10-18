#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <malloc.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <setjmp.h>
#include <errno.h>
#include <capstone/virtual.h>
#include "vm.h"
#include <capstone/capability.h>

/* This program tests the runtime's own allocator, so every allocation must
 * reach it. Applications compile hosted, where clang may remove a malloc/free
 * pair it can see through and treat the removed allocation as successful: the
 * recycling loop below then allocated nothing, and the failed-realloc check
 * compiled to a constant. A call through a volatile function pointer is one the
 * compiler cannot see through, whatever the SDK's flags. (no_builtin on the
 * calling functions does not prevent the elision.) */
static void *(*volatile contract_malloc)(size_t) = malloc;
static void *(*volatile contract_calloc)(size_t, size_t) = calloc;
static void *(*volatile contract_realloc)(void *, size_t) = realloc;
static void (*volatile contract_free)(void *) = free;
#define malloc(n) contract_malloc(n)
#define calloc(n, m) contract_calloc(n, m)
#define realloc(p, n) contract_realloc(p, n)
#define free(p) contract_free(p)

extern int malloc_trim(size_t);
static void *volatile stale;
static jmp_buf jump;
static __thread unsigned thread_value = 41;
static void fail(const char *what) { fprintf(stderr, "VM_CONTRACT_FAIL:%s\n", what); exit(1); }
static void require(int ok, const char *what) { if (!ok) fail(what); }
static volatile unsigned virtual_thread_value;
static void *current_thread_pointer(void)
{
    void *tp;
    __asm__ volatile("movc %0, tp" : "=r"(tp));
    return tp;
}
static void *virtual_thread_entry(void *arg)
{
    volatile unsigned *value = arg;
    *value = 42;
    write(1, "VIRTUAL_THREAD_CHILD\n", 21);
    capstone_virtual_thread_exit(NULL);
}
#define THREAD_OBJECTS 384
static void *child_objects[THREAD_OBJECTS];
static void *heap_thread_entry(void *unused)
{
    (void)unused;
    for (unsigned i = 0; i < THREAD_OBJECTS; ++i) {
        child_objects[i] = malloc(4096);
        require(child_objects[i] != NULL, "child malloc");
        memset(child_objects[i], 0x5a, 4096);
    }
    virtual_thread_value = 42;
    /* The parent owns these objects after join, in the same namespace. */
    capstone_virtual_thread_exit(NULL);
}
static void vm_contract(void)
{
    const int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    char *empty = mmap(NULL, 64UL << 20, PROT_NONE, flags, -1, 0);
    require(empty != MAP_FAILED && !munmap(empty, 64UL << 20), "unused PROT_NONE retire");
    char *p = mmap(NULL, 8193, PROT_READ, flags, -1, 0);
    require(p != MAP_FAILED && p[8192] == 0, "read-only first touch");
    require(!mprotect(p, 12288, PROT_READ | PROT_WRITE), "restore write rights");
    p[0] = 17; p[8192] = 23;
    require(!mprotect(p + 4096, 4096, PROT_NONE), "guard page");
    require(p[0] == 17 && p[8192] == 23, "guard neighbours");
    /* Rights cannot reach past the mapping's last page. With power-of-two
     * grants the page after it is padding of this grant; with page-rounded
     * grants (exact bounds) it may be another mapping of the same domain, so
     * the range straddles the end instead: it can never be one mapping's. */
    errno = 0;
    require(mprotect(p + 8192, 8192, PROT_READ) == -1 && errno == EINVAL,
            "rights cannot extend past the mapping");
    errno = 0;
    require(munmap(p + 4096, 4096) == -1 && errno == EINVAL,
            "partial unmap refused");
    require(!mprotect(p, 12288, PROT_NONE) && !munmap(p, 8193), "protected whole retire");
    /* Real padding, under either grant shape: an application mapping aligned
     * beyond its size keeps the rest of its grant (bytes = alignment) out of
     * reach, so mprotect is checked against the visible bytes, not the grant. */
    sublet_cap aligned;
    require(!cap_vm_acquire(&aligned, 4096, 16384, PROT_READ | PROT_WRITE, 7, CAP_VM_MAPPING),
            "aligned mapping");
    char *a = cap_vm_copyable(&aligned);
    require(!(__builtin_capstone_cap_get_cursor(a) & 16383), "aligned mapping alignment");
    require(!mprotect(a, 4096, PROT_READ), "aligned mapping protect");
    errno = 0;
    require(mprotect(a + 4096, 4096, PROT_READ) == -1 && errno == EINVAL,
            "padding cannot gain rights");
    errno = 0;
    require(mprotect(a + 8192, 4096, PROT_READ) == -1 && errno == EINVAL,
            "padding past the visible end cannot gain rights");
    require(!munmap(a, 4096), "aligned mapping retire");
    char *q = mmap(NULL, 4096, PROT_READ | PROT_WRITE, flags, -1, 0);
    require(q != MAP_FAILED, "executable mapping");
    /* Execute this mapping as an explicit C context. Ordinary gp-free calls
     * retain their image PCC; cross-mapping call/return needs a separate ABI. */
    const uint32_t code[] = {0x02a00293, 0x00552023, 0x00400893, 0x00000073, 0x00100073};
    memcpy(q, code, sizeof code); /* li t0,42; sw t0,0(a0); exit ECALL */
    require(!mprotect(q, 4096, PROT_READ | PROT_EXEC), "execute rights");
    __asm__ volatile("fence.i" ::: "memory");
    char *stack = mmap(NULL, 65536, PROT_READ | PROT_WRITE, flags, -1, 0);
    require(stack != MAP_FAILED, "executable context stack");
    virtual_thread_value = 0;
    long tid = capstone_virtual_thread_create((void *(*)(void *))q,
                        (void *)&virtual_thread_value, stack + 65536,
                        current_thread_pointer());
    require(tid > 0 && !capstone_virtual_thread_join(tid) && virtual_thread_value == 42,
            "execute through mapping PCC");
    require(!munmap(stack, 65536) && !munmap(q, 4096), "RX retire");
    char *old = malloc(80);
    require(old != NULL, "realloc original");
    old[0] = 33;
    errno = 0;
    require(!realloc(old, CV_MAX_REGION_BYTES + 1) && errno == ENOMEM && old[0] == 33,
            "realloc failure preserves original");
    free(old);
    puts("VIRTUAL_VM_OK protections guards requested_length unused_pages execute rollback");
}
/* A region above the first profile's 256 MiB, a process above its 1 GiB, both
 * ceilings, and the release of a freed region's charge. free clears the whole
 * payload, so only the 600 MiB region is freed; the others stay sparse until
 * exit, where the module retires them. */
static void vm_large_contract(void)
{
    size_t big = 300UL << 20, bigger = 600UL << 20, half = 1536UL << 20, last = 800UL << 20;
    char *a = malloc(big);
    require(a != NULL, "malloc above 256 MiB");
    a[0] = 1; a[big - 1] = 2;
    char *b = realloc(a, bigger);
    require(b != NULL && b[0] == 1 && b[big - 1] == 2, "realloc keeps the data");
    b[bigger - 1] = 3;
    char *c = malloc(half), *d = malloc(half);
    require(c != NULL && d != NULL, "registered VA above 1 GiB");
    c[0] = 4; c[half - 1] = 5; d[0] = 6; d[half - 1] = 7;
    /* 600 + 1536 + 1536 + 800 MiB is above 4 GiB, startup regions or not. */
    errno = 0;
    require(!malloc(last) && errno == ENOMEM, "the process's ceiling");
    errno = 0;
    require(!malloc(CV_MAX_REGION_BYTES + 1) && errno == ENOMEM, "the region ceiling");
    require(b[bigger - 1] == 3 && c[half - 1] == 5 && d[half - 1] == 7, "data after the refusals");
    /* Without the 600 MiB charge, 3072 + 800 MiB leaves 224 MiB for the
     * startup regions: a charge that free leaked would refuse this again. */
    free(b);
    char *e = malloc(last);
    require(e != NULL, "free releases the region's charge");
    e[0] = 8; e[last - 1] = 9;
    puts("VIRTUAL_VM_LARGE_OK region aggregate realloc release");
}
/* More registered mappings than the 8192 the module's table once held: every
 * allocation above mallocng's mmap threshold is a mapping of its own. Every
 * second one is freed and replaced, eight grow through the remap path, and the
 * data is checked in an order unrelated to addresses. They stay live at exit,
 * so the launcher's stats record shows the registered count. */
static char mapping_byte(unsigned i, unsigned at) { return (char)((i & 1 ? i : ~i) >> at); }
static void vm_mappings_contract(void)
{
    enum { count = 9000, stride = 7919 }; /* coprime: every index once */
    static char *held[count];
    const size_t bytes = 140000, grown = 600000;
    for (unsigned i = 0; i < count; ++i) {
        held[i] = malloc(bytes);
        require(held[i] != NULL, "mapping above 8192");
        held[i][0] = (char)i; held[i][bytes - 1] = (char)(i >> 8);
    }
    for (unsigned k = 0, i = 0; k < count; ++k, i = (i + stride) % count)
        require(held[i][0] == (char)i && held[i][bytes - 1] == (char)(i >> 8), "mapping data");
    for (unsigned i = 0; i < count; i += 2)
        free(held[i]);
    for (unsigned i = 0; i < count; i += 2) {
        held[i] = malloc(bytes);
        require(held[i] != NULL, "mapping refill");
        held[i][0] = mapping_byte(i, 0); held[i][bytes - 1] = mapping_byte(i, 8);
    }
    unsigned moved = 0;
    for (unsigned i = 1; i < count; i += count / 8) {
        char *before = held[i], *p = realloc(held[i], grown);
        require(p != NULL, "mapping remap");
        moved += p != before;
        held[i] = p; p[grown - 1] = 5;
    }
    for (unsigned k = 0, i = 0; k < count; ++k, i = (i + stride) % count)
        require(held[i][0] == mapping_byte(i, 0) && held[i][bytes - 1] == mapping_byte(i, 8),
                "mapping data after refill and remap");
    for (unsigned i = 1; i < count; i += count / 8)
        require(held[i][grown - 1] == 5, "remapped tail");
    printf("VIRTUAL_VM_MAPPINGS_MOVED %u\n", moved);
    puts("VIRTUAL_VM_MAPPINGS_OK 9000");
}
int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "ok";
    if (!strcmp(mode, "vm")) { vm_contract(); return 0; }
    if (!strcmp(mode, "vm-large")) { vm_large_contract(); return 0; }
    if (!strcmp(mode, "vm-mappings")) { vm_mappings_contract(); return 0; }
    if (!strcmp(mode, "vm-ro-write") || !strcmp(mode, "vm-none-read") ||
        !strcmp(mode, "vm-guard") || !strcmp(mode, "vm-padding")) {
        char *p = mmap(NULL, 8193, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(p != MAP_FAILED, "VM fault setup");
        *p = 11;
        if (!strcmp(mode, "vm-ro-write")) {
            require(!mprotect(p, 12288, PROT_READ), "RO denial setup");
            puts("VIRTUAL_VM_ACCESS:write_ro"); fflush(stdout);
            __asm__ volatile(".global cap_vm_fault_ro\ncap_vm_fault_ro:\nsb zero, 0(%0)"
                             : : "r"(p) : "memory");
        } else if (!strcmp(mode, "vm-none-read")) {
            require(!mprotect(p, 12288, PROT_NONE), "NONE denial setup");
            puts("VIRTUAL_VM_ACCESS:read_none"); fflush(stdout);
            __asm__ volatile(".global cap_vm_fault_none\ncap_vm_fault_none:\nlbu zero, 0(%0)"
                             : : "r"(p) : "memory");
        } else if (!strcmp(mode, "vm-guard")) {
            require(!mprotect(p + 4096, 4096, PROT_NONE), "guard denial setup");
            puts("VIRTUAL_VM_ACCESS:guard"); fflush(stdout);
            __asm__ volatile(".global cap_vm_fault_guard\ncap_vm_fault_guard:\nlbu zero, 0(%0)"
                             : : "r"(p + 4096) : "memory");
        } else {
            puts("VIRTUAL_VM_ACCESS:padding"); fflush(stdout);
            __asm__ volatile(".global cap_vm_fault_padding\ncap_vm_fault_padding:\nlbu zero, 0(%0)"
                             : : "r"(p + 12288) : "memory");
        }
        fail("VM denial survived");
    }
    if (!strcmp(mode, "nested") || !strcmp(mode, "nested-stale")) {
        char *allocation = malloc(4096);
        require(allocation != NULL, "nested block allocation");
        char *parent = capstone_cap_derive(allocation, 0, 4096);
        char *p = capstone_cap_derive(parent, 0, 128);
        char *sibling = capstone_cap_derive(parent, 128, 128);
        p[0] = 17; sibling[0] = 31;
        stale = p;
        capstone_cap_revoke_child(parent, p);
        parent[0] = 23;
        require(sibling[0] == 31, "sibling survives child retirement");
        char *replacement = capstone_cap_derive(parent, 0, 128);
        require((uintptr_t)replacement == (uintptr_t)p && replacement[0] == 23,
                "same-address successor survives");
        /* Free the outer malloc object: its child and grandchild must die. */
        stale = capstone_cap_derive(replacement, 0, 16);
        free(allocation);
        if (!strcmp(mode, "nested-stale")) {
            puts("VIRTUAL_NESTED_ACCESS:stale"); fflush(stdout);
            __asm__ volatile(".global cap_vm_fault_nested\ncap_vm_fault_nested:\nlbu zero, 0(%0)"
                             : : "r"(stale) : "memory");
            fail("nested descendant survived outer free");
        }
        puts("VIRTUAL_NESTED_OK");
        return 0;
    }
    if (!strcmp(mode, "heap-threads")) {
        size_t bytes = 65536;
        void *stack = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(stack != MAP_FAILED, "heap thread stack");
        long tid = capstone_virtual_thread_create(heap_thread_entry, NULL,
                      (char *)stack + bytes, current_thread_pointer());
        require(tid > 0, "heap thread create");
        void *parent_objects[THREAD_OBJECTS];
        for (unsigned i = 0; i < THREAD_OBJECTS; ++i) {
            parent_objects[i] = malloc(4096);
            require(parent_objects[i] != NULL, "parent malloc");
            memset(parent_objects[i], 0xa5, 4096);
        }
        require(!capstone_virtual_thread_join(tid), "heap thread join");
        require(virtual_thread_value == 42, "child published objects");
        for (unsigned i = 0; i < THREAD_OBJECTS; ++i) {
            require(((unsigned char *)parent_objects[i])[4095] == 0xa5 &&
                    ((unsigned char *)child_objects[i])[4095] == 0x5a,
                    "shared heap isolation");
            free(parent_objects[i]);
            /* Ownership and revocation are shared by the mm, not the thread. */
            free(child_objects[i]);
        }
        (void)malloc_trim(0); /* musl selects group release during free. */
        require(!munmap(stack, bytes), "heap thread stack retire");
        puts("VIRTUAL_HEAP_THREADS_OK shared_ownership metadata_growth arena_growth");
        return 0;
    }
    if (!strcmp(mode, "sparse-churn") || !strcmp(mode, "sparse-stale")) {
        void *reserved = mmap(NULL, 64UL << 20, PROT_NONE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        void **slot = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(reserved != MAP_FAILED && slot != MAP_FAILED, "sparse mappings");
        *slot = malloc(64);
        require(*slot != NULL, "collected pointer");
        free(*slot);
        require(!mprotect(slot, 4096, PROT_NONE), "protected stale tag storage");
        unsigned char *held = malloc(256);
        require(held != NULL, "held allocation");
        memset(held, 0x5a, 256);
        for (unsigned i = 0; i < 200000; ++i) {
            unsigned char *p = malloc(64);
            require(p != NULL, "recycling allocation");
            p[0] = i; p[63] = i >> 8;
            require(p[0] == (unsigned char)i && p[63] == (unsigned char)(i >> 8),
                    "recycling contents");
            free(p);
            require(held[i & 255] == 0x5a, "live object survived collection");
            if ((i + 1) % 50000 == 0) {
                printf("VIRTUAL_CHURN_PROGRESS mode=%s allocations=%u\n", mode, i + 1);
                fflush(stdout);
            }
        }
        if (!strcmp(mode, "sparse-stale")) {
            require(!mprotect(slot, 4096, PROT_READ), "restore stale tag storage");
            void *p = *slot;
            puts("VIRTUAL_VM_ACCESS:collected"); fflush(stdout);
            __asm__ volatile(".global cap_vm_fault_collected\ncap_vm_fault_collected:\nlbu zero, 0(%0)"
                             : : "r"(p) : "memory");
            fail("collected stale pointer survived");
        }
        free(held);
        require(!munmap(slot, 4096) && !munmap(reserved, 64UL << 20), "sparse retire");
        puts("VIRTUAL_SPARSE_CHURN_OK allocations=200000 protected_stale_tags");
        return 0;
    }
    if (!strcmp(mode, "threads")) {
        size_t bytes = 64 * 1024;
        void *stack = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(stack != MAP_FAILED, "thread stack");
        long tid = capstone_virtual_thread_create(virtual_thread_entry,
                                                   (void *)&virtual_thread_value,
                                                   (char *)stack + bytes,
                                                   current_thread_pointer());
        require(tid > 0, "thread create");
        for (unsigned i = 0; i < 100000000 && virtual_thread_value != 42; ++i)
            __asm__ volatile("" ::: "memory");
        require(virtual_thread_value == 42, "thread ran");
        require(!capstone_virtual_thread_join(tid), "thread join");
        require(!munmap(stack, bytes), "thread stack retire");
        puts("VIRTUAL_THREADS_OK shared_mm lifetime_root quantum");
        return 0;
    }
    if (!strcmp(mode, "spin")) {
        write(1, "VIRTUAL_SPIN_READY\n", 19);
        for (;;) __asm__ volatile("" ::: "memory");
    }
    if (!strcmp(mode, "stale") || !strcmp(mode, "bounds")) {
        char *p = malloc(256);
        require(p != NULL, "fault allocation");
        stale = p;
        if (!strcmp(mode, "stale")) {
            free(p);
            char *q = malloc(256);
            require(q != NULL, "replacement");
            *q = 7;
            (void)*(volatile char *)stale;
        } else {
            size_t n = malloc_usable_size(p);
            ((volatile char *)p)[n] = 1;
        }
        fail("unsafe access survived");
    }
    if (!strcmp(mode, "retired")) {
        char *p = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(p != MAP_FAILED, "old mapping");
        *p = 42; stale = p;
        unsigned long address = __builtin_capstone_cap_get_cursor(p);
        require(!munmap(p, 4096), "retire");
        char *q = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(q != MAP_FAILED, "new mapping");
        require(__builtin_capstone_cap_get_cursor(q) == address, "same VA reuse");
        *q = 23;
        write(1, "VIRTUAL_VA_REUSED\n", 18);
        (void)*(volatile char *)stale;
        fail("retired access survived");
    }
    require(argc == 3 && !strcmp(argv[2], "argument"), "argv");
    require(getenv("CAPSTONE_VM_CONTRACT") &&
            !strcmp(getenv("CAPSTONE_VM_CONTRACT"), "environment"), "environment");
    require(getpid() > 0, "pid");
    require(++thread_value == 42, "TLS");
    if (!setjmp(jump)) longjmp(jump, 7);
    char path[80];
    snprintf(path, sizeof(path), "/tmp/virtual-contract-%ld", (long)getpid());
    int fd = open(path, O_CREAT | O_TRUNC | O_RDWR, 0600);
    require(fd >= 0, "open");
    struct flock lock = {.l_type = F_WRLCK, .l_whence = SEEK_SET, .l_len = 1};
    require(!fcntl(fd, F_SETLK, &lock), "capability lock argument");
    lock.l_type = F_UNLCK;
    require(!fcntl(fd, F_SETLK, &lock), "unlock");
    require(write(fd, "Linux-backed IO", 15) == 15, "write");
    require(lseek(fd, 0, SEEK_SET) == 0, "seek");
    char text[16] = {0};
    require(read(fd, text, 15) == 15 && !strcmp(text, "Linux-backed IO"), "read");
    require(!close(fd) && !unlink(path), "close/unlink");
    char *p[12];
    for (unsigned i = 0; i < 12; ++i) {
        p[i] = calloc(1, 131072);
        require(p[i] && !p[i][131071], "heap growth/zero");
        memset(p[i], i + 1, 131072);
    }
    for (unsigned i = 0; i < 12; ++i) {
        require(p[i][0] == i + 1 && p[i][131071] == i + 1, "heap preserved");
        free(p[i]);
    }
    (void)malloc_trim(0); /* musl selects group release during free. */
    char *m = mmap(NULL, 8192, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(m != MAP_FAILED && !m[4096], "demand mapping");
    m[0] = 41; m[4096] = m[0] + 1;
    require(m[4096] == 42 && !munmap(m, 8192), "mapping retirement");
    puts("VIRTUAL_APPLICATION_OK argv env files heap_growth heap_retirement mmap");
    return 0;
}
