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

extern int malloc_trim(size_t);
extern unsigned long __capstone_sublet_malloc_linear(size_t, sublet_cap *);
extern void __capstone_sublet_free_linear(unsigned long);
extern void __capstone_sublet_heap_stats(unsigned long [9]);
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
    errno = 0;
    require(mprotect(p + 12288, 4096, PROT_READ) == -1 && errno == EINVAL,
            "padding cannot gain rights");
    errno = 0;
    require(munmap(p + 4096, 4096) == -1 && errno == EINVAL,
            "partial unmap refused");
    require(!mprotect(p, 12288, PROT_NONE) && !munmap(p, 8193), "protected whole retire");
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
    require(!realloc(old, (1024UL << 20) + 1) && errno == ENOMEM && old[0] == 33,
            "realloc failure preserves original");
    free(old);
    puts("VIRTUAL_VM_OK protections guards requested_length unused_pages execute rollback");
}
int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "ok";
    if (!strcmp(mode, "vm")) { vm_contract(); return 0; }
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
    if (!strcmp(mode, "linear") || !strcmp(mode, "linear-stale")) {
        sublet_cap loan;
        unsigned long before[9], after[9];
        __capstone_sublet_heap_stats(before);
        unsigned long base = __capstone_sublet_malloc_linear(4096, &loan);
        require(base && base == sublet_base(&loan) && sublet_end(&loan) >= base + 4096,
                "linear block grant");
        char *p = sublet_take(&loan);
        memset(p, 0x5a, 4096);
        stale = p;
        __capstone_sublet_free_linear(base);
        __capstone_sublet_heap_stats(after);
        require(after[0] == before[0] + 1 && after[1] == before[1] + 1 &&
                after[7] >= before[7] + 1 && after[2] == 0, "linear block retire counters"); /* Nested group retirement can also revoke its parent. */
        char *replacement = malloc(4096);
        require(replacement != NULL, "linear block reuse");
        replacement[0] = 23;
        if (!strcmp(mode, "linear-stale")) {
            puts("VIRTUAL_LINEAR_ACCESS:stale"); fflush(stdout);
            __asm__ volatile(".global cap_vm_fault_linear\ncap_vm_fault_linear:\nlbu zero, 0(%0)"
                             : : "r"(stale) : "memory");
            fail("linear descendant survived retirement");
        }
        require(replacement[0] == 23, "linear replacement live");
        free(replacement);
        (void)malloc_trim(0); /* musl selects group release during free. */
        puts("VIRTUAL_LINEAR_OK");
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
