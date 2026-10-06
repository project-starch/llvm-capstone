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

extern int malloc_trim(size_t);
static void *volatile stale;
static jmp_buf jump;
static __thread unsigned thread_value = 41;
static void fail(const char *what) { fprintf(stderr, "VM_CONTRACT_FAIL:%s\n", what); exit(1); }
static void require(int ok, const char *what) { if (!ok) fail(what); }
int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "ok";
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
    require(malloc_trim(0), "return arenas to Linux");
    char *m = mmap(NULL, 8192, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(m != MAP_FAILED && !m[4096], "demand mapping");
    m[0] = 41; m[4096] = m[0] + 1;
    require(m[4096] == 42 && !munmap(m, 8192), "mapping retirement");
    puts("VIRTUAL_APPLICATION_OK argv env files heap_growth heap_retirement mmap");
    return 0;
}
