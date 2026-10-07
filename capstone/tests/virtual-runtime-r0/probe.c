#define _GNU_SOURCE
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wire.h"
extern const char r0_stale[], r0_stale_end[];
extern const char r0_program[], r0_fault[], r0_ecall[], r0_program_end[];
#define CHECK(x) do { if (!(x)) { perror("R0 syscall"); fprintf(stderr, "R0_FAIL:%d: %s\n", __LINE__, #x); exit(1); } } while (0)
static void *mapping(size_t n) {
    void *p = mmap(NULL, n, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    CHECK(p != MAP_FAILED); memset(p, 0, n); return p;
}
int main(void) {
    int fd = open("/dev/capstone-r0", O_RDWR);
    size_t bytes = r0_program_end - r0_program;
    char *code = mapping(4096), *stack = mapping(4096), *tls = mapping(4096);
    unsigned count = 0;
    CHECK(fd >= 0);
    memcpy(code, r0_program, bytes);
    __builtin___clear_cache(code, code+bytes);
    CHECK(!mprotect(code, 4096, PROT_READ|PROT_EXEC));
    /* Touch intervening pages before moving the last page next to the first.
     * The module checks the actual PFNs; adjacency is never inferred from VA. */
    char *data = NULL;
    for (unsigned attempt = 0; attempt != 16; ++attempt) {
        char *pool = mapping(16*4096);
        CHECK(mremap(pool+15*4096, 4096, 4096, MREMAP_MAYMOVE|MREMAP_FIXED,
                     pool+4096) == pool+4096);
        CHECK(!munmap(pool+8192, 13*4096));
        struct r0_request r = {.code=(uintptr_t)code, .data=(uintptr_t)pool,
            .stack=(uintptr_t)stack, .tls=(uintptr_t)tls, .code_bytes=4096,
            .data_bytes=8192, .data_perms=6, .code_perms=5};
        CHECK(!ioctl(fd, R0_RUN, &r));
        CHECK(r.kind == 2 && r.cause == 11 && r.result == 42 &&
              r.pc == (uintptr_t)code + (r0_ecall-r0_program));
        CHECK(*(uint64_t *)pool == 0x123 && *(uint64_t *)(pool+4096) == 0x123);
        CHECK(*(uint64_t *)(stack+4064) == 0x123 && *(uint64_t *)tls == 0x123);
        if (r.scattered) { data=pool; break; }
        CHECK(!munmap(pool, 8192));
    }
    CHECK(data != NULL);
    puts("R0:PASS scattered_linux_mappings"); ++count;
    const char *names[] = {"cap_bounds", "pte_write", "cap_execute", "pte_execute", "fresh_retry"};
    for (unsigned test = 0; test != 5; ++test) {
        struct r0_request r = {.code=(uintptr_t)code, .data=(uintptr_t)data,
            .stack=(uintptr_t)stack, .tls=(uintptr_t)tls, .code_bytes=4096,
            .data_bytes=test==0 ? 4096 : 8192, .data_perms=6,
            .code_perms=test==2 ? 4 : 5};
        *(uint64_t *)(data+4096)=0;
        if (test == 1) CHECK(!mprotect(data+4096, 4096, PROT_READ));
        if (test == 3) CHECK(!mprotect(code, 4096, PROT_READ));
        CHECK(!ioctl(fd, R0_RUN, &r));
        unsigned causes[] = {28, 15, 27, 12, 11};
        uintptr_t pc = (uintptr_t)code + (test<2 ? r0_fault-r0_program :
                                             test==4 ? r0_ecall-r0_program : 0);
        CHECK(r.kind == 2 && r.cause == causes[test] && r.pc == pc);
        if (test < 2) CHECK(r.address == (uintptr_t)data+4096);
        if (test < 4) CHECK(*(uint64_t *)(data+4096) == 0);
        else CHECK(r.result == 42 && *(uint64_t *)(data+4096) == 0x123);
        CHECK(!mprotect(data+4096, 4096, PROT_READ|PROT_WRITE));
        CHECK(!mprotect(code, 4096, PROT_READ|PROT_EXEC));
        printf("R0:PASS %s\n", names[test]); ++count;
    }
    /* Retain the previous invocation's spilled bytes across namespace
     * destruction and fresh local IDs. They must carry no authority. */
    CHECK(!mprotect(code,4096,PROT_READ|PROT_WRITE));
    memcpy(code,r0_stale,r0_stale_end-r0_stale);
    __builtin___clear_cache(code,code+4096);
    CHECK(!mprotect(code,4096,PROT_READ|PROT_EXEC));
    struct r0_request stale = {.code=(uintptr_t)code, .data=(uintptr_t)data,
        .stack=(uintptr_t)stack, .tls=(uintptr_t)tls, .code_bytes=4096,
        .data_bytes=8192, .data_perms=6, .code_perms=5};
    CHECK(!ioctl(fd,R0_RUN,&stale));
    CHECK(stale.kind==2 && stale.cause==24 && stale.pc==(uintptr_t)code+4);
    puts("R0:PASS destroyed_context_tags_cleared"); ++count;
    /* The final program is compiled by the Capstone C compiler, including
     * its ordinary capability stack spills and pointer arithmetic. */
    int image = open("/mnt/r0/c-entry.bin", O_RDONLY);
    CHECK(image >= 0);
    CHECK(!mprotect(code,4096,PROT_READ|PROT_WRITE));
    ssize_t got = read(image, code,4096); CHECK(got>0 && got<4096);
    CHECK(!close(image));
    __builtin___clear_cache(code,code+got);
    CHECK(!mprotect(code,4096,PROT_READ|PROT_EXEC));
    *(uint64_t *)data=0; *(uint64_t *)(data+4096)=0;
    struct r0_request r = {.code=(uintptr_t)code, .data=(uintptr_t)data,
        .stack=(uintptr_t)stack, .tls=(uintptr_t)tls, .code_bytes=4096,
        .data_bytes=8192, .data_perms=6, .code_perms=5, .c_entry=1};
    CHECK(!ioctl(fd,R0_RUN,&r));
    CHECK(r.kind==2 && r.cause==11 && r.result==42);
    CHECK(*(uint64_t *)data==0x123 && *(uint64_t *)(data+4096)==0x123);
    puts("R0:PASS capstone_compiled_c"); ++count;
    CHECK(!close(fd));
    CHECK(!munmap(data,8192)); CHECK(!munmap(code,4096));
    CHECK(!munmap(stack,4096)); CHECK(!munmap(tls,4096));
    printf("VIRTUAL_RUNTIME_R0_OK tests=%u\n",count);
    return 0;
}
