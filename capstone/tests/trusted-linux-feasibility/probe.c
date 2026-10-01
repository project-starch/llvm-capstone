/* Ordinary Linux control for the trusted-Linux feasibility gate.
 * This binary does not enable Capstone protection. */
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

static void fail(const char *step)
{
    fprintf(stderr, "CAPSTONE_FEASIBILITY_FAIL:%s:%d\n", step, errno);
    exit(1);
}

int main(void)
{
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        fail("pagesize");
    }

    unsigned char *mapping = mmap(NULL, (size_t)page_size * 2,
                                  PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) {
        fail("mmap");
    }
    mapping[0] = 0x41;
    mapping[page_size] = 0x42; /* Fault in a second ordinary Linux page. */
    if (mprotect(mapping + page_size, (size_t)page_size, PROT_READ) != 0) {
        fail("mprotect");
    }
    if (mapping[0] != 0x41 || mapping[page_size] != 0x42) {
        fail("mapping-content");
    }

    unsigned char *first = malloc(64);
    if (!first) {
        fail("malloc");
    }
    memset(first, 0x5a, 64);
    uintptr_t old_address = (uintptr_t)first;
    free(first);
    unsigned char *second = malloc(64);
    if (!second) {
        fail("malloc-reuse");
    }
    memset(second, 0x63, 64);
    int same_address = (uintptr_t)second == old_address;

    int fds[2];
    if (pipe(fds) != 0) {
        fail("pipe");
    }
    pid_t child = fork();
    if (child < 0) {
        fail("fork");
    }
    if (child == 0) {
        close(fds[0]);
        ssize_t written = write(fds[1], "linux", 5);
        _exit(written == 5 ? 0 : 1);
    }
    close(fds[1]);
    char buffer[16] = {0};
    ssize_t count = read(fds[0], buffer, sizeof(buffer));
    close(fds[0]);
    int status = 0;
    if (waitpid(child, &status, 0) != child ||
        !WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        count != 5 || memcmp(buffer, "linux", 5) != 0) {
        fail("read-fork-wait");
    }

    if (munmap(mapping, (size_t)page_size * 2) != 0) {
        fail("munmap");
    }
    free(second);
    printf("CAPSTONE_FEASIBILITY_BASELINE_OK same_address=%d\n", same_address);
    return 0;
}
