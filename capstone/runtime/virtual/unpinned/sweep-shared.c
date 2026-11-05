/* A namespace open must not scrub a tagged page still shared with a child.
 * Build with the namespace-probe.h fixture from the unpinned qualification
 * tests. This trusted native client enters no capability domain. */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include "namespace-probe.h"

int main(void)
{
    void *memory = mmap(0, 4096, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int probe = open("/dev/onq-namespace-probe", O_RDWR);
    int vm = open("/dev/capstone-vm", O_RDWR);
    struct onq_namespace_sample sample = {.address = (unsigned long)memory};
    int ready[2], done[2], status;
    char byte = 0;
    if (memory == MAP_FAILED || probe < 0 || vm < 0) return 2;
    if (ioctl(probe, ONQ_NS_SEED, &sample) || !sample.tag) return 3;
    close(vm);
    if (pipe(ready) || pipe(done)) return 4;
    pid_t child = fork();
    if (child < 0) return 5;
    if (!child) {
        close(ready[1]); close(done[0]);
        if (read(ready[0], &byte, 1) != 1) _exit(6);
        if (ioctl(probe, ONQ_NS_SAMPLE, &sample)) _exit(7);
        byte = !!sample.tag;
        if (write(done[1], &byte, 1) != 1) _exit(8);
        if (ioctl(probe, ONQ_NS_CLEAR, &sample)) _exit(9);
        _exit(0);
    }
    close(ready[0]); close(done[1]);
    errno = 0;
    vm = open("/dev/capstone-vm", O_RDWR);
    int refused = vm < 0 && errno == EOPNOTSUPP;
    if (vm >= 0) close(vm);
    if (write(ready[1], &byte, 1) != 1 || read(done[0], &byte, 1) != 1) return 10;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status)) return 11;
    printf("SWEEP_SHARED refused=%d child_tag=%d\n", refused, byte);
    /* With the child gone, a complete exclusive sweep must work again. */
    vm = open("/dev/capstone-vm", O_RDWR);
    if (vm < 0 || ioctl(probe, ONQ_NS_SAMPLE, &sample)) return 12;
    int clean = sample.tag == 0;
    if (ioctl(probe, ONQ_NS_CLEAR, &sample)) return 13;
    close(vm); close(probe); munmap(memory, 4096);
    printf("SWEEP_SHARED exclusive_clean=%d\n", clean);
    return refused && byte && clean ? 0 : 1;
}
