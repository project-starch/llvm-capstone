/* Native client consuming the namespace and page-reference test doubles.
 * A missing put_page must fail the same assertions as the production module. */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include "namespace-probe.h"
#include "sweep-ref-probe.h"

static int vm_open(void) { return open("/dev/capstone-vm", O_RDWR); }
static void *drain_page;

static int sample_refs(int fd, struct sweep_ref_sample *value)
{
    unsigned char resident = 0;
    /* Linux 6.1 madvise_cold drains the local LRU batches before walking the
     * requested range. Use our untouched middle page so that walk cannot add
     * a new deactivation reference to either observed page. No kernel export
     * or production instrumentation is required. One hart only. */
    if (mincore(drain_page, 4096, &resident) || (resident & 1) ||
        madvise(drain_page, 4096, MADV_COLD)) {
        printf("SWEEP_REF_SETUP resident=%u errno=%d\n", resident, errno);
        return -1;
    }
    int rc = ioctl(fd, SWEEP_REF_SAMPLE, value);
    printf("SWEEP_REF_SAMPLE rc=%d references=%lu mappings=%lu\n",
           rc, value->references, value->mappings);
    return rc;
}

int main(void)
{
    volatile unsigned long *memory = mmap(0, 3 * 4096, PROT_READ | PROT_WRITE,
                                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int ns = open("/dev/onq-namespace-probe", O_RDWR);
    int refs = open("/dev/sweep-ref-probe", O_RDWR);
    int vm = vm_open(), ready[2], done[2], status;
    unsigned long shared_frame;
    char byte = 0;
    struct onq_namespace_sample tag = {.address = (unsigned long)memory};
    struct sweep_ref_sample before = {.address = (unsigned long)memory}, after = before;
    struct sweep_ref_sample shared_before = {.address = (unsigned long)memory + 2 * 4096};
    struct sweep_ref_sample shared_after = shared_before;
    if (memory == MAP_FAILED || ns < 0 || refs < 0 || vm < 0) return 2;
    drain_page = (char *)memory + 4096;
    if (ioctl(ns, ONQ_NS_SEED, &tag) || !tag.tag) return 3;
    close(vm);
    if (sample_refs(refs, &before) || before.mappings != 1 ||
        before.references != 2 || (before.pfn << 12) != tag.frame) return 4;
    /* The scanner must see the PFN of an inaccessible but present PTE. */
    if (mprotect((void *)memory, 4096, PROT_NONE)) return 5;
    vm = vm_open();
    if (vm < 0) return 6;
    close(vm);
    tag.tag = ~0UL; /* An unchanged output field must fail either assertion. */
    if (mprotect((void *)memory, 4096, PROT_READ | PROT_WRITE) ||
        ioctl(ns, ONQ_NS_SAMPLE, &tag) || sample_refs(refs, &after)) return 7;
    int clean = !tag.tag;
    int same = before.pfn == after.pfn && after.mappings == 1;
    int balanced = before.references == after.references && same;
    printf("SWEEP_REFS success_before=%lu success_after=%lu protnone_clean=%d same_page=%d\n",
           before.references, after.references, clean, same);
    /* Two tagged pages share a VMA. After fork, COW the first in the parent;
     * the later page remains shared and forces a failure after an earlier
     * exclusive page has been held. Both processes retain their mappings. */
    if (ioctl(ns, ONQ_NS_SEED, &tag)) return 8;
    tag.address = (unsigned long)memory + 2 * 4096;
    if (ioctl(ns, ONQ_NS_SEED, &tag) || pipe(ready) || pipe(done)) return 9;
    shared_frame = tag.frame;
    pid_t child = fork();
    if (child < 0) return 10;
    if (!child) {
        close(ready[1]); close(done[0]);
        if (write(done[1], &byte, 1) != 1 || read(ready[0], &byte, 1) != 1) _exit(11);
        _exit(0);
    }
    close(ready[0]); close(done[1]);
    if (read(done[0], &byte, 1) != 1) return 12;
    memory[0] = 0; /* COW and scalar clear; seed the now-exclusive copy. */
    tag.address = (unsigned long)memory;
    if (ioctl(ns, ONQ_NS_SEED, &tag) || sample_refs(refs, &before) ||
        before.mappings != 1 || before.references != 2 || (before.pfn << 12) != tag.frame ||
        sample_refs(refs, &shared_before) || shared_before.mappings != 2 ||
        shared_before.references != 3 || (shared_before.pfn << 12) != shared_frame) return 13;
    errno = 0;
    vm = vm_open();
    int refused = vm < 0 && errno == EOPNOTSUPP;
    if (vm >= 0) close(vm);
    tag.tag = ~0UL;
    if (sample_refs(refs, &after) || ioctl(ns, ONQ_NS_SAMPLE, &tag) ||
        sample_refs(refs, &shared_after)) return 14;
    same = before.pfn == after.pfn && after.mappings == 1;
    int shared_same = shared_before.pfn == shared_after.pfn && shared_after.mappings == 2;
    int rollback = before.references == after.references && tag.tag == 1 && same &&
                   shared_before.references == shared_after.references && shared_same;
    printf("SWEEP_REFS failure_before=%lu failure_after=%lu refused=%d tag_preserved=%lu same_page=%d\n",
           before.references, after.references, refused, tag.tag, same);
    printf("SWEEP_REFS shared_before=%lu shared_after=%lu same_page=%d mappings=%lu\n",
           shared_before.references, shared_after.references, shared_same, shared_after.mappings);
    if (write(ready[1], &byte, 1) != 1 || waitpid(child, &status, 0) != child ||
        !WIFEXITED(status) || WEXITSTATUS(status)) return 15;
    if (ioctl(ns, ONQ_NS_CLEAR, &tag)) return 16;
    tag.address = (unsigned long)memory + 2 * 4096;
    if (ioctl(ns, ONQ_NS_CLEAR, &tag)) return 17;
    close(ns); close(refs); munmap((void *)memory, 3 * 4096);
    return clean && balanced && refused && rollback ? 0 : 1;
}
