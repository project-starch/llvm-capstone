/* Native client for the isolated namespace fixture. It enters no domain. */
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "namespace-probe.h"
int main(void)
{
    void *memory = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int probe = open("/dev/onq-namespace-probe", O_RDWR);
    int vm = open("/dev/capstone-vm", O_RDWR);
    struct onq_namespace_sample before = {.address = (unsigned long)memory}, after;
    if (memory == MAP_FAILED || probe < 0 || vm < 0) return 2;
    if (ioctl(probe, ONQ_NS_SEED, &before) || before.tag != 1) return 3;
    close(vm);
    after = before;
    if (ioctl(probe, ONQ_NS_SAMPLE, &after) || after.tag != 1) return 4;
    vm = open("/dev/capstone-vm", O_RDWR);
    if (vm < 0 || ioctl(probe, ONQ_NS_SAMPLE, &after)) return 5;
    int ok = after.tag == 0 && after.low == before.low && after.high == before.high;
    printf("NAMESPACE tag_before=%lu tag_after=%lu bytes_equal=%d\n", before.tag, after.tag,
           after.low == before.low && after.high == before.high);
    printf("NAMESPACE frames=%lx:%lx\n", before.frame, after.frame);
    if (ioctl(probe, ONQ_NS_CLEAR, &after)) return 6;
    int direct = after.tag == 0 && after.low == before.low && after.high == before.high;
    printf("NAMESPACE direct_scalar_clear=%d bytes_equal=%d\n", direct,
           after.low == before.low && after.high == before.high);
    if (!direct) return 7;
    close(vm);
    close(probe);
    munmap(memory, 4096);
    puts(ok ? "NAMESPACE PASS" : "NAMESPACE FAIL");
    return ok ? 0 : 1;
}
