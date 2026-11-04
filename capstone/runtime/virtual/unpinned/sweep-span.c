/* A namespace scrub must reach tagged pages in later PTE tables and later VMAs.
 * Native client for the namespace-probe.h fixture (it enters no domain). The
 * default namespace-check seeds fit one PTE table per mapping, so a sweep that
 * stopped at the end of its first PTE table, or at the first VMA that held a
 * tagged page, would still pass it. This client seeds storage tags at:
 *   - four 2 MiB-aligned PTE tables of one 8 MiB mapping: table 0 holds only an
 *     UNTAGGED resident page, tables 1-3 hold tagged pages (table 2 two of them,
 *     one early and one near its end);
 *   - both sides of a PROT_NONE page inside a second mapping, so tagged pages sit
 *     in two VMAs that are separated by a third;
 * closes and reopens the namespace, and requires every seeded page to read
 * tag-free with both representation words unchanged. */
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "namespace-probe.h"

#define PAGE 4096UL
#define TABLE (512 * PAGE)
#define MAX_SEEDS 32

static struct onq_namespace_sample seeds[MAX_SEEDS];
static const char *where[MAX_SEEDS];
static int nseeds;

static int seed(int probe, unsigned long page, unsigned long offset, const char *label)
{
    struct onq_namespace_sample *s = &seeds[nseeds];
    if (nseeds == MAX_SEEDS) return -1;
    *s = (struct onq_namespace_sample){.address = page, .offset = offset};
    where[nseeds] = label;
    if (ioctl(probe, ONQ_NS_SEED, s) || s->tag == 0) return -1;
    ++nseeds;
    return 0;
}

int main(void)
{
    char *raw = mmap(0, 5 * TABLE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char *guarded = mmap(0, 5 * PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char *base;
    int probe = open("/dev/onq-namespace-probe", O_RDWR);
    int vm = open("/dev/capstone-vm", O_RDWR);
    if (raw == MAP_FAILED || guarded == MAP_FAILED || probe < 0 || vm < 0) return 2;
    base = (char *)(((unsigned long)raw + TABLE - 1) & ~(TABLE - 1));
    /* Table 0: resident, no tag. A sweep that gives up after a table with no tagged page misses the rest. */
    base[0] = 1;
    base[3 * PAGE] = 1;
    base[TABLE + 5 * PAGE] = 1;
    base[2 * TABLE + 7 * PAGE] = 1;
    base[2 * TABLE + 500 * PAGE] = 1;
    base[3 * TABLE + 9 * PAGE] = 1;
    if (seed(probe, (unsigned long)base + TABLE + 5 * PAGE, 16, "table1")) return 3;
    if (seed(probe, (unsigned long)base + 2 * TABLE + 7 * PAGE, 0, "table2-early")) return 3;
    if (seed(probe, (unsigned long)base + 2 * TABLE + 500 * PAGE, 4080, "table2-late")) return 3;
    if (seed(probe, (unsigned long)base + 3 * TABLE + 9 * PAGE, 2048, "table3")) return 3;
    guarded[0] = 1; guarded[4 * PAGE] = 1;
    if (seed(probe, (unsigned long)guarded, 0, "vma-low")) return 3;
    if (seed(probe, (unsigned long)guarded + 4 * PAGE, 0, "vma-high")) return 3;
    if (mprotect(guarded + 2 * PAGE, PAGE, PROT_NONE)) return 2;
    close(vm);
    for (int i = 0; i < nseeds; ++i) {
        struct onq_namespace_sample now = {.address = seeds[i].address};
        if (ioctl(probe, ONQ_NS_SAMPLE, &now) || now.tag == 0) return 4;
    }
    vm = open("/dev/capstone-vm", O_RDWR);
    if (vm < 0) return 5;
    int tag_free = 1, equal = 1;
    for (int i = 0; i < nseeds; ++i) {
        struct onq_namespace_sample after = seeds[i];
        if (ioctl(probe, ONQ_NS_SAMPLE, &after)) return 5;
        equal &= after.low == seeds[i].low && after.high == seeds[i].high;
        if (after.tag) {
            tag_free = 0;
            printf("SPAN_TAGGED %s page=%lx offset=%lu\n", where[i], after.address, after.offset);
        }
    }
    printf("SPAN tables=4 vmas_with_tags=3 seeds=%d tag_after_clean=%d bytes_equal=%d\n", nseeds, tag_free, equal);
    close(vm);
    close(probe);
    return tag_free && equal ? 0 : 1;
}
