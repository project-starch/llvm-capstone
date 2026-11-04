/* Native client for the isolated namespace fixture. It enters no domain.
 *
 * namespace-check            seeds storage tags at granules 0, 16 and 4080 of several pages in
 *                            several mappings (see below), closes and reopens the namespace, and
 *                            requires EVERY seeded page to read tag-free with both representation
 *                            words unchanged;
 * namespace-check single     the first fixture: one granule at offset 0 of one page. A scrub that
 *                            stops after its first page or granule passes it, which is why the
 *                            default is the multi-seed run.
 *
 * Mappings seeded: an anonymous mapping of four pages; a second, separate mapping whose pages are
 * first touched read-only (the zero page) and then made writable and written; a stack page; a page
 * of .bss. Four mappings, ten pages, twenty-one granules.
 */
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "namespace-probe.h"

#define PAGE 4096UL
#define MAX_SEEDS 64

static char bss_page[2 * PAGE] __attribute__((aligned(4096)));

struct seed {
    struct onq_namespace_sample before;
    const char *where;
};

static struct seed seeds[MAX_SEEDS];
static int nseeds;

static int seed_granule(int probe, unsigned long page, unsigned long offset, const char *where)
{
    struct seed *s = &seeds[nseeds];
    if (nseeds == MAX_SEEDS) return -1;
    s->before = (struct onq_namespace_sample){.address = page, .offset = offset};
    s->where = where;
    if (ioctl(probe, ONQ_NS_SEED, &s->before) || s->before.tag == 0) return -1;
    ++nseeds;
    return 0;
}

int main(int argc, char **argv)
{
    int single = argc > 1 && !strcmp(argv[1], "single");
    static const unsigned long offsets[] = {0, 16, 4080};
    char *anon = mmap(0, 4 * PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char *late = mmap(0, 2 * PAGE, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char stack_storage[2 * PAGE];
    char *stack = (char *)(((unsigned long)stack_storage + PAGE - 1) & ~(PAGE - 1));
    int probe = open("/dev/onq-namespace-probe", O_RDWR);
    int vm = open("/dev/capstone-vm", O_RDWR);
    unsigned long pages[16], npages = 0;
    if (anon == MAP_FAILED || late == MAP_FAILED || probe < 0 || vm < 0) return 2;
    if (single) {
        if (seed_granule(probe, (unsigned long)anon, 0, "anon")) return 3;
    } else {
        /* the read-only-then-written pages: a read maps the zero page, mprotect + a store breaks it */
        volatile char sink = late[0] + late[PAGE];
        (void)sink;
        if (mprotect(late, 2 * PAGE, PROT_READ | PROT_WRITE)) return 2;
        late[0] = 1; late[PAGE] = 2;
        stack[0] = 1; stack[PAGE] = 2;
        bss_page[0] = 1; bss_page[PAGE] = 2;
        for (int p = 0; p < 4; ++p)
            for (int o = 0; o < 3; ++o)
                if (seed_granule(probe, (unsigned long)anon + p * PAGE, offsets[o], "anon")) return 3;
        for (int p = 0; p < 2; ++p) {
            if (seed_granule(probe, (unsigned long)late + p * PAGE, 16, "late")) return 3;
            if (seed_granule(probe, (unsigned long)late + p * PAGE, 4080, "late")) return 3;
        }
        if (seed_granule(probe, (unsigned long)stack, 4080, "stack")) return 3;
        if (seed_granule(probe, (unsigned long)stack + PAGE, 0, "stack")) return 3;
        if (seed_granule(probe, (unsigned long)bss_page, 0, "bss")) return 3;
        if (seed_granule(probe, (unsigned long)bss_page, 16, "bss")) return 3;
        if (seed_granule(probe, (unsigned long)bss_page + PAGE, 2048, "bss")) return 3;
    }
    for (int i = 0; i < nseeds; ++i) {
        int known = 0;
        for (unsigned long j = 0; j < npages; ++j) known |= pages[j] == seeds[i].before.address;
        if (!known) pages[npages++] = seeds[i].before.address;
    }
    close(vm);
    /* after the close, before the reopen: every seeded page must still hold its tags */
    int survive = 1;
    for (unsigned long j = 0; j < npages; ++j) {
        struct onq_namespace_sample probe_page = {.address = pages[j]};
        if (ioctl(probe, ONQ_NS_SAMPLE, &probe_page) || probe_page.tag == 0) survive = 0;
    }
    if (!survive) return 4;
    vm = open("/dev/capstone-vm", O_RDWR);
    if (vm < 0) return 5;
    int tag_free = 1, equal = 1;
    for (int i = 0; i < nseeds; ++i) {
        struct onq_namespace_sample after = seeds[i].before;
        if (ioctl(probe, ONQ_NS_SAMPLE, &after)) return 5;
        int same = after.low == seeds[i].before.low && after.high == seeds[i].before.high;
        equal &= same;
        if (after.tag) {
            tag_free = 0;
            printf("NAMESPACE_TAGGED %s page=%lx offset=%lu\n", seeds[i].where, after.address, after.offset);
        }
    }
    printf("NAMESPACE mappings=%s pages=%lu seeds=%d tag_after_clean=%d bytes_equal=%d\n",
           single ? "1" : "4", npages, nseeds, tag_free, equal);
    /* the scalar clear works directly, as the control that the tags are clearable at all: clear
     * every seeded granule by hand, then every page must read tag-free */
    int cleared = 1, equal_direct = 1;
    for (int i = 0; i < nseeds; ++i) {
        struct onq_namespace_sample d = seeds[i].before, kept;
        if (ioctl(probe, ONQ_NS_SAMPLE, &d)) return 6;
        kept = d;
        if (ioctl(probe, ONQ_NS_CLEAR, &d)) return 6;
        equal_direct &= d.low == kept.low && d.high == kept.high;
    }
    for (unsigned long j = 0; j < npages; ++j) {
        struct onq_namespace_sample d = {.address = pages[j]};
        if (ioctl(probe, ONQ_NS_SAMPLE, &d)) return 6;
        cleared &= d.tag == 0;
    }
    printf("NAMESPACE direct_scalar_clear=%d bytes_equal=%d\n", cleared, equal_direct);
    if (!cleared || !equal_direct) return 7;
    close(vm);
    close(probe);
    int ok = tag_free && equal;
    puts(ok ? "NAMESPACE PASS" : "NAMESPACE FAIL");
    return ok ? 0 : 1;
}
