/* tagleak: does a frame reach a new capability context still carrying a tag from an earlier context?
 * fill <MB>: store a capability in every 16-byte granule of a fresh MB-sized block, record the block's PFNs
 *            (from /proc/self/pagemap) in fill.pfn, exit. Run with retire_exit_sweep=2, its frames go back
 *            to Linux with their tags.
 * scan <MB>: take a fresh block, count tagged granules in it, read its PFNs, count how many came from the
 *            fill (recycled frames: the test is loaded only if that count is high), then store one capability
 *            and re-count it (positive control: the detector sees a tag it made).
 * Prints one line: TAGLEAK <mode> mb=.. pages=.. tagged=.. tagged_pages=.. recycled=.. tagged_recycled=.. ctl=.. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>

static uint64_t pfn_of(int fd, uintptr_t va) {
    uint64_t e = 0;
    if (pread(fd, &e, 8, (off_t)(va / 4096) * 8) != 8) return 0;
    return (e >> 63) & 1 ? e & ((1ULL << 55) - 1) : 0;
}

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    int fill = argv[1][0] == 'f';
    size_t mb = strtoul(argv[2], 0, 10), n = mb << 20, pages = n / 4096, gpp = 4096 / sizeof(void *);
    char *raw = malloc(n + 4096);
    if (!raw) { printf("TAGLEAK %s malloc-failed\n", argv[1]); return 3; }
    void **p = (void **)(raw + (4096 - ((uintptr_t)raw & 4095)) % 4096);
    unsigned long tagged = 0, tagged_pages = 0, recycled = 0, tagged_recycled = 0, nopfn = 0;
    unsigned char *pt = calloc(pages, 1);
    if (fill) {
        for (size_t i = 0; i < pages * gpp; i++) ((void *volatile *)p)[i] = (void *)p;
    } else {
        for (size_t pg = 0; pg < pages; pg++) {
            unsigned long t = 0;
            for (size_t g = 0; g < gpp; g++) t += __builtin_capstone_cap_get_tag(((void *volatile *)p)[pg * gpp + g]) != 0;
            tagged += t; tagged_pages += t != 0; pt[pg] = t != 0;
        }
    }
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) { printf("TAGLEAK %s no-pagemap\n", argv[1]); return 4; }
    if (fill) {
        FILE *f = fopen("/tmp/fill.pfn", "w");
        for (size_t pg = 0; pg < pages; pg++) { uint64_t x = pfn_of(fd, (uintptr_t)p + pg * 4096); nopfn += !x; fprintf(f, "%llu\n", (unsigned long long)x); }
        fclose(f);
        /* the fill's own tags, read back: the detector's positive control on the fill side */
        for (size_t i = 0; i < pages * gpp; i += 97) tagged += __builtin_capstone_cap_get_tag(((void *volatile *)p)[i]) != 0;
        printf("TAGLEAK fill mb=%zu pages=%zu sampled_tagged=%lu of %zu nopfn=%lu\n", mb, pages, tagged, (pages * gpp + 96) / 97, nopfn);
        return 0;
    }
    /* scan: which of this block's frames did the fill hold? */
    size_t nf = 0, cap = 1 << 16; uint64_t *fp = malloc(cap * 8), x;
    FILE *f = fopen("/tmp/fill.pfn", "r");
    if (f) { unsigned long long v; while (fscanf(f, "%llu", &v) == 1) { if (nf == cap) fp = realloc(fp, (cap *= 2) * 8); fp[nf++] = v; } fclose(f); }
    /* sort-free membership: a bitmap over PFNs below 2^20 (4 GiB) */
    unsigned char *bm = calloc(1 << 17, 1);
    for (size_t i = 0; i < nf; i++) if (fp[i] && fp[i] < (1u << 20)) bm[fp[i] >> 3] |= 1 << (fp[i] & 7);
    for (size_t pg = 0; pg < pages; pg++) {
        x = pfn_of(fd, (uintptr_t)p + pg * 4096); nopfn += !x;
        if (x && x < (1u << 20) && (bm[x >> 3] >> (x & 7) & 1)) { recycled++; tagged_recycled += pt[pg]; }
    }
    ((void *volatile *)p)[5] = (void *)p;
    unsigned long ctl = __builtin_capstone_cap_get_tag(((void *volatile *)p)[5]) != 0;
    printf("TAGLEAK scan mb=%zu pages=%zu tagged=%lu tagged_pages=%lu fill_pfns=%zu recycled=%lu tagged_recycled=%lu nopfn=%lu ctl=%lu\n",
           mb, pages, tagged, tagged_pages, nf, recycled, tagged_recycled, nopfn, ctl);
    return 0;
}
