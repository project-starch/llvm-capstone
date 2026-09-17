/* RISC-V Linux loader. Shared regions hold trace, report, and heap; none of
 * them is part of the domain image. Copies stage kernel I/O through malloc
 * memory because this platform cannot read(2) directly into region mappings.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libcapstone.h"
#include "replay-format.h"

int main(int argc, char **argv)
{
    if (argc != 4) { fprintf(stderr, "usage: replay-host DOMAIN TRACE REPORT\n"); return 2; }
    setbuf(stdout, NULL);
    FILE *f = fopen(argv[2], "rb");
    unsigned char *staging = malloc(FFTRACE_BYTES);
    if (!f || !staging) return 2;
    size_t bytes = fread(staging, 1, FFTRACE_BYTES, f);
    if (ferror(f) || fgetc(f) != EOF || bytes < sizeof(struct ff_header)) return 3;
    fclose(f);
    struct ff_header *h = (struct ff_header *)staging;
    uint64_t expected_count = h->count;
    if (h->magic != FFTRACE_MAGIC || h->count >
        (FFTRACE_BYTES - sizeof *h) / sizeof(struct ff_event) ||
        bytes != sizeof *h + h->count * sizeof(struct ff_event)) return 3;
    if (capstone_init()) return 4;
    dom_id_t dom = create_dom(argv[1], NULL);
    if ((long)dom < 0) return 5;
    region_id_t ro = create_region(FFREPORT_BYTES);
    region_id_t ra = create_region(FFARENA_BYTES);
    region_id_t rt = create_region(FFTRACE_BYTES);
    if ((long)ro < 0 || (long)ra < 0 || (long)rt < 0) return 6;
    struct ff_header *out = map_region(ro, FFREPORT_BYTES);
    void *trace = map_region(rt, FFTRACE_BYTES);
    if (!out || out == (void *)-1 || !trace || trace == (void *)-1) return 7;
    memset(out, 0, sizeof *out);
    memcpy(trace, staging, bytes);
    printf("FFREPLAY loaded bytes=%lu events=%llu\n", (unsigned long)bytes,
           (unsigned long long)h->count);
    shared_region_annotated(dom, ro, 1, 0);
    shared_region_annotated(dom, ra, 1, 0);
    shared_region_annotated(dom, rt, 1, 0);
    unsigned long result = call_dom(dom);
    printf("FFREPLAY return=%lu status=%llu events=%llu arena=%llu\n", result,
           (unsigned long long)out->status, (unsigned long long)out->count,
           (unsigned long long)out->arena_used);
    if (out->magic != FFTRACE_MAGIC || out->count > h->count) return 8;
    size_t written = sizeof *out + out->count * sizeof(struct ff_event);
    memcpy(staging, out, written);
    f = fopen(argv[3], "wb");
    if (!f || fwrite(staging, 1, written, f) != written || fclose(f)) return 9;
    int ok = result == 42043 && !out->status && out->count == expected_count;
    ok = ok && written == bytes;
    capstone_cleanup();
    free(staging);
    return ok ? 0 : 1;
}
