/* Linux staging loader. The payload grant is linear; trace, report, and
 * allocator metadata remain in distinct nonlinear shared regions.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libcapstone.h"
#include "trace.h"
int main(int argc, char **argv)
{
    if (argc != 5) return 2;
    setbuf(stdout, NULL);
    unsigned mode = strtoul(argv[4], NULL, 0);
    if (mode > 2) return 2;
    FILE *f = fopen(argv[2], "rb");
    unsigned char *staging = malloc(FF2_FILE_BYTES);
    if (!f || !staging) return 2;
    size_t bytes = fread(staging, 1, FF2_FILE_BYTES, f);
    if (ferror(f) || fgetc(f) != EOF || bytes < sizeof(struct ff2_header)) return 3;
    fclose(f);
    struct ff2_header *h = (struct ff2_header *)staging;
    uint64_t expected = h->count;
    if (h->magic != FF2_MAGIC || expected > (FF2_FILE_BYTES - sizeof *h) / sizeof(struct ff2_event) ||
        bytes != sizeof *h + expected * sizeof(struct ff2_event)) return 3;
    if (capstone_init()) return 4;
    dom_id_t dom = create_dom(argv[1], NULL);
    if ((long)dom < 0) return 5;
    region_id_t ro = create_region(FF2_FILE_BYTES);
    region_id_t rm = create_region(FF2_META_BYTES);
    region_id_t rt = create_region(FF2_FILE_BYTES);
    region_id_t rp = create_region(FF2_PAYLOAD_BYTES);
    if ((long)ro < 0 || (long)rm < 0 || (long)rt < 0 || (long)rp < 0) return 6;
    struct ff2_header *out = map_region(ro, FF2_FILE_BYTES);
    void *input = map_region(rt, FF2_FILE_BYTES);
    if (!out || out == (void *)-1 || !input || input == (void *)-1) return 7;
    memset(out, 0, sizeof *out); out->mode = mode;
    memcpy(input, staging, bytes);
    printf("FF2 loaded bytes=%lu events=%llu mode=%u\n", (unsigned long)bytes, (unsigned long long)expected, mode);
    shared_region_annotated(dom, ro, 1, 0);
    shared_region_annotated(dom, rm, 1, 0);
    shared_region_annotated(dom, rt, 1, 0);
    shared_region_annotated(dom, rp, 1, 1);
    unsigned long result = call_dom(dom);
    printf("FF2 return=%lu status=%llu events=%llu metadata=%llu payload=%llu\n", result,
           (unsigned long long)out->status, (unsigned long long)out->count,
           (unsigned long long)out->metadata_used, (unsigned long long)out->payload_used);
    if (out->magic != FF2_MAGIC || out->count > expected) return 8;
    size_t written = sizeof *out + out->count * sizeof(struct ff2_event);
    memcpy(staging, out, written);
    f = fopen(argv[3], "wb");
    if (!f || fwrite(staging, 1, written, f) != written || fclose(f)) return 9;
    int ok = result == 42044 && !out->status && out->count == expected;
    capstone_cleanup(); free(staging);
    return ok ? 0 : 1;
}
