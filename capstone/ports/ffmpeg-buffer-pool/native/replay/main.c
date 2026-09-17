#include "../../shared/replay-engine.inc"

#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
static jmp_buf failure;
_Noreturn void ff2_fail(unsigned code)
{
    report->status = code;
    fprintf(stderr, "FF2 fail=%u event=%llu op=%llu\n", code,
            (unsigned long long)cursor, (unsigned long long)(cursor < trace->count ? peek()->op : 0));
    longjmp(failure, 1);
}
int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4) return 2;
    struct ff2_header *input = calloc(1, FF2_FILE_BYTES);
    report = calloc(1, FF2_FILE_BYTES);
    void *meta = aligned_alloc(64, FF2_META_BYTES);
    void *payload = aligned_alloc(64, FF2_PAYLOAD_BYTES);
    FILE *f = fopen(argv[1], "rb");
    if (!input || !report || !meta || !payload || !f) return 2;
    size_t n = fread(input, 1, FF2_FILE_BYTES, f);
    if (ferror(f) || fgetc(f) != EOF || n < sizeof *input ||
        input->count > (FF2_FILE_BYTES - sizeof *input) / sizeof(struct ff2_event) ||
        n != sizeof *input + input->count * sizeof(struct ff2_event)) return 3;
    fclose(f); trace = input;
    report->mode = argc == 4 ? strtoul(argv[3], NULL, 0) : 0;
    if (!setjmp(failure)) { ff2_payload_init(payload, FF2_PAYLOAD_BYTES); replay(meta); }
    f = fopen(argv[2], "wb");
    size_t bytes = sizeof *report + report->count * sizeof(struct ff2_event);
    if (!f || fwrite(report, 1, bytes, f) != bytes || fclose(f)) return 4;
    printf("FF2 status=%llu events=%llu metadata=%llu payload=%llu\n",
           (unsigned long long)report->status, (unsigned long long)report->count,
           (unsigned long long)report->metadata_used, (unsigned long long)report->payload_used);
    return report->status ? 1 : 0;
}
