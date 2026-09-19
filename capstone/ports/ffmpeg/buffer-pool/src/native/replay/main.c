#include "replay-engine.h"

#ifdef FFPOOL_CHERI
#include <malloc_np.h>
#endif
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef FFPOOL_POISONCAP
#include <sys/mman.h>
#endif
static jmp_buf failure;
static struct ff2_header *report;
_Noreturn void ff2_fail(unsigned code) {
  report->status = code;
  fprintf(stderr, "FF2 fail=%u event=%llu op=%llu\n", code,
          (unsigned long long)ff2_replay_cursor(),
          (unsigned long long)ff2_replay_operation());
  longjmp(failure, 1);
}
int main(int argc, char **argv) {
  if (argc < 3 || argc > 4)
    return 2;
  struct ff2_header *input = calloc(1, FF2_FILE_BYTES);
  report = calloc(1, FF2_FILE_BYTES);
  void *meta = aligned_alloc(64, FF2_META_BYTES);
#ifdef FFPOOL_POISONCAP
  /* The trusted manager needs backing authority to manage inner lifetimes. */
  void *payload = mmap(NULL, FF2_PAYLOAD_BYTES, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON, -1, 0);
  if (payload == MAP_FAILED)
    return 2;
#else
  void *payload = aligned_alloc(64, FF2_PAYLOAD_BYTES);
#endif
  FILE *f = fopen(argv[1], "rb");
  if (!input || !report || !meta || !payload || !f)
    return 2;
#ifdef FFPOOL_CHERI
  printf("FF2 CHERI runtime_revoke=%u pointer_bytes=%zu\n",
         (unsigned)malloc_revoke_enabled(), sizeof(void *));
  fflush(stdout);
#endif
  size_t n = fread(input, 1, FF2_FILE_BYTES, f);
  if (ferror(f) || fgetc(f) != EOF || n < sizeof *input ||
      input->count >
          (FF2_FILE_BYTES - sizeof *input) / sizeof(struct ff2_event) ||
      n != sizeof *input + input->count * sizeof(struct ff2_event))
    return 3;
  fclose(f);
  report->mode = argc == 4 ? strtoul(argv[3], NULL, 0) : 0;
  if (!setjmp(failure)) {
    ff2_payload_init(payload, FF2_PAYLOAD_BYTES);
    ff2_replay_run(input, report, meta);
  }
  f = fopen(argv[2], "wb");
  size_t bytes = sizeof *report + report->count * sizeof(struct ff2_event);
  if (!f || fwrite(report, 1, bytes, f) != bytes || fclose(f))
    return 4;
  printf("FF2 status=%llu events=%llu metadata=%llu payload=%llu\n",
         (unsigned long long)report->status, (unsigned long long)report->count,
         (unsigned long long)report->metadata_used,
         (unsigned long long)report->payload_used);
  return report->status ? 1 : 0;
}
