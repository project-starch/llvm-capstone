/* What the patched slabs.c references beyond its allocator: the settings
 * object memcached.c would have defined, and the four libc services the
 * freestanding shim renamed. Each service ends the run with its own code
 * rather than returning quietly, so a case that came to depend on one cannot
 * pass unnoticed; getenv is the exception, answering nothing, because upstream
 * consults it on every slabs_init for a test-suite knob. */
#include "mc_slabs_shim.h"
#include "mc_slabs_libc.h"
#include "port.h"

/* memcached.c settings_init, 1.6.45 lines 224-258: the defaults a server
 * starts with, for the fields slabs.c reads. slab_chunk_size_max is
 * slab_page_size / 2 there too. */
struct settings settings = {
    .maxbytes = 64 * 1024 * 1024, /* default is 64MB */
    .verbose = 0,
    .factor = 1.25,
    .chunk_size = 48, /* space for a modest key and value */
    .item_size_max = 1024 * 1024, /* The famous 1MB upper limit. */
    .slab_chunk_size_max = 1024 * 1024 / 2,
    .slab_page_size = 1024 * 1024, /* chunks are split from 1MB pages. */
    .slab_reassign = true,
};

int mc_fprintf(FILE *stream, const char *format, ...) {
  (void)stream;
  (void)format;
  mcp_fail(609);
}
_Noreturn void mc_exit(int code) {
  (void)code;
  mcp_fail(610);
}
char *mc_getenv(const char *name) {
  (void)name;
  return NULL;
}
bool safe_strtoll(const char *str, int64_t *out) {
  (void)str;
  (void)out;
  mcp_fail(611);
}
