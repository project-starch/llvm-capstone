/* wm-test.c: unit test of wmem's lifetime patch (0001), the block and block_fast allocators built
 * with the port's GLib shim and inject.h.
 *   wm-test stale        a block object read after wmem_free: must fault (cause 25)
 *   wm-test stale-reset  a block_fast object read after wmem_free_all: must fault (cause 25)
 *   wm-test fail-issue   a failed CDERIVE for an object: g_error (wm_fail 209), never NULL
 *   wm-test fail-block   a failed CDERIVE for a block's generation: g_error (wm_fail 209)
 *   wm-test fail-renew   a failed CRENEW at wmem_free_all: g_error (wm_fail 209), so no object of
 *                        the old generation outlives the reset */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wmem_core.h"
#include "port.h"

long lifec_fail_left, lifec_fail_size, lifec_failed;
unsigned long lifec_failed_at;

_Noreturn void wm_fail(unsigned code) { printf("WMTEST wm_fail %u (failed=%ld)\n", code, lifec_failed); fflush(stdout); exit(code == 209 ? 0 : 3); }

static void arm(void) { lifec_fail_size = 0; lifec_fail_left = 1; }

int main(int argc, char **argv)
{
    const char *m = argc > 1 ? argv[1] : "";
    wmem_init();
    if (!strcmp(m, "stale")) {
        wmem_allocator_t *a = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK);
        volatile char *p = wmem_alloc(a, 64);
        void *keep = wmem_alloc(a, 64);
        p[0] = 7;
        wmem_free(a, (void *)p);
        printf("WMTEST stale: read after free\n"); fflush(stdout);
        printf("WMTEST stale: read %d, no fault\n", p[0]);
        (void)keep;
        return 1;
    }
    if (!strcmp(m, "stale-reset")) {
        wmem_allocator_t *a = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK_FAST);
        volatile char *p = wmem_alloc(a, 64);
        p[0] = 7;
        wmem_free_all(a);
        printf("WMTEST stale-reset: read after reset\n"); fflush(stdout);
        printf("WMTEST stale-reset: read %d, no fault\n", p[0]);
        return 1;
    }
    if (!strcmp(m, "fail-issue")) {
        wmem_allocator_t *a = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK);
        void *keep = wmem_alloc(a, 64);
        arm();
        char *p = wmem_alloc(a, 64);
        printf("WMTEST fail-issue: wmem_alloc returned %s\n", p ? "a pointer" : "NULL"); fflush(stdout);
        memset(p, 0x5a, 64);
        printf("WMTEST fail-issue: wrote through it\n");
        (void)keep;
        return 1;
    }
    if (!strcmp(m, "fail-block")) {
        wmem_allocator_t *a = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK);
        arm();
        char *p = wmem_alloc(a, 64);
        printf("WMTEST fail-block: wmem_alloc returned %s\n", p ? "a pointer" : "NULL"); fflush(stdout);
        memset(p, 0x5a, 64);
        printf("WMTEST fail-block: wrote through it\n");
        return 1;
    }
    if (!strcmp(m, "fail-renew")) {
        wmem_allocator_t *a = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK_FAST);
        volatile char *p = wmem_alloc(a, 64);
        p[0] = 7;
        arm();
        wmem_free_all(a);
        printf("WMTEST fail-renew: reset returned\n"); fflush(stdout);
        printf("WMTEST fail-renew: read %d through an object of the reset pool, no fault\n", p[0]);
        return 1;
    }
    return 2;
}
