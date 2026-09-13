/* Create the domain, share one region for the level below's arena, call it once, print what came
 * back. Deliberately smaller than the other ports' guests: this domain runs a fixed driver rather
 * than a resumable suite, so there is nothing to resume and nothing to capture.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "capstone.h"
#include "lib/libcapstone.h"

#define NGX_ARENA_BYTES (1UL << 20)   /* one mebibyte: the driver's thousand cycles fit easily */
#define NGX_PERM_INOUT  0x1UL
#define NGX_REV_SHARED  0x2UL

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s DOMAIN\n", argv[0]);
        return 2;
    }
    int rc = capstone_init();
    if (rc != 0) {
        fprintf(stderr, "Failed to initialise Capstone: %d\n", rc);
        return rc;
    }
    dom_id_t dom = create_dom(argv[1], NULL);
    printf("Created domain ID = %lu\n", dom);

    region_id_t arena = create_region(NGX_ARENA_BYTES);
    void *mapped = map_region(arena, NGX_ARENA_BYTES);
    if (mapped == NULL) {
        fprintf(stderr, "Failed to map the arena region\n");
        capstone_cleanup();
        return 3;
    }
    memset(mapped, 0, NGX_ARENA_BYTES);
    shared_region_annotated(dom, arena, NGX_PERM_INOUT, NGX_REV_SHARED);

    unsigned long v = call_dom(dom);
    printf("ngx retval = %lu\n", v);
    capstone_cleanup();
    return 0;
}
