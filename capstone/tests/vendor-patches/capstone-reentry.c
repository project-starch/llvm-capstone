#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lib/libcapstone.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Please provide the domain file name!\n");
        return 1;
    }

    int retval = capstone_init();
    if(retval) {
        fprintf(stderr, "Failed to initialise Capstone\n");
        return retval;
    }

    char* file_name = argv[1];

    unsigned times = 1; // how many times to run the domain
    if (argc >= 3) {
        times = atoi(argv[2]);
    }

    dom_id_t dom_id;
    region_id_t region_id;
    if (argc >= 4) {
        dom_id = create_dom(file_name, argv[3]);
    } else {
        dom_id = create_dom(file_name, NULL);
    }
    printf("Created domain ID = %lu\n", dom_id);

    /* REENTRY LOADER. Same annotated-share mechanism as capstone-diag, but it shares
     * TWO regions, which means the domain is ENTERED TWICE -- the share is the entry.
     * That is the only way to exercise the glue's __test_reentry path, and it is the
     * shape sqlite_capstone_domain.c actually has: two CAPSTONE_DPI_REGION_SHARE
     * entries that stash host capabilities in globals, then a run entry.
     *
     * Purpose: gate S2 (idempotent entry). Before S2 both entry points rebuilt the
     * cap table and re-ran every initializer, so a global written on entry 1 was
     * reset before entry 2 could read it. The paired rung counts its entries in a
     * global, so its returned value is wrong unless that global SURVIVES a domreturn.
     *
     * Original diag comment follows, since the share mechanics are identical:
     * DIAG LOADER: the region IS shared here, unlike capstone-test.user.
     *
     * Why a separate loader instead of enabling this in capstone-test.c: that file
     * is the loader for the ENTIRE QEMU corpus (82 BEEBS, RV8, CoreMark, SQLite,
     * the authority suite). Sharing a region changes what the domain's first
     * argument IS, so enabling it there risks moving where every existing domain
     * finds its result -- a regression across everything the project validates on.
     * A separate binary has zero regression surface.
     *
     * What it buys (issue I-3): diagnostic rungs write raw per-probe values into
     * res[3..47]. Without a shared region a domain gets only an 8-byte return slot,
     * so those writes fault and every *_diag / rawhazard* probe is BOARD-ONLY. That
     * is why two board boots produced one data point on 2026-07-28. With this
     * loader they run under QEMU in seconds.
     *
     * Run it via run-domain-smoke.py --domain-loader, from the 9p host share; no
     * guest image rebuild is required. */
    region_id = create_region(4096);
    unsigned char *rmap = map_region(region_id, 4096);
    if (!rmap) { fprintf(stderr, "map_region failed\n"); return 1; }
    memset(rmap, 0, 4096);

    region_id_t region2 = create_region(4096);
    unsigned char *rmap2 = map_region(region2, 4096);
    if (!rmap2) { fprintf(stderr, "map_region 2 failed\n"); return 1; }
    memset(rmap2, 0, 4096);

    /* Entry 1. */
    shared_region_annotated(dom_id, region_id, 0x1u, 0x2u);
    unsigned long *r1 = (unsigned long *)rmap;
    printf("entry1 retval = %lu\n", r1[0]);

    /* Entry 2 -- reached through the glue's __test_reentry path. */
    shared_region_annotated(dom_id, region2, 0x1u, 0x2u);
    unsigned long *r2 = (unsigned long *)rmap2;
    printf("entry2 retval = %lu\n", r2[0]);
    printf("REENTRY RESULT entry1=%lu entry2=%lu\n", r1[0], r2[0]);

    retval = capstone_cleanup();
    if (retval) { fprintf(stderr, "Failed to clean up Capstone\n"); return retval; }
    return 0;
}
