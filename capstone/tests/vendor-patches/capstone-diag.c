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

    /* DIAG LOADER: the region IS shared here, unlike capstone-test.user.
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
    /* MAP and zero the region before sharing, matching ladder_perf_ctl, whose
       comment says "the share IS the entry". Sharing WITHOUT mapping first left the
       domain seeing only the 8-byte return slot -- tried both plain and annotated
       shares, both faulted at res[3]. */
    unsigned char *rmap = map_region(region_id, 4096);
    if (!rmap) { fprintf(stderr, "map_region failed\n"); return 1; }
    for (unsigned k = 0; k < 4096; k++) rmap[k] = 0;
    /* ANNOTATED share, matching ladder_perf_ctl. Plain share_region() does NOT
       become the domain's first argument -- tried it, the domain still saw only the
       8-byte return slot and faulted at res[3]. The annotation is what makes the
       region the entry argument, which is exactly how the FPGA controller delivers
       res[0..47] to a diagnostic rung. */

    /* THE SHARE IS THE ENTRY. ladder_perf_ctl says so in as many words, and it is
       why four earlier attempts failed: sharing the region and THEN calling call_dom
       runs the domain via the plain call path, whose first argument is the 8-byte
       return slot -- so res[3..] faulted every time. The annotated share itself
       enters the domain with the REGION as its argument, so there is no call_dom
       here and the results are read back out of the mapped region. */
    shared_region_annotated(dom_id, region_id, 0x1u /*ANNOT_PERM_INOUT*/,
                            0x2u /*REV_SHARED*/);

    unsigned long *res = (unsigned long *)rmap;
    printf("Called dom (1-th time) retval = %lu\n", res[0]);
    /* Diagnostic slots, the whole point of this loader: res[3..47] raw. */
    int any = 0;
    for (int i = 0; i < 45; i++) if (res[3 + i]) { any = 1; break; }
    if (any) {
        printf("DEBUG");
        for (int i = 0; i < 45; i++) printf(" dbg%d=%lu", i, res[3 + i]);
        printf("\n");
    }

    retval = capstone_cleanup();
    if(retval) {
        fprintf(stderr, "Failed to clean up Capstone\n");
        return retval;
    }

    return 0;
}
