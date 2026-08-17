/* Guest-side loader for resumable MicroPython suite runs.
 *
 * A capability fault stops the entire QEMU boot.  Share one control page before the first normal
 * call so the domain can start at an arbitrary test-table index; the host can then reboot and
 * continue at the following test without rebuilding the large interpreter image.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lib/libcapstone.h"

#define MPY_TEST_START_MAGIC 0x4d50595354415254ULL
#define MPY_CONTROL_PERM_INOUT 0x1UL
#define MPY_CONTROL_REV_SHARED 0x2UL

struct mpy_hostcall_v0 {
    unsigned long long phase, opcode, offset, length;
    long long result, error;
};

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s DOMAIN START COUNT\n", argv[0]);
        return 2;
    }

    unsigned long start = strtoul(argv[2], NULL, 0);
    unsigned long count = strtoul(argv[3], NULL, 0);
    int rc = capstone_init();
    if (rc != 0) {
        fprintf(stderr, "Failed to initialise Capstone: %d\n", rc);
        return rc;
    }

    dom_id_t dom_id = create_dom(argv[1], NULL);
    printf("Created domain ID = %lu\n", dom_id);

    region_id_t control_id = create_region(4096);
    struct mpy_hostcall_v0 *control = map_region(control_id, 4096);
    if (control == NULL) {
        fprintf(stderr, "Failed to map resume control region\n");
        capstone_cleanup();
        return 3;
    }
    memset(control, 0, 4096);
    control->phase = MPY_TEST_START_MAGIC;
    control->offset = start;
    shared_region_annotated(dom_id, control_id, MPY_CONTROL_PERM_INOUT,
                            MPY_CONTROL_REV_SHARED);

    for (unsigned long i = 0; i < count; ++i) {
        unsigned long dom_retval = call_dom(dom_id);
        printf("Called dom (%lu-th time) retval = %lu\n", start + i + 1, dom_retval);
        fflush(stdout);
    }

    rc = capstone_cleanup();
    if (rc != 0) {
        fprintf(stderr, "Failed to clean up Capstone: %d\n", rc);
    }
    return rc;
}
