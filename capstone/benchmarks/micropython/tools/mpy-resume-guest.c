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
#define MPY_OUTPUT_SIZE 4096UL

struct mpy_hostcall_v0 {
    unsigned long long phase, opcode, offset, length;
    long long result, error;
};

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s DOMAIN START COUNT [--dump-output]\n", argv[0]);
        return 2;
    }
    int dump_output = argc == 5 && strcmp(argv[4], "--dump-output") == 0;
    if (argc == 5 && !dump_output) {
        fprintf(stderr, "unknown option: %s\n", argv[4]);
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

    unsigned char *output = NULL;
    if (dump_output) {
        region_id_t output_id = create_region(MPY_OUTPUT_SIZE);
        output = map_region(output_id, MPY_OUTPUT_SIZE);
        if (output == NULL) {
            fprintf(stderr, "Failed to map output capture region\n");
            capstone_cleanup();
            return 3;
        }
        memset(output, 0, MPY_OUTPUT_SIZE);
        shared_region_annotated(dom_id, output_id, MPY_CONTROL_PERM_INOUT,
                                MPY_CONTROL_REV_SHARED);
    }

    for (unsigned long i = 0; i < count; ++i) {
        unsigned long dom_retval = call_dom(dom_id);
        printf("Called dom (%lu-th time) retval = %lu\n", start + i + 1, dom_retval);
        if (dump_output) {
            static const char hex[] = "0123456789abcdef";
            size_t output_len = control->length;
            if (output_len > MPY_OUTPUT_SIZE) {
                output_len = MPY_OUTPUT_SIZE;
            }
            printf("MPYOUT %lu %zu ", start + i, output_len);
            for (size_t j = 0; j < output_len; ++j) {
                putchar(hex[output[j] >> 4]);
                putchar(hex[output[j] & 0xf]);
            }
            putchar('\n');
        }
        fflush(stdout);
    }

    rc = capstone_cleanup();
    if (rc != 0) {
        fprintf(stderr, "Failed to clean up Capstone: %d\n", rc);
    }
    return rc;
}
