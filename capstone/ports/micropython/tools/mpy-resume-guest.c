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
#define MPY_CONTROL_REV_TRANSFERRED 0x3UL
#define MPY_OUTPUT_SIZE 4096UL
/* The collector's heap, matching MPY_HEAP_SIZE in port/mpy_domain.c. They are not required to
   agree: the domain takes the region's own bounds, so a larger region is simply a larger heap and
   a smaller one is a smaller heap. */
#define MPY_POOL_SIZE (384UL * 1024UL)

struct mpy_hostcall_v0 {
    unsigned long long phase, opcode, offset, length;
    long long result, error;
};

int main(int argc, char **argv) {
    int dump_output = 0, pool_linear = 0;
    for (int i = 4; i < argc; ++i) {
        if (strcmp(argv[i], "--dump-output") == 0) dump_output = 1;
        else if (strcmp(argv[i], "--pool-linear") == 0) pool_linear = 1;
        else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
    }
    if (argc < 4) {
        fprintf(stderr, "usage: %s DOMAIN START COUNT [--dump-output] [--pool-linear]\n", argv[0]);
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

    /* SHARE ORDER IS THE CAPTURE ORDER IN THE DOMAIN: 0 control, 1 output, 2 pool. The output
       region is shared even when nobody asked to dump it, because otherwise the index of the pool
       would depend on a command-line flag, and a pool that lands where the domain expects output
       is the kind of mistake that reports as a capability fault three layers away. Four kilobytes
       is the price of that not being possible. */
    region_id_t output_id = create_region(MPY_OUTPUT_SIZE);
    unsigned char *output = map_region(output_id, MPY_OUTPUT_SIZE);
    if (output == NULL) {
        fprintf(stderr, "Failed to map output capture region\n");
        capstone_cleanup();
        return 3;
    }
    memset(output, 0, MPY_OUTPUT_SIZE);
    shared_region_annotated(dom_id, output_id, MPY_CONTROL_PERM_INOUT,
                            MPY_CONTROL_REV_SHARED);

    /* The collector's heap. REV_SHARED arrives NONLIN and is ordinary memory, which is what the
       arm without the discipline wants; --pool-linear hands it over REV_TRANSFERRED instead, the
       annotation SQLite's arena uses, so it arrives LIN and csmrev will accept it. One binary
       serves both arms and the flag says which. */
    region_id_t pool_id = create_region(MPY_POOL_SIZE);
    if (map_region(pool_id, MPY_POOL_SIZE) == NULL) {
        fprintf(stderr, "Failed to map the collector's pool region\n");
        capstone_cleanup();
        return 3;
    }
    shared_region_annotated(dom_id, pool_id, MPY_CONTROL_PERM_INOUT,
                            pool_linear ? MPY_CONTROL_REV_TRANSFERRED
                                        : MPY_CONTROL_REV_SHARED);

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
