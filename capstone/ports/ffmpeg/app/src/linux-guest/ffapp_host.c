/* Guest-side host for the FFmpeg domain: shares the two HostCall v0 regions and services
 * the file and stdout opcodes. It is musl-capstone's stdio-probe host
 * (ports/musl-capstone/stdio-probe/stdio_probe_host.c) with a program in place of a probe:
 * every opcode goes through the shared switch in host_service.h, so file semantics are the
 * ones the musl probes were validated against.
 *
 * THE ROUND BOUND is raised from the probe's 64, not removed. FFmpeg reads the input in
 * <=4064-byte rounds (the payload region minus the header) and every stdout line is a
 * round, so the short workload needs a few hundred. A bound still turns a domain that
 * never reaches DONE into a reported failure instead of a harness timeout.
 *
 * The verdict is the program's own: capstone_main returns the milestone it stopped at
 * (ffapp_decode.h), and the host passes only if that equals the one this image was built
 * to reach. The MD5 lines themselves are compared by the runner, against the native
 * reference, outside the guest. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libcapstone.h"
#include "hostcall_stdout_probe.h"
#include "host_service.h"

#define FFAPP_MAX_ROUNDS 200000u
#define FFAPP_REGION_SIZE HOSTCALL_STDOUT_PROBE_REGION_SIZE

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s <ffapp.dom> <expected-stage> [verbose]\n", argv[0]);
        return 2;
    }
    long long expected = atoll(argv[2]);

    if (capstone_init()) {
        fprintf(stderr, "ffapp-host: capstone_init failed\n");
        return 1;
    }
    dom_id_t domain = create_dom(argv[1], NULL);
    if ((long)domain < 0) {
        fprintf(stderr, "ffapp-host: create_dom failed (%ld)\n", (long)domain);
        capstone_cleanup();
        return 1;
    }

    region_id_t metadata_region = create_region(FFAPP_REGION_SIZE);
    region_id_t payload_region = create_region(FFAPP_REGION_SIZE);
    struct hostcall_v0 *metadata =
        (struct hostcall_v0 *)map_region(metadata_region, FFAPP_REGION_SIZE);
    char *payload = (char *)map_region(payload_region, FFAPP_REGION_SIZE);
    if (!metadata || !payload) {
        fprintf(stderr, "ffapp-host: map_region failed\n");
        capstone_cleanup();
        return 1;
    }
    memset(metadata, 0, FFAPP_REGION_SIZE);
    memset(payload, 0, FFAPP_REGION_SIZE);

    shared_region_annotated(domain, metadata_region,
                            HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                            HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
    shared_region_annotated(domain, payload_region,
                            HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                            HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
#ifdef FFAPP_HEAP_REGION_BYTES
    /* The Sublet heap arm: a third region, TRANSFERRED, so it arrives LINEAR (mrev needs that,
       and REV_SHARED cannot give it) and the host keeps no authority over it. hostcall.c parks
       it for the heap (CAPSTONE_PROGRAM_REGIONS). Created zeroed (dma_alloc_pages). */
    region_id_t heap_region = create_region(FFAPP_HEAP_REGION_BYTES);
    if (heap_region == (region_id_t)-1) {
        fprintf(stderr, "ffapp-host: create_region(%lu) for the heap failed\n",
                (unsigned long)FFAPP_HEAP_REGION_BYTES);
        capstone_cleanup();
        return 1;
    }
    shared_region_annotated(domain, heap_region, HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
                            0x3UL /* REV_TRANSFERRED, as the SQLite and nginx arenas */);
#endif

    static struct hc_host host;
    host.tag = "ffapp";
    /* Quiet by default: a few hundred rounds, and the stage lines say enough. Verbose
       prints every request, which is what separates "slow" from "stuck" when a stage
       does not return: rounds that keep coming are progress, rounds that stop are not. */
    host.verbose = argc == 4;

    unsigned serviced = 0;
    for (unsigned round = 0; round < FFAPP_MAX_ROUNDS; ++round) {
        (void)call_dom(domain);

        /* Snapshot before acting: metadata stays INOUT+SHARED, so re-reading it
           mid-service is a TOCTOU (HostCall v0 design note). */
        struct hostcall_v0 request;
        hostcall_snapshot_request(&request, metadata);

        if (request.phase == HC_V0_PHASE_DONE) {
            printf("ffapp-host: DONE, serviced %u request(s), capstone_main = %lld, expected %lld\n",
                   serviced, (long long)request.result, expected);
            fflush(stdout);
            hostcall_cleanup_open_handles(host.slots, HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES);
            capstone_cleanup();
            if (request.result == expected) {
                printf("__CAPSTONE_FFAPP_STAGE_REACHED__ %lld\n", expected);
                fflush(stdout);
                return 0;
            }
            fprintf(stderr, "ffapp-host: FAILED at status %lld\n", (long long)request.result);
            return 1;
        }
        if (request.phase != HC_V0_PHASE_REQ) {
            fprintf(stderr, "ffapp-host: unexpected phase %llu\n",
                    (unsigned long long)request.phase);
            break;
        }
        if (!hostcall_payload_range_valid(&request)) {
            fprintf(stderr, "ffapp-host: request out of bounds\n");
            break;
        }
        if (hc_host_service(&host, &request, metadata, payload) < 0) {
            fprintf(stderr, "ffapp-host: unexpected opcode %llu\n",
                    (unsigned long long)request.opcode);
            hc_host_error(metadata, ENOSYS);
        }
        metadata->phase = HC_V0_PHASE_RESP;
        ++serviced;
    }

    fprintf(stderr, "ffapp-host: did not reach DONE (%u rounds serviced)\n", serviced);
    hostcall_cleanup_open_handles(host.slots, HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES);
    capstone_cleanup();
    return 1;
}
