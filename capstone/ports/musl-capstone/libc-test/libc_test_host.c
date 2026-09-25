/* Host for one libc-test domain: create it, share the two regions, service
 * whatever it asks, and report the test's status on one parseable line.
 *
 * No round-count oracle, and no content oracle either: the test IS the oracle.
 * libc-test's main() returns 0 only when every check passed, and the wrapper
 * folds the unserved-syscall count into the status, so metadata->result at DONE
 * is the whole verdict. This host's job is to get it out unaltered.
 *
 * The round bound is large and finite. A test that prints a thousand lines
 * makes a thousand rounds and must not be cut off; a test that never reaches
 * DONE must not run until the harness timeout, because that log is
 * indistinguishable from a stall. The guest-side `timeout` is the second net.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "libcapstone.h"
#include "host_service.h"

#define LT_MAX_ROUNDS 200000u
#define LT_REGION HOSTCALL_STDOUT_PROBE_REGION_SIZE

static const char *base(const char *p) {
  const char *s = strrchr(p, '/');
  return s ? s + 1 : p;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) { fprintf(stderr, "usage: %s <test.dom> [seconds]\n", argv[0]); return 2; }
  const char *name = base(argv[1]);
  /* The per-test timeout lives here, not in a guest `timeout` command: the
     buildroot image has no such applet, and the first batch reported rc=127 for
     every test because of it. alarm() needs nothing from the guest but the
     kernel; an unanswered SIGALRM ends this process with 128+14, which the
     summary reads as TIMEOUT. */
  if (argc == 3) alarm((unsigned)atoi(argv[2]));

  if (capstone_init()) { fprintf(stderr, "libc-test %s: capstone_init failed\n", name); return 3; }
  dom_id_t dom = create_dom(argv[1], NULL);
  if ((long)dom < 0) { fprintf(stderr, "libc-test %s: create_dom failed (%ld)\n", name, (long)dom); capstone_cleanup(); return 3; }

  region_id_t mr = create_region(LT_REGION), pr = create_region(LT_REGION);
  struct hostcall_v0 *metadata = (struct hostcall_v0 *)map_region(mr, LT_REGION);
  char *payload = (char *)map_region(pr, LT_REGION);
  if (!metadata || !payload) { fprintf(stderr, "libc-test %s: map_region failed\n", name); capstone_cleanup(); return 3; }
  memset(metadata, 0, LT_REGION); memset(payload, 0, LT_REGION);
  shared_region_annotated(dom, mr, HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT, HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
  shared_region_annotated(dom, pr, HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT, HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
#ifdef LT_HEAP_REGION_BYTES
  /* A heap for the domain's Sublet heap (runtime/sublet_heap.c), for a port that builds this host
     with it (the tshark port's sublet arm): a third region, TRANSFERRED, so it arrives LINEAR and
     this host keeps no authority over it. hostcall.c parks it (CAPSTONE_PROGRAM_REGIONS). As the
     FFmpeg app's host does it (ports/ffmpeg/app/src/linux-guest/ffapp_host.c). Without the define
     this host is unchanged. */
  region_id_t hr = create_region(LT_HEAP_REGION_BYTES);
  if (hr == (region_id_t)-1) {
    fprintf(stderr, "libc-test %s: create_region(%lu) for the heap failed\n", name, (unsigned long)LT_HEAP_REGION_BYTES);
    capstone_cleanup(); return 3;
  }
  shared_region_annotated(dom, hr, HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT, 0x3UL /* REV_TRANSFERRED */);
#endif

  static struct hc_host host;
  host.tag = name; host.verbose = 0;

  unsigned rounds = 0;
  for (; rounds < LT_MAX_ROUNDS; ++rounds) {
    (void)call_dom(dom);
    struct hostcall_v0 req;
    hostcall_snapshot_request(&req, metadata);

    if (req.phase == HC_V0_PHASE_DONE) {
      long long st = (long long)req.result;
      /* One line, fixed shape, parsed by run-libc-test.sh. UNSERVED is the bit
         the wrapper sets when a syscall the test needed had no opcode. */
      printf("LT-RESULT %s status=%lld rounds=%u %s%s\n", name, st, rounds,
             st == 0 ? "PASS" : "FAIL", (st & 0x100) ? " UNSERVED" : "");
      fflush(stdout);
      hostcall_cleanup_open_handles(host.slots, HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES);
      capstone_cleanup();
      return st == 0 ? 0 : 1;
    }
    /* call_dom returns the domain's retval and says nothing about a fault. A
       domain that halted never advances the phase: it is still the RESP this
       host wrote last round. That is the only host-side signature of a halt
       that returns from the ioctl at all; a halt that does not return is the
       guest loop's problem, not this process's. */
    if (rounds > 0 && req.phase == HC_V0_PHASE_RESP) {
      printf("LT-RESULT %s status=-1 rounds=%u FAIL HALTED\n", name, rounds);
      fflush(stdout); capstone_cleanup(); return 1;
    }
    if (req.phase == HC_V0_PHASE_ERROR) {
      printf("LT-RESULT %s status=-1 rounds=%u FAIL DOMAIN-ERROR\n", name, rounds);
      fflush(stdout); capstone_cleanup(); return 1;
    }
    if (req.phase != HC_V0_PHASE_REQ) {
      printf("LT-RESULT %s status=-1 rounds=%u FAIL PHASE=%llu\n", name, rounds, (unsigned long long)req.phase);
      fflush(stdout); capstone_cleanup(); return 1;
    }
    if (!hostcall_payload_range_valid(&req)) {
      printf("LT-RESULT %s status=-1 rounds=%u FAIL BAD-RANGE\n", name, rounds);
      fflush(stdout); capstone_cleanup(); return 1;
    }
    if (hc_host_service(&host, &req, metadata, payload) < 0)
      hc_host_error(metadata, ENOSYS);
    metadata->phase = HC_V0_PHASE_RESP;
  }
  printf("LT-RESULT %s status=-1 rounds=%u FAIL NO-DONE\n", name, rounds);
  fflush(stdout);
  hostcall_cleanup_open_handles(host.slots, HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES);
  capstone_cleanup();
  return 1;
}
