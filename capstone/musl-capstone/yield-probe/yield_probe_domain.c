/* Does a PURE-CAPABILITY domain survive a hostcall and resume where it left off?
 *
 * Everything in the musl port depends on the answer. A syscall must return to
 * the instruction after itself with the C stack intact; the shared entry glue
 * instead RESTARTS domain_main on every entry, and the existing HostCall v0
 * probes dodge the question by running an S-mode payload in a nested domain,
 * which is not pure-cap and so cannot carry a libc.
 *
 * THE CHECK IS BUILT TO DISTINGUISH RESUME FROM RESTART, because those two look
 * identical if you only test that "the domain ran again":
 *
 *   1. Round 2 sends a DIFFERENT message. A restart would re-send message 1 and
 *      message 2 would never appear.
 *   2. A local variable set before the yield is verified after it. It lives in
 *      the frame the hardware register-file swap is supposed to preserve, so a
 *      restart, or a lost stack, produces MARKER-LOST rather than a clean pass.
 *   3. A file-scope counter counts entries. It distinguishes "resumed" from
 *      "restarted but happened to reach the same place".
 *
 * A pass therefore requires all three of: message 1 once, message 2 once, and
 * the marker intact.
 */
#include "../../tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe.h"

/* shared_region_annotated() enters the domain with func == 1 and the region
   capability in the first argument; see the SQLite domain, which uses the same
   convention. */
#define CAPSTONE_DPI_REGION_SHARE 1

#define YIELD_PROBE_MSG1 "yield-probe: round 1 before yield\n"
#define YIELD_PROBE_MSG2 "yield-probe: round 2 AFTER RESUME, stack intact\n"
#define YIELD_PROBE_MSG_LOST "yield-probe: MARKER-LOST after resume\n"
#define YIELD_PROBE_MARKER 0x00C0FFEE5A5A1234UL

extern void __capstone_yield(void);

/* PADDING, NOT DEAD WEIGHT. The monitor's create_domain splits the code
 * capability at what looks like a fixed 0x1000 offset, so a loadable image
 * smaller than that makes the SPLIT degenerate and QEMU aborts in
 * helper_cssplit on `mid < bounds.end`. This probe's real image is ~1.2 KB,
 * squarely in that hole. The project already recorded the countermeasure --
 * "__pad keeps image > 0x1000 so the monitor SPLIT is non-degenerate" -- and
 * gp-free-domain/captable_app.c uses the same idiom.
 *
 * `retain` and the reference from domain_main are both needed: --gc-sections
 * would otherwise drop an unreferenced array and silently restore the failure
 * this exists to avoid. ISSUES.md has the open question of whether the fix
 * belongs in the monitor instead; until then, padding here is the cheap path.
 */
__attribute__((used, retain)) volatile const unsigned long __pad[512] = {1};

static volatile struct hostcall_v0 *metadata;
static volatile char *payload;
static unsigned shared_region_count;
static unsigned entry_count;

static unsigned long put_payload(const char *text) {
  unsigned long n = 0;
  while (text[n] && n + 1 < HOSTCALL_STDOUT_PROBE_REGION_SIZE) {
    payload[n] = text[n];
    n++;
  }
  return n;
}

/* Publish one WRITE_STDOUT request and hand control to the host. Returns
   nonzero if the host serviced exactly the bytes we asked for. */
static int hostcall_write_stdout(const char *text) {
  unsigned long n = put_payload(text);
  metadata->opcode = HC_V0_OP_WRITE_STDOUT;
  metadata->offset = 0;
  metadata->length = n;
  metadata->result = 0;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_REQ;

  __capstone_yield();

  return metadata->phase == HC_V0_PHASE_RESP && metadata->error == 0 &&
         metadata->result == (hostcall_s64_t)n;
}

void domain_main(unsigned *res, unsigned func) {
  unsigned long marker;

  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (shared_region_count == 0)
      metadata = (volatile struct hostcall_v0 *)res;
    else if (shared_region_count == 1)
      payload = (volatile char *)res;
    ++shared_region_count;
    return;
  }

  ++entry_count;
  if (!metadata || !payload)
    return;
  if (__pad[0] != 1) /* keeps the padding referenced; see __pad above */
    return;

  marker = YIELD_PROBE_MARKER;

  if (!hostcall_write_stdout(YIELD_PROBE_MSG1)) {
    metadata->phase = HC_V0_PHASE_ERROR;
    return;
  }

  /* Reached only by resuming inside hostcall_write_stdout and returning from
     it, i.e. only if the call frame survived the domain boundary. */
  if (marker != YIELD_PROBE_MARKER) {
    (void)hostcall_write_stdout(YIELD_PROBE_MSG_LOST);
    metadata->phase = HC_V0_PHASE_ERROR;
    return;
  }

  if (!hostcall_write_stdout(YIELD_PROBE_MSG2)) {
    metadata->phase = HC_V0_PHASE_ERROR;
    return;
  }

  metadata->phase = HC_V0_PHASE_DONE;
  metadata->opcode = HC_V0_OP_NONE;
  /* entry_count is reported so a restart cannot masquerade as a resume: a
     resumed run enters domain_main exactly once after the two shares. */
  metadata->result = (hostcall_s64_t)entry_count;
  if (res)
    *res = 0;
}
