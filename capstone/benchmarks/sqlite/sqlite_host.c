#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "sqlite_hostcall.h"

/* MARKERS WITHOUT STDIO, AND SHORT ENOUGH TO ESCAPE.
 *
 * Two independent properties, both forced by the board capture (2026-07-30, three
 * SQLite attempts): after libcapstone's last line the console shows at most
 * "sqlite-host: cre" -- 16 bytes, the 8250 TX FIFO depth -- and then the *bootrom*
 * banner, i.e. the core RESETS. Only what is already sitting in the FIFO ever
 * transmits, so a marker longer than 16 bytes is a marker that cannot be read.
 *
 * 1. write(2), not fprintf(). Removes glibc's vfprintf engine (buffering mode
 *    selection, the unbuffered-stderr stack buffer, __printf_buffer's dispatch) from
 *    the one instrument that has to survive the failure. This is exactly what every
 *    freestanding ladder controller already does successfully on this board
 *    (rtl-smoke/ladder_perf_ctl.c `puts_`), so the marker path becomes identical to a
 *    proven-working one. No varargs, no locale, no buffering, one syscall.
 * 2. Every marker is <= 16 bytes, so one FIFO load carries a whole marker and the
 *    LAST marker on the console names the last phase reached. Values go on their own
 *    following line, so a lost value can never cost a lost phase.
 *
 * Phases: A dom-ok, B before create_region #1, C before #2, D both mapped,
 * E before share #1, F before share #2, G before call_dom, H after call_dom,
 * X the failure path. */
static void mark(const char *s) {
  unsigned long n = 0;
  while (s[n])
    n++;
  (void)write(STDOUT_FILENO, s, n);   /* fd 1: stderr resets the core */
}

/* Decimal, no varargs: hand-rolled so the value-carrying markers keep their meaning
   without reintroducing stdio. Single write, so it cannot tear between digits. */
static void mark_u(const char *prefix, unsigned long v) {
  char buf[24];
  int i = (int)sizeof(buf);
  if (!v)
    buf[--i] = '0';
  while (v) {
    buf[--i] = (char)('0' + (v % 10));
    v /= 10;
  }
  mark(prefix);
  (void)write(STDOUT_FILENO, buf + i, sizeof(buf) - (unsigned long)i);
  mark("\n");
}

static int fail_cleanup(const char *message, unsigned long value) {
  /* X FIRST, on purpose. "sqlite-host: create_dom failed (observed=...)",
     "sqlite-host: create_dom ok (id=...)" and "sqlite-host: create_region #1" all
     share their first 16 characters, so the previous markers could not tell a FAILED
     create_dom from a successful one -- and a failed create_dom means the ioctl
     returned an error, which is a completely different bug from a wedge. */
  mark("SQ: X/fail\n");
  mark_u("SQ: obs=", value);
  mark(message);
  mark("\n");
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    fprintf(stderr, "usage: %s <sqlite-domain.dom> [probe-stage]\n", argv[0]);
    return 2;
  }
  /* RUNTIME PROBE SELECTION (optional 2nd argument).
     Every probe used to be its OWN binary (-DCAPSTONE_SQLITE_STAGE=N), so each measurement
     drew a fresh ticket in the SHA5 stall lottery with a different image -- x101 lost that
     lottery 5 times running and its question is still unmeasured. Passing the stage at run
     time instead lets ONE domain image answer several questions, and lets the probe run in
     the image that is already known to enter. Omitted -> the domain keeps its built-in
     stage, so every existing invocation behaves exactly as before. */
  unsigned long probe_stage = 0;
  int have_probe_stage = 0;
  if (argc == 3) {
    probe_stage = strtoul(argv[2], NULL, 0);
    have_probe_stage = 1;
  }
  if (capstone_init()) {
    mark("SQ: no-init\n");
    return 1;
  }

  dom_id_t domain = create_dom(argv[1], NULL);
  if ((long)domain < 0)
    return fail_cleanup("create_dom failed", (unsigned long)domain);

  /* PHASE MARKERS. On silicon a monitor fault is C_PRINT + while(1), and C_PRINT
     goes to the RTL trace, not the UART -- so a wedge in create_dom and a wedge in
     the domain's entry glue look identical from the console: silence after
     libcapstone's last line. These two lines separate them, which is the whole
     difference between debugging the monitor and debugging the glue. */
  mark("SQ: A/dom-ok\n");
  mark_u("SQ: id=", (unsigned long)domain);

  /* FINE-GRAINED PHASE MARKERS. The board wedges somewhere between "create_dom ok"
     and "entering domain", and the monitor's region path has two silent `while(1)`s
     (split_out_cap: sbi_capstone.c:236, and :246 whose comment is "matching region.
     We don't support this for now"). SQLite is the first domain to create and share
     TWO regions, so no ladder rung covers this. One marker per call turns a 6-call
     gap into a single named culprit, which is worth one board run. */
  /* Attribute the faulting pc to an object. The first diagnosed run trapped ILLEGAL
     INSTRUCTION at (32-bit-truncated) 0xBDB30E3C during create_region -- neither the
     monitor (0x8001xxxx) nor the domain (0x10000). This host is a dynamic PIE, so print a
     known libc address and a known host-image address; subtracting identifies which
     mapping contains the fault, without parsing /proc/self/maps in a wedging process. */
  mark_u("SQ: libc=", (unsigned long)(void *)&printf);
  mark_u("SQ: self=", (unsigned long)(void *)&main);
  mark("SQ: B/mkregion1\n");
  region_id_t metadata_region = create_region(SQLITE_HC_REGION_SIZE);
  mark("SQ: C/mkregion2\n");
  region_id_t payload_region = create_region(SQLITE_HC_REGION_SIZE);
  mark("SQ: D/mapped\n");
  mark_u("SQ: r1=", (unsigned long)metadata_region);
  mark_u("SQ: r2=", (unsigned long)payload_region);
  struct sqlite_hostcall_v0 *metadata =
      (struct sqlite_hostcall_v0 *)map_region(metadata_region,
                                              SQLITE_HC_REGION_SIZE);
  char *payload = (char *)map_region(payload_region, SQLITE_HC_REGION_SIZE);
  if (!metadata || !payload)
    return fail_cleanup("map_region failed", 0);

  memset(metadata, 0, SQLITE_HC_REGION_SIZE);
  memset(payload, 0, SQLITE_HC_REGION_SIZE);
  /* Publish the probe selector AFTER the memset and BEFORE the domain runs. Magic-guarded so
     an unset region is indistinguishable from today's behaviour. */
  if (have_probe_stage)
    metadata->opcode = 0x5A6E0000UL | (probe_stage & 0xffUL);
  mark("SQ: E/share1\n");
  shared_region_annotated(domain, metadata_region,
                          SQLITE_HC_ANNOTATION_PERM_INOUT,
                          SQLITE_HC_ANNOTATION_REV_SHARED);
  mark("SQ: F/share2\n");
  shared_region_annotated(domain, payload_region,
                          SQLITE_HC_ANNOTATION_PERM_INOUT,
                          SQLITE_HC_ANNOTATION_REV_SHARED);

  mark("SQ: G/enter\n");
  unsigned long result = call_dom(domain);
  mark("SQ: H/return\n");
  if (metadata->length > 0 && metadata->length <= SQLITE_HC_REGION_SIZE) {
    (void)write(STDOUT_FILENO, payload, (size_t)metadata->length);
    fflush(stdout);
  }
  if (result != SQLITE_HC_RET_DONE)
    return fail_cleanup("unexpected domain return", result);

  if (capstone_cleanup()) {
    mark("SQ: no-cleanup\n");
    return 1;
  }
  return 0;
}
