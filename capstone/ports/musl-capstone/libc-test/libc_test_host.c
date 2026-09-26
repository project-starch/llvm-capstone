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

/* Where the rounds go: one counter set per opcode and one per opened path,
 * printed after the LT-RESULT line as LT-HIST and LT-FILE lines. The result
 * parsers stop at the end of the LT-RESULT line, so they see nothing new.
 *
 * `req` is the payload length the domain asked for, `moved` what the service
 * transferred (the READ, WRITE, STDOUT and DIR_READ results), `full` the rounds
 * whose request filled the whole region: those are the rounds a larger region
 * would fold into fewer. The first LT-HIST line checks itself against the
 * round count of the LT-RESULT line, so a histogram that lost rounds says so
 * (MISMATCH) instead of reading as a smaller total. */
#define LT_HIST_OPS 40
#define LT_HIST_FILES 32
#define LT_HIST_PATH 96
struct lt_op_hist { unsigned rounds, full; unsigned long long req, moved, max; };
struct lt_file_hist { /* one per distinct path, over all its opens */
  char path[LT_HIST_PATH];
  unsigned opens, rd_rounds, wr_rounds;
  unsigned long long rd_bytes, wr_bytes;
};
static struct lt_op_hist lt_ops[LT_HIST_OPS];
static unsigned lt_ops_other; /* rounds with an opcode past the table */
static struct lt_file_hist lt_files[LT_HIST_FILES];
static unsigned lt_files_n, lt_files_lost;
/* token -> index into lt_files while the handle is open, -1 otherwise */
static int lt_cur[HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES + 1];

static int lt_file_index(const char *path) {
  char key[LT_HIST_PATH];
  size_t n = strnlen(path, LT_HIST_PATH - 1); /* truncated on purpose */
  memcpy(key, path, n);
  key[n] = '\0';
  for (unsigned i = 0; i < lt_files_n; i++)
    if (strcmp(lt_files[i].path, key) == 0) return (int)i;
  if (lt_files_n == LT_HIST_FILES) { lt_files_lost++; return -1; }
  memcpy(lt_files[lt_files_n].path, key, n + 1);
  return (int)lt_files_n++;
}

static const char *lt_op_name(unsigned op) {
  switch (op) {
  case HC_V0_OP_NONE: return "NONE";
  case HC_V0_OP_WRITE_STDOUT: return "WRITE_STDOUT";
  case HC_V0_OP_WRITE_GUEST_TMPFILE: return "WRITE_GUEST_TMPFILE";
  case HC_V0_OP_READ_GUEST_TMPFILE: return "READ_GUEST_TMPFILE";
  case HC_V0_OP_FILE_OPEN: return "FILE_OPEN";
  case HC_V0_OP_FILE_READ: return "FILE_READ";
  case HC_V0_OP_FILE_WRITE: return "FILE_WRITE";
  case HC_V0_OP_FILE_CLOSE: return "FILE_CLOSE";
  case HC_V0_OP_FILE_STAT_BASIC: return "FILE_STAT_BASIC";
  case HC_V0_OP_FILE_SYNC: return "FILE_SYNC";
  case HC_V0_OP_FILE_TRUNCATE: return "FILE_TRUNCATE";
  case HC_V0_OP_PATH_ACCESS: return "PATH_ACCESS";
  case HC_V0_OP_PATH_DELETE: return "PATH_DELETE";
  case HC_V0_OP_CLOCK_GETTIME: return "CLOCK_GETTIME";
  case HC_V0_OP_DIR_READ: return "DIR_READ";
  case HC_V0_OP_PATH_RENAME: return "PATH_RENAME";
  case HC_V0_OP_PATH_MKDIR: return "PATH_MKDIR";
  case HC_V0_OP_PATH_READLINK: return "PATH_READLINK";
  case HC_V0_OP_PATH_SYMLINK: return "PATH_SYMLINK";
  case HC_V0_OP_PATH_STAT: return "PATH_STAT";
  case HC_V0_OP_PATH_CHMOD: return "PATH_CHMOD";
  case HC_V0_OP_FILE_FLOCK: return "FILE_FLOCK";
  default: return "?";
  }
}

/* The handle a request names, read BEFORE the service runs: a READ lands its
   data at the request's offset, which may be 0, on top of the request header. */
static hostcall_u64_t lt_request_handle(const struct hostcall_v0 *req, const char *payload) {
  switch (req->opcode) {
  case HC_V0_OP_FILE_READ: return ((const struct hc_file_read_req_v0 *)payload)->handle;
  case HC_V0_OP_FILE_WRITE: return ((const struct hc_file_write_req_v0 *)payload)->handle;
  case HC_V0_OP_FILE_CLOSE: return ((const struct hc_file_close_req_v0 *)payload)->handle;
  default: return 0;
  }
}

static void lt_hist_record(const struct hc_host *h, const struct hostcall_v0 *req,
                           const struct hostcall_v0 *resp, hostcall_u64_t handle) {
  unsigned op = (unsigned)req->opcode;
  if (op >= LT_HIST_OPS) { lt_ops_other++; return; }
  struct lt_op_hist *o = &lt_ops[op];
  o->rounds++;
  o->req += req->length;
  if (req->length > o->max) o->max = req->length;
  /* "Full" is a request that reaches the END of the region, not one of the
     region's size: a READ's data sits behind its 32-byte header, so its largest
     chunk is 4064 bytes and a test against 4096 could never fire (it did not,
     on the first three boots: full=0 with max=4064 on every row). */
  if (req->offset + req->length == LT_REGION) o->full++;
  unsigned long long moved = 0;
  if (resp->error == 0 && resp->result > 0 &&
      (op == HC_V0_OP_WRITE_STDOUT || op == HC_V0_OP_FILE_READ ||
       op == HC_V0_OP_FILE_WRITE || op == HC_V0_OP_DIR_READ))
    moved = (unsigned long long)resp->result;
  o->moved += moved;

  const hostcall_u64_t max_handle = HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES;
  if (op == HC_V0_OP_FILE_OPEN && resp->error == 0 && resp->result >= 1 &&
      (hostcall_u64_t)resp->result <= max_handle) {
    int i = lt_file_index(h->path);
    if (i >= 0) lt_files[i].opens++;
    lt_cur[resp->result] = i;
    return;
  }
  if (handle < 1 || handle > max_handle || lt_cur[handle] < 0) return;
  struct lt_file_hist *f = &lt_files[lt_cur[handle]];
  if (op == HC_V0_OP_FILE_READ) { f->rd_rounds++; f->rd_bytes += moved; }
  else if (op == HC_V0_OP_FILE_WRITE) { f->wr_rounds++; f->wr_bytes += moved; }
  else if (op == HC_V0_OP_FILE_CLOSE && resp->error == 0) lt_cur[handle] = -1;
}

static void lt_hist_report(const char *name, unsigned rounds) {
  unsigned long long total = lt_ops_other, req = 0, moved = 0;
  for (unsigned op = 0; op < LT_HIST_OPS; op++) {
    total += lt_ops[op].rounds; req += lt_ops[op].req; moved += lt_ops[op].moved;
  }
  printf("LT-HIST %s rounds=%llu %s req=%llu moved=%llu region=%lu\n", name, total,
         total == rounds ? "MATCH" : "MISMATCH", req, moved, (unsigned long)LT_REGION);
  for (unsigned op = 0; op < LT_HIST_OPS; op++) {
    const struct lt_op_hist *o = &lt_ops[op];
    if (!o->rounds) continue;
    printf("LT-HIST %s op=%u:%s rounds=%u full=%u req=%llu moved=%llu max=%llu\n", name, op,
           lt_op_name(op), o->rounds, o->full, o->req, o->moved, o->max);
  }
  if (lt_ops_other) printf("LT-HIST %s op=other rounds=%u\n", name, lt_ops_other);
  for (unsigned i = 0; i < lt_files_n; i++) {
    const struct lt_file_hist *f = &lt_files[i];
    printf("LT-FILE %s opens=%u rd=%u/%llu wr=%u/%llu path=%s\n", name, f->opens, f->rd_rounds,
           f->rd_bytes, f->wr_rounds, f->wr_bytes, f->path);
  }
  if (lt_files_lost) printf("LT-FILE %s lost=%u opens (table of %u paths full)\n", name, lt_files_lost, LT_HIST_FILES);
  fflush(stdout);
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
  for (unsigned i = 0; i <= HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES; i++) lt_cur[i] = -1;

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
      lt_hist_report(name, rounds);
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
      fflush(stdout); lt_hist_report(name, rounds); capstone_cleanup(); return 1;
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
    hostcall_u64_t handle = lt_request_handle(&req, payload);
    if (hc_host_service(&host, &req, metadata, payload) < 0)
      hc_host_error(metadata, ENOSYS);
    lt_hist_record(&host, &req, metadata, handle);
    metadata->phase = HC_V0_PHASE_RESP;
  }
  printf("LT-RESULT %s status=-1 rounds=%u FAIL NO-DONE\n", name, rounds);
  fflush(stdout);
  lt_hist_report(name, rounds);
  hostcall_cleanup_open_handles(host.slots, HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES);
  capstone_cleanup();
  return 1;
}
