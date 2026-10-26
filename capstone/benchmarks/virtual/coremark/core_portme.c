/* CoreMark-derived workload, virtual-profile port (see core_portme.h). One harness iteration is one
 * call of upstream main() (compiled as coremark_main): it allocates the data block, initialises the
 * three algorithms, runs ITERATIONS iterations between start_time() and stop_time() -- the harness
 * bracket, so only the iterations are timed, as upstream -- and reports. The verdict is read from
 * the report itself: the parameters must be the 2K performance run (TOTAL_DATA_SIZE 2000, 666 bytes per algorithm,
 * seeds 0, 0, 0x66: upstream known_id 3), and the list, matrix and state CRCs must match the
 * expected values for the list geometry (below). With 32-byte nodes they are NOT upstream's table:
 * a known deviation, identical in both arms; core_main.c's own "ERROR! ... crc" lines say so. "Must execute for at least 10 secs" is upstream's reporting rule for scores, which
 * this port never publishes, and does not enter the verdict. */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "coremark.h"
#include "bench-harness.h"

#if VALIDATION_RUN
volatile ee_s32 seed1_volatile = 0x3415;
volatile ee_s32 seed2_volatile = 0x3415;
volatile ee_s32 seed3_volatile = 0x66;
#endif
#if PERFORMANCE_RUN
volatile ee_s32 seed1_volatile = 0x0;
volatile ee_s32 seed2_volatile = 0x0;
volatile ee_s32 seed3_volatile = 0x66;
#endif
#if PROFILE_RUN
volatile ee_s32 seed1_volatile = 0x8;
volatile ee_s32 seed2_volatile = 0x8;
volatile ee_s32 seed3_volatile = 0x8;
#endif
volatile ee_s32 seed4_volatile = ITERATIONS;
volatile ee_s32 seed5_volatile = 0;
ee_u32 default_num_contexts = 1;

void start_time(void) { bench_begin(); }
void stop_time(void) { bench_end(); }
CORE_TICKS get_time(void) { return (CORE_TICKS)bench_window_ns(); }
secs_ret time_in_secs(CORE_TICKS ticks) { return (secs_ret)ticks / 1e9; }

void *portable_malloc(ee_size_t size) { return malloc(size); }
void portable_free(void *p) { free(p); }

/* What the report said, per iteration set. */
static struct {
  int perf_params, crc_lines, matrix_err, state_err, list_err, type_err;
  unsigned crclist, crcmatrix, crcstate, crcfinal;
} rep;

static void record(const char *line) {
  unsigned v;
  if (strstr(line, "2K performance run parameters"))
    rep.perf_params = 1;
  if (sscanf(line, "[0]crclist : 0x%x", &v) == 1) {
    rep.crclist = v;
    rep.crc_lines++;
  }
  if (sscanf(line, "[0]crcmatrix : 0x%x", &v) == 1) {
    rep.crcmatrix = v;
    rep.crc_lines++;
  }
  if (sscanf(line, "[0]crcstate : 0x%x", &v) == 1) {
    rep.crcstate = v;
    rep.crc_lines++;
  }
  if (sscanf(line, "[0]crcfinal : 0x%x", &v) == 1) {
    rep.crcfinal = v;
    rep.crc_lines++;
  }
  if (strstr(line, "ERROR! matrix crc"))
    rep.matrix_err = 1;
  if (strstr(line, "ERROR! state crc"))
    rep.state_err = 1;
  if (strstr(line, "ERROR! list crc"))
    rep.list_err = 1;
  if (strstr(line, "ERROR: ") || strstr(line, "ERROR! Please"))
    rep.type_err = 1;
}

/* Silent: the report is formatted and read here (the verdict), never written. On the board stdout is
 * a UART and every write a delegated call; upstream prints ~15 lines per main() call. The port's
 * only output is the harness's BENCH lines. -DCM_ECHO_REPORT prints the report (debugging only). */
int ee_printf(const char *fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  record(buf);
#ifdef CM_ECHO_REPORT
  fputs(buf, stdout);
#endif
  return n;
}

void portable_init(core_portable *p, int *argc, char *argv[]) {
  (void)argc;
  (void)argv;
  memset(&rep, 0, sizeof rep);
  p->portable_id = 1;
}

void portable_fini(core_portable *p) { p->portable_id = 0; }

int coremark_main(void);

static void unit(void) { coremark_main(); }

/* The expected first-iteration CRCs of the 2K performance run, by list-node size. All three depend
 * on the list: the matrix and state CRCs are first computed from list data (calc_func), so a list
 * with other nodes changes them too, not only the list CRC.
 *   16: upstream's table, core_main.c list/matrix/state_known_crc[3].
 *   32: the purecap geometry (16 list nodes, not 31). Computed with CM_LIST_NODE_BYTES=32 by
 *       x86-64 gcc -O2 and by the riscv64 buildroot gcc -O2 (QEMU), both LP64, so independent of
 *       this target's compiler and capability ABI. */
#if CM_LIST_NODE_BYTES == 16
#define CM_EXPECT_LIST 0xe714
#define CM_EXPECT_MATRIX 0x1fd7
#define CM_EXPECT_STATE 0x8e3a
#elif CM_LIST_NODE_BYTES == 32
#define CM_EXPECT_LIST 0x70a5
#define CM_EXPECT_MATRIX 0xdb4b
#define CM_EXPECT_STATE 0x905f
#else
#error "no expected CRCs for this CM_LIST_NODE_BYTES"
#endif
/* A node size smaller than the real list_head would overlap the list with the matrix block. */
#ifndef CM_REPRODUCE_OVERLAP /* build.py --coremark-unmodified reproduces the overlap on purpose */
_Static_assert(CM_LIST_NODE_BYTES >= sizeof(list_head), "CM_LIST_NODE_BYTES below sizeof(list_head)");
#endif

static int post(unsigned long long *check) {
  *check = ((unsigned long long)rep.crclist << 48) | ((unsigned long long)rep.crcmatrix << 32) |
           ((unsigned long long)rep.crcstate << 16) | rep.crcfinal;
  return rep.perf_params && rep.crc_lines == 4 && rep.crclist == CM_EXPECT_LIST &&
         rep.crcmatrix == CM_EXPECT_MATRIX && rep.crcstate == CM_EXPECT_STATE && !rep.type_err;
}

int main(int argc, char **argv) { return bench_run_self("coremark/coremark", argc, argv, unit, post); }
