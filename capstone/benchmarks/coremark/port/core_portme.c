#include "coremark.h"

#ifndef ITERATIONS
#define ITERATIONS 10
#endif

#ifndef COREMARK_DEFAULT_EXECS
#define COREMARK_DEFAULT_EXECS 0
#endif

#if VALIDATION_RUN
volatile ee_s32 seed1_volatile = 0x3415;
volatile ee_s32 seed2_volatile = 0x3415;
volatile ee_s32 seed3_volatile = 0x66;
#elif PERFORMANCE_RUN
volatile ee_s32 seed1_volatile = 0x0;
volatile ee_s32 seed2_volatile = 0x0;
volatile ee_s32 seed3_volatile = 0x66;
#else
volatile ee_s32 seed1_volatile = 0x8;
volatile ee_s32 seed2_volatile = 0x8;
volatile ee_s32 seed3_volatile = 0x8;
#endif

volatile ee_s32 seed4_volatile = ITERATIONS;
volatile ee_s32 seed5_volatile = COREMARK_DEFAULT_EXECS;

ee_u32 default_num_contexts = 1;

static CORE_TICKS g_start_time;
static CORE_TICKS g_stop_time;

void *portable_malloc(ee_size_t size) {
  (void)size;
  return 0;
}

void portable_free(void *p) {
  (void)p;
}

void start_time(void) {
  g_start_time = 0;
  g_stop_time = 0;
}

void stop_time(void) {
  g_stop_time = 10000;
}

CORE_TICKS get_time(void) {
  return g_stop_time - g_start_time;
}

secs_ret time_in_secs(CORE_TICKS ticks) {
  return (secs_ret)(ticks / 1000u);
}

void portable_init(core_portable *p, int *argc, char *argv[]) {
  (void)argc;
  (void)argv;
  if (sizeof(ee_ptr_int) != sizeof(void *))
    return;
  if (sizeof(ee_u32) != 4)
    return;
  p->portable_id = 1;
}

void portable_fini(core_portable *p) {
  p->portable_id = 0;
}

int ee_printf(const char *fmt, ...) {
  (void)fmt;
  return 0;
}

