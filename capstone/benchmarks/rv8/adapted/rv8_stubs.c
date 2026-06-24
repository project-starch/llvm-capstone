/*
 * No-op stubs for the hosted services rv8-bench programs reference but that are
 * irrelevant to correctness on the Capstone domain: timing (gettimeofday),
 * stdout reporting (printf), and process exit. Real computation/string/memory
 * helpers come from the trio/beebs freestanding libs and rv8_malloc.c.
 */
#include "rv8_capstone_preamble.h"

int gettimeofday(struct timeval *tv, void *tz) {
  (void)tz;
  if (tv) {
    tv->tv_sec = 0;
    tv->tv_usec = 0;
  }
  return 0;
}

int printf(const char *fmt, ...) {
  (void)fmt;
  return 0;
}

void exit(int code) {
  (void)code;
  for (;;) {
  }
}
