#ifndef PG_REPLAY_ENGINE_H
#define PG_REPLAY_ENGINE_H
#include "a11trace.h"
#include <stddef.h>

struct replay_counts {
  unsigned long create, alloc, free, realloc, reset, delete;
  unsigned long peak;    /* objects alive at once, most */
  unsigned long checked; /* objects whose contents were read back */
  /* what the backend's manager asked of the level below, from the trace */
  unsigned long was_alloc, was_free, was_realloc, was_peak;
  int have_was;
};

/* Platform callbacks allocate zeroed identity tables and report fatal errors.
 * The tables belong to the replay, outside the manager's measured heap. */
void *replay_alloc(size_t bytes);
_Noreturn void replay_die(const char *message);
_Noreturn void replay_die_at(unsigned long event, const char *message,
                             unsigned long identity);
void replay_run(struct a11_rec *records, unsigned long count,
                struct replay_counts *counts);
#endif
