#ifndef PG_DOMAIN_RUNTIME_H
#define PG_DOMAIN_RUNTIME_H
#include "replay-engine.h"
#include <stddef.h>

#define CAPSTONE_DPI_REGION_SHARE 1U

struct pg_hostcall_v0 {
  unsigned long long phase, opcode, offset, length;
  long long result, error;
};

void pg_domain_payload(char *base, unsigned long *length,
                       unsigned long capacity);
void pg_domain_text(const char *s);
void pg_domain_uint(unsigned long v);

/* The region sizes the host was told to make. Both halves must agree, so they
   are build parameters and the host publishes what it used. */
#ifndef PG_REPLAY_PAYLOAD_SIZE
#define PG_REPLAY_PAYLOAD_SIZE 65536UL
#endif
#ifndef PG_REPLAY_ARENA_SIZE
#define PG_REPLAY_ARENA_SIZE (32UL * 1024UL * 1024UL)
#endif
#ifndef PG_REPLAY_TRACE_SIZE
#define PG_REPLAY_TRACE_SIZE (64UL * 1024UL * 1024UL)
#endif

/* One state instance per domain executable. Entry handlers receive the shared
 * regions and initialize these fields before running the manager. */
extern volatile struct pg_hostcall_v0 *meta;
extern volatile char *payload;
extern unsigned shares;
extern char *scratch_next, *scratch_end;
extern unsigned *domain_result;
_Noreturn void give_up(unsigned code);
void fail(const char *message);
void row(const char *name, unsigned long expected, unsigned long actual);
void pg_domain_entry(unsigned *result, unsigned function);
#endif
