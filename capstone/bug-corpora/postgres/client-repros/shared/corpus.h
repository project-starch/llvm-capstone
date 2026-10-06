/* What a case in client-repros needs, so a case.c is a complete translation
 * unit. Sibling of ../../mmgr-repros/shared/corpus.h, and deliberately not the
 * same header: that corpus drives PostgreSQL's memory-context allocators and
 * its seam is palloc, while every object here comes from libc. The whole point
 * of this corpus is that difference, so the two seams stay separate.
 *
 * shared/driver.c supplies the entry point and the probes. shared/upstream_*.c
 * supply the real 17.5 functions each case drives, copied verbatim from the pin
 * with the file and line recorded in each one's header comment.
 */
#ifndef PGCLIENT_CORPUS_H
#define PGCLIENT_CORPUS_H

#include <stddef.h>

/* A case declares the number its directory carries. The driver refuses a
 * fixture that names another case rather than silently running it. */
#define PGCLIENT_CASE(n)                                                       \
  const unsigned pgclient_case_number = (n);                                   \
  void pgclient_case_run(void)

extern const unsigned pgclient_case_number;
void pgclient_case_run(void);

/* Allocation. Plain libc, on purpose and without a wrapper that could round a
 * size up: a case's object must end exactly where it says it ends, or an
 * overread of one byte lands in padding and nothing sees it. */
void *pgclient_malloc(size_t n);
void  pgclient_free(void *p);
void  pgclient_memcpy(void *dst, const void *src, size_t n);

/* The marker that must be the LAST thing before the defective access. Its
 * presence in the output is the evidence that the setup ran, so a silent arm
 * can be told apart from a case that never reached the defect. */
void pgclient_mark(void);

/* Recorded facts, printed so a run carries its own arithmetic rather than
 * relying on the reader to trust the prose. */
void pgclient_note_signed(const char *label, long v);
void pgclient_note_overrun(const void *p, size_t legitimate, size_t asked);
void pgclient_note_overread(const void *p, size_t legitimate, size_t asked);

/* A value of fewer than two bytes, as a server returns for `select '' :: text`.
 * Allocated exactly, so pval + 2 is already outside it. */
char *pgclient_short_value(void);

_Noreturn void pgclient_give_up(unsigned long code);
#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      pgclient_give_up(n);                                                      \
  } while (0)

/* pg_dump's Oid, and the limit it wrongly assumes the backend enforces.
 * FUNC_MAX_ARGS is 100 at pg_config_manual.h:43 in the 17.5 tree. */
typedef unsigned int Oid;
#define InvalidOid ((Oid) 0)
#define FUNC_MAX_ARGS 100

/* A catalog value of exactly n oids, the shape pg_proc.protrftypes can take. */
char *pgclient_oid_list(int n);
/* A sink the oid walk feeds, so the loop cannot be optimised away. */
void  pgclient_consume_oid(Oid v);

/* --- the real 17.5 functions the cases drive; see shared/upstream_*.c --- */
unsigned ecpg_hex_decode(const char *src, unsigned len, char *dst);
int      pg_utf8_string_len(const char *source);
void     parseOidArray(const char *str, Oid *array, int arraysize);

/* libpq's connection, reduced to the fields pqGetnchar touches; see
 * shared/upstream_libpq.c for why it is reduced. */
struct pgclient_conn;
typedef struct pgclient_conn PGconn;
int      pqGetnchar(char *s, size_t len, PGconn *conn);
void     pgclient_fill_input(size_t nbytes);
PGconn  *pgclient_conn(void);

/* pg_basebackup's streamer, reduced to the fields the two buffering helpers
 * touch; see shared/upstream_bbstreamer.c. */
typedef struct { char *data; int len; int maxlen; } StringInfoData;
typedef struct bbstreamer { StringInfoData bbs_buffer; } bbstreamer;
typedef int bool_compat_unused_;
#ifndef __cplusplus
#include <stdbool.h>
#endif
void pgclient_si_init(StringInfoData *si);
void bbstreamer_buffer_bytes(bbstreamer *streamer, const char **data, int *len,
                             int nbytes);
bool bbstreamer_buffer_until(bbstreamer *streamer, const char **data, int *len,
                             int target_bytes);
void pgclient_consume_content(const char *data, int len);

#endif
