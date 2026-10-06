/* Entry point and probes for client-repros. One program per case, as the
 * contract says: a capability fault ends the process, so a case that provokes
 * one cannot also report results beside it.
 */
#include "corpus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Noreturn void pgclient_give_up(unsigned long code) {
  fprintf(stderr, "CONTROL-FAILED %lx\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}

void *pgclient_malloc(size_t n) {
  void *p = malloc(n);
  if (!p) pgclient_give_up(1);
  return p;
}

void pgclient_free(void *p) { free(p); }
void pgclient_memcpy(void *d, const void *s, size_t n) { memcpy(d, s, n); }

void pgclient_mark(void) {
  printf("PG_DEFECT case=%u mark\n", pgclient_case_number);
  fflush(stdout);
}

void pgclient_note_signed(const char *label, long v) {
  printf("PG_NOTE %s=%ld\n", label, v);
  fflush(stdout);
}

void pgclient_note_overrun(const void *p, size_t legitimate, size_t asked) {
  printf("PG_NOTE overrun object=%p legitimate=%zu asked=%zu\n",
         p, legitimate, asked);
  fflush(stdout);
}

void pgclient_note_overread(const void *p, size_t legitimate, size_t asked) {
  printf("PG_NOTE overread object=%p legitimate=%zu asked=%zu\n",
         p, legitimate, asked);
  fflush(stdout);
}

char *pgclient_short_value(void) {
  /* What libpq hands ecpg for a zero-length text value: a 1-byte buffer
   * holding just the terminator. Allocated at exactly 1 byte so that the
   * `pval + 2` at data.c:532 is already past the end. */
  char *v = pgclient_malloc(1);
  v[0] = '\0';
  return v;
}

char *pgclient_oid_list(int n) {
  /* "1 2 3 ... n", the textual form parseOidArray is handed out of the
   * catalog. Sized exactly, so nothing downstream depends on slack. */
  size_t cap = (size_t) n * 12 + 1;
  char *s = pgclient_malloc(cap);
  size_t off = 0;
  for (int i = 0; i < n; i++)
    off += (size_t) snprintf(s + off, cap - off, i ? " %d" : "%d", i + 1);
  return s;
}

volatile Oid pgclient_oid_sink;
void pgclient_consume_oid(Oid v) { pgclient_oid_sink = v; }

int main(int argc, char **argv) {
  unsigned want;
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc != 2) {
    fprintf(stderr, "usage: %s <case-number>\n", argv[0]);
    return 75;
  }
  want = (unsigned) strtoul(argv[1], NULL, 10);
  if (want != pgclient_case_number) {
    fprintf(stderr, "CONTROL-FAILED this image is case %u, not %u\n",
            pgclient_case_number, want);
    return 75;
  }
  printf("case %u BEGIN\n", pgclient_case_number);
  pgclient_case_run();
  printf("case %u RETURNED\n", pgclient_case_number);
  return 0;
}
