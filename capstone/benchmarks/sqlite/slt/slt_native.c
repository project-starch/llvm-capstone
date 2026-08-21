/* Native harness for the SQLLogicTest runner -- THE BASELINE, and the reason the runner
 * is a header rather than domain code.
 *
 * This links the SAME slt_runner.h against the SAME SQLite 3.53.3 amalgamation the domain
 * uses, on the build machine. Its output is what the domain's output is compared against.
 * A record that fails here and in the domain is a corpus-versus-engine artifact and says
 * nothing about capabilities; a record that passes here and fails there is the only kind
 * of result this exercise exists to find.
 *
 * It is also where the runner is developed and negative-tested, because it runs in
 * milliseconds where a QEMU domain run takes minutes.
 *
 *   slt_native <file.test> [...]        -- one summary line per file, plus a TOTAL line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sqlite3.h"
#include "slt_runner.h"

static void out_stdout(void *ctx, const char *text) {
  (void)ctx;
  fputs(text, stdout);
}

static char *slurp(const char *path, unsigned long *len) {
  FILE *f = fopen(path, "rb");
  char *buf;
  long n;
  if (!f) return 0;
  fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
  buf = (char *)malloc((size_t)n + 1);
  if (!buf) { fclose(f); return 0; }
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return 0; }
  fclose(f);
  buf[n] = '\0';
  *len = (unsigned long)n;
  return buf;
}

int main(int argc, char **argv) {
  unsigned max_values = SLT_MAX_VALUES;
  int i, first = 1;
  slt_stats tot;
  const char *env = getenv("SLT_MAX_VALUES");
  if (env) max_values = (unsigned)strtoul(env, 0, 0);
  memset(&tot, 0, sizeof tot);
  tot.completed = 1;

  if (argc < 2) {
    fprintf(stderr, "usage: %s <file.test> [...]\n", argv[0]);
    return 2;
  }
  for (i = first; i < argc; i++) {
    unsigned long len = 0;
    char *buf = slurp(argv[i], &len);
    slt_stats st;
    if (!buf) { fprintf(stderr, "cannot read %s\n", argv[i]); return 2; }
    printf("SLT-FILE %s\n", argv[i]);
    slt_run(buf, len, out_stdout, 0, max_values, &st);
    slt_report(out_stdout, 0, &st);
    free(buf);
    tot.records += st.records;
    tot.stmt_pass += st.stmt_pass; tot.stmt_fail += st.stmt_fail;
    tot.query_pass += st.query_pass; tot.query_fail += st.query_fail;
    tot.skip_big += st.skip_big; tot.skip_cond += st.skip_cond;
    tot.oom += st.oom;
    tot.parse_err += st.parse_err;
    if (!st.completed) tot.completed = 0;
  }
  printf("SLT-TOTAL ");
  slt_report(out_stdout, 0, &tot);
  return (tot.stmt_fail || tot.query_fail || tot.parse_err) ? 1 : 0;
}
