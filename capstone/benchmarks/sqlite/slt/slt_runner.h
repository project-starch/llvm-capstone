#ifndef CAPSTONE_SLT_RUNNER_H
#define CAPSTONE_SLT_RUNNER_H

/* A SQLLogicTest runner that compiles UNCHANGED for the host and for a capability domain.
 *
 * THAT IS THE POINT OF THE FILE, not an incidental convenience. The number this project
 * needs is not "SQLite passes N% of SQLLogicTest" -- it is "SQLite in a pure-capability
 * domain agrees with the SAME SQLite built natively, over the same records". Only a
 * difference is attributable to the capability port; an absolute rate is contaminated by
 * corpus-versus-engine artifacts that have nothing to do with us. Measured, before this
 * file existed: evidence/slt_lang_aggfunc.test has records that stock SQLite fails on any
 * machine, because the corpus's expected float rendering disagrees with C's at 1e18. If
 * the runner differed between the two sides, that artifact would be indistinguishable
 * from a capability defect.
 *
 * So: no capability-specific code here, and no libc beyond memcpy/strlen/strcmp.
 * Formatting goes through sqlite3_snprintf (the amalgamation implements its own printf,
 * so no host snprintf is needed); allocation goes through sqlite3_malloc, which on the
 * domain side comes out of the memsys5 arena that already exists.
 *
 * WHAT IT DELIBERATELY DOES NOT DO:
 *   - it never reports a record it could not evaluate as a pass. Records skipped for size
 *     or by skipif/onlyif are counted in their own buckets and printed separately.
 *   - it never renders "nothing happened" as a clean result: the summary carries the
 *     record count, and `completed` says whether the input was consumed to the end.
 *
 * The rendering, sort and hash rules were calibrated against the corpus BEFORE this was
 * written -- a Python model of these exact rules reproduces all 12,516 records of
 * select1-5 with stock SQLite. Mutating any one rule (NULL text, rowsort, valuesort,
 * float precision) makes thousands of records fail, so the rules are pinned by evidence
 * rather than by memory of the spec.
 */

#include "slt_md5.h"

#ifndef SLT_ENGINE_NAME
#define SLT_ENGINE_NAME "sqlite"
#endif

/* Result sets larger than this are reported as SKIPPED-BIG rather than compared.
 * WHY A CAP EXISTS AT ALL: the silicon domain's whole SQLite heap is 256 KiB
 * (build-sqlite-silicon.sh:44), and the corpus contains a record with 94,080 values.
 * Measured over the subset: at 4096 values the cap costs 184 of 7,393 hash records
 * (2.5%), and every one of them is rowsort -- no nosort or valuesort record in the
 * subset exceeds it. The native baseline uses the SAME cap by default so that the two
 * sides compare like for like; raise it there to measure what the cap hides. */
#ifndef SLT_MAX_VALUES
#define SLT_MAX_VALUES 4096u
#endif
#ifndef SLT_MAX_VALUE_BYTES
#define SLT_MAX_VALUE_BYTES (192u * 1024u)
#endif
/* Only the first few failures are described; the counts are always complete. A domain
 * that fails every record would otherwise fill the output region with detail and lose
 * the summary line, which is the one thing that must survive. */
#ifndef SLT_MAX_REPORTED
#define SLT_MAX_REPORTED 8u
#endif

/* PER-QUERY INSTRUMENT HOOKS. No-ops by default, so the native baseline and the ordinary
 * domain build are unaffected and this header stays capability-agnostic.
 *
 * They exist because arming an instrument ONCE for the whole file measures the wrong thing:
 * SQLite's opcode counter is cumulative across every sqlite3VdbeExec, so a clamp armed at
 * entry stops CREATE TABLE rather than the query under test. Measured, not theorised -- a
 * clamp of 20 armed at entry reported "no such table: t1" for every statement. Arming and
 * resetting per query is what makes a clamp ladder point at the query it claims to. */
#ifndef SLT_VDBE_ARM
#define SLT_VDBE_ARM()    do { } while (0)
#endif
#ifndef SLT_VDBE_DISARM
#define SLT_VDBE_DISARM() do { } while (0)
#endif

typedef void (*slt_out_fn)(void *ctx, const char *text);

typedef struct {
  unsigned records;        /* records dispatched (statement + query)          */
  unsigned stmt_pass, stmt_fail;
  unsigned query_pass, query_fail;
  unsigned skip_big;       /* result set exceeded the cap -- NOT a pass       */
  unsigned oom;            /* SQLITE_NOMEM -- a RESOURCE limit, NOT a mismatch */
  unsigned skip_cond;      /* skipif/onlyif excluded it -- NOT a pass         */
  unsigned parse_err;      /* unrecognised record -- NOT a pass               */
  unsigned reported;
  int completed;           /* 1 = the whole input was consumed                */
  int open_failed;
} slt_stats;

/* ---------------------------------------------------------------- value accumulator */
/* A flat text buffer plus 32-bit offsets, NOT an array of pointers. On this target a
 * pointer is a 128-bit capability, so a pointer array costs 16 bytes per value against a
 * 256 KiB heap; offsets cost 4. At the 4096-value cap that is 16 KiB instead of 64 KiB,
 * and it also collapses thousands of small allocations into two growable blocks. */
typedef struct {
  char *txt; unsigned tlen, tcap;
  unsigned *off; unsigned n, ocap;
  int overflow;
} slt_vals;

static void slt_vals_init(slt_vals *v) {
  v->txt = 0; v->tlen = 0; v->tcap = 0;
  v->off = 0; v->n = 0; v->ocap = 0; v->overflow = 0;
}
static void slt_vals_free(slt_vals *v) {
  if (v->txt) sqlite3_free(v->txt);
  if (v->off) sqlite3_free(v->off);
  slt_vals_init(v);
}
/* Appends one NUL-terminated rendered value. Sets `overflow` and stops on any cap or
   allocation failure, so the caller reports SKIPPED rather than a short comparison. */
static void slt_vals_push(slt_vals *v, const char *s, unsigned max_values) {
  unsigned len = 0;
  while (s[len]) len++;
  if (v->overflow) return;
  if (v->n >= max_values || v->tlen + len + 1u > SLT_MAX_VALUE_BYTES) {
    v->overflow = 1; return;
  }
  if (v->n == v->ocap) {
    unsigned nc = v->ocap ? v->ocap * 2u : 64u;
    unsigned *no = (unsigned *)sqlite3_realloc64(v->off, (sqlite3_uint64)nc * sizeof(unsigned));
    if (!no) { v->overflow = 1; return; }
    v->off = no; v->ocap = nc;
  }
  if (v->tlen + len + 1u > v->tcap) {
    unsigned nc = v->tcap ? v->tcap : 1024u;
    char *nt;
    while (nc < v->tlen + len + 1u) nc *= 2u;
    nt = (char *)sqlite3_realloc64(v->txt, (sqlite3_uint64)nc);
    if (!nt) { v->overflow = 1; return; }
    v->txt = nt; v->tcap = nc;
  }
  v->off[v->n++] = v->tlen;
  { unsigned k; for (k = 0; k <= len; k++) v->txt[v->tlen + k] = s[k]; }
  v->tlen += len + 1u;
}

/* ------------------------------------------------------------------------- sorting */
/* Heapsort: in place, O(n log n), and it needs NO scratch array -- which matters on a
   256 KiB heap that is already holding the values themselves. */
typedef int (*slt_cmp_fn)(const slt_vals *v, unsigned ncol, unsigned a, unsigned b);

static int slt_cmp_value(const slt_vals *v, unsigned ncol, unsigned a, unsigned b) {
  (void)ncol;
  return strcmp(v->txt + v->off[a], v->txt + v->off[b]);
}
/* Rows compare column by column, which is what makes rowsort stable across engines: two
   rows differing only in a later column must still order deterministically. */
static int slt_cmp_row(const slt_vals *v, unsigned ncol, unsigned a, unsigned b) {
  unsigned k;
  for (k = 0; k < ncol; k++) {
    int c = strcmp(v->txt + v->off[a * ncol + k], v->txt + v->off[b * ncol + k]);
    if (c) return c;
  }
  return 0;
}

static void slt_sift(unsigned *idx, unsigned n, unsigned root,
                     const slt_vals *v, unsigned ncol, slt_cmp_fn cmp) {
  for (;;) {
    unsigned big = root, l = 2u * root + 1u, r = l + 1u, t;
    if (l < n && cmp(v, ncol, idx[l], idx[big]) > 0) big = l;
    if (r < n && cmp(v, ncol, idx[r], idx[big]) > 0) big = r;
    if (big == root) return;
    t = idx[root]; idx[root] = idx[big]; idx[big] = t;
    root = big;
  }
}
static void slt_heapsort(unsigned *idx, unsigned n,
                         const slt_vals *v, unsigned ncol, slt_cmp_fn cmp) {
  unsigned i;
  if (n < 2u) return;
  for (i = n / 2u; i-- > 0;) slt_sift(idx, n, i, v, ncol, cmp);
  for (i = n; i-- > 1;) {
    unsigned t = idx[0]; idx[0] = idx[i]; idx[i] = t;
    slt_sift(idx, i, 0, v, ncol, cmp);
  }
}

/* ----------------------------------------------------------------------- rendering */
/* THE FOUR RULES BELOW ARE THE COMPARATOR. Each is pinned by a mutation experiment
 * against the corpus (see the header comment): changing NULL to anything else fails
 * 2,857 records, dropping rowsort fails 4,525, dropping valuesort fails 1,505. Float
 * precision and the empty-string rendering are NOT exercised by select1-5 -- only by
 * evidence/slt_lang_aggfunc.test, which is in the subset for exactly that reason. */
static void slt_render(char *buf, int bufsz, sqlite3_stmt *st, int col, char type) {
  if (sqlite3_column_type(st, col) == SQLITE_NULL) {
    sqlite3_snprintf(bufsz, buf, "NULL");
    return;
  }
  if (type == 'I') {
    sqlite3_snprintf(bufsz, buf, "%lld", (sqlite3_int64)sqlite3_column_int64(st, col));
  } else if (type == 'R') {
    sqlite3_snprintf(bufsz, buf, "%.3f", sqlite3_column_double(st, col));
  } else {
    const unsigned char *z = sqlite3_column_text(st, col);
    int k = 0;
    if (!z || !z[0]) { sqlite3_snprintf(bufsz, buf, "(empty)"); return; }
    while (z[k] && k < bufsz - 1) {
      unsigned char c = z[k];
      buf[k] = (c < 0x20u || c > 0x7eu) ? '@' : (char)c;   /* non-printables -> '@' */
      k++;
    }
    buf[k] = '\0';
  }
}

/* -------------------------------------------------------------------- line scanning */
typedef struct { const char *p; unsigned long n, i; } slt_scan;

/* Returns 1 and fills (*s,*len) with the next line, \r stripped; 0 at end of input. */
static int slt_line(slt_scan *sc, const char **s, unsigned *len) {
  unsigned long start;
  if (sc->i >= sc->n) return 0;
  start = sc->i;
  while (sc->i < sc->n && sc->p[sc->i] != '\n') sc->i++;
  *s = sc->p + start;
  *len = (unsigned)(sc->i - start);
  if (*len && (*s)[*len - 1] == '\r') (*len)--;
  if (sc->i < sc->n) sc->i++;                     /* step over the newline */
  return 1;
}
static int slt_blank(const char *s, unsigned len) {
  unsigned i;
  for (i = 0; i < len; i++)
    if (s[i] != ' ' && s[i] != '\t') return 0;
  return 1;
}
/* Copies whitespace-separated field `k` of a line into buf. Returns 0 if absent. */
static int slt_field(const char *s, unsigned len, unsigned k, char *buf, unsigned bufsz) {
  unsigned i = 0, f = 0;
  for (;;) {
    while (i < len && (s[i] == ' ' || s[i] == '\t')) i++;
    if (i >= len) return 0;
    { unsigned st = i;
      while (i < len && s[i] != ' ' && s[i] != '\t') i++;
      if (f == k) {
        unsigned m = i - st, j;
        /* TRUNCATION IS AN ERROR, NOT A SHORTER FIELD. select5.test declares queries with
           a 64-character type string, and a 64-byte buffer silently kept 63 of them --
           so ncol was one too small, the last column of every row was never read, and
           twelve records failed with "nvalue=63 want=64". A wrong answer that looks like
           a SQLite defect is the worst possible failure mode for this runner, so a field
           that does not fit is refused and counted as a parse error. */
        if (m > bufsz - 1u) return 0;
        for (j = 0; j < m; j++) buf[j] = s[st + j];
        buf[m] = '\0';
        return 1;
      }
      f++; }
  }
}

static void slt_emit_u(slt_out_fn out, void *ctx, const char *prefix, unsigned v) {
  char b[32];
  int i = (int)sizeof(b);
  b[--i] = '\0';
  if (!v) b[--i] = '0';
  while (v) { b[--i] = (char)('0' + v % 10u); v /= 10u; }
  out(ctx, prefix);
  out(ctx, b + i);
}

/* ------------------------------------------------------------------------ the runner */
/* Runs one SQLLogicTest file held in `input`. Creates and destroys its own in-memory
 * connection, so one call is one self-contained file -- which is what makes the
 * one-call_dom-per-file design work: SLT files create their own tables and share no
 * state, and domain re-entry would rebuild the cap table and destroy an in-memory db
 * anyway (start-gp-captable-generic.S:30).
 *
 * Returns the number of failing records; 0 means every record that could be evaluated
 * agreed with the corpus. The caller must still read `completed` and the skip counts:
 * zero failures with zero records is not a pass. */
static unsigned slt_run(const char *input, unsigned long input_len,
                        slt_out_fn out, void *ctx,
                        unsigned max_values, slt_stats *st) {
  sqlite3 *db = 0;
  slt_scan sc;
  const char *line; unsigned len;
  int skip_next = 0;
  unsigned lineno = 0;
  char tok[64];

  st->records = st->stmt_pass = st->stmt_fail = 0;
  st->query_pass = st->query_fail = 0;
  st->skip_big = st->skip_cond = st->parse_err = st->reported = 0;
  st->oom = 0;
  st->completed = 0; st->open_failed = 0;

  if (sqlite3_open(":memory:", &db) != SQLITE_OK) {
    st->open_failed = 1;
    if (db) sqlite3_close(db);
    return 1;
  }

  sc.p = input; sc.n = input_len; sc.i = 0;
  while (slt_line(&sc, &line, &len)) {
    lineno++;
    if (slt_blank(line, len) || line[0] == '#') continue;
    if (!slt_field(line, len, 0, tok, sizeof tok)) continue;

    if (!strcmp(tok, "hash-threshold")) continue;   /* generation-side only */
    /* `halt` IS A RECORD, so a pending skipif/onlyif applies to it. Getting this wrong is
       not a corner case: evidence/slt_lang_aggfunc.test opens with
           skipif sqlite
           halt
       which means "every engine except ours stops here". Honouring the halt before
       consuming the flag stopped the file after 5 of its 80 records and still reported
       completed=1 -- a clean summary from a file that never ran. Found only because the
       native baseline was compared against an independent model of the same corpus. */
    if (!strcmp(tok, "halt")) {
      if (skip_next) { skip_next = 0; st->skip_cond++; continue; }
      break;
    }
    if (!strcmp(tok, "skipif") || !strcmp(tok, "onlyif")) {
      char engine[64];
      int is_skip = (tok[0] == 's');
      if (!slt_field(line, len, 1, engine, sizeof engine)) continue;
      if (is_skip ? !strcmp(engine, SLT_ENGINE_NAME) : strcmp(engine, SLT_ENGINE_NAME))
        skip_next = 1;
      continue;
    }

    if (!strcmp(tok, "statement")) {
      char want[16];
      const char *sql; unsigned long sql_start, sql_end;
      int want_ok, rc, got_ok = 1, last_rc = SQLITE_OK;
      sqlite3_stmt *s2 = 0;
      const char *tail;
      if (!slt_field(line, len, 1, want, sizeof want)) { st->parse_err++; continue; }
      want_ok = !strcmp(want, "ok");
      sql_start = sc.i;
      sql_end = sql_start;
      while (slt_line(&sc, &line, &len)) {
        lineno++;
        if (slt_blank(line, len)) break;
        sql_end = sc.i;
      }
      if (skip_next) { skip_next = 0; st->skip_cond++; continue; }
      st->records++;
      sql = input + sql_start;
      /* prepare_v2 takes a byte count, so the SQL is used IN PLACE -- the shared region
         is never modified and never copied onto the heap. */
      { int nbyte = (int)(sql_end - sql_start);
        tail = sql;
        while (nbyte > 0 && got_ok) {
          rc = sqlite3_prepare_v2(db, tail, nbyte, &s2, &tail);
          if (rc != SQLITE_OK) { last_rc = rc; got_ok = 0; break; }
          if (!s2) break;                       /* trailing whitespace/comment only */
          do { rc = sqlite3_step(s2); } while (rc == SQLITE_ROW);
          if (rc != SQLITE_DONE) { last_rc = rc; got_ok = 0; }
          sqlite3_finalize(s2); s2 = 0;
          nbyte = (int)(sql_end - (unsigned long)(tail - input));
        }
      }
      /* SQLITE_NOMEM IS A RESOURCE LIMIT, NOT A WRONG ANSWER, and scoring it as a failure
         would be the single most misleading thing this runner could do. The domain's whole
         SQLite arena is 256 KiB (build-sqlite-silicon.sh:44) while the native baseline has
         the machine's; select4 exhausts the former around a third of the way in. Counting
         those as mismatches produced "query_fail=2759" for a build that had not computed a
         single wrong value. A record that ran out of memory was not evaluated, and goes in
         its own bucket alongside skip_big -- never into a pass bucket either. */
      if ((last_rc & 0xff) == SQLITE_NOMEM) { st->oom++; continue; }
      if (got_ok == want_ok) st->stmt_pass++;
      else {
        st->stmt_fail++;
        if (st->reported < SLT_MAX_REPORTED) {
          st->reported++;
          slt_emit_u(out, ctx, "SLT-FAIL stmt line=", lineno);
          out(ctx, want_ok ? " wanted=ok got=error: " : " wanted=error got=ok: ");
          out(ctx, sqlite3_errmsg(db));
          out(ctx, "\n");
        }
      }
      continue;
    }

    if (!strcmp(tok, "query")) {
      char types[256], mode[32];   /* corpus maximum is 64 columns; measured, all 622 files */
      unsigned long sql_start, sql_end;
      unsigned ncol, i, want_n = 0;
      int rc, is_hash = 0, failed = 0, saw_sep = 0;
      sqlite3_stmt *s2 = 0;
      slt_vals v;
      unsigned exp_first = 0, exp_count = 0;
      const char *exp_lines[SLT_MAX_REPORTED];  /* only used for small value-form sets */
      unsigned exp_lens[SLT_MAX_REPORTED];
      char want_hash[40];
      unsigned long exp_start;

      if (!slt_field(line, len, 1, types, sizeof types)) { st->parse_err++; continue; }
      if (!slt_field(line, len, 2, mode, sizeof mode)) sqlite3_snprintf(sizeof mode, mode, "nosort");
      ncol = (unsigned)strlen(types);

      sql_start = sc.i; sql_end = sql_start;
      while (slt_line(&sc, &line, &len)) {
        lineno++;
        if (len == 4 && line[0] == '-' && line[1] == '-' && line[2] == '-' && line[3] == '-') {
          saw_sep = 1; break;
        }
        if (slt_blank(line, len)) break;
        sql_end = sc.i;
      }
      exp_start = sc.i;
      /* The expected block is scanned first so that a skipped or failed record still
         leaves the cursor at the next record rather than mid-file. */
      while (slt_line(&sc, &line, &len)) {
        lineno++;
        if (slt_blank(line, len)) break;
        if (exp_count == 0) {
          char c0[64];
          if (slt_field(line, len, 1, c0, sizeof c0) && !strcmp(c0, "values") &&
              slt_field(line, len, 0, c0, sizeof c0)) {
            unsigned k = 0; want_n = 0;
            while (c0[k] >= '0' && c0[k] <= '9') want_n = want_n * 10u + (unsigned)(c0[k++] - '0');
            if (slt_field(line, len, 4, want_hash, sizeof want_hash)) is_hash = 1;
          }
        }
        if (exp_count < SLT_MAX_REPORTED) { exp_lines[exp_count] = line; exp_lens[exp_count] = len; }
        exp_count++;
      }
      (void)exp_first; (void)exp_start;

      if (skip_next) { skip_next = 0; st->skip_cond++; continue; }
      st->records++;
      if (!saw_sep) {
        st->parse_err++;
        continue;
      }

      rc = sqlite3_prepare_v2(db, input + sql_start, (int)(sql_end - sql_start), &s2, 0);
      if ((rc & 0xff) == SQLITE_NOMEM) {
        st->oom++;
        if (s2) sqlite3_finalize(s2);
        continue;
      }
      if (rc != SQLITE_OK || !s2) {
        st->query_fail++;
        if (st->reported < SLT_MAX_REPORTED) {
          st->reported++;
          slt_emit_u(out, ctx, "SLT-FAIL query line=", lineno);
          out(ctx, " prepare: "); out(ctx, sqlite3_errmsg(db)); out(ctx, "\n");
        }
        if (s2) sqlite3_finalize(s2);
        continue;
      }

      /* CROSS-CHECK the declared column count against the engine's. This is the second
         detector for the truncation bug above, and it is kept because the two fail
         independently: a mis-sized buffer trips this, and a corpus record whose type
         string genuinely disagrees with its SQL trips it too. Either way the record must
         not be scored. */
      if (sqlite3_column_count(s2) != (int)ncol) {
        st->query_fail++;
        if (st->reported < SLT_MAX_REPORTED) {
          st->reported++;
          slt_emit_u(out, ctx, "SLT-FAIL query line=", lineno);
          slt_emit_u(out, ctx, " ncol_declared=", ncol);
          slt_emit_u(out, ctx, " ncol_engine=", (unsigned)sqlite3_column_count(s2));
          out(ctx, "\n");
        }
        sqlite3_finalize(s2);
        continue;
      }

      slt_vals_init(&v);
      SLT_VDBE_ARM();          /* this query only; see the hook definition above */
      while ((rc = sqlite3_step(s2)) == SQLITE_ROW && !v.overflow) {
        char cell[256];
        for (i = 0; i < ncol; i++) {
          slt_render(cell, (int)sizeof cell, s2, (int)i, types[i]);
          slt_vals_push(&v, cell, max_values);
        }
      }
      SLT_VDBE_DISARM();
      sqlite3_finalize(s2);

      if (v.overflow) {                    /* NOT a pass, and counted on its own */
        st->skip_big++;
        slt_vals_free(&v);
        continue;
      }
      if ((rc & 0xff) == SQLITE_NOMEM) { st->oom++; slt_vals_free(&v); continue; }
      if (rc != SQLITE_DONE) {
        st->query_fail++;
        if (st->reported < SLT_MAX_REPORTED) {
          st->reported++;
          slt_emit_u(out, ctx, "SLT-FAIL query line=", lineno);
          out(ctx, " step: "); out(ctx, sqlite3_errmsg(db)); out(ctx, "\n");
        }
        slt_vals_free(&v);
        continue;
      }

      /* Sort, per the record's declared mode. */
      if (!strcmp(mode, "rowsort") && ncol && v.n >= ncol) {
        unsigned nrow = v.n / ncol, k;
        unsigned *idx = (unsigned *)sqlite3_malloc64((sqlite3_uint64)nrow * sizeof(unsigned));
        if (!idx) { st->skip_big++; slt_vals_free(&v); continue; }
        for (k = 0; k < nrow; k++) idx[k] = k;
        slt_heapsort(idx, nrow, &v, ncol, slt_cmp_row);
        /* Rewrite the offset array into sorted order, in place via a second pass. */
        { unsigned *no = (unsigned *)sqlite3_malloc64((sqlite3_uint64)v.n * sizeof(unsigned));
          if (!no) { sqlite3_free(idx); st->skip_big++; slt_vals_free(&v); continue; }
          for (k = 0; k < nrow; k++) {
            unsigned c;
            for (c = 0; c < ncol; c++) no[k * ncol + c] = v.off[idx[k] * ncol + c];
          }
          sqlite3_free(v.off); v.off = no; }
        sqlite3_free(idx);
      } else if (!strcmp(mode, "valuesort")) {
        unsigned k;
        unsigned *idx = (unsigned *)sqlite3_malloc64((sqlite3_uint64)v.n * sizeof(unsigned));
        if (!idx) { st->skip_big++; slt_vals_free(&v); continue; }
        for (k = 0; k < v.n; k++) idx[k] = k;
        slt_heapsort(idx, v.n, &v, 1u, slt_cmp_value);
        { unsigned *no = (unsigned *)sqlite3_malloc64((sqlite3_uint64)(v.n ? v.n : 1u) * sizeof(unsigned));
          if (!no) { sqlite3_free(idx); st->skip_big++; slt_vals_free(&v); continue; }
          for (k = 0; k < v.n; k++) no[k] = v.off[idx[k]];
          sqlite3_free(v.off); v.off = no; }
        sqlite3_free(idx);
      }

      if (is_hash) {
        struct slt_md5 m;
        char got[33];
        slt_md5_init(&m);
        for (i = 0; i < v.n; i++) {
          const char *s = v.txt + v.off[i];
          unsigned l = 0;
          while (s[l]) l++;
          slt_md5_update(&m, s, l);
          slt_md5_update(&m, "\n", 1);
        }
        slt_md5_final(&m, got);
        if (v.n == want_n && !strcmp(got, want_hash)) st->query_pass++;
        else {
          st->query_fail++;
          if (st->reported < SLT_MAX_REPORTED) {
            st->reported++;
            slt_emit_u(out, ctx, "SLT-FAIL query line=", lineno);
            slt_emit_u(out, ctx, " nvalue=", v.n);
            slt_emit_u(out, ctx, " want=", want_n);
            out(ctx, " md5="); out(ctx, got);
            out(ctx, " want="); out(ctx, want_hash); out(ctx, "\n");
          }
        }
      } else {
        failed = (v.n != exp_count);
        if (!failed) {
          for (i = 0; i < v.n && i < SLT_MAX_REPORTED; i++) {
            const char *g = v.txt + v.off[i];
            unsigned gl = 0, j;
            unsigned el = exp_lens[i];
            const char *e = exp_lines[i];
            while (g[gl]) gl++;
            while (el && (e[el - 1] == ' ' || e[el - 1] == '\t')) el--;
            if (gl != el) { failed = 1; break; }
            for (j = 0; j < gl; j++)
              if (g[j] != e[j]) { failed = 1; break; }
            if (failed) break;
          }
          /* Beyond SLT_MAX_REPORTED values the expected lines were not retained, so the
             comparison is completed by hashing BOTH sides -- never by assuming a match. */
          if (!failed && v.n > SLT_MAX_REPORTED) {
            struct slt_md5 mg, me;
            char hg[33], he[33];
            slt_scan es; const char *el2; unsigned ell;
            slt_md5_init(&mg); slt_md5_init(&me);
            for (i = 0; i < v.n; i++) {
              const char *s = v.txt + v.off[i];
              unsigned l = 0;
              while (s[l]) l++;
              slt_md5_update(&mg, s, l); slt_md5_update(&mg, "\n", 1);
            }
            es.p = input; es.n = sc.i; es.i = exp_start;
            while (slt_line(&es, &el2, &ell)) {
              if (slt_blank(el2, ell)) break;
              while (ell && (el2[ell - 1] == ' ' || el2[ell - 1] == '\t')) ell--;
              slt_md5_update(&me, el2, ell); slt_md5_update(&me, "\n", 1);
            }
            slt_md5_final(&mg, hg); slt_md5_final(&me, he);
            failed = strcmp(hg, he) != 0;
          }
        }
        if (!failed) st->query_pass++;
        else {
          st->query_fail++;
          if (st->reported < SLT_MAX_REPORTED) {
            st->reported++;
            slt_emit_u(out, ctx, "SLT-FAIL query line=", lineno);
            slt_emit_u(out, ctx, " nvalue=", v.n);
            slt_emit_u(out, ctx, " nexpected=", exp_count);
            out(ctx, "\n");
          }
        }
      }
      slt_vals_free(&v);
      continue;
    }

    st->parse_err++;
    if (st->reported < SLT_MAX_REPORTED) {
      st->reported++;
      slt_emit_u(out, ctx, "SLT-PARSE line=", lineno);
      out(ctx, " record="); out(ctx, tok); out(ctx, "\n");
    }
  }

  st->completed = 1;
  sqlite3_close(db);
  return st->stmt_fail + st->query_fail + st->parse_err;
}

/* The authoritative summary. Counts live HERE, in the payload, not in a return-value
   bitfield: files run to thousands of records and would clamp into a byte. */
static void slt_report(slt_out_fn out, void *ctx, const slt_stats *st) {
  out(ctx, "SLT-SUMMARY");
  slt_emit_u(out, ctx, " records=", st->records);
  slt_emit_u(out, ctx, " stmt_pass=", st->stmt_pass);
  slt_emit_u(out, ctx, " stmt_fail=", st->stmt_fail);
  slt_emit_u(out, ctx, " query_pass=", st->query_pass);
  slt_emit_u(out, ctx, " query_fail=", st->query_fail);
  slt_emit_u(out, ctx, " skip_big=", st->skip_big);
  slt_emit_u(out, ctx, " oom=", st->oom);
  slt_emit_u(out, ctx, " skip_cond=", st->skip_cond);
  slt_emit_u(out, ctx, " parse_err=", st->parse_err);
  out(ctx, st->completed ? " completed=1" : " completed=0");
  out(ctx, st->open_failed ? " open_failed=1\n" : "\n");
}

#endif
