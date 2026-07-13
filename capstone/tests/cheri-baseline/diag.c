/* Diagnostic: pinpoint where/why a purecap SQLite program faults.
 * Installs SA_SIGINFO handlers for the fault signals, prints a step marker
 * before each SQLite operation, and on a fault prints signal + si_code +
 * fault address (so we can see alignment) and the last step reached. */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>
#include "sqlite3.h"

static volatile const char *step = "start";

static void handler(int sig, siginfo_t *si, void *uc) {
  (void)uc;
  char buf[256];
  int n = snprintf(buf, sizeof buf,
      "DIAG FAULT sig=%d code=%d addr=%p laststep=%s\n",
      sig, si->si_code, (void *)si->si_addr, (const char *)step);
  write(2, buf, n);
  _exit(100 + sig);
}

static void mark(const char *s) {
  step = s;
  char buf[128]; int n = snprintf(buf, sizeof buf, "DIAG step=%s\n", s);
  write(2, buf, n);
}

int main(void) {
  struct sigaction sa; memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = handler; sa.sa_flags = SA_SIGINFO;
  sigaction(SIGBUS, &sa, 0); sigaction(SIGSEGV, &sa, 0);
  sigaction(34 /*SIGPROT*/, &sa, 0); sigaction(SIGILL, &sa, 0);

  sqlite3 *db = 0; sqlite3_stmt *st = 0;
  mark("open");     if (sqlite3_open(":memory:", &db) != SQLITE_OK) { mark("open-fail"); return 2; }
  mark("exec");     sqlite3_exec(db, "CREATE TABLE t(a); INSERT INTO t VALUES(123)", 0, 0, 0);
  mark("prepare");  sqlite3_prepare_v2(db, "SELECT a FROM t", -1, &st, 0);
  mark("step");     sqlite3_step(st);
  mark("column");   (void)sqlite3_column_int(st, 0);
  mark("finalize"); sqlite3_finalize(st);
  mark("close");    sqlite3_close(db);
  mark("done");
  printf("DIAG OK\n");
  return 0;
}
