/* The backend symbols resowner.c and the memory managers reference, and
 * nothing else.  Each one is here because the link demanded it; the list is
 * the claim that this corpus pulls in resowner.c and the seven manager files
 * and NOT a backend.
 *
 * errstart/errfinish are the interesting ones.  In a backend, elog(ERROR)
 * reports and then siglongjmps to the enclosing PG_TRY or to the main loop,
 * and that unwinding is exactly the mechanism both cases depend on.  So they
 * are implemented as the longjmp they are, not as an abort: an errfinish that
 * ended the process would make the second release unreachable and the corpus
 * would measure nothing.
 */
#include "postgres.h"
#include "storage/ipc.h"
#include "storage/lock.h"
#include "utils/memutils.h"
#include <setjmp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- the error jump target, owned by the driver ---------------------- */
sigjmp_buf ro_error_jump;
volatile sig_atomic_t ro_error_armed = 0;
static int ro_cur_elevel = 0;

volatile sig_atomic_t InterruptPending = false;
volatile sig_atomic_t LogMemoryContextPending = false;
volatile sig_atomic_t IdleInTransactionSessionTimeoutPending = false;
int MyProcPid = 0;
int32 CritSectionCount = 0;

bool stack_is_too_deep(void) { return false; }

bool
errstart(int elevel, const char *domain)
{
	(void) domain;
	ro_cur_elevel = elevel;
	return true;
}

bool
errstart_cold(int elevel, const char *domain)
{
	return errstart(elevel, domain);
}

void
errfinish(const char *filename, int lineno, const char *funcname)
{
	(void) filename;
	(void) lineno;
	fprintf(stderr, "  [elog] %s\n", funcname ? funcname : "?");
	if (ro_cur_elevel >= ERROR)
	{
		if (ro_error_armed)
			siglongjmp(ro_error_jump, 1);
		fprintf(stderr, "  [elog] ERROR with no handler armed\n");
		exit(70);
	}
}

static int
a_say(const char *fmt, va_list ap)
{
	fputs("  [elog] ", stderr);
	vfprintf(stderr, fmt, ap);
	fputc('\n', stderr);
	return 0;
}

int
errmsg(const char *fmt,...)
{
	va_list ap; va_start(ap, fmt); a_say(fmt, ap); va_end(ap); return 0;
}

int
errmsg_internal(const char *fmt,...)
{
	va_list ap; va_start(ap, fmt); a_say(fmt, ap); va_end(ap); return 0;
}

int errdetail(const char *fmt,...) { (void) fmt; return 0; }
int errcode(int sqlerrcode) { (void) sqlerrcode; return 0; }
int errhidestmt(bool hide_stmt) { (void) hide_stmt; return 0; }
int errhidecontext(bool hide_ctx) { (void) hide_ctx; return 0; }

void
ExceptionalCondition(const char *conditionName, const char *fileName,
					 int lineNumber)
{
	fprintf(stderr, "ASSERT %s at %s:%d\n", conditionName, fileName, lineNumber);
	abort();
}

char *
psprintf(const char *fmt,...)
{
	va_list ap;
	char	   *buf;
	int			need;

	va_start(ap, fmt);
	need = vsnprintf(NULL, 0, fmt, ap);
	va_end(ap);
	if (need < 0)
		need = 0;
	buf = palloc((Size) need + 1);
	va_start(ap, fmt);
	vsnprintf(buf, (size_t) need + 1, fmt, ap);
	va_end(ap);
	return buf;
}

/* pg_qsort comes from upstream src/port/qsort.c, which the build compiles.
 * Do not shim it: port.h redirects qsort() to pg_qsort(), so a shim that calls
 * qsort() calls itself and the stack overflows (measured 2026-11-10). */

/* resowner's lock-release phase. No case owns a lock, so these are reached
 * with nothing to do; they are here for the link, not for the sequence. */
void LockReleaseCurrentOwner(struct LOCALLOCK **locallocks, int nlocks)
{ (void) locallocks; (void) nlocks; }
void LockReassignCurrentOwner(struct LOCALLOCK **locallocks, int nlocks)
{ (void) locallocks; (void) nlocks; }
void ProcReleaseLocks(bool isCommit) { (void) isCommit; }
void ReleasePredicateLocks(bool isCommit, bool isReadOnlySafe)
{ (void) isCommit; (void) isReadOnlySafe; }
void on_shmem_exit(pg_on_exit_callback function, Datum arg)
{ (void) function; (void) arg; }

int
pg_mbcliplen(const char *mbstr, int len, int limit)
{
	(void) mbstr;
	return len < limit ? len : limit;
}
