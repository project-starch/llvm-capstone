/* PostgreSQL's memory manager, outside PostgreSQL: what the seven files of
 * src/backend/utils/mmgr reference and the backend would have provided.
 *
 * Sixteen symbols, and the list is not a guess: it is what the linker asks for
 * when the seven objects are linked with an empty main, and nothing more is
 * here than the linker asked for. They fall into four groups.
 *
 *   the elog family   nine entry points. The memory manager reports two kinds
 *                     of thing through them: out of memory, which is fatal
 *                     here and takes the process down loudly, and a debug
 *                     message, which is dropped. Nothing between the two ever
 *                     reaches them from these files.
 *   three globals     the interrupt flags a backend sets from a signal handler
 *                     and its pid. Nothing sets them here, so the checks that
 *                     read them see what a quiet backend would see.
 *   the stack check   a backend refuses to recurse near its stack limit.
 *                     Nothing here recurses on data, so it never is.
 *   printf helpers    PostgreSQL routes its own formatting through these, for
 *                     the sake of platforms whose printf differs. On the host
 *                     they are the platform's, by definition.
 *
 * A freestanding build replaces the last group and keeps the rest.
 */
#include "postgres.h"
#include "miscadmin.h"
#include "mb/pg_wchar.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/* port.h redirects the whole printf family to PostgreSQL's own, this file
   included. Each one is undefined here, above every call, or a stub calls
   itself through the macro and the linker asks for the rest of the family. */
#undef printf
#undef vprintf
#undef snprintf
#undef fprintf
#undef vsnprintf
#undef vfprintf

/* ---- the three globals -------------------------------------------------- */
volatile sig_atomic_t InterruptPending = false;
volatile sig_atomic_t LogMemoryContextPending = false;
int MyProcPid = 0;

/* ---- the stack check ---------------------------------------------------- */
bool stack_is_too_deep(void) { return false; }

/* ---- the elog family ----------------------------------------------------
 * errstart decides whether a report will be made at all. Everything below
 * ERROR is dropped, so a debug message costs nothing; ERROR and above are
 * kept, and errfinish ends the process. A backend would longjmp to its own
 * handler and carry on; there is no handler here, and an allocator that has
 * decided it cannot go on has nothing to say to a replay that would be true.
 */
static int a_elevel;
static char a_msg[1024];

bool errstart(int elevel, const char *domain)
{
    (void) domain;
    a_elevel = elevel;
    a_msg[0] = '\0';
    return elevel >= ERROR;
}
bool errstart_cold(int elevel, const char *domain) { return errstart(elevel, domain); }

void errfinish(const char *filename, int lineno, const char *funcname)
{
    fflush(stdout);
    fprintf(stderr, "pg-mmgr: %s at %s:%d in %s\n",
            a_msg[0] ? a_msg : "(no message)", filename ? filename : "?", lineno,
            funcname ? funcname : "?");
    abort();
}

static int a_say(const char *fmt, va_list ap)
{
    vsnprintf(a_msg, sizeof a_msg, fmt, ap);
    return 0;
}
int errmsg(const char *fmt,...)
{
    va_list ap; va_start(ap, fmt); int r = a_say(fmt, ap); va_end(ap); return r;
}
int errmsg_internal(const char *fmt,...)
{
    va_list ap; va_start(ap, fmt); int r = a_say(fmt, ap); va_end(ap); return r;
}
int errdetail(const char *fmt,...) { (void) fmt; return 0; }
int errcode(int sqlerrcode) { (void) sqlerrcode; return 0; }
int errhidestmt(bool hide_stmt) { (void) hide_stmt; return 0; }
int errhidecontext(bool hide_ctx) { (void) hide_ctx; return 0; }

/* ---- pg_mbcliplen ------------------------------------------------------
 * Clips a multibyte string to a byte limit without splitting a character.
 * The memory manager calls it on a context's name when it prints statistics.
 * Names here come from the trace and are ASCII, so the byte limit is the
 * character limit.
 */
int pg_mbcliplen(const char *mbstr, int len, int limit)
{
    (void) mbstr;
    return len < limit ? len : limit;
}

/* ---- the printf helpers ------------------------------------------------ */
int pg_snprintf(char *str, size_t count, const char *fmt,...)
{
    va_list ap; va_start(ap, fmt); int r = vsnprintf(str, count, fmt, ap); va_end(ap); return r;
}
int pg_fprintf(FILE *stream, const char *fmt,...)
{
    va_list ap; va_start(ap, fmt); int r = vfprintf(stream, fmt, ap); va_end(ap); return r;
}
/* Not reached from the manager, but port.h redirects every caller that
   includes it, and a program built on this port is such a caller. */
int pg_printf(const char *fmt,...)
{
    va_list ap; va_start(ap, fmt); int r = vprintf(fmt, ap); va_end(ap); return r;
}
