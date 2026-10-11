/* pqGetnchar, verbatim from PostgreSQL 17.5
 * src/interfaces/libpq/fe-misc.c:164-176. Not one line is changed.
 *
 * The defect is what the guard at :167 bounds. `conn->inEnd - conn->inCursor`
 * is how many bytes are AVAILABLE TO READ; the memcpy at :170 then writes len
 * bytes into `s` with nothing said about how big `s` is. It cannot be said,
 * because PQfn (fe-exec.c:2980) takes `int *result_buf` and has no size
 * parameter at all, so the destination's extent never reaches this function.
 *
 * PGconn here is reduced to the three fields this function touches. That is
 * deliberate: the real struct is several hundred lines of connection state,
 * none of which participates, and carrying it would obscure that the guard
 * and the copy disagree about which buffer they are talking about.
 */
#include "corpus.h"
#include <string.h>

#ifndef EOF
#define EOF (-1)
#endif

struct pgclient_conn {
	char	   *inBuffer;
	int			inStart;
	int			inCursor;
	int			inEnd;
};

typedef struct pgclient_conn PGconn;

int pqGetnchar(char *s, size_t len, PGconn *conn)
{
	if (len > (size_t) (conn->inEnd - conn->inCursor))
		return EOF;

	memcpy(s, conn->inBuffer + conn->inCursor, len);
	/* no terminating null */

	conn->inCursor += len;

	return 0;
}

/* --- the harness side: a connection whose input buffer already holds the
 * oversized reply, which is the state pqFunctionCall3 reaches on a hostile
 * 'V' message. --- */
static PGconn pgclient_the_conn;

void pgclient_fill_input(size_t nbytes) {
	char *buf = pgclient_malloc(nbytes);
	memset(buf, 0x41, nbytes);
	pgclient_the_conn.inBuffer = buf;
	pgclient_the_conn.inStart = 0;
	pgclient_the_conn.inCursor = 0;
	pgclient_the_conn.inEnd = (int) nbytes;
}

PGconn *pgclient_conn(void) { return &pgclient_the_conn; }
