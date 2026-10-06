/* bbstreamer_buffer_bytes and bbstreamer_buffer_until, verbatim from
 * PostgreSQL 17.5 src/bin/pg_basebackup/bbstreamer.h:156-196.
 *
 * These two are the whole defect, and they are worth reading together:
 *
 *   bbstreamer_buffer_bytes(streamer, data, len, nbytes)
 *       appendBinaryStringInfo(&streamer->bbs_buffer, *data, nbytes);
 *       *len  -= nbytes;
 *       *data += nbytes;          <-- the caller's cursor MOVES
 *
 * so when bbstreamer_buffer_until returns true, the bytes it buffered are in
 * bbs_buffer and `data` points PAST them. bbstreamer_tar.c:225-228 then
 * forwards `data` -- not bbs_buffer.data -- with length pad_bytes_expected.
 * When the padding ran to the end of the chunk, `len` is 0 and `data` is one
 * past the end of the caller's input.
 *
 * StringInfo is a stand-in rather than src/common/stringinfo.c: the buffer's
 * growth policy takes no part in the defect, and what matters is only that
 * appendBinaryStringInfo copies the bytes somewhere ELSE, which is why the
 * advanced `data` no longer points at them.
 */
#include "corpus.h"
#include <stdlib.h>
#include <string.h>

void pgclient_si_init(StringInfoData *si) {
	si->maxlen = 256;
	si->data = pgclient_malloc((size_t) si->maxlen);
	si->len = 0;
}

static void appendBinaryStringInfo(StringInfoData *si, const char *d, int n) {
	if (si->len + n > si->maxlen) {
		while (si->len + n > si->maxlen) si->maxlen *= 2;
		char *nb = pgclient_malloc((size_t) si->maxlen);
		memcpy(nb, si->data, (size_t) si->len);
		free(si->data);
		si->data = nb;
	}
	memcpy(si->data + si->len, d, (size_t) n);
	si->len += n;
}

void bbstreamer_buffer_bytes(bbstreamer *streamer, const char **data, int *len,
							 int nbytes)
{
	appendBinaryStringInfo(&streamer->bbs_buffer, *data, nbytes);
	*len -= nbytes;
	*data += nbytes;
}

bool bbstreamer_buffer_until(bbstreamer *streamer, const char **data, int *len,
							 int target_bytes)
{
	int			buflen = streamer->bbs_buffer.len;

	if (buflen >= target_bytes)
	{
		/* Target length already reached; nothing to do. */
		return true;
	}

	if (buflen + *len < target_bytes)
	{
		/* Not enough data to reach target length; buffer all of it. */
		bbstreamer_buffer_bytes(streamer, data, len, *len);
		return false;
	}

	/* Buffer just enough to reach the target length. */
	bbstreamer_buffer_bytes(streamer, data, len, target_bytes - buflen);
	return true;
}

/* The next streamer in the chain. bbstreamer_tar.c:225 hands it a pointer and
 * a length; all it does is consume that many bytes, which is what makes the
 * wrong pointer an out-of-bounds read rather than merely wrong output. */
volatile char pgclient_sink_byte;
void pgclient_consume_content(const char *data, int len) {
	for (int i = 0; i < len; i++)
		pgclient_sink_byte = data[i];
}
