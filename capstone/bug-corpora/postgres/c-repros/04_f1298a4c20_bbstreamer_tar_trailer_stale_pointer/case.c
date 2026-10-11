#include "corpus.h"

PGCLIENT_CASE(4) {
/* pg_basebackup reads past its input when forwarding a tar member trailer.
 * Upstream fix f1298a4c20; live at the 17.5 pin.
 *
 *   bbstreamer_tar.c:220   if (!bbstreamer_buffer_until(streamer, &data, &len,
 *                                   mystreamer->pad_bytes_expected))
 *                              return;
 *   bbstreamer_tar.c:225   bbstreamer_content(mystreamer->base.bbs_next,
 *                                   &mystreamer->member,
 *                                   data, mystreamer->pad_bytes_expected,
 *                                   BBSTREAMER_MEMBER_TRAILER);
 *
 * buffer_until takes data and len BY ADDRESS, and bbstreamer.h:157-164 shows
 * what it does with them: appendBinaryStringInfo copies the bytes into the
 * streamer's OWN buffer, then `*data += nbytes`. So at :225 the padding lives
 * in bbs_buffer and `data` points past it. When the padding completed the
 * chunk, `len` is 0 and `data` is one past the end of the input. The next
 * streamer is then told to read pad_bytes_expected bytes from there.
 *
 * Upstream passes mystreamer->base.bbs_buffer.data. The pin passes data.
 *
 * WHY THIS IS NON-NESTED. pg_basebackup is a client: its palloc is
 * fe_memutils.c's, which is pg_malloc, which is malloc. The chunk this reads
 * past is a libc allocation.
 *
 * REDUCTION. buffer_until and buffer_bytes are the real ones. What is not
 * reproduced is the tar parsing that reaches BBSTREAMER_MEMBER_TRAILER and
 * decides pad_bytes_expected -- that needs a server streaming a base backup.
 * The case puts the parser in the state :220 reaches and then performs :225
 * exactly as written. */

  const int pad_bytes_expected = 24;

  /* The input chunk, sized so the padding ends exactly at its end -- the
   * condition that turns the advanced pointer from "wrong bytes" into "off
   * the end of the allocation". Allocated exactly, no slack. */
  const int chunk_len = pad_bytes_expected;
  char *chunk = pgclient_malloc((size_t) chunk_len);
  for (int i = 0; i < chunk_len; i++) chunk[i] = 0;

  bbstreamer streamer;
  pgclient_si_init(&streamer.bbs_buffer);

  const char *data = chunk;
  int len = chunk_len;

  /* bbstreamer_tar.c:220. Returns true: the whole chunk is the padding. */
  if (!bbstreamer_buffer_until(&streamer, &data, &len, pad_bytes_expected))
    pgclient_give_up(2);           /* the case needs the completed branch */

  pgclient_note_signed("remaining_len", len);          /* 0 */
  pgclient_note_signed("data_past_end", data == chunk + chunk_len);  /* 1 */

  /* bbstreamer_tar.c:225-228, transcribed. `data`, not bbs_buffer.data. */
  pgclient_expect_fault_in((const void *) &pgclient_consume_content,
                           "pgclient_consume_content");
  pgclient_mark();                   /* the defect's own line is the next one */
  pgclient_consume_content(data, pad_bytes_expected);

  /* Reached only where nothing bounds the read: every byte the sink took came
   * from beyond the chunk. */
  pgclient_note_overread(chunk, (size_t) chunk_len,
                         (size_t) chunk_len + (size_t) pad_bytes_expected);
  pgclient_free(streamer.bbs_buffer.data);
  pgclient_free(chunk);
}
