#ifndef CAPSTONE_MUSL_FILE_PROBE_H
#define CAPSTONE_MUSL_FILE_PROBE_H

/* What both sides of the file probe agree on.
 *
 * The path is in the guest's /tmp because the helper resolves paths in its own
 * working directory: there is no *at family behind this protocol, so a relative
 * path would mean "relative to the helper" and an absolute one says what it
 * means. */
#define MUSL_FILE_PROBE_PATH "/tmp/musl_file_probe.txt"
#define MUSL_FILE_PROBE_MISSING_PATH "/tmp/musl_file_probe_absent.txt"
#define MUSL_FILE_PROBE_MESSAGE "musl file io through hostcall v0\n"
#define MUSL_FILE_PROBE_MESSAGE_LEN (sizeof(MUSL_FILE_PROBE_MESSAGE) - 1)

/* capstone_main() status, read by the host out of metadata->result at DONE. */
#define FP_OK                 0
#define FP_OPEN_FAILED        1
#define FP_WRITE_FAILED       2
#define FP_SEEK_FAILED        3
#define FP_READ_FAILED        4
#define FP_CONTENT_MISMATCH   5
#define FP_CLOSE_FAILED       6
#define FP_MISSING_OPENED     7  /* a path that is not there was opened */
#define FP_MISSING_WRONG_ERR  8  /* it failed, but not with ENOENT */
#define FP_CLOSED_FD_READ     9  /* reading a closed descriptor succeeded */
#define FP_CLOSED_WRONG_ERR  10  /* it failed, but not with EBADF */

/* The large transfer. 10 000 bytes against a payload window of 4096 with a
 * 32-byte header, so 4064 per round: two full chunks and a partial third of
 * 1872. Chosen NOT to be a multiple of the chunk size, so the tail path runs
 * too. Every byte carries its own offset, so a chunk delivered out of order, at
 * the wrong file offset, or truncated changes the comparison; matching counts
 * alone would not see any of those. */
#define MUSL_FILE_PROBE_BIG_BYTES 10000UL
#define MUSL_FILE_PROBE_CHUNK (4096UL - 32UL)
#define MUSL_FILE_PROBE_BIG_ROUNDS \
  ((MUSL_FILE_PROBE_BIG_BYTES + MUSL_FILE_PROBE_CHUNK - 1) / MUSL_FILE_PROBE_CHUNK)
#define MUSL_FILE_PROBE_BYTE_AT(i) ((unsigned char)(((i) * 7 + ((i) >> 8)) & 0xff))

/* The round count is derived, not typed, because it is the assertion that the
 * chunking happened: OPEN, WRITE, READ, CLOSE and one refused OPEN are five,
 * the refused read is zero because the domain's own table answers it, and the
 * large arm adds an OPEN, its write rounds, its read rounds and a CLOSE. A
 * transfer that went out in one round, or in one more than it should, changes
 * this number. */
#define MUSL_FILE_PROBE_EXPECTED_ROUNDS \
  (5 + 2 + 2 * MUSL_FILE_PROBE_BIG_ROUNDS)

#define FP_BIG_OPEN_FAILED   11
#define FP_BIG_WRITE_SHORT   12  /* write() returned less than asked, no short-write cause exists here */
#define FP_BIG_SEEK_FAILED   13
#define FP_BIG_READ_SHORT    14
#define FP_BIG_MISMATCH      15  /* a byte differs; the first differing offset is the status' upper bits */
#define FP_BIG_CLOSE_FAILED  16

#endif
