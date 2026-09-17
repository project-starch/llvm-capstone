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

/* OPEN, WRITE, READ, CLOSE, and one refused OPEN. The refused read never
 * reaches the helper, because the domain's own descriptor table rejects it. */
#define MUSL_FILE_PROBE_EXPECTED_ROUNDS 5

#endif
