#ifndef CAPSTONE_MUSL_WRITE_PROBE_H
#define CAPSTONE_MUSL_WRITE_PROBE_H

/* The one string both sides of this probe agree on.
 *
 * It lives in a header rather than being typed twice because the host does not
 * merely print what arrives, it COMPARES. A host that prints is a proof that
 * some bytes crossed; a host that compares is a proof that these bytes did, and
 * the difference matters when the payload region is shared memory that starts
 * out zeroed and could just as well be echoing itself. */
#define MUSL_WRITE_PROBE_MESSAGE "musl write through hostcall v0\n"
#define MUSL_WRITE_PROBE_MESSAGE_LEN (sizeof(MUSL_WRITE_PROBE_MESSAGE) - 1)

/* capstone_main() status, read by the host out of metadata->result at DONE. */
#define WP_OK                0
#define WP_SHORT_WRITE       1
#define WP_WRITE_FAILED      2
#define WP_BADFD_SUCCEEDED   3
#define WP_BADFD_WRONG_ERRNO 4

/* The bad file descriptor of the negative control. hc_write serves only 1 and
 * 2, so this must come back -1/EBADF without ever reaching the host. */
#define MUSL_WRITE_PROBE_BAD_FD 7

/* Exactly one WRITE_STDOUT round is expected: the message is far below the
 * 4 KiB payload region, so it is not chunked, and the bad-fd arm is refused
 * inside the domain and never becomes a request. A second round would mean the
 * chunk loop or the fd check is not doing what this probe claims. */
#define MUSL_WRITE_PROBE_EXPECTED_ROUNDS 1

#endif
