/* musl's open, write, lseek, read and close, through HostCall v0 file service.
 *
 * WHAT THIS ADDS TO THE WRITE PROBE. write-probe proves one opcode that needs
 * no state: WRITE_STDOUT takes bytes and a length and nothing else. File I/O is
 * where the protocol and POSIX disagree, and the disagreement is the point.
 * FILE_READ and FILE_WRITE carry an explicit file_offset; read() and write() do
 * not, they advance an implicit position. The domain keeps that position, so
 * this probe is what shows the position is actually kept: it writes at 0, seeks
 * back to 0, reads, and compares. A domain that forgot to advance would read
 * back its own first bytes and still "succeed" on the counts alone.
 *
 * THREE ARMS, and the second and third are the ones that can fail quietly.
 *
 *   1. The round trip. open, write, lseek, read, compare, close.
 *   2. A path that is not there. The helper answers -ENOENT, and the test is
 *      that the error crosses the boundary AS an error and arrives in errno.
 *      This is the arm that would catch the sign convention going wrong: the
 *      wire spec says error carries a negative errno, a positive one would
 *      reach musl as a successful syscall, and open() would return a small
 *      positive number that looks like a descriptor.
 *   3. A read on the descriptor just closed. This one never reaches the helper:
 *      the domain's own table refuses it. It is here because the table is new
 *      code and a stale slot would otherwise be found by a workload rather than
 *      by a probe.
 */
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#include "file_probe.h"

int capstone_main(void)
{
    char buf[MUSL_FILE_PROBE_MESSAGE_LEN];

    int fd = open(MUSL_FILE_PROBE_PATH, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        return FP_OPEN_FAILED;

    if (write(fd, MUSL_FILE_PROBE_MESSAGE, MUSL_FILE_PROBE_MESSAGE_LEN)
        != (ssize_t)MUSL_FILE_PROBE_MESSAGE_LEN)
        return FP_WRITE_FAILED;

    if (lseek(fd, 0, SEEK_SET) != 0)
        return FP_SEEK_FAILED;

    if (read(fd, buf, sizeof buf) != (ssize_t)sizeof buf)
        return FP_READ_FAILED;

    if (memcmp(buf, MUSL_FILE_PROBE_MESSAGE, sizeof buf) != 0)
        return FP_CONTENT_MISMATCH;

    if (close(fd) != 0)
        return FP_CLOSE_FAILED;

    /* Arm 2: the helper's error, across the boundary, into errno. */
    errno = 0;
    int missing = open(MUSL_FILE_PROBE_MISSING_PATH, O_RDONLY);
    if (missing >= 0)
        return FP_MISSING_OPENED;
    if (errno != ENOENT)
        return FP_MISSING_WRONG_ERR;

    /* Arm 3: the domain's own table, which never asks the helper. */
    errno = 0;
    if (read(fd, buf, sizeof buf) >= 0)
        return FP_CLOSED_FD_READ;
    if (errno != EBADF)
        return FP_CLOSED_WRONG_ERR;

    /* Arm 4: more than one payload window. Until this arm existed the chunk
       loops in hc_file_rw had never run more than once, and the tail path
       never at all: the longest message any probe sent was 34 bytes against a
       window of 4064. Static buffers, because 20 000 bytes on a domain stack
       is a question this probe is not asking. */
    static unsigned char big_out[MUSL_FILE_PROBE_BIG_BYTES];
    static unsigned char big_in[MUSL_FILE_PROBE_BIG_BYTES];
    for (unsigned long i = 0; i < MUSL_FILE_PROBE_BIG_BYTES; i++) {
        big_out[i] = MUSL_FILE_PROBE_BYTE_AT(i);
        big_in[i] = 0;
    }

    int bfd = open(MUSL_FILE_PROBE_PATH, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (bfd < 0)
        return FP_BIG_OPEN_FAILED;
    if (write(bfd, big_out, MUSL_FILE_PROBE_BIG_BYTES)
        != (ssize_t)MUSL_FILE_PROBE_BIG_BYTES)
        return FP_BIG_WRITE_SHORT;
    if (lseek(bfd, 0, SEEK_SET) != 0)
        return FP_BIG_SEEK_FAILED;
    if (read(bfd, big_in, MUSL_FILE_PROBE_BIG_BYTES)
        != (ssize_t)MUSL_FILE_PROBE_BIG_BYTES)
        return FP_BIG_READ_SHORT;
    for (unsigned long i = 0; i < MUSL_FILE_PROBE_BIG_BYTES; i++)
        if (big_in[i] != big_out[i])
            return FP_BIG_MISMATCH | (int)(i << 8);
    if (close(bfd) != 0)
        return FP_BIG_CLOSE_FAILED;

    return FP_OK;
}
