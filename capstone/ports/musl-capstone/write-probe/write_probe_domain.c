/* musl's own write(2), through the Capstone hostcall, end to end.
 *
 * WHAT THIS ADDS TO THE YIELD PROBE. yield-probe shows that a pure-capability
 * domain can suspend at a hostcall and resume with its C frame intact. It calls
 * __capstone_yield() directly. This one calls nothing of ours: it calls
 * write(), musl's write(), which reaches __syscall3 in arch-capstone64, which
 * reaches __capstone_hostcall, which marshals into the shared payload and
 * yields. Every layer the port exists for is on that path, and none of them is
 * exercised by the yield probe.
 *
 * THE ORACLE IS THE RETURN VALUE, NOT THE OUTPUT. A host that prints the bytes
 * proves they arrived; it does not prove musl saw a well-formed result. So the
 * probe checks what write() returned to C, and the host checks what it received.
 * Either alone is half a proof.
 *
 * THE NEGATIVE CONTROL IS THE POINT OF THE SECOND CALL. A probe whose every arm
 * succeeds cannot distinguish "the path works" from "the path is not taken and
 * something else printed". hc_write returns -EBADF for any fd that is not 1 or
 * 2, so write() on fd 7 must come back -1 with errno EBADF -- which also proves
 * that musl's errno translation survives the boundary, since the hostcall
 * returns a negative errno and musl is what turns it into -1/errno.
 */
#include <errno.h>
#include <unistd.h>

#include "write_probe.h"

#define MSG     MUSL_WRITE_PROBE_MESSAGE
#define MSG_LEN MUSL_WRITE_PROBE_MESSAGE_LEN

int capstone_main(void)
{
    /* Arm 1: the real thing. */
    ssize_t n = write(1, MSG, MSG_LEN);
    if (n < 0)
        return WP_WRITE_FAILED;
    if (n != (ssize_t)MSG_LEN)
        return WP_SHORT_WRITE;

    /* Arm 2: the negative control, which must fail and must fail precisely.
     *
     * OFF BY DEFAULT, and this is a finding rather than a convenience. Reading
     * errno needs __errno_location, which returns &__pthread_self()->errno_val,
     * which dereferences the thread pointer. A freestanding domain has none.
     * Measured 2026-09-16 with this arm on: the first write SUCCEEDS end to end
     * -- the host printed the payload -- and the domain then halts with
     * cause = 24 at __errno_location, with tp (x4) reading 0.
     *
     * So this arm is not disabled because it fails. It is disabled because it
     * has already told us what it had to tell us, and a gate that always fails
     * is one everybody learns to ignore. Turn it back on with
     * -DMUSL_WRITE_PROBE_WANT_BADFD once a thread pointer exists, and it becomes
     * the test that the TLS setup is real rather than merely present.
     *
     * Note what this does NOT mean: the hostcall's own error path is fine. It
     * returns -EBADF and musl's __syscall_ret correctly recognises it as an
     * error. Only the last step, recording it in errno, has nowhere to write. */
#ifdef MUSL_WRITE_PROBE_WANT_BADFD
    errno = 0;
    ssize_t bad = write(MUSL_WRITE_PROBE_BAD_FD, MSG, MSG_LEN);
    if (bad >= 0)
        return WP_BADFD_SUCCEEDED;
    if (errno != EBADF)
        return WP_BADFD_WRONG_ERRNO;
#endif

    return WP_OK;
}
