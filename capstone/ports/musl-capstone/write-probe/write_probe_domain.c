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

#define MSG "musl write through hostcall v0\n"
#define MSG_LEN (sizeof(MSG) - 1)

/* Status codes; the host reads this out of metadata->result. */
#define WP_OK                0
#define WP_SHORT_WRITE       1
#define WP_WRITE_FAILED      2
#define WP_BADFD_SUCCEEDED   3
#define WP_BADFD_WRONG_ERRNO 4

int capstone_main(void)
{
    /* Arm 1: the real thing. */
    ssize_t n = write(1, MSG, MSG_LEN);
    if (n < 0)
        return WP_WRITE_FAILED;
    if (n != (ssize_t)MSG_LEN)
        return WP_SHORT_WRITE;

    /* Arm 2: the negative control, which must fail and must fail precisely. */
    errno = 0;
    ssize_t bad = write(7, MSG, MSG_LEN);
    if (bad >= 0)
        return WP_BADFD_SUCCEEDED;
    if (errno != EBADF)
        return WP_BADFD_WRONG_ERRNO;

    return WP_OK;
}
