/* musl's stdio in a domain: fprintf, fflush, fgets, printf.
 *
 * WHY THIS IS NOT THE FILE PROBE WITH MORE STEPS. file-probe exercises the
 * syscalls; this one exercises the layer above them, and that layer talks to a
 * different pair of syscalls. musl's __stdio_write gathers the FILE's buffer
 * and the caller's bytes into a two-entry iovec and issues writev; __stdio_read
 * issues readv. A domain that serves read and write and not their vectored
 * forms passes every file test and has no working printf.
 *
 * The oracle is content, not counts. stdio decides for itself when to flush, so
 * the number of rounds depends on buffer sizes and is not a property worth
 * asserting. What is worth asserting is that the bytes formatted on one side
 * come back identical on the other, through the buffer, the vector and the
 * wire.
 *
 * The %f is deliberate: it is what drags in the soft-float long-double builtins
 * that musl's vfprintf references, and those were the last eleven undefined
 * symbols this port had.
 */
#include <stdio.h>
#include <string.h>

#include "stdio_probe.h"

int capstone_main(void)
{
    char buf[128];

    FILE *f = fopen(MUSL_STDIO_PROBE_PATH, "w");
    if (!f)
        return SP_FOPEN_W_FAILED;
    if (fprintf(f, "stdio %d %f through writev\n", 42, 2.5) < 0)
        return SP_FPRINTF_FAILED;
    if (fclose(f) != 0)
        return SP_FCLOSE_W_FAILED;

    f = fopen(MUSL_STDIO_PROBE_PATH, "r");
    if (!f)
        return SP_FOPEN_R_FAILED;
    if (!fgets(buf, sizeof buf, f))
        return SP_FGETS_FAILED;
    if (strcmp(buf, MUSL_STDIO_PROBE_LINE) != 0)
        return SP_CONTENT_MISMATCH;

    /* fseek to the end and back tells us whether the FILE layer's own idea of
       the position agrees with the domain's descriptor table. */
    if (fseek(f, 0, SEEK_SET) != 0)
        return SP_SEEK_END_WRONG;
    if (fclose(f) != 0)
        return SP_FCLOSE_R_FAILED;

    /* And the same layer to stdout, which takes the WRITE_STDOUT path instead
       of a handle, so both destinations are covered. */
    if (printf("%s", MUSL_STDIO_PROBE_STDOUT_LINE) < 0)
        return SP_PRINTF_FAILED;
    if (fflush(stdout) != 0)
        return SP_PRINTF_FAILED;

    return SP_OK;
}
