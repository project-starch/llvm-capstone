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
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

unsigned long __capstone_unserved_count(void);
long __capstone_unserved_at(unsigned long i);

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

    /* The rest of the file service, at the layer a program actually uses it
       from. Each of these is one opcode that existed in the protocol and had
       no domain side until now. */
    int fd = open(MUSL_STDIO_PROBE_PATH, O_RDWR);
    if (fd < 0)
        return SP_FOPEN_R_FAILED;

    /* SEEK_END is the one seek that costs a round, because the size lives with
       the helper. The line written above is what it must report. */
    off_t end = lseek(fd, 0, SEEK_END);
    if (end < 0)
        return SP_SEEK_END_WRONG;
    if (end != (off_t)sizeof(MUSL_STDIO_PROBE_LINE) - 1)
        return SP_SEEK_END_SIZE;

    if (fsync(fd) != 0)
        return SP_FSYNC_FAILED;

    if (ftruncate(fd, 8) != 0)
        return SP_TRUNCATE_FAILED;
    /* Asked again, so the answer comes from the helper's fstat and not from
       our own bookkeeping: a truncate that returned 0 and did nothing would
       pass on the return value alone. */
    if (lseek(fd, 0, SEEK_END) != 8)
        return SP_TRUNCATE_SIZE;
    close(fd);

    if (access(MUSL_STDIO_PROBE_PATH, F_OK) != 0)
        return SP_ACCESS_FAILED;
    if (unlink(MUSL_STDIO_PROBE_PATH) != 0)
        return SP_UNLINK_FAILED;
    if (access(MUSL_STDIO_PROBE_PATH, F_OK) == 0)
        return SP_ACCESS_AFTER;

    /* Last, and the reason it is last: anything the program asked for that had
       no opcode is recorded rather than merely refused, and this is where the
       list is read. Printing it needs stdio, which is why it cannot happen
       inside the hostcall itself. */
    if (__capstone_unserved_count() != 0) {
        printf("stdio-probe: UNSERVED syscalls:");
        for (unsigned long i = 0; i < __capstone_unserved_count(); i++)
            printf(" %ld", __capstone_unserved_at(i));
        printf("\n");
        fflush(stdout);
        return SP_UNSERVED;
    }

    return SP_OK;
}
