#ifndef CAPSTONE_MUSL_STDIO_PROBE_H
#define CAPSTONE_MUSL_STDIO_PROBE_H

/* The stdio layer, which is where musl stops calling write() and read().
 * __stdio_write builds a two-entry iovec and issues writev; __stdio_read
 * issues readv. A domain can therefore pass every file-I/O test and still have
 * no working printf, which is why this probe exists separately. */
#define MUSL_STDIO_PROBE_PATH "/tmp/musl_stdio_probe.txt"
#define MUSL_STDIO_PROBE_LINE "stdio 42 2.500000 through writev\n"
#define MUSL_STDIO_PROBE_STDOUT_LINE "musl printf through hostcall v0\n"

#define SP_OK                0
#define SP_FOPEN_W_FAILED    1
#define SP_FPRINTF_FAILED    2
#define SP_FCLOSE_W_FAILED   3
#define SP_FOPEN_R_FAILED    4
#define SP_FGETS_FAILED      5
#define SP_CONTENT_MISMATCH  6
#define SP_FCLOSE_R_FAILED   7
#define SP_PRINTF_FAILED     8
#define SP_SEEK_END_WRONG    9
#define SP_SEEK_END_SIZE    10  /* SEEK_END did not report the file's size */
#define SP_FSYNC_FAILED     11
#define SP_TRUNCATE_FAILED  12
#define SP_TRUNCATE_SIZE    13  /* truncate reported success and did nothing */
#define SP_ACCESS_FAILED    14  /* the file exists and access() says otherwise */
#define SP_UNLINK_FAILED    15
#define SP_ACCESS_AFTER     16  /* it is gone and access() still finds it */
#define SP_UNSERVED         17  /* a syscall went unserved; the number is printed */

#endif
