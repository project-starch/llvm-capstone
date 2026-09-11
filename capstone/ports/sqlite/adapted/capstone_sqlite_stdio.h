#ifndef CAPSTONE_SQLITE_STDIO_H
#define CAPSTONE_SQLITE_STDIO_H
/* The stdio a domain does not have, for benchmarks that expect one.
 *
 * WHY THIS EXISTS. capstone_sqlite_libc.h defines _STDIO_H and a dummy FILE purely to keep glibc
 * out of a freestanding capability domain, and declares no output functions at all -- SQLite itself
 * needs none. speedtest1 does: it reports through printf/fprintf and dies through fprintf+exit, so
 * nothing links until these exist.
 *
 * WHAT IT IS NOT. It is not a printf. The conversions below are the ones MEASURED to reach these
 * calls in speedtest1.c (comment-stripped, with the WASM/__linux__/_WIN32 blocks removed, since
 * counting code that never compiles is how a shim ends up carrying features nobody calls):
 *
 *     %d  %s  %u  %x        with width, '-' left-justify, '0' zero-pad,
 *     %llu                  precision, and '*' star-precision
 *     seen literally: %-28s  %.*s  %.48s  %02x  %03d  %4d  %5d  %d  %llu  %s
 *
 * NO FLOATING-POINT CONVERSION IS NEEDED. speedtest1's one %f goes through SQLite's own
 * sqlite3_vmprintf, not through here. An earlier draft of this work listed %c, %f and %g -- none
 * is reached -- and omitted %u, %x and the 'll' modifier, which are. That list was written from
 * memory; this one was measured.
 *
 * THE SINK IS THE DOMAIN'S. Output goes wherever the including domain sends it (the shared payload
 * region, via its own output_text). The domain provides capstone_stdio_sink; this file provides the
 * formatting. That split keeps the shim usable by any domain and keeps it out of the region-bounds
 * logic, which differs between SLT and non-SLT builds.
 *
 * TRUNCATION IS COUNTED, NEVER SILENT. The payload region is finite and its writer stops at the
 * limit. A report that quietly loses its tail looks exactly like a report that had nothing to say,
 * so the sink returns what it accepted and capstone_stdio_dropped accumulates the rest. A run that
 * is reported must show that counter at zero. */

/* Provided by the domain. Returns the number of bytes actually accepted, which may be less than n
 * when the payload region is full; anything short is counted as dropped. */
unsigned long capstone_stdio_sink(const char *text, unsigned long n);

/* Bytes the sink refused, cumulative. Non-zero invalidates the run's output, not the run. */
extern unsigned long capstone_stdio_dropped;

/* Also the domain's. Called by exit(); a domain cannot end a process, and SPINNING THERE WOULD
 * WEDGE THE CORE -- which on the board takes every later stage of the boot with it. The domain
 * implements this to unwind back to its entry point and report, so a fatal error becomes a result
 * instead of a lost session. If it returns anyway, exit() spins as the last resort. */
void capstone_stdio_on_exit(int code);

/* The two stream objects speedtest1 names. They carry no state: both route to the same sink, so
 * stdout and stderr are distinguishable only by address, which is all the callers need. */
/* capstone_sqlite_libc.h names this struct but never defines it -- SQLite only ever passes FILE* as
 * an opaque handle, so an incomplete type is exactly right there. This shim has to INSTANTIATE two
 * of them, so it completes the type. One member, never read; the objects are distinguished by
 * address. (A first version tried to instantiate the incomplete type and did not compile, in the
 * test harness or the real build.) */
#ifndef CAPSTONE_SQLITE_VFS_H
/* Only when nothing else completes it -- i.e. the standalone gate harness. In the real build
 * capstone_sqlite_vfs.h has already completed this struct with the VFS's own members and a second
 * definition is an error, which is exactly what the first trial compile of the amalgam TU said. */
struct capstone_sqlite_file { int capstone_stream_id; };
#endif
typedef struct capstone_sqlite_file FILE;
extern FILE *const capstone_stdout;
extern FILE *const capstone_stderr;
#define stdout capstone_stdout
#define stderr capstone_stderr

/* Enough of <stdarg.h> for the vfprintf the callers use, without pulling a host header in. */
typedef __builtin_va_list capstone_va_list;

int capstone_printf(const char *fmt, ...);
int capstone_fprintf(FILE *stream, const char *fmt, ...);
int capstone_vfprintf(FILE *stream, const char *fmt, capstone_va_list ap);
int capstone_fflush(FILE *stream);
void capstone_exit(int code);
int capstone_atoi(const char *s);

/* The file operations exist because speedtest1 REFERENCES them, not because it uses them on our
 * path: --hashfile and --script are never passed, but a reference is a link-time symbol whether or
 * not it is reachable. They fail cleanly rather than pretending to work; a domain has no files. */
FILE *capstone_fopen(const char *path, const char *mode);
int capstone_fclose(FILE *stream);
unsigned long capstone_fwrite(const void *ptr, unsigned long size, unsigned long n, FILE *stream);
int capstone_unlink(const char *path);

#define printf   capstone_printf
#define fprintf  capstone_fprintf
#define vfprintf capstone_vfprintf
#define fflush   capstone_fflush
#define exit     capstone_exit
#define atoi     capstone_atoi
#define fopen    capstone_fopen
#define fclose   capstone_fclose
#define fwrite   capstone_fwrite
#define unlink   capstone_unlink

/* speedtest1 includes <unistd.h> unconditionally for unlink(). That header is NOT in
 * capstone_sqlite_libc.h's guard list and neither build passes -nostdinc, so it resolves to host
 * glibc and drags in 97 host headers -- measured 2026-09-10, and it compiles, which is what makes it
 * dangerous rather than obvious. Guarding it here is the fix. */
#define _UNISTD_H 1

#endif /* CAPSTONE_SQLITE_STDIO_H */
