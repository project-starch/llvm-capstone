/* inline-io-contract: the adversarial test of lane AB's in-module I/O (experiment/in-module-io).
 *
 * The risk is the confused deputy: the kernel writes into memory whose authority was revoked, or
 * whose range was recycled, while the system call was in flight. Each case prints one line,
 *   INLINE_IO case=<name> hold=<0|1> ... verdict=<PASS|FAIL|LANDED|REFUSED|...>
 * and the run ends with INLINE_IO_DONE. The test reads the module's inline_hold parameter, so ONE
 * binary serves both arms: with the hold (1) the confused-deputy cases must PASS; without it (0,
 * the NEGATIVE CONTROL) the recycle case must read LANDED, or the test has proved nothing.
 *
 *   recycle     thread A blocks in a DIRECT read() on a pipe into X+64 (X: its own mmap arena);
 *               main unmaps X (munmap -> CV_RETIRE -> CSRETIRE), maps Y (the launcher reuses the
 *               range: same address), fills Y with 0xAA, then writes 256 bytes into the pipe.
 *               hold=1: munmap must answer EBUSY, Y must lie elsewhere, no byte of Y may change,
 *               the read must complete into X (still held, still readable through its capability),
 *               and a later munmap of X must succeed (the hold was released).
 *               hold=0: expected munmap 0, Y at X's address, and the pipe's bytes IN Y (LANDED).
 *   span        a DIRECT read into a buffer that crosses from one mapping into the next one: refused
 *               (EFAULT), nothing consumed from the pipe, nothing written.
 *   outside     DIRECT reads into an unmapped address and into the image's text: refused.
 *   stale       a DIRECT read naming a buffer whose mapping was unmapped before the call: refused
 *               while the range is free (stale-free); once the range is recycled into a new mapping
 *               (stale-recycled) the module, which checks the ARENA and not the capability, accepts
 *               it and the bytes land in the new mapping. That is the limit section 2.4 of the
 *               report states, shown, not passed: verdict LANDED is the expected reading. The
 *               libc-side capability check at submission (io_buffer_ok) is printed beside it.
 *   heap-object thread A blocks in a DIRECT read into a malloc'd object B; main frees B and
 *               mallocs B2 of the same size; then the pipe is written. The hold keeps the heap
 *               ARENA, not the object: where B2 overlaps B the bytes land in B2 (LANDED, the same
 *               limit inside one arena), and the rest overwrite whatever the allocator keeps there
 *               (its in-band slot header: the final free(B2) may then stop the program in the
 *               allocator's own check, cause 24 at its a_crash store).
 *   heap-libc   the same with the libc's own blocking read() (exchange path, served in-module or by
 *               the launcher): the copy-back goes through B's revoked capability and faults (cause
 *               25) in thread A -- the use-after-free is caught, nothing lands. Runs on the control
 *               platform too (no module parameter needed).
 * Build with the SDK (-lpthread); run with two supervisor slots. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include "inline-io.h"

#define PAGE 4096UL
static int hold = -1;
static int failures;

static unsigned long at(const void *p) { return __builtin_capstone_cap_get_cursor((void *)p); }
static void nap(long ms)
{
    struct timespec t = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&t, 0);
}
static int module_param(const char *name)
{
    char path[128], text[32] = {0};
    snprintf(path, sizeof path, "/sys/module/capstone_vm/parameters/%s", name);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    long n = read(fd, text, sizeof text - 1);
    close(fd);
    if (n <= 0) return -1;
    return text[0] == 'Y' || text[0] == '1' ? 1 : text[0] == 'N' || text[0] == '0' ? 0 : atoi(text);
}
static long counter(const char *name)
{
    char path[128], text[32] = {0};
    snprintf(path, sizeof path, "/sys/module/capstone_vm/parameters/%s", name);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    long n = read(fd, text, sizeof text - 1);
    close(fd);
    return n > 0 ? atol(text) : -1;
}
static void verdict(const char *name, int ok, const char *what)
{
    if (!ok) ++failures;
    printf("INLINE_IO case=%s hold=%d verdict=%s %s\n", name, hold, ok ? "PASS" : "FAIL", what);
}
static size_t changed(const unsigned char *p, size_t n, unsigned char fill)
{
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) k += p[i] != fill;
    return k;
}
static void pattern(unsigned char *p, size_t n) { for (size_t i = 0; i < n; ++i) p[i] = (unsigned char)(0x5c ^ i); }

struct blocked { struct io_transport t; int fd; void *buffer; size_t bytes; long result; };
static void *blocked_read(void *opaque)
{
    struct blocked *b = opaque;
    b->result = io_direct(&b->t, 63, b->fd, b->buffer, b->bytes, 0);
    return 0;
}

static void case_recycle(void)
{
    struct blocked b = {0};
    int p[2];
    unsigned char data[256], *x, *y;
    pthread_t a;
    if (pipe(p) || io_transport_init(&b.t)) { verdict("recycle", 0, "setup"); return; }
    x = mmap(0, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (x == MAP_FAILED) { verdict("recycle", 0, "mmap X"); return; }
    memset(x, 0x11, PAGE);
    b.fd = p[0]; b.buffer = x + 64; b.bytes = sizeof data;
    if (pthread_create(&a, 0, blocked_read, &b)) { verdict("recycle", 0, "pthread_create"); return; }
    nap(500);                                  /* A is inside the module's pipe_read by now */
    int um = munmap(x, PAGE), um_errno = um ? errno : 0;
    y = mmap(0, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (y == MAP_FAILED) { verdict("recycle", 0, "mmap Y"); return; }
    memset(y, 0xAA, PAGE);
    pattern(data, sizeof data);
    long w = write(p[1], data, sizeof data);
    pthread_join(a, 0);
    size_t in_y = changed(y, PAGE, 0xAA);
    int same = at(y) == at(x);
    char what[320];
    if (hold == 1) {
        int x_ok = um && um_errno == EBUSY && !memcmp(x + 64, data, sizeof data);
        int later = munmap(x, PAGE);
        snprintf(what, sizeof what, "munmap=%d errno=%d y_at_x=%d write=%ld read=%ld bytes_in_y=%zu "
                 "x_has_data=%d later_munmap=%d", um, um_errno, same, w, b.result, in_y, x_ok, later);
        verdict("recycle", x_ok && !same && in_y == 0 && w == 256 && b.result == 256 && later == 0, what);
    } else {
        snprintf(what, sizeof what, "munmap=%d errno=%d y_at_x=%d write=%ld read=%ld bytes_in_y=%zu "
                 "y_matches_pipe=%d", um, um_errno, same, w, b.result, in_y, !memcmp(y + 64, data, sizeof data));
        /* the NEGATIVE CONTROL: the write must land in Y, exactly where X+64 was and nowhere
         * else, or this test cannot see the defect. (A byte of the pattern may equal Y's fill, so
         * the landing is the comparison, not the count.) */
        size_t outside = changed(y, 64, 0xAA) + changed(y + 64 + sizeof data, PAGE - 64 - sizeof data, 0xAA);
        int landed = um == 0 && same && outside == 0 && !memcmp(y + 64, data, sizeof data);
        if (!landed) ++failures;
        printf("INLINE_IO case=recycle hold=0 verdict=%s %s\n", landed ? "LANDED" : "CONTROL_DID_NOT_FIRE", what);
    }
    munmap(y, PAGE);
    close(p[0]); close(p[1]);
}

static void case_span(struct io_transport *t)
{
    /* Two ADJACENT mappings: the launcher reserves 2 x size and trims the upper half, so an 8 KiB
     * mapping leaves an 8 KiB hole above it that the next 4 KiB request's 8 KiB reservation fits
     * exactly (top-down mmap): that mapping starts where the 8 KiB one ends. Tried a few times. */
    enum { TRIES = 6 };
    unsigned char *c8[TRIES], *c4[TRIES];
    int p[2], pair = -1;
    unsigned char data[200], back[200];
    char what[256];
    for (int i = 0; i < TRIES; ++i) {
        c8[i] = mmap(0, 2 * PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        c4[i] = mmap(0, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (c8[i] == MAP_FAILED || c4[i] == MAP_FAILED) { verdict("span", 0, "mmap"); return; }
        memset(c8[i], 0x33, 2 * PAGE); memset(c4[i], 0x33, PAGE);
        if (pair < 0 && at(c4[i]) == at(c8[i]) + 2 * PAGE) pair = i;
    }
    if (pipe(p)) { verdict("span", 0, "pipe"); return; }
    pattern(data, sizeof data);
    if (write(p[1], data, sizeof data) != (long)sizeof data) { verdict("span", 0, "fill"); return; }
    if (pair >= 0) {
        /* 100 bytes before the end of the 8 KiB mapping, 200 long: half in the 4 KiB one. */
        long r = io_direct_at(t, 63, p[0], at(c8[pair]) + 2 * PAGE - 100, sizeof data, 0);
        long again = read(p[0], back, sizeof back);
        size_t touched = changed(c8[pair], 2 * PAGE, 0x33) + changed(c4[pair], PAGE, 0x33);
        snprintf(what, sizeof what, "adjacent=1 at=0x%lx|0x%lx result=%ld bytes_written=%zu pipe_still_full=%d",
                 at(c8[pair]), at(c4[pair]), r, touched,
                 again == (long)sizeof data && !memcmp(back, data, sizeof data));
        verdict("span", r == -EFAULT && touched == 0 && again == (long)sizeof data, what);
    } else {
        snprintf(what, sizeof what, "adjacent=0 (no pair of %d was adjacent; not run)", TRIES);
        verdict("span", 0, what);
    }
    for (int i = 0; i < TRIES; ++i) { munmap(c8[i], 2 * PAGE); munmap(c4[i], PAGE); }
    close(p[0]); close(p[1]);
}

static void case_outside(struct io_transport *t)
{
    int p[2];
    unsigned char data[64];
    if (pipe(p)) { verdict("outside", 0, "pipe"); return; }
    pattern(data, sizeof data);
    if (write(p[1], data, sizeof data) != (long)sizeof data) { verdict("outside", 0, "fill"); return; }
    /* an address nothing is mapped at, and the image's own read+execute text */
    long r1 = io_direct_at(t, 63, p[0], 0x10000UL, 16, 0);
    long r2 = io_direct_at(t, 63, p[0], at((void *)&case_outside), 16, 0);
    unsigned char back[64];
    long again = read(p[0], back, sizeof back);
    char what[200];
    snprintf(what, sizeof what, "unmapped=%ld text=%ld pipe_still_full=%d", r1, r2,
             again == (long)sizeof data && !memcmp(back, data, sizeof data));
    verdict("outside", r1 == -EFAULT && r2 == -EFAULT && again == (long)sizeof data, what);
    close(p[0]); close(p[1]);
}

static void case_stale(struct io_transport *t)
{
    int p[2];
    unsigned char data[128], back[128];
    char what[256];
    if (pipe(p)) { verdict("stale", 0, "pipe"); return; }
    pattern(data, sizeof data);
    unsigned char *x = mmap(0, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (x == MAP_FAILED) { verdict("stale", 0, "mmap"); return; }
    memset(x, 0x22, PAGE);
    if (munmap(x, PAGE)) { verdict("stale", 0, "munmap"); return; }
    int libc_ok = io_buffer_ok(x, sizeof data, 2);
    if (write(p[1], data, sizeof data) != (long)sizeof data) { verdict("stale", 0, "fill"); return; }
    long r = io_direct(t, 63, p[0], x, sizeof data, 0);       /* the range is free: no arena */
    long again = read(p[0], back, sizeof back);
    snprintf(what, sizeof what, "result=%ld libc_check_at_submission=%d pipe_still_full=%d", r, libc_ok,
             again == (long)sizeof data);
    printf("INLINE_IO case=stale-free hold=%d verdict=%s %s\n", hold,
           r == -EFAULT && again == (long)sizeof data ? "PASS" : "FAIL", what);
    if (!(r == -EFAULT && again == (long)sizeof data)) ++failures;
    /* now the range recycled: a new mapping at the same address */
    unsigned char *y = mmap(0, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (y == MAP_FAILED) { verdict("stale", 0, "mmap Y"); return; }
    memset(y, 0xAA, PAGE);
    if (write(p[1], data, sizeof data) != (long)sizeof data) { verdict("stale", 0, "fill 2"); return; }
    r = io_direct(t, 63, p[0], x, sizeof data, 0);
    size_t in_y = changed(y, PAGE, 0xAA);
    snprintf(what, sizeof what, "y_at_x=%d result=%ld bytes_in_y=%zu libc_check_at_submission=%d",
             at(y) == at(x), r, in_y, io_buffer_ok(x, sizeof data, 2));
    /* Not a pass/fail of the module: the reading of the arena-granularity limit. */
    printf("INLINE_IO case=stale-recycled hold=%d verdict=%s %s\n", hold,
           at(y) != at(x) ? "NOT_RECYCLED" : in_y ? "LANDED" : "REFUSED", what);
    munmap(y, PAGE);
    close(p[0]); close(p[1]);
}

static void case_heap_object(void)
{
    struct blocked b = {0};
    int p[2];
    unsigned char data[256];
    pthread_t a;
    char what[256];
    if (pipe(p) || io_transport_init(&b.t)) { verdict("heap-object", 0, "setup"); return; }
    unsigned char *obj = malloc(sizeof data);
    if (!obj) { verdict("heap-object", 0, "malloc"); return; }
    memset(obj, 0x44, sizeof data);
    unsigned long obj_at = at(obj);
    b.fd = p[0]; b.buffer = obj; b.bytes = sizeof data;
    if (pthread_create(&a, 0, blocked_read, &b)) { verdict("heap-object", 0, "pthread_create"); return; }
    nap(500);
    printf("INLINE_IO case=heap-object note=freeing_under_a_blocked_direct_read\n");
    fflush(stdout);
    free(obj);
    printf("INLINE_IO case=heap-object note=freed\n");
    fflush(stdout);
    unsigned char *obj2 = malloc(sizeof data);
    if (!obj2) { verdict("heap-object", 0, "malloc 2"); return; }
    memset(obj2, 0xAA, sizeof data);
    pattern(data, sizeof data);
    printf("INLINE_IO case=heap-object note=reallocated same_address=%d obj=0x%lx obj2=0x%lx\n",
           at(obj2) == obj_at, obj_at, at(obj2));
    fflush(stdout);
    long w = write(p[1], data, sizeof data);
    pthread_join(a, 0);
    printf("INLINE_IO case=heap-object note=joined write=%ld read=%ld\n", w, b.result);
    fflush(stdout);
    size_t in2 = changed(obj2, sizeof data, 0xAA);
    int same = at(obj2) == obj_at;
    snprintf(what, sizeof what, "same_address=%d new_object_at=%+ld write=%ld read=%ld bytes_in_new_object=%zu",
             same, (long)(at(obj2) - obj_at), w, b.result, in2);
    printf("INLINE_IO case=heap-object hold=%d verdict=%s %s\n", hold,
           in2 ? "LANDED" : same ? "UNTOUCHED" : "MOVED", what);
    fflush(stdout);
    printf("INLINE_IO case=heap-object note=freeing_obj2\n");
    fflush(stdout);
    free(obj2);
    close(p[0]); close(p[1]);
}

/* Localisation variants of heap-object (not part of the verdict): the same free with no I/O in
 * flight (a thread only sleeping), after a DIRECT read that completed, and with the libc's own
 * blocking read in the thread -- today's path, whose copy-back goes through the capability. */
static void *sleeper(void *opaque) { (void)opaque; nap(1500); return 0; }
struct libc_blocked { int fd; void *buffer; size_t bytes; long result; };
static void *libc_read(void *opaque)
{
    struct libc_blocked *b = opaque;
    b->result = read(b->fd, b->buffer, b->bytes);
    return 0;
}
static void case_heap_variants(const char *which, struct io_transport *t)
{
    unsigned char data[256];
    int p[2];
    pthread_t a;
    pattern(data, sizeof data);
    if (pipe(p)) { verdict(which, 0, "pipe"); return; }
    unsigned char *obj = malloc(sizeof data);
    memset(obj, 0x44, sizeof data);
    if (!strcmp(which, "heap-sleep")) {
        pthread_create(&a, 0, sleeper, 0);
        nap(300);
        free(obj);
        pthread_join(a, 0);
        printf("INLINE_IO case=%s hold=%d verdict=PASS freed_with_a_sleeping_thread=1\n", which, hold);
    } else if (!strcmp(which, "heap-done")) {
        long w = write(p[1], data, sizeof data);
        long r = io_direct(t, 63, p[0], obj, sizeof data, 0);
        int ok = !memcmp(obj, data, sizeof data);
        free(obj);
        printf("INLINE_IO case=%s hold=%d verdict=%s write=%ld read=%ld data_ok=%d freed=1\n", which, hold,
               ok && r == 256 ? "PASS" : "FAIL", w, r, ok);
    } else {
        struct libc_blocked b = {p[0], obj, sizeof data, 0};
        pthread_create(&a, 0, libc_read, &b);
        nap(500);
        free(obj);
        unsigned char *obj2 = malloc(sizeof data);
        memset(obj2, 0xAA, sizeof data);
        printf("INLINE_IO case=%s hold=%d note=freed_under_a_blocked_libc_read same_address=%d\n", which, hold,
               at(obj2) == at(obj));
        fflush(stdout);
        long w = write(p[1], data, sizeof data);
        printf("INLINE_IO case=%s note=pipe_written write=%ld\n", which, w);
        fflush(stdout);
        pthread_join(a, 0);
        printf("INLINE_IO case=%s note=joined\n", which);
        fflush(stdout);
        printf("INLINE_IO case=%s hold=%d verdict=%s write=%ld read=%ld bytes_in_new_object=%zu\n", which, hold,
               changed(obj2, sizeof data, 0xAA) ? "LANDED" : "UNTOUCHED", w, b.result,
               changed(obj2, sizeof data, 0xAA));
    }
    close(p[0]); close(p[1]);
}

int main(int argc, char **argv)
{
    struct io_transport t;
    hold = module_param("inline_hold");
    int enabled = module_param("in_module_io");
    printf("INLINE_IO_START hold=%d in_module_io=%d\n", hold, enabled);
    const char *only = argc > 1 ? argv[1] : "";
    /* The libc-only variants also run on a module without in-module I/O (the control platform). */
    int libc_only = !strcmp(only, "heap-libc") || !strcmp(only, "heap-sleep");
    if (libc_only && enabled != 1) {
        case_heap_variants(only, &t);
        printf("INLINE_IO_DONE failures=%d\n", failures);
        return failures ? 1 : 0;
    }
    if (hold < 0 || enabled != 1 || io_transport_init(&t)) {
        printf("INLINE_IO_DONE failures=1 (module parameters unreadable or in-module I/O off)\n");
        return 1;
    }
    if (!*only || !strcmp(only, "recycle")) case_recycle();
    if (!*only || !strcmp(only, "span")) case_span(&t);
    if (!*only || !strcmp(only, "outside")) case_outside(&t);
    if (!*only || !strcmp(only, "stale")) case_stale(&t);
    if (!*only || !strcmp(only, "heap-object")) case_heap_object();
    if (!strncmp(only, "heap-", 5) && strcmp(only, "heap-object")) case_heap_variants(only, &t);
    printf("INLINE_IO_COUNTERS served_v1=%ld served_direct=%ld answered=%ld fallback=%ld "
           "hold_refusals=%ld stopped=%ld\n", counter("inline_served_v1"), counter("inline_served_direct"),
           counter("inline_answered"), counter("inline_fallback"), counter("inline_hold_refusals"),
           counter("inline_stopped"));
    printf("INLINE_IO_DONE failures=%d\n", failures);
    fflush(stdout);
    return failures ? 1 : 0;
}
