/* Shared measurement harness of the virtual-profile microbenchmarks (CoreMark-derived, RV8, BEEBS).
 *
 * One process runs W warm-up iterations and then I timed iterations of the program's whole
 * benchmark unit (W and I from argv[1] and argv[2], defaults 1 and 5). Each iteration is
 * bracketed by bench_mark() (the perf module's cumulative cycle and instret counters) and one
 * clock_gettime(CLOCK_MONOTONIC) at each end, and prints, outside the bracket:
 *
 *   BENCH <suite>/<name> it=<k> cycles=<d> instret=<d> ns=<d> check=<v>
 *
 * with it=w0.. for warm-ups. The exit status is 0 only if every iteration verified.
 *
 * Counters: /sys/module/capstone_vm/parameters/bench_{cycles,instret}, opened once at start-up
 * (outside timing) and pread from offset 0 at each mark. Absent files (QEMU, a module without
 * them) read as 0/0 and the harness prints `BENCH-INFO <id> counters=absent` once.
 *
 * Counting build (-DBENCH_COUNT_ALLOCS, linked with --wrap=malloc,free,realloc,calloc): used ONLY
 * for the QEMU functional pass, never for a measured image. It prints, per iteration and for the
 * whole run, `ALLOC <id> it=<k> malloc=<n> free=<n> realloc=<n> calloc=<n>`. */
#ifndef BENCH_HARNESS_H
#define BENCH_HARNESS_H

/* A unit runs one whole benchmark iteration. It returns nothing; its verdict is read after the
 * bracket by `post`, which stores the check value and returns 1 if the iteration verified. */
typedef void (*bench_unit_fn)(void);
typedef int (*bench_post_fn)(unsigned long long *check);

/* The generic driver: parse argv, open the counters, run W+I bracketed iterations of `unit`,
 * report each, and return the process exit status. */
int bench_run(const char *id, int argc, char **argv, bench_unit_fn unit, bench_post_fn post);

/* For a program whose timed region sits inside its own code (CoreMark's start_time/stop_time):
 * `bench_run_self` does not bracket `unit`; the program calls bench_begin()/bench_end() itself,
 * exactly once each per iteration (the run fails if it does not). */
int bench_run_self(const char *id, int argc, char **argv, bench_unit_fn unit, bench_post_fn post);
void bench_begin(void);
void bench_end(void);
/* The last bracket's elapsed time (after bench_end), for a program that reports it itself. */
long long bench_window_ns(void);

/* FNV-1a over a byte range, for state checks of programs without their own verification. */
unsigned long long bench_fnv(unsigned long long h, const void *p, unsigned long n);
#define BENCH_FNV_INIT 1469598103934665603ULL

#endif
