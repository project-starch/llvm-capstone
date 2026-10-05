/* B3c (docs/plans/b0-silicon-delegated-runtime.md): the board's clocks, and the oracle harness's
 * `stop_seconds=inf`. A NATIVE program (the board's glibc), run by b0run.sh's b3-clock rung.
 *
 * mc-harness computes stop_seconds as t1 - t0, both from now() = tv_sec + tv_nsec / 1e9 on CLOCK_MONOTONIC, and
 * keeps t0 in a callee-saved FP register across kill + waitpid of the memcached domain. No integer clock reading
 * converts to inf, so either an FP operation misbehaved or the register did not survive the wait. Arms:
 *   1 clocks       CLOCK_REALTIME and CLOCK_MONOTONIC as raw integers, twice across a 1 s sleep.
 *   2 arith        200,000 evaluations of now()'s shape on live readings; results that are not finite or go
 *                  backwards are counted, and the first one's operands are printed in hex.
 *   3 selftest     arm 4's instrument with fs5 changed on purpose: it must report exactly 1 changed register.
 *   4 hold-native  known values in fs0-fs11 across waitpid of a native child (`sleep 1`); changed ones counted.
 *   5 hold-domain  the harness's sequence: memcached under capstone-job + capstone-exec, 3 s, then t0, SIGTERM,
 *                  waitpid with fs0-fs11 held, t1; stop_seconds printed as the harness prints it, with t0/t1 in hex.
 * Exit status is a bitmask: 1 arith anomaly, 2 native hold changed, 4 domain hold changed, 8 a clock call failed,
 * 16 t1 - t0 not finite in arm 5, 32 the selftest did not fire. */
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int clock_failed;

static __attribute__((noinline)) double now(void)
{
	struct timespec t;
	if (clock_gettime(CLOCK_MONOTONIC, &t)) clock_failed = 1;
	return t.tv_sec + t.tv_nsec / 1e9;
}

static uint64_t bits(double d) { uint64_t u; memcpy(&u, &d, sizeof u); return u; }

static void clocks(const char *when)
{
	struct timespec r, m;
	int rr = clock_gettime(CLOCK_REALTIME, &r), rm = clock_gettime(CLOCK_MONOTONIC, &m);
	if (rr || rm) clock_failed = 1;
	printf("B3c clocks %s: realtime rc %d %lld.%09ld monotonic rc %d %lld.%09ld\n", when, rr, (long long)r.tv_sec,
	       r.tv_nsec, rm, (long long)m.tv_sec, m.tv_nsec);
}

static int arith(void)
{
	int bad = 0;
	double prev = now();
	for (int i = 0; i < 200000; i++) {
		struct timespec t;
		if (clock_gettime(CLOCK_MONOTONIC, &t)) clock_failed = 1;
		volatile double scale = 1e9;
		double d = t.tv_sec + t.tv_nsec / scale;
		if (!isfinite(d) || d < prev) {
			if (!bad)
				printf("B3c arith first anomaly at %d: sec %lld nsec %ld scale %016llx -> %016llx (prev %016llx)\n", i,
				       (long long)t.tv_sec, t.tv_nsec, (unsigned long long)bits(scale),
				       (unsigned long long)bits(d), (unsigned long long)bits(prev));
			bad++;
		}
		prev = d;
	}
	printf("B3c arith: %d anomalies of 200000\n", bad);
	return bad != 0;
}

/* hold_wait(pid, in[12], out[13], selftest): load in[] into fs0-fs11 (the callee-saved FP registers), waitpid(pid),
 * store fs0-fs11 to out[0..11] and the child's status to out[12]. The caller's fs0-fs11 are saved and restored around
 * it. In assembly because a C version cannot pin the registers: GCC spilled the held values to the stack, so only
 * fs0 was ever live across the wait. selftest != 0 changes fs5 after the load, as a positive control. */
long hold_wait(long pid, const uint64_t *in, uint64_t *out, long selftest);
__asm__(
".text\n"
".globl hold_wait\n"
".type hold_wait, @function\n"
"hold_wait:\n"
"  addi sp, sp, -128\n"
"  sd ra, 0(sp)\n"
"  sd s0, 8(sp)\n"
"  sd s1, 16(sp)\n"
"  fsd fs0, 24(sp)\n"
"  fsd fs1, 32(sp)\n"
"  fsd fs2, 40(sp)\n"
"  fsd fs3, 48(sp)\n"
"  fsd fs4, 56(sp)\n"
"  fsd fs5, 64(sp)\n"
"  fsd fs6, 72(sp)\n"
"  fsd fs7, 80(sp)\n"
"  fsd fs8, 88(sp)\n"
"  fsd fs9, 96(sp)\n"
"  fsd fs10, 104(sp)\n"
"  fsd fs11, 112(sp)\n"
"  mv s0, a2\n"
"  mv s1, a3\n"
"  fld fs0, 0(a1)\n"
"  fld fs1, 8(a1)\n"
"  fld fs2, 16(a1)\n"
"  fld fs3, 24(a1)\n"
"  fld fs4, 32(a1)\n"
"  fld fs5, 40(a1)\n"
"  fld fs6, 48(a1)\n"
"  fld fs7, 56(a1)\n"
"  fld fs8, 64(a1)\n"
"  fld fs9, 72(a1)\n"
"  fld fs10, 80(a1)\n"
"  fld fs11, 88(a1)\n"
"  beqz s1, 1f\n"
"  fadd.d fs5, fs5, fs5          /* the self-test: one register changed on purpose */\n"
"1:\n"
"  addi a1, sp, 120\n"
"  li a2, 0\n"
"  call waitpid\n"
"  fsd fs0, 0(s0)\n"
"  fsd fs1, 8(s0)\n"
"  fsd fs2, 16(s0)\n"
"  fsd fs3, 24(s0)\n"
"  fsd fs4, 32(s0)\n"
"  fsd fs5, 40(s0)\n"
"  fsd fs6, 48(s0)\n"
"  fsd fs7, 56(s0)\n"
"  fsd fs8, 64(s0)\n"
"  fsd fs9, 72(s0)\n"
"  fsd fs10, 80(s0)\n"
"  fsd fs11, 88(s0)\n"
"  lw a1, 120(sp)\n"
"  sw a1, 96(s0)                 /* out[12] low word: the child's status */\n"
"  fld fs0, 24(sp)\n"
"  fld fs1, 32(sp)\n"
"  fld fs2, 40(sp)\n"
"  fld fs3, 48(sp)\n"
"  fld fs4, 56(sp)\n"
"  fld fs5, 64(sp)\n"
"  fld fs6, 72(sp)\n"
"  fld fs7, 80(sp)\n"
"  fld fs8, 88(sp)\n"
"  fld fs9, 96(sp)\n"
"  fld fs10, 104(sp)\n"
"  fld fs11, 112(sp)\n"
"  ld ra, 0(sp)\n"
"  ld s0, 8(sp)\n"
"  ld s1, 16(sp)\n"
"  addi sp, sp, 128\n"
"  ret\n"
".size hold_wait, .-hold_wait\n"
);

static int hold(const char *arm, char *const argv[], int term_after_s, int selftest)
{
	uint64_t in[12], out[13];
	for (int i = 0; i < 12; i++) in[i] = bits(now() + 1000 * i);
	pid_t pid = fork();
	if (pid == 0) {
		execv(argv[0], argv);
		_exit(127);
	}
	double t0 = 0, t1 = 0;
	if (term_after_s) {
		sleep(term_after_s);
		t0 = now();
		kill(pid, SIGTERM);
	}
	hold_wait(pid, in, out, selftest);
	if (term_after_s) t1 = now();
	int changed = 0;
	for (int i = 0; i < 12; i++)
		if (out[i] != in[i]) {
			if (changed < 4)
				printf("B3c %s: fs%d %016llx, was %016llx\n", arm, i, (unsigned long long)out[i],
				       (unsigned long long)in[i]);
			changed++;
		}
	printf("B3c %s: child status %d, %d of 12 callee-saved FP registers changed across the wait\n", arm,
	       (int)(uint32_t)out[12], changed);
	if (term_after_s) {
		printf("B3c %s: stop_seconds=%.2f t0 %016llx t1 %016llx\n", arm, t1 - t0, (unsigned long long)bits(t0),
		       (unsigned long long)bits(t1));
		if (!isfinite(t1 - t0)) changed |= 1 << 16;
	}
	return changed;
}

int main(int argc, char **argv)
{
	int rc = 0;
	setvbuf(stdout, NULL, _IONBF, 0);
	clocks("first");
	sleep(1);
	clocks("after 1 s");
	if (arith()) rc |= 1;
	char *native[] = {"/bin/sleep", "1", NULL};
	if (hold("selftest", native, 0, 1) != 1) rc |= 32;
	if (hold("hold-native", native, 0, 0)) rc |= 2;
	if (argc > 1) {
		int h = hold("hold-domain", argv + 1, 3, 0);
		if (h & 0xffff) rc |= 4;
		if (h >> 16) rc |= 16;
	}
	if (clock_failed) rc |= 8;
	printf("B3c probe rc=%d\n", rc);
	return rc;
}
