/* B3f (docs/plans/b0-silicon-delegated-runtime.md): where do FP registers loaded with `fld` lose their values?
 * A NATIVE program (the board's glibc), run by b0run.sh's b3-fphold rung.
 *
 * On caplifive_r51r52_0568f93a9 (boot C, 2026-10-11), b3-clock-probe.c's hold_wait read the CALLER's fs0..fs11 back
 * after `waitpid`, not the twelve values it had just loaded with `fld`, 12 of 12 in all three of its arms. Yet
 * mc-harness's t0 (an `fadd` result) survived the same kind of wait in fs0. Each variant below changes ONE thing
 * between "load twelve values" and "store them back", so the first variant to read the caller's values names the
 * step that loses them:
 *   0 nothing in between (fld then fsd at once)
 *   1 a plain function call (no trap)
 *   2 getppid (a trap, no blocking, normally no context switch)
 *   3 sched_yield (a trap that may switch)
 *   4 waitpid on a child that sleeps 1 s (blocks; the original arm)
 *   5 as 4, but 32 nops between the save-fsds and the load-flds (the back-to-back fsd/fld of the same FPRs)
 *   6 as 4, but the twelve values are COMPUTED by fadd from one fld, not loaded (mc-harness's case)
 *   8 as 4, but the child SPINS for 1 s instead of sleeping (the hart never idles)
 *   9 as 4, with getppid between the loads and waitpid (mc-harness has kill() there)
 *  10 no child: the parent itself nanosleeps 1 s (blocks and idles, no waitpid)
 *  11 no child: the parent busy-loops 1 s with no syscall (timer interrupts only, nothing to switch to)
 *   7 the POSITIVE CONTROL: as 0, with fs5 doubled by fadd before the store. It must read exactly 1 of 12 changed;
 *     anything else means the instrument cannot be trusted, and sets bit 7.
 * Each variant prints `B3f v<n>: <k> of 12 changed` and the first changed value. Exit status is a bitmask, bit n =
 * variant n read something other than its expectation (0 changed; exactly 1 for variant 7). 0 = all as expected. */
#include <sched.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* fphold(in[12], out[12], mode, pid): load in[] into fs0..fs11 (or compute from in[0] for mode 6), do the mode's
 * step, store fs0..fs11 to out[]. The caller's fs0..fs11 are saved and restored around it. */
long fphold(const uint64_t *in, uint64_t *out, long mode, long pid);
long dummy_fn(long x) { return x + 1; }
/* Integer-only helpers (no FP), so the callee-saved FPRs stay the ABI's: whatever they read after is not theirs. */
static long mono_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000L + t.tv_nsec / 1000000L; }
void nap1s(void) { struct timespec t = {1, 0}; nanosleep(&t, NULL); }
void spin1s(void) { long end = mono_ms() + 1000; while (mono_ms() < end) { } }
__asm__(
".text\n.globl fphold\n.type fphold, @function\nfphold:\n"
"  addi sp, sp, -160\n  sd ra, 0(sp)\n  sd s0, 8(sp)\n  sd s1, 16(sp)\n  sd s2, 128(sp)\n"
"  fsd fs0, 24(sp)\n  fsd fs1, 32(sp)\n  fsd fs2, 40(sp)\n  fsd fs3, 48(sp)\n  fsd fs4, 56(sp)\n  fsd fs5, 64(sp)\n"
"  fsd fs6, 72(sp)\n  fsd fs7, 80(sp)\n  fsd fs8, 88(sp)\n  fsd fs9, 96(sp)\n  fsd fs10, 104(sp)\n  fsd fs11, 112(sp)\n"
"  mv s0, a1\n  mv s1, a2\n  mv s2, a3\n"
"  li t0, 5\n  bne s1, t0, 2f\n"
"  .rept 32\n  nop\n  .endr\n"
"2:li t0, 6\n  bne s1, t0, 3f\n"
"  fld fs0, 0(a0)\n  fadd.d fs1, fs0, fs0\n  fadd.d fs2, fs1, fs0\n  fadd.d fs3, fs2, fs0\n  fadd.d fs4, fs3, fs0\n"
"  fadd.d fs5, fs4, fs0\n  fadd.d fs6, fs5, fs0\n  fadd.d fs7, fs6, fs0\n  fadd.d fs8, fs7, fs0\n  fadd.d fs9, fs8, fs0\n"
"  fadd.d fs10, fs9, fs0\n  fadd.d fs11, fs10, fs0\n"
"  fsd fs0, 0(s0)\n  fsd fs1, 8(s0)\n  fsd fs2, 16(s0)\n  fsd fs3, 24(s0)\n  fsd fs4, 32(s0)\n  fsd fs5, 40(s0)\n"
"  fsd fs6, 48(s0)\n  fsd fs7, 56(s0)\n  fsd fs8, 64(s0)\n  fsd fs9, 72(s0)\n  fsd fs10, 80(s0)\n  fsd fs11, 88(s0)\n"
"  addi s0, s0, 96\n  j 4f\n"
"3:fld fs0, 0(a0)\n  fld fs1, 8(a0)\n  fld fs2, 16(a0)\n  fld fs3, 24(a0)\n  fld fs4, 32(a0)\n  fld fs5, 40(a0)\n"
"  fld fs6, 48(a0)\n  fld fs7, 56(a0)\n  fld fs8, 64(a0)\n  fld fs9, 72(a0)\n  fld fs10, 80(a0)\n  fld fs11, 88(a0)\n"
"4:li t0, 7\n  bne s1, t0, 8f\n  fadd.d fs5, fs5, fs5\n  j 9f\n"
"8:beqz s1, 9f\n"
"  li t0, 1\n  bne s1, t0, 5f\n  li a0, 41\n  call dummy_fn\n  j 9f\n"
"5:li t0, 2\n  bne s1, t0, 6f\n  call getppid\n  j 9f\n"
"6:li t0, 3\n  bne s1, t0, 7f\n  call sched_yield\n  j 9f\n"
"7:li t0, 10\n  bne s1, t0, 70f\n  call nap1s\n  j 9f\n"
"70:li t0, 11\n  bne s1, t0, 71f\n  call spin1s\n  j 9f\n"
"71:li t0, 9\n  bne s1, t0, 72f\n  call getppid\n"
"72:mv a0, s2\n  addi a1, sp, 120\n  li a2, 0\n  call waitpid\n"
"9:fsd fs0, 0(s0)\n  fsd fs1, 8(s0)\n  fsd fs2, 16(s0)\n  fsd fs3, 24(s0)\n  fsd fs4, 32(s0)\n  fsd fs5, 40(s0)\n"
"  fsd fs6, 48(s0)\n  fsd fs7, 56(s0)\n  fsd fs8, 64(s0)\n  fsd fs9, 72(s0)\n  fsd fs10, 80(s0)\n  fsd fs11, 88(s0)\n"
"  fld fs0, 24(sp)\n  fld fs1, 32(sp)\n  fld fs2, 40(sp)\n  fld fs3, 48(sp)\n  fld fs4, 56(sp)\n  fld fs5, 64(sp)\n"
"  fld fs6, 72(sp)\n  fld fs7, 80(sp)\n  fld fs8, 88(sp)\n  fld fs9, 96(sp)\n  fld fs10, 104(sp)\n  fld fs11, 112(sp)\n"
"  ld ra, 0(sp)\n  ld s0, 8(sp)\n  ld s1, 16(sp)\n  ld s2, 128(sp)\n  addi sp, sp, 160\n  ret\n"
".size fphold, .-fphold\n");

static uint64_t bits(double d) { uint64_t u; memcpy(&u, &d, sizeof u); return u; }

int main(void)
{
	int rc = 0;
	setvbuf(stdout, NULL, _IONBF, 0);
	for (long mode = 0; mode <= 11; mode++) {
		uint64_t in[12], out[24], want[12];
		/* Distinct, non-trivial patterns, and none equal to anything a caller register is likely to hold. */
		for (int i = 0; i < 12; i++) in[i] = 0x4010000000000000ull + 0x0000010000000000ull * (uint64_t)(i + 1) + (uint64_t)mode;
		memset(out, 0, sizeof out);
		pid_t pid = 0;
		if (mode == 4 || mode == 5 || mode == 6 || mode == 9) {
			pid = fork();
			if (pid == 0) { execl("/bin/sleep", "sleep", "1", (char *)NULL); _exit(127); }
		}
		if (mode == 8) {
			pid = fork();
			if (pid == 0) { spin1s(); _exit(0); }
		}
		fphold(in, out, mode, pid);
		/* What should come back: the loaded values, or for mode 6 the fadd chain from in[0] (computed again here, in
		 * the same arithmetic, so a wrong FPU result would show as a mismatch in mode 6 only). */
		if (mode == 6) {
			double x = 0, v[12];
			memcpy(&x, &in[0], sizeof x);
			v[0] = x;
			for (int i = 1; i < 12; i++) v[i] = v[i - 1] + x;
			for (int i = 0; i < 12; i++) want[i] = bits(v[i]);
		} else {
			memcpy(want, in, sizeof want);
		}
		const uint64_t *got = mode == 6 ? out + 12 : out;
		int changed = 0, first = -1;
		for (int i = 0; i < 12; i++)
			if (got[i] != want[i]) { if (first < 0) first = i; changed++; }
		if (first >= 0)
			printf("B3f v%ld: %d of 12 changed; first fs%d %016llx, want %016llx\n", mode, changed, first,
			       (unsigned long long)got[first], (unsigned long long)want[first]);
		else
			printf("B3f v%ld: 0 of 12 changed\n", mode);
		if (mode == 6) {
			int pre = 0;
			for (int i = 0; i < 12; i++) pre += out[i] != want[i];
			printf("B3f v6: before the wait %d of 12 differ from the fadd chain\n", pre);
		}
		if (mode == 7 ? changed != 1 : changed != 0) rc |= 1 << mode;
	}
	printf("B3f probe rc=%d\n", rc);
	return rc;
}
