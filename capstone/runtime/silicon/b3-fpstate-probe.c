/* B3i (docs/plans/b0-silicon-delegated-runtime.md, ISSUES S-19): what native FP state the kernel keeps on silicon.
 * A NATIVE program (the board's glibc), run by b0run.sh's b3-fpstate rung. The audit's discriminators, in one boot:
 * - Before the twelve values I are loaded into fs0..fs11, the registers hold a distinct pattern C. Each register is
 *   classified after the window as I (held), C (the loads themselves undone), 0 (an older kernel copy: main's FPRs
 *   are zero), or ? (anything else). All twelve are printed.
 * - getrusage's voluntary and involuntary context-switch counts around each window, so a switch is counted.
 * - A signal raised right after the loads. Linux copies the FP state into the frame only if the trap-entry sstatus
 *   says FS Dirty (save_fp_state), so a frame of I means FS tracking works, and a frame of 0 means it does not.
 * - A fork right after the loads. The child inherits the FP state through arch_dup_task_struct, also FS-checked, and
 *   dumps its fs0..fs11 into a shared page.
 * Windows:
 *   n  nothing
 *   w  waitpid on a child that sleeps 1 s
 *   l  a pure integer loop (~1 s on the board)
 *   z  nanosleep 10 ms
 *   s  the signal
 *   f  the fork
 *   k  the control (fs5 doubled, nothing else). It must read I except fs5.
 * Exit status: 0 if every window reads all-I (k: all-I but fs5), else 1. The printed lines are the reading. */
#define _GNU_SOURCE
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

static uint64_t I[12], C[12];
static volatile uint64_t g_frame[12];
static volatile int g_frame_set;
static uint64_t *g_shared;

/* fpwin(callerpat, in, out, mode, pid): save the caller's fs0..fs11, load C then I, do the window `mode`, store
 * fs0..fs11 to out[], restore the caller's. Modes: 0 n, 1 w, 2 l, 3 z, 4 s, 5 f, 6 k. */
long fpwin(const uint64_t *c, const uint64_t *in, uint64_t *out, long mode, long pid);
void win_signal(void);
void win_fork(void);
void win_nap10ms(void) { struct timespec t = {0, 10000000}; nanosleep(&t, NULL); }
__asm__(
".text\n.globl fpwin\n.type fpwin, @function\nfpwin:\n"
"  addi sp, sp, -160\n  sd ra, 0(sp)\n  sd s0, 8(sp)\n  sd s1, 16(sp)\n  sd s2, 128(sp)\n"
"  fsd fs0, 24(sp)\n  fsd fs1, 32(sp)\n  fsd fs2, 40(sp)\n  fsd fs3, 48(sp)\n  fsd fs4, 56(sp)\n  fsd fs5, 64(sp)\n"
"  fsd fs6, 72(sp)\n  fsd fs7, 80(sp)\n  fsd fs8, 88(sp)\n  fsd fs9, 96(sp)\n  fsd fs10, 104(sp)\n  fsd fs11, 112(sp)\n"
"  mv s0, a2\n  mv s1, a3\n  mv s2, a4\n"
"  fld fs0, 0(a0)\n  fld fs1, 8(a0)\n  fld fs2, 16(a0)\n  fld fs3, 24(a0)\n  fld fs4, 32(a0)\n  fld fs5, 40(a0)\n"
"  fld fs6, 48(a0)\n  fld fs7, 56(a0)\n  fld fs8, 64(a0)\n  fld fs9, 72(a0)\n  fld fs10, 80(a0)\n  fld fs11, 88(a0)\n"
"  fld fs0, 0(a1)\n  fld fs1, 8(a1)\n  fld fs2, 16(a1)\n  fld fs3, 24(a1)\n  fld fs4, 32(a1)\n  fld fs5, 40(a1)\n"
"  fld fs6, 48(a1)\n  fld fs7, 56(a1)\n  fld fs8, 64(a1)\n  fld fs9, 72(a1)\n  fld fs10, 80(a1)\n  fld fs11, 88(a1)\n"
"  beqz s1, 9f\n"
"  li t0, 1\n  bne s1, t0, 2f\n  mv a0, s2\n  addi a1, sp, 120\n  li a2, 0\n  call waitpid\n  j 9f\n"
"2:li t0, 2\n  bne s1, t0, 3f\n  li t1, 6000000\n4:addi t1, t1, -1\n  bnez t1, 4b\n  j 9f\n"
"3:li t0, 3\n  bne s1, t0, 5f\n  call win_nap10ms\n  j 9f\n"
"5:li t0, 4\n  bne s1, t0, 6f\n  call win_signal\n  j 9f\n"
"6:li t0, 5\n  bne s1, t0, 7f\n  call win_fork\n  j 9f\n"
"7:fadd.d fs5, fs5, fs5\n"
"9:fsd fs0, 0(s0)\n  fsd fs1, 8(s0)\n  fsd fs2, 16(s0)\n  fsd fs3, 24(s0)\n  fsd fs4, 32(s0)\n  fsd fs5, 40(s0)\n"
"  fsd fs6, 48(s0)\n  fsd fs7, 56(s0)\n  fsd fs8, 64(s0)\n  fsd fs9, 72(s0)\n  fsd fs10, 80(s0)\n  fsd fs11, 88(s0)\n"
"  fld fs0, 24(sp)\n  fld fs1, 32(sp)\n  fld fs2, 40(sp)\n  fld fs3, 48(sp)\n  fld fs4, 56(sp)\n  fld fs5, 64(sp)\n"
"  fld fs6, 72(sp)\n  fld fs7, 80(sp)\n  fld fs8, 88(sp)\n  fld fs9, 96(sp)\n  fld fs10, 104(sp)\n  fld fs11, 112(sp)\n"
"  ld ra, 0(sp)\n  ld s0, 8(sp)\n  ld s1, 16(sp)\n  ld s2, 128(sp)\n  addi sp, sp, 160\n  ret\n"
".size fpwin, .-fpwin\n"
/* fpdump(out): store fs0..fs11 to out[], touching nothing else. */
".globl fpdump\n.type fpdump, @function\nfpdump:\n"
"  fsd fs0, 0(a0)\n  fsd fs1, 8(a0)\n  fsd fs2, 16(a0)\n  fsd fs3, 24(a0)\n  fsd fs4, 32(a0)\n  fsd fs5, 40(a0)\n"
"  fsd fs6, 48(a0)\n  fsd fs7, 56(a0)\n  fsd fs8, 64(a0)\n  fsd fs9, 72(a0)\n  fsd fs10, 80(a0)\n  fsd fs11, 88(a0)\n"
"  ret\n.size fpdump, .-fpdump\n");
void fpdump(uint64_t *out);

/* fs0, fs1 are f8, f9; fs2..fs11 are f18..f27. */
static const int FREG[12] = {8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};

static void on_usr1(int sig, siginfo_t *si, void *ctx)
{
	(void)sig; (void)si;
	ucontext_t *uc = ctx;
	for (int i = 0; i < 12; i++) g_frame[i] = uc->uc_mcontext.__fpregs.__d.__f[FREG[i]];
	g_frame_set = 1;
}
void win_signal(void) { kill(getpid(), SIGUSR1); }
void win_fork(void)
{
	pid_t p = fork();
	if (p == 0) { fpdump(g_shared); _exit(0); }   /* the child's first act: its inherited fs0..fs11 */
	int st;
	waitpid(p, &st, 0);
}

static char cls(uint64_t v, int i) { return v == I[i] ? 'I' : v == C[i] ? 'C' : v == 0 ? '0' : '?'; }
static void row(const char *name, const uint64_t *v, char *out)
{
	for (int i = 0; i < 12; i++) out[i] = cls(v[i], i);
	out[12] = 0;
	(void)name;
}
static long switches(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_nvcsw * 1000000L + r.ru_nivcsw; }

int main(void)
{
	setvbuf(stdout, NULL, _IONBF, 0);
	g_shared = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	struct sigaction sa;
	memset(&sa, 0, sizeof sa);
	sa.sa_sigaction = on_usr1;
	sa.sa_flags = SA_SIGINFO;
	sigaction(SIGUSR1, &sa, NULL);
	for (int i = 0; i < 12; i++) {
		I[i] = 0x4010000000000000ull + 0x0000010000000000ull * (uint64_t)(i + 1) + 0x77;
		C[i] = 0x4030000000000000ull + 0x0000010000000000ull * (uint64_t)(i + 1) + 0x33;
	}
	static const char *names = "nwlzsfk";
	int bad = 0;
	for (long mode = 0; mode <= 6; mode++) {
		uint64_t out[12];
		char r[13];
		pid_t pid = 0;
		memset(out, 0, sizeof out);
		g_frame_set = 0;
		for (int i = 0; i < 12; i++) g_shared[i] = 0xdeadull;
		if (mode == 1) {
			pid = fork();
			if (pid == 0) { execl("/bin/sleep", "sleep", "1", (char *)NULL); _exit(127); }
		}
		long s0 = switches();
		fpwin(C, I, out, mode, pid);
		long s1 = switches();
		row(names + mode, out, r);
		printf("B3i %c: regs %s  switches vol %ld invol %ld\n", names[mode], r, (s1 / 1000000L) - (s0 / 1000000L),
		       (s1 % 1000000L) - (s0 % 1000000L));
		if (mode == 4) {
			uint64_t f[12];
			for (int i = 0; i < 12; i++) f[i] = g_frame[i];
			row("frame", f, r);
			printf("B3i s: signal frame %s (handler ran %d)\n", r, g_frame_set);
		}
		if (mode == 5) {
			uint64_t f[12];
			for (int i = 0; i < 12; i++) f[i] = g_shared[i];
			row("child", f, r);
			printf("B3i f: fork child %s\n", r);
		}
		int ok = 1;
		for (int i = 0; i < 12; i++) {
			char c = cls(out[i], i);
			if (mode == 6 ? (i == 5 ? c == 'I' : c != 'I') : c != 'I') ok = 0;
		}
		if (!ok) bad = 1;
	}
	printf("B3i probe rc=%d\n", bad);
	return bad;
}
