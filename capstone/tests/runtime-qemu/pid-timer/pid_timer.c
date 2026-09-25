/* getpid, getppid, umask, setitimer and getitimer in a musl domain: the
 * process identity a one-process domain answers for itself, and the timer
 * that is accepted and never fires.
 *
 * getpid is 1 and stable, getppid 0; umask returns the previous mask and
 * keeps the new one; setitimer succeeds and reports no earlier timer,
 * getitimer reads it as unarmed, alarm() reports nothing pending, and an
 * unknown timer is EINVAL. run.sh then checks the domain's exit report:
 * "capstone-domain: NO-OP syscalls: 103x2 102" -- the two setitimers (one
 * direct, one through alarm) and the getitimer, in first-seen order -- which
 * is how a no-op stays visible. Every check prints one line; the last line
 * counts the failures and main returns that count. */
#define _XOPEN_SOURCE 700
#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

static int failures;

static void check(const char *name, int ok, const char *detail)
{
	printf("PID-TIMER %s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
	if (!ok)
		failures++;
}

int main(void)
{
	char detail[160];
	int rc;

	errno = 0;
	pid_t p1 = getpid(), p2 = getpid();
	snprintf(detail, sizeof detail, "pid=%ld again=%ld errno=%d (want 1)", (long)p1, (long)p2, errno);
	check("getpid", p1 == 1 && p2 == 1, detail);

	errno = 0;
	pid_t pp = getppid();
	snprintf(detail, sizeof detail, "ppid=%ld errno=%d (want 0)", (long)pp, errno);
	check("getppid", pp == 0, detail);

	mode_t m1 = umask(077);
	mode_t m2 = umask(022);
	snprintf(detail, sizeof detail, "first=%o (want 22) second=%o (want 77)", (unsigned)m1, (unsigned)m2);
	check("umask", m1 == 022 && m2 == 077, detail);

	struct itimerval it = { { 0, 0 }, { 10, 0 } }, old;
	old.it_value.tv_sec = 99;
	errno = 0;
	rc = setitimer(ITIMER_REAL, &it, &old);
	snprintf(detail, sizeof detail, "rc=%d errno=%d old=%ld.%06ld (want 0.0)", rc, rc ? errno : 0,
	         (long)old.it_value.tv_sec, (long)old.it_value.tv_usec);
	check("setitimer", rc == 0 && old.it_value.tv_sec == 0 && old.it_value.tv_usec == 0, detail);

	struct itimerval cur;
	cur.it_value.tv_sec = 99;
	rc = getitimer(ITIMER_REAL, &cur);
	snprintf(detail, sizeof detail, "rc=%d value=%ld.%06ld (want 0.0: it never runs)", rc,
	         (long)cur.it_value.tv_sec, (long)cur.it_value.tv_usec);
	check("getitimer-unarmed", rc == 0 && cur.it_value.tv_sec == 0 && cur.it_value.tv_usec == 0, detail);

	unsigned left = alarm(5);
	snprintf(detail, sizeof detail, "left=%u (want 0)", left);
	check("alarm", left == 0, detail);

	errno = 0;
	rc = setitimer(99, &it, 0);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want EINVAL=%d)", rc, errno, EINVAL);
	check("setitimer-bad-which", rc < 0 && errno == EINVAL, detail);

	printf("PID-TIMER-DONE failures=%d\n", failures);
	return failures;
}
