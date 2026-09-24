/* An ordinary main() inside a musl domain.
 *
 * The port runtime's domain_main (musl-capstone/runtime/hostcall.c) calls
 * capstone_main(); a C program has main(). This is the adapter, and the same
 * shape as libc-test's libc_test_domain.c without the test harness: an empty
 * environment rather than musl's null __environ (which __libc_start_main would
 * have set, and a domain does not run), and one argv entry.
 *
 * configure links every conftest through it, so HAVE_<func> means "resolves in
 * a real domain image"; for those links argv and the environment do not
 * matter. The interpreter links CPython's own Programs/python.c main() here.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* What the interpreter is started with. A domain has no command line and no
 * environment of its own, so they are fixed here; the paths are the helper's
 * (the guest's), where the runner mounts its share at /mnt/host.
 *
 *   PYTHONHOME  where getpath finds lib/python313.zip, the stdlib: one file
 *               opened once, where a stdlib directory costs a listing and an
 *               open per import. (Listing works since the hostcall's
 *               DIR_READ; large reads since its bounce buffer.)
 *   -P          keeps the script's directory off sys.path, where it would be
 *               searched before the zip.
 *   -S          no site: the first boot runs without it.
 *   argv[0]     an ABSOLUTE path under PYTHONHOME. getpath makes the program
 *               name absolute, and a bare "python3" with no PATH falls back to
 *               abspath('.'), i.e. getcwd, which the hostcall does not serve
 *               ("failed to make path absolute", measured). The file need not
 *               exist: getpath tolerates a realpath that fails on a missing one.
 */
#ifndef CPY_DOMAIN_HOME
#define CPY_DOMAIN_HOME "/mnt/host/pyhome"
#endif
#ifndef CPY_DOMAIN_SCRIPT
#define CPY_DOMAIN_SCRIPT "/mnt/host/main.py"
#endif

/* Extra variables for one run, read when the domain starts: one NAME=value per
 * line. run-cpython-domain.sh writes the file in the guest before each run, so
 * one image serves every variant. Building a variable in instead
 * (CPY_DOMAIN_EXTRA_ENV) changes the image's layout, not just the variable:
 * one changed string in domain_entry.o moved 44743 bytes of the linked image
 * (seed 0 against seed 3, measured). Each variable taken is echoed to stderr,
 * so a run shows what it actually ran with. */
#ifndef CPY_DOMAIN_ENV_FILE
#define CPY_DOMAIN_ENV_FILE "/tmp/cpy-domain.env"
#endif
enum { ENV_MAX = 8, ENV_LINE = 256 };

extern char **__environ;
int main(int argc, char **argv);

static void read_env_file(char **envp, int *n)
{
	static char lines[ENV_MAX][ENV_LINE];
	FILE *f = fopen(CPY_DOMAIN_ENV_FILE, "r");
	if (!f)
		return;
	for (int i = 0; i < ENV_MAX && fgets(lines[i], ENV_LINE, f); i++) {
		lines[i][strcspn(lines[i], "\n")] = 0;
		if (!strchr(lines[i], '='))
			continue;
		envp[(*n)++] = lines[i];
		fprintf(stderr, "CPY-ENV %s\n", lines[i]);
	}
	char rest[ENV_LINE];
	if (fgets(rest, sizeof rest, f))
		fprintf(stderr, "CPY-ENV IGNORED: more than %d lines in %s\n", ENV_MAX, CPY_DOMAIN_ENV_FILE);
	fclose(f);
}

int capstone_main(void)
{
	static char *argv[] = { CPY_DOMAIN_HOME "/bin/python3", "-P", "-S", CPY_DOMAIN_SCRIPT, 0 };
	/* PYTHONHOME, CPY_DOMAIN_EXTRA_ENV if built in (e.g.
	   -DCPY_DOMAIN_EXTRA_ENV='"PYTHONVERBOSE=1"'), the env file's lines, NULL. */
	static char *envp[2 + ENV_MAX + 1] = { "PYTHONHOME=" CPY_DOMAIN_HOME };
	int n = 1;
#ifdef CPY_DOMAIN_EXTRA_ENV
	envp[n++] = CPY_DOMAIN_EXTRA_ENV;
#endif
	read_env_file(envp, &n);
	envp[n] = 0;
	__environ = envp;
	/* Returning from main is exit(): atexit handlers run and stdio is flushed.
	   Returning from capstone_main would do neither, and musl buffers stdout
	   fully in a domain (the tty ioctl fails), so everything after the first
	   line of output was lost. exit() reaches domain_main through the runtime's
	   exit hostcall (C-56 made that path safe without a hook of our own). */
	exit(main((int)(sizeof argv / sizeof argv[0]) - 1, argv));
}
