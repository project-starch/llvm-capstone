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
#include <stdlib.h>

/* What the interpreter is started with. A domain has no command line and no
 * environment of its own, so they are fixed here; the paths are the helper's
 * (the guest's), where the runner mounts its share at /mnt/host.
 *
 *   PYTHONHOME  where getpath finds lib/python313.zip, the stdlib; import
 *               reads a zip with open/read/lseek/fstat, all served. A stdlib
 *               DIRECTORY on sys.path would not work yet: listing it needs
 *               getdents64, which the hostcall does not serve.
 *   -P          keeps the script's directory off sys.path, for that reason.
 *   -S          no site: the first boot runs without it.
 */
#ifndef CPY_DOMAIN_HOME
#define CPY_DOMAIN_HOME "/mnt/host/pyhome"
#endif
#ifndef CPY_DOMAIN_SCRIPT
#define CPY_DOMAIN_SCRIPT "/mnt/host/main.py"
#endif

extern char **__environ;
int main(int argc, char **argv);

int capstone_main(void)
{
	static char *argv[] = { "python3", "-P", "-S", CPY_DOMAIN_SCRIPT, 0 };
	static char *envp[] = { "PYTHONHOME=" CPY_DOMAIN_HOME, 0 };
	__environ = envp;
	/* Returning from main is exit(): atexit handlers run and stdio is flushed.
	   Returning from capstone_main would do neither, and musl buffers stdout
	   fully in a domain (the tty ioctl fails), so everything after the first
	   line of output was lost. exit() reaches domain_main through the runtime's
	   exit hostcall (C-56 made that path safe without a hook of our own). */
	exit(main((int)(sizeof argv / sizeof argv[0]) - 1, argv));
}
