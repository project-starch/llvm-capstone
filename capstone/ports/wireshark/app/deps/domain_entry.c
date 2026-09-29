/* An ordinary main() inside a musl domain, for the tshark port and its third-party libraries.
 *
 * The port runtime's domain_main (musl-capstone/runtime/hostcall.c) calls capstone_main(); a C
 * program has main(). A domain has no command line and no environment of its own, so they are
 * read, when the domain starts, from two files in the guest:
 *
 *   /tmp/domain.argv   one argument per line, argv[0] first (make it absolute: getcwd is not
 *                      served, so a relative program path cannot be resolved)
 *   /tmp/domain.env    one NAME=value per line
 *
 * One image then serves every run, and the runner writes the files before each one. Building
 * them into the image instead would change its layout with every variant: CPython's port
 * measured one changed string moving 44,743 bytes of the image (interpreter/toolchain/
 * domain_entry.c). Each argument and variable taken is echoed to stderr, so a run shows what it
 * ran with.
 *
 * The environment is read by __capstone_domain_environ(), which the runtime calls BEFORE the
 * program's constructors (hostcall.c, ISSUES C-64): in C, environ exists before any constructor
 * runs, and GLib's reads it. The command line is read by capstone_main, before main.
 *
 * With neither file present: argv {"domain"} and an empty environment. configure links every
 * conftest through this adapter, and reads nothing from the files there.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv);
char **__capstone_domain_environ(void);

enum { ARGS_MAX = 32, ENVS_MAX = 16, LINE_MAX_BYTES = 512 };

static int read_lines(const char *path, char lines[][LINE_MAX_BYTES], int max)
{
	FILE *f = fopen(path, "r");
	int n = 0;
	if (!f)
		return 0;
	while (n < max && fgets(lines[n], LINE_MAX_BYTES, f)) {
		size_t len = strlen(lines[n]);
		if (len && lines[n][len - 1] == '\n')
			lines[n][--len] = '\0';
		if (len)
			n++;
	}
	fclose(f);
	return n;
}

char **__capstone_domain_environ(void)
{
	static char envl[ENVS_MAX][LINE_MAX_BYTES];
	static char *envp[ENVS_MAX + 1];
	int envc = read_lines("/tmp/domain.env", envl, ENVS_MAX);
	int i;

	for (i = 0; i < envc; i++) {
		envp[i] = envl[i];
		fprintf(stderr, "domain_entry: env %s\n", envl[i]);
	}
	envp[envc] = NULL;
	return envp;
}

int capstone_main(void)
{
	static char argl[ARGS_MAX][LINE_MAX_BYTES];
	static char *argv[ARGS_MAX + 1];
	static char prog[] = "domain";
	int argc = read_lines("/tmp/domain.argv", argl, ARGS_MAX);
	int i;

	for (i = 0; i < argc; i++) {
		argv[i] = argl[i];
		fprintf(stderr, "domain_entry: argv[%d]=%s\n", i, argl[i]);
	}
	if (argc == 0)
		argv[argc++] = prog;
	argv[argc] = NULL;
	exit(main(argc, argv));
}
