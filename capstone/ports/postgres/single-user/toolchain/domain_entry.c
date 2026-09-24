/* An ordinary main() inside a musl domain, for the PostgreSQL backend.
 *
 * The port runtime's domain_main (musl-capstone/runtime/hostcall.c) calls
 * capstone_main(); the backend has main(). This is the adapter, the same shape
 * as the CPython port's: an empty environment rather than musl's null
 * __environ, and an argv.
 *
 * The backend is started differently for each initdb step -- `--boot` with the
 * catalog description on its input, then `--single` with the setup SQL -- and
 * one changed string in the image moves its whole layout (the CPython port
 * measured 44743 bytes for one digit). So neither the arguments nor the
 * environment are built in: the domain reads them from two files the host
 * writes before each run, one entry per line,
 *
 *   /mnt/host/pg-args    argv[1..]; argv[0] is PGSU_ARGV0, the image on the share
 *   /mnt/host/pg-env     NAME=value, exported before main runs
 *
 * and echoes each line it took to stderr, so a run shows what it ran with.
 * configure links every conftest through this file too; for those links the
 * files do not exist and main gets argv[0] alone. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef PGSU_ARGS_FILE
#define PGSU_ARGS_FILE "/mnt/host/pg-args"
#endif
#ifndef PGSU_ENV_FILE
#define PGSU_ENV_FILE "/mnt/host/pg-env"
#endif
#ifndef PGSU_START_DIR
#define PGSU_START_DIR "/mnt/host"
#endif
/* argv[0] has to name a file: find_my_exec stats it, resolves it with
   realpath, and takes the share directory from it -- <dir of argv[0]>/../share
   when the directory is named bin, which is why run-domain.sh stages the image
   under bin/ and the native share/ beside it. */
#ifndef PGSU_ARGV0
#define PGSU_ARGV0 "/mnt/host/bin/postgres.dom"
#endif
enum { LINES_MAX = 32, LINE_MAX_BYTES = 512 };
#include <unistd.h>

extern char **__environ;
int main(int argc, char **argv);

static int read_lines(const char *path, char lines[][LINE_MAX_BYTES], char **out, int start)
{
	int n = start;
	FILE *f = fopen(path, "r");
	if (!f)
		return n;
	while (n < LINES_MAX - 1 && fgets(lines[n], LINE_MAX_BYTES, f)) {
		size_t len = strlen(lines[n]);
		while (len && (lines[n][len - 1] == '\n' || lines[n][len - 1] == '\r'))
			lines[n][--len] = '\0';
		if (len == 0)
			continue;
		out[n] = lines[n];
		n++;
	}
	fclose(f);
	return n;
}

int capstone_main(void)
{
	static char argl[LINES_MAX][LINE_MAX_BYTES];
	static char envl[LINES_MAX][LINE_MAX_BYTES];
	static char *argv[LINES_MAX];
	static char *envp[LINES_MAX];

	argv[0] = PGSU_ARGV0;
	int argc = read_lines(PGSU_ARGS_FILE, argl, argv, 1);
	argv[argc] = 0;
	int envc = read_lines(PGSU_ENV_FILE, envl, envp, 0);
	envp[envc] = 0;
	__environ = envp;

	for (int i = 1; i < argc; i++)
		fprintf(stderr, "PGSU-ARG %s\n", argv[i]);
	for (int i = 0; i < envc; i++)
		fprintf(stderr, "PGSU-ENV %s\n", envp[i]);
	/* A domain starts with no working directory (getcwd says ENOENT until the
	   first chdir), and the backend asks for one before it does anything else
	   (find_my_exec). The share the runner mounts is the natural one. */
	if (chdir(PGSU_START_DIR) != 0)
		fprintf(stderr, "PGSU-WARN chdir(%s) failed\n", PGSU_START_DIR);
	return main(argc, argv);
}
