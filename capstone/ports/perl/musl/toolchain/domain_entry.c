/* An ordinary main() inside a musl domain, for perl (whose main() is perlmain.c's).
 *
 * The port runtime's domain_main (musl-capstone/runtime/hostcall.c) calls
 * capstone_main(); perl has main(). This is the adapter, the mruby port's
 * toolchain/domain_entry.c (itself the PostgreSQL port's) with its names changed
 * and nothing else: an empty environment rather than musl's null __environ, and an
 * argv read from files the host writes before each run, one entry per line,
 *
 *   /mnt/host/dom-args   argv[1..]; argv[0] is PERLD_ARGV0, the image on the share
 *   /mnt/host/dom-env    NAME=value, exported before main runs
 *
 * so one image serves every script and every test selection. Each line taken is
 * echoed to stderr, so a run shows what it ran with.
 *
 * stderr is unbuffered and stdout line-buffered before main: a domain that
 * halts on a capability fault never flushes, and what it printed last is what
 * locates the fault. A diagnostic written with fprintf(stderr) just before a
 * fault was lost that way while this port was brought up. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef PERLD_ARGS_FILE
#define PERLD_ARGS_FILE "/mnt/host/dom-args"
#endif
#ifndef PERLD_ENV_FILE
#define PERLD_ENV_FILE "/mnt/host/dom-env"
#endif
#ifndef PERLD_START_DIR
#define PERLD_START_DIR "/mnt/host"
#endif
#ifndef PERLD_ARGV0
#define PERLD_ARGV0 "/mnt/host/bin/perl.dom"
#endif
enum { LINES_MAX = 64, LINE_MAX_BYTES = 512 };

extern char **__environ;
int main(int argc, char **argv);
#ifdef PERLD_SUBLET_HEAP
void __capstone_sublet_heap_stats(unsigned long out[9]);
#endif
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

	setvbuf(stderr, NULL, _IONBF, 0);
	setvbuf(stdout, NULL, _IOLBF, 0);
	argv[0] = PERLD_ARGV0;
	int argc = read_lines(PERLD_ARGS_FILE, argl, argv, 1);
	argv[argc] = 0;
	int envc = read_lines(PERLD_ENV_FILE, envl, envp, 0);
	envp[envc] = 0;
	__environ = envp;

	for (int i = 1; i < argc; i++)
		fprintf(stderr, "PERLD-ARG %s\n", argv[i]);
	for (int i = 0; i < envc; i++)
		fprintf(stderr, "PERLD-ENV %s\n", envp[i]);
	/* A domain starts with no working directory (getcwd says ENOENT until the
	   first chdir); the share the runner mounts is the natural one. */
	if (chdir(PERLD_START_DIR) != 0)
		fprintf(stderr, "PERLD-WARN chdir(%s) failed\n", PERLD_START_DIR);
	int rc = main(argc, argv);
#ifdef PERLD_SUBLET_HEAP
	/* What the revoking heap (runtime/sublet_heap.c) spent. split + mrev is the
	   revocation-node count, which silicon caps at 65,532 per boot. Printed only
	   when main returns; a program that ends in exit() does not reach it. */
	unsigned long hs[9];
	__capstone_sublet_heap_stats(hs);
	printf("PERLD-HEAP alloc=%lu free=%lu merge=%lu peak-live=%lu split=%lu mrev=%lu "
	       "delin=%lu revoke=%lu init=%lu\n",
	       hs[0], hs[1], hs[2], hs[3], hs[4], hs[5], hs[6], hs[7], hs[8]);
#endif
	fflush(NULL);
	return rc;
}
