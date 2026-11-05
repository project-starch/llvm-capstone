/* Which defence ends these runs? The corpus's aborting cases die by SIGABRT raised by the
 * process itself (ktrace: thr_kill(SIGABRT), code=SI_LWP) with nothing written to stderr.
 * This probe asks whether CheriBSD's own allocator does that for an invalid free, so the
 * reading can say which check fired instead of guessing. Each stage prints before it acts. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
	char *p;
	int stage = (argc > 1) ? atoi(argv[1]) : 0;

	p = malloc(32);
	memset(p, 'a', 32);
	switch (stage) {
	case 1:
		printf("PROBE 1: double free\n"); fflush(stdout);
		free(p); free(p);
		printf("PROBE 1: survived\n");
		break;
	case 2:
		printf("PROBE 2: read after free\n"); fflush(stdout);
		free(p);
		printf("PROBE 2: read back %c, survived\n", p[0]);
		break;
	case 3:
		printf("PROBE 3: write after free\n"); fflush(stdout);
		free(p);
		p[0] = 'z';
		printf("PROBE 3: survived\n");
		break;
	case 4:
		printf("PROBE 4: free of an interior pointer\n"); fflush(stdout);
		free(p + 8);
		printf("PROBE 4: survived\n");
		break;
	default:
		printf("PROBE 0: malloc/free only\n"); fflush(stdout);
		free(p);
		printf("PROBE 0: survived\n");
	}
	fflush(stdout);
	return 0;
}
