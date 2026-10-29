/* pair N0 N1 K: K rounds of p = malloc(N0); realloc(p, N1). The granule of repr_round(N1) is g. A round
 * whose base is a multiple of g reallocs and prints its outcome; the FIRST round whose base is not
 * prints PRED-TRAP and reallocs too (predicted: the process dies there with the realloc trap). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned long gran(unsigned long len) { return len < 4096 ? 1 : 1UL << (63 - __builtin_clzl(len) - 9); }
int main(int argc, char **argv)
{
	unsigned long n0 = strtoul(argv[1], 0, 0), n1 = strtoul(argv[2], 0, 0);
	int k = atoi(argv[3]);
	unsigned long g = gran(n1);
	for (int i = 0; i < k; i++) {
		char *p = malloc(n0);
		if (!p) { printf("PAIR malloc NULL\n"); return 2; }
		memset(p, 0x5a, n0);
		unsigned long a = __builtin_capstone_cap_get_cursor(p);
		if (a % g) { printf("PAIR round %d a%%g=%lu g=%lu PRED-TRAP\n", i, a % g, g); fflush(stdout); }
		char *q = realloc(p, n1);
		if (!q) { printf("PAIR realloc NULL\n"); return 3; }
		unsigned long b = __builtin_capstone_cap_get_cursor(q);
		printf("PAIR round %d a%%g=%lu %s ok byte=%x\n", i, a % g, a == b ? "in-place" : "moved", (unsigned char)q[n0 - 1]);
		fflush(stdout);
		free(q);
	}
	printf("PAIR DONE no trap\n");
	return 0;
}
