/* The `bounds-malloc` control of the cheribsd-default configuration (bug-corpora/tools/arms.json).
 *
 * A protection arm that cannot be shown to fire is not a column, so every invocation that scores
 * cases must also run this: it reads far past a malloc'd block and MUST die by SIGPROT. If it
 * completes, CHERI's per-object bounds are not in force in that guest and every silence in the
 * same run is void -- which is the whole point of declaring it in `controls_for_missed`.
 *
 * It is deliberately not a use-after-free: the knob this corpus varies is revocation, and a
 * control must not depend on the knob under test.
 */
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
	volatile char *p = malloc(32);
	int i;

	if (p == NULL)
		return 2;
	for (i = 0; i < 32; i++)
		p[i] = (char)i;
	printf("CONTROL bounds-malloc: 32-byte block, now reading offset 4096\n");
	fflush(stdout);
	/* Out of the block's bounds by a mile: on purecap this traps. */
	i = (int)p[4096];
	printf("CONTROL bounds-malloc: NO FAULT (read %d) -- bounds are not in force\n", i);
	fflush(stdout);
	return 1;
}
