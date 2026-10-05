/* B1 (docs/plans/b0-silicon-delegated-runtime.md): the first minted context of a gp-captable delegated application.
 *
 * One pthread, created and joined, returning a value computed from its argument. pthread_create reaches context.c's
 * __clone, which mints a context from the arena the glue split off at the first entry, seals it with the monitor's
 * parked code capability (B1.3) and the creator's gp (B1.1), and offers it to the launcher; the launcher adopts and
 * steps it. Distinct exit codes say where it stopped:
 *   0 joined with the expected value; 3 pthread_create failed (its errno-style code is printed); 4 pthread_join
 *   failed; 5 joined with a wrong value (printed). */
#include <pthread.h>
#include <stdio.h>

#ifdef CAPSTONE_CLONE_DIAG
struct capstone_clone_diag { long step, arena_type, arena_bytes, transport, offered, id, r; };
extern struct capstone_clone_diag __capstone_clone_diag;
static void diag(void)
{
	struct capstone_clone_diag *d = &__capstone_clone_diag;
	printf("B1 diag: step %ld arena_type %ld arena_bytes %ld transport %ld offered %ld id %ld r %ld\n",
	       d->step, d->arena_type, d->arena_bytes, d->transport, d->offered, d->id, d->r);
}
#else
static void diag(void) {}
#endif

static void *worker(void *arg)
{
	long v = (long)arg;
	return (void *)(v * 3 + 1);
}

int main(void)
{
	pthread_t t;
	void *ret = 0;
	int r = pthread_create(&t, 0, worker, (void *)41L);
	if (r) {
		printf("B1: pthread_create failed: %d\n", r);
		diag();
		return 3;
	}
	r = pthread_join(t, &ret);
	if (r) {
		printf("B1: pthread_join failed: %d\n", r);
		return 4;
	}
	diag();
	printf("B1: thread returned %ld\n", (long)ret);
	return (long)ret == 124 ? 0 : 5;
}
