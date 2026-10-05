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
#ifdef CAPSTONE_B1_PROBES
#include <errno.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <capstone/capability.h>
extern void *__capstone_silicon_code_cap;
extern capstone_cap_slot __capstone_context_arena;
/* B1e: the capabilities the context path depends on, as silicon sees them. LCC's type query is total; the other
   selectors are asked only of a tagged value. */
static void capinfo(const char *what, void *c)
{
	unsigned long t, cur = 0, b = 0, e = 0;
	__asm__ volatile ("lcc %0, %1, 1" : "=r"(t) : "r"(c));
	if (t != 7) {
		__asm__ volatile ("lcc %0, %1, 2" : "=r"(cur) : "r"(c));
		__asm__ volatile ("lcc %0, %1, 3" : "=r"(b) : "r"(c));
		__asm__ volatile ("lcc %0, %1, 4" : "=r"(e) : "r"(c));
	}
	printf("B1 cap: %s type %lu cursor %lx base %lx end %lx\n", what, t, cur, b, e);
}
#endif

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

#ifdef CAPSTONE_B1_PROBES
/* B1 on silicon (B1d): pthread_create failed before __clone (B1c: step 0), i.e. in musl's stack mapping. Each step of
   that mapping is done here by hand, then a thread with a small stack, so one boot separates the hypotheses. */
static void probes(void)
{
	size_t sz = 4096 + ((131072 + 4096 + 4095) & ~(size_t)4095);
	printf("B1 probe: level0 arena %lu (as built)\n", (unsigned long)CAPSTONE_LEVEL0_ARENA_BYTES);
	capinfo("code", __capstone_silicon_code_cap);
	printf("B1 cap: arena type %lu base %lx end %lx\n", capstone_cap_type(&__capstone_context_arena),
	       capstone_cap_base(&__capstone_context_arena), capstone_cap_end(&__capstone_context_arena));
	{
		unsigned long gb, ge, sb, se;
		__asm__ volatile ("lcc %0, gp, 3" : "=r"(gb));
		__asm__ volatile ("lcc %0, gp, 4" : "=r"(ge));
		__asm__ volatile ("lcc %0, sp, 3" : "=r"(sb));
		__asm__ volatile ("lcc %0, sp, 4" : "=r"(se));
		printf("B1 cap: gp base %lx end %lx; sp base %lx end %lx\n", gb, ge, sb, se);
	}
	errno = 0;
	void *m = malloc(sz + 4096);
	printf("B1 probe: malloc(%lu) %s errno %d\n", (unsigned long)(sz + 4096), m ? "ok" : "NULL", errno);
	free(m);
	errno = 0;
	void *map = mmap(0, sz, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
	printf("B1 probe: mmap(%lu) %s errno %d\n", (unsigned long)sz, map == MAP_FAILED ? "FAILED" : "ok", errno);
	if (map != MAP_FAILED) {
		errno = 0;
		int pr = mprotect((char *)map + 4096, sz - 4096, PROT_READ | PROT_WRITE);
		printf("B1 probe: mprotect rc %d errno %d\n", pr, errno);
		munmap(map, sz);
	}
}
static int small_thread(void)
{
	pthread_attr_t a;
	pthread_t t;
	void *ret = 0;
	pthread_attr_init(&a);
	pthread_attr_setstacksize(&a, 16384);
	int r = pthread_create(&t, &a, worker, (void *)41L);
	printf("B1 probe: small-stack pthread_create %d\n", r);
	diag();
	if (r)
		return 3;
	r = pthread_join(t, &ret);
	printf("B1 probe: small-stack join %d value %ld\n", r, (long)ret);
	return r ? 4 : ((long)ret == 124 ? 0 : 5);
}
#endif

int main(void)
{
	pthread_t t;
	void *ret = 0;
#ifdef CAPSTONE_B1_PROBES
	probes();
#endif
	int r = pthread_create(&t, 0, worker, (void *)41L);
	if (r) {
		printf("B1: pthread_create failed: %d\n", r);
		diag();
#ifdef CAPSTONE_B1_PROBES
		return 10 + small_thread();
#endif
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
