/* mmap, munmap and System V shared memory in a musl domain, served from the
 * domain's own allocator (runtime/mmap_shm_level0.c).
 *
 * Anonymous mappings: page-aligned, zeroed, distinct, writable, unmapped whole
 * and refused in part; the refusals (a file mapping, MAP_FIXED, MAP_HUGETLB,
 * more than the arena holds) with their errnos. Segments: IPC_PRIVATE and a
 * key, IPC_CREAT|IPC_EXCL on an existing key (EEXIST), lookup by key, a
 * missing key (ENOENT), attach and detach with the attach count in IPC_STAT,
 * IPC_RMID with the id gone afterwards. run.sh compiles level0 with a 512 KiB
 * arena for this domain: an undeclared domain's block is sized from its image,
 * and a larger arena pushed it past what the module gives.
 *
 * The refusals are judged by errno, not by the pointer: MAP_FAILED is (void *)-1,
 * which crosses a call as an integer, and C-32's movc on the caller's side turns
 * an integer into NULL, so a pointer comparison against MAP_FAILED can hold for
 * the wrong reason. errno is set by the override itself and cannot. Every check prints one line; the last line counts
 * the failures and main returns that count. */
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <unistd.h>

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif

static int failures;

static void check(const char *name, int ok, const char *detail)
{
	printf("MMAP-SHM %s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
	if (!ok)
		failures++;
}

static int all_zero(const unsigned char *p, size_t n)
{
	for (size_t i = 0; i < n; i++)
		if (p[i])
			return 0;
	return 1;
}

static int pattern_ok(unsigned char *p, size_t n)
{
	for (size_t i = 0; i < n; i++)
		p[i] = (unsigned char)(i * 7 + 3);
	for (size_t i = 0; i < n; i++)
		if (p[i] != (unsigned char)(i * 7 + 3))
			return 0;
	return 1;
}

int main(void)
{
	char detail[200];
	const size_t A = 64 * 1024, B = 128 * 1024;

	errno = 0;
	unsigned char *a = mmap(0, A, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	snprintf(detail, sizeof detail, "p=%s errno=%d", a == MAP_FAILED ? "MAP_FAILED" : "ok", a == MAP_FAILED ? errno : 0);
	check("mmap", a != MAP_FAILED, detail);
	if (a == MAP_FAILED) {
		printf("MMAP-SHM-DONE failures=%d\n", failures);
		return failures;
	}
	snprintf(detail, sizeof detail, "misalign=%lu", (unsigned long)((__UINTPTR_TYPE__)a & 4095));
	check("page-aligned", ((__UINTPTR_TYPE__)a & 4095) == 0, detail);
	check("zeroed", all_zero(a, A), "");
	check("writable", pattern_ok(a, A), "");

	unsigned char *b = mmap(0, B, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	int distinct = b != MAP_FAILED && (b + B <= a || a + A <= b);
	snprintf(detail, sizeof detail, "b=%s distinct=%d", b == MAP_FAILED ? "MAP_FAILED" : "ok", distinct);
	check("second-mapping-distinct", distinct, detail);
	check("second-zeroed-and-writable", b != MAP_FAILED && all_zero(b, B) && pattern_ok(b, B), "");

	errno = 0;
	int rc = munmap(a, A);
	snprintf(detail, sizeof detail, "rc=%d errno=%d", rc, rc ? errno : 0);
	check("munmap-whole", rc == 0, detail);
	errno = 0;
	rc = munmap(b, 4096);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want EINVAL=%d)", rc, errno, EINVAL);
	check("munmap-partial-refused", rc < 0 && errno == EINVAL, detail);
	errno = 0;
	rc = munmap(a, A);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want EINVAL=%d)", rc, errno, EINVAL);
	check("munmap-unknown-refused", rc < 0 && errno == EINVAL, detail);
	rc = munmap(b, B);
	snprintf(detail, sizeof detail, "rc=%d", rc);
	check("munmap-second", rc == 0, detail);

	errno = 0;
	void *p = mmap(0, 4096, PROT_READ, MAP_SHARED, 3, 0);
	snprintf(detail, sizeof detail, "errno=%d (want ENODEV=%d)", errno, ENODEV);
	check("file-mapping-refused", p == MAP_FAILED && errno == ENODEV, detail);
	errno = 0;
	p = mmap((void *)0x10000, 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
	snprintf(detail, sizeof detail, "errno=%d (want EINVAL=%d)", errno, EINVAL);
	check("map-fixed-refused", p == MAP_FAILED && errno == EINVAL, detail);
	errno = 0;
	p = mmap(0, 2 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
	snprintf(detail, sizeof detail, "errno=%d (want ENOMEM=%d)", errno, ENOMEM);
	check("hugetlb-refused", p == MAP_FAILED && errno == ENOMEM, detail);
	errno = 0;
	p = mmap(0, 8 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	snprintf(detail, sizeof detail, "errno=%d (want ENOMEM=%d)", errno, ENOMEM);
	check("beyond-arena-enomem", p == MAP_FAILED && errno == ENOMEM, detail);

	/* System V: the private segment PostgreSQL uses as its interlock. */
	errno = 0;
	int id = shmget(IPC_PRIVATE, 56, IPC_CREAT | 0600);
	snprintf(detail, sizeof detail, "id=%d errno=%d", id, id < 0 ? errno : 0);
	check("shmget-private", id >= 0, detail);
	unsigned char *s = shmat(id, 0, 0);
	snprintf(detail, sizeof detail, "p=%s zero=%d", s == (void *)-1 ? "-1" : "ok", s != (void *)-1 && all_zero(s, 56));
	check("shmat", s != (void *)-1 && all_zero(s, 56), detail);
	struct shmid_ds ds;
	memset(&ds, 0x55, sizeof ds);
	rc = shmctl(id, IPC_STAT, &ds);
	snprintf(detail, sizeof detail, "rc=%d segsz=%lu nattch=%lu", rc, (unsigned long)ds.shm_segsz, ds.shm_nattch);
	check("shmctl-stat-attached", rc == 0 && ds.shm_segsz == 56 && ds.shm_nattch == 1, detail);
	s[0] = 0x42;
	rc = shmdt(s);
	shmctl(id, IPC_STAT, &ds);
	snprintf(detail, sizeof detail, "rc=%d nattch=%lu", rc, ds.shm_nattch);
	check("shmdt", rc == 0 && ds.shm_nattch == 0, detail);
	rc = shmctl(id, IPC_RMID, 0);
	errno = 0;
	int rc2 = shmctl(id, IPC_STAT, &ds);
	snprintf(detail, sizeof detail, "rmid=%d stat-after=%d errno=%d (want EINVAL=%d)", rc, rc2, errno, EINVAL);
	check("shmctl-rmid", rc == 0 && rc2 < 0 && errno == EINVAL, detail);

	/* A keyed segment: found again by key, refused with EXCL, gone with RMID. */
	errno = 0;
	int k = shmget(0x5eed, 64 * 1024, IPC_CREAT | IPC_EXCL | 0600);
	snprintf(detail, sizeof detail, "id=%d errno=%d", k, k < 0 ? errno : 0);
	check("shmget-key", k >= 0, detail);
	errno = 0;
	rc = shmget(0x5eed, 64 * 1024, IPC_CREAT | IPC_EXCL | 0600);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want EEXIST=%d)", rc, errno, EEXIST);
	check("shmget-excl-refused", rc < 0 && errno == EEXIST, detail);
	rc = shmget(0x5eed, 0, 0);
	snprintf(detail, sizeof detail, "rc=%d (want %d)", rc, k);
	check("shmget-lookup", rc == k, detail);
	errno = 0;
	rc = shmget(0x7777, 0, 0);
	snprintf(detail, sizeof detail, "rc=%d errno=%d (want ENOENT=%d)", rc, errno, ENOENT);
	check("shmget-missing", rc < 0 && errno == ENOENT, detail);
	unsigned char *t = shmat(k, 0, 0);
	int ok = t != (void *)-1 && all_zero(t, 64 * 1024) && pattern_ok(t, 64 * 1024);
	snprintf(detail, sizeof detail, "p=%s ok=%d", t == (void *)-1 ? "-1" : "ok", ok);
	check("shmat-key-writable", ok, detail);
	rc = shmctl(k, IPC_RMID, 0);
	rc2 = shmdt(t);
	errno = 0;
	int rc3 = shmget(0x5eed, 0, 0);
	snprintf(detail, sizeof detail, "rmid=%d dt=%d lookup-after=%d errno=%d (want ENOENT=%d)", rc, rc2, rc3, errno, ENOENT);
	check("rmid-then-detach-frees", rc == 0 && rc2 == 0 && rc3 < 0 && errno == ENOENT, detail);

	printf("MMAP-SHM-DONE failures=%d\n", failures);
	return failures;
}
