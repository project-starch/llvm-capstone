/* B6 (docs/plans/b0-silicon-delegated-runtime.md): Sublet's child lifetimes from a delegated application. The
 * runtime's own helpers (capstone/capability.h: capstone_cap_derive, capstone_cap_revoke_child) issue CDERIVE and
 * CREVOKE under Linux and the monitor, on the heap a domain gets from malloc.
 *
 * A delegated application, built by build-b0-hello.sh (B0_APP=b6-sublet), run by b0run.sh. Every reading is taken
 * with lcc or through a capability that is still live, so every arm RETURNS: nothing is ever dereferenced through a
 * revoked child (that would end the domain), and no CREVOKE is issued by a holder without MANAGE (cause 27).
 *
 *   P  malloc(8192): the parent. Must be NONLIN, or nothing else runs (rc 200 + type).
 *   G  g = derive(p, 0, 4096): a group. NONLIN, perm (p & 7) | MANAGE, bounds exactly [p, p + 4096).
 *   C  c = derive(g, 256, 512): a child of the group, [g + 256, g + 768).
 *   S  a store through c reads back through c and through g at +256 (one set of bytes, two lifetimes).
 *   K  k = derive(c, 0, 64): a grandchild, minted with MANAGE by a child.
 *   R  revoke_child(g, c): c and k read invalid (lcc x0), g stays valid, and the bytes stay (read through g).
 *   N  c2 = derive(g, 256, 512) after the kill: a fresh child over the same range, valid, stores through it.
 *   E  revoke_child(g, c2): c2 invalid, g valid.
 * Exit status: bit k set if arm k (G=0 C=1 S=2 K=3 R=4 N=5 E=6) read anything other than its expectation; the
 * printed lines are the reading. Sizes are chosen exact under the full scheme (below 4 KiB, or 4096 at 16-byte
 * alignment), so QEMU's physical profile (which refuses an unrepresentable child with 29) and the RTL (which rounds
 * outward) must agree on every value. */
#include <stdio.h>
#include <stdlib.h>
#include <capstone/capability.h>

#define BASE(p) __builtin_capstone_cap_get_base(p)
#define END(p) __builtin_capstone_cap_get_end(p)

static unsigned long lcc_valid(void *p)
{
	unsigned long v;
	__asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x0" : "=r"(v) : "r"(p));
	return v;
}

static unsigned long lcc_type(void *p)
{
	unsigned long v;
	__asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x1" : "=r"(v) : "r"(p));
	return v;
}

static unsigned long lcc_perm(void *p)
{
	unsigned long v;
	__asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x5" : "=r"(v) : "r"(p));
	return v;
}

static void show(const char *name, void *c, void *rel)
{
	printf("B6 %s: valid %lu type %lu perm %lu [%+ld, %+ld)\n", name, lcc_valid(c), lcc_type(c), lcc_perm(c),
	       (long)(BASE(c) - BASE(rel)), (long)(END(c) - BASE(rel)));
}

int main(void)
{
	char *p = malloc(8192);
	if (!p) {
		printf("B6: malloc(8192) failed\n");
		return 255;
	}
	unsigned long pt = lcc_type(p), pp = lcc_perm(p);
	printf("B6 P: valid %lu type %lu perm %lu len %ld base%%16 %lu\n", lcc_valid(p), pt, pp,
	       (long)(END(p) - BASE(p)), BASE(p) & 15);
	if (pt != CAPSTONE_CAP_NONLINEAR) {
		printf("B6 rc=%lu\n", 200 + pt);
		return (int)(200 + pt);
	}
	int rc = 0;
	unsigned long want = (pp & 7) | CAPSTONE_PERM_MANAGE;

	char *g = capstone_cap_derive(p, 0, 4096);
	show("G", g, p);
	if (lcc_valid(g) != 1 || lcc_type(g) != CAPSTONE_CAP_NONLINEAR || lcc_perm(g) != want || BASE(g) != BASE(p) ||
	    END(g) != BASE(p) + 4096)
		rc |= 1 << 0;

	char *c = capstone_cap_derive(g, 256, 512);
	show("C", c, g);
	if (lcc_valid(c) != 1 || lcc_perm(c) != want || BASE(c) != BASE(g) + 256 || END(c) != BASE(g) + 768)
		rc |= 1 << 1;

	*(volatile long *)c = 0x1234;
	long s1 = *(volatile long *)c, s2 = *(volatile long *)(g + 256);
	printf("B6 S: through c 0x%lx, through g+256 0x%lx\n", s1, s2);
	if (s1 != 0x1234 || s2 != 0x1234)
		rc |= 1 << 2;

	char *k = capstone_cap_derive(c, 0, 64);
	show("K", k, g);
	if (lcc_valid(k) != 1 || lcc_perm(k) != want || BASE(k) != BASE(g) + 256 || END(k) != BASE(g) + 320)
		rc |= 1 << 3;

	/* A revoked child's validity is read here, before any call: a callee that spills and restores the register
	 * holding it hands back a non-capability under QEMU, whose ldc untags a revoked capability (ISSUES Q-11; lcc
	 * then raised 24 in the first build, a39ca7382563fef5). The RTL forwards it unchanged. */
	capstone_cap_revoke_child(g, c);
	unsigned long vc = lcc_valid(c), vk = lcc_valid(k), vg = lcc_valid(g);
	long s3 = *(volatile long *)(g + 256);
	printf("B6 R: after revoke_child(g, c): c valid %lu, k valid %lu, g valid %lu, g+256 0x%lx\n", vc, vk, vg, s3);
	if (vc != 0 || vk != 0 || vg != 1 || s3 != 0x1234)
		rc |= 1 << 4;

	char *c2 = capstone_cap_derive(g, 256, 512);
	*(volatile long *)c2 = 0x5678;
	long s4 = *(volatile long *)(g + 256);
	show("N", c2, g);
	printf("B6 N: g+256 0x%lx\n", s4);
	if (lcc_valid(c2) != 1 || BASE(c2) != BASE(g) + 256 || END(c2) != BASE(g) + 768 || s4 != 0x5678)
		rc |= 1 << 5;

	capstone_cap_revoke_child(g, c2);
	unsigned long vc2 = lcc_valid(c2), vg2 = lcc_valid(g);
	printf("B6 E: after revoke_child(g, c2): c2 valid %lu, g valid %lu\n", vc2, vg2);
	if (vc2 != 0 || vg2 != 1)
		rc |= 1 << 6;

	printf("B6 rc=%d\n", rc);
	return rc;
}
