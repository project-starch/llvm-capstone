/* SPDX-License-Identifier: GPL-2.0-only */
#include <stdbool.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#define PAGE_SHIFT 12
#include "sublet-block-state.h"
#define P (1UL << PAGE_SHIFT)
#define CHECK(x) do { if (!(x)) { \
	fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)
static unsigned int revokes;
static bool fault_revoke;
static int remove_pages(struct sublet_block_state *b, struct sublet_ref_info *r,
		unsigned long address, unsigned long bytes)
{
	int error = request_operand(b, r, address, bytes);
	if (error)
		return error;
	if (pages_last(b, address, bytes)) {
		if (fault_revoke)
			return -EFAULT;
		++revokes;
	}
	pages_commit(b, address, bytes, false);
	return 0;
}
int main(void)
{
	unsigned char present[4] = {1, 1, 1, 1};
	struct sublet_block_state b = { P, 4 * P, 4, present, false, false, false };
	struct sublet_ref_info r = { 2, P, 5 * P, 1, true, true, false };

	/* This oracle deliberately fails in the depth-check mutant. */
	r.depth = 2;
	CHECK(remove_pages(&b, &r, P, 4 * P) == -EINVAL);
	CHECK(b.mapped == 4 && revokes == 0);
	r.depth = 1;
	r.tagged = false;
	CHECK(remove_pages(&b, &r, P, 4 * P) == -EFAULT);
	r.tagged = true; r.live = false;
	CHECK(remove_pages(&b, &r, P, 4 * P) == -EFAULT);
	r.live = true;
	b.kernel_owned = true;
	CHECK(remove_pages(&b, &r, P, 4 * P) == -EPERM);
	CHECK(request_operand(&b, &r, P, P) == -EPERM);
	b.kernel_owned = false;
	b.quarantined = true;
	CHECK(remove_pages(&b, &r, P, P) == -EINVAL);
	b.quarantined = false;
	CHECK(remove_pages(&b, &r, P, P) == 0);
	CHECK(!b.ended && b.mapped == 3 && !present[0] && revokes == 0);
	r.descendants = true;
	CHECK(refill_rule(&r) == -EINVAL);
	r.descendants = false;
	CHECK(refill_rule(&r) == 0);
	pages_commit(&b, P, P, true);
	CHECK(b.mapped == 4 && present[0]);
	CHECK(remove_pages(&b, &r, P, 2 * P) == 0);
	CHECK(remove_pages(&b, &r, P, P) == 0); /* a hole changes no count */
	CHECK(b.mapped == 2 && revokes == 0);
	/* A renewed, narrowed depth-1 reference can remove its last pages. */
	r.id = 99; r.base = 3 * P;
	CHECK(remove_pages(&b, &r, P, P) == -EFAULT);
	fault_revoke = true;
	CHECK(remove_pages(&b, &r, 3 * P, 2 * P) == -EFAULT);
	CHECK(b.mapped == 2 && !b.ended && revokes == 0);
	fault_revoke = false;
	CHECK(remove_pages(&b, &r, 3 * P, 2 * P) == 0);
	CHECK(b.ended && b.mapped == 0 && revokes == 1);
	CHECK(remove_pages(&b, &r, 3 * P, 2 * P) == -EINVAL);
	CHECK(!request_call(&b, P, 0));
	CHECK(!request_call(&b, P + 1, P));
	CHECK(!request_call(&b, 4 * P, 2 * P));
	puts("PASS: partial removal, last-page revoke, depth, ownership, bounds, refill, revoke fault");
	return 0;
}
