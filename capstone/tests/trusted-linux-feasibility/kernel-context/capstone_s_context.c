// SPDX-License-Identifier: GPL-2.0-only
/* Exercise the candidate S-mode context instructions inside the Linux image. */
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>

static unsigned long slot[2] __aligned(16);

static int __init capstone_s_context_init(void)
{
	unsigned long expected = 0x7788;
	unsigned long loaded;

	asm volatile(
		".insn s 0x5b, 4, %[value], 0(%[address])\n"
		".insn i 0x5b, 3, %[result], %[address], 0\n"
		: [result] "=&r" (loaded)
		: [value] "r" (expected), [address] "r" (slot)
		: "memory");
	if (loaded != expected || slot[0] != expected || slot[1] != 0)
		return -EINVAL;
	pr_warn("CAPSTONE_S_CONTEXT_PRESELECT_OK\n");
	return 0;
}

static void __exit capstone_s_context_exit(void)
{
}

module_init(capstone_s_context_init);
module_exit(capstone_s_context_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Candidate S-mode context operation control");
