// SPDX-License-Identifier: GPL-2.0-only
/* EXPERIMENT: direct copy_page pricing and free/reallocate tag invariant.
 * Load only in a private test VM before starting any protected domain. */
#include <linux/module.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <linux/interrupt.h>
#include <asm/csr.h>

#define START() asm volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory")
#define STOP()  asm volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory")
#define REPS 1024

static bool any_tag(unsigned long pa)
{
    unsigned long flags, any;
    local_irq_save(flags);
    csr_write(0x5c8, pa);
    any = csr_read(0x5cb);
    local_irq_restore(flags);
    return any != 0;
}

static int __init raw_page_probe_init(void)
{
    struct page *src = alloc_page(GFP_KERNEL | __GFP_ZERO);
    struct page *dst = alloc_page(GFP_KERNEL | __GFP_ZERO);
    struct page *table = alloc_page(GFP_KERNEL | __GFP_ZERO);
    struct page *reuse;
    unsigned long flags, oldroot, id, pa;
    unsigned long desc[6] __aligned(16);
    unsigned long *header;
    int i, bad = 0;

    if (!src || !dst || !table) {
        if (src) __free_page(src);
        if (dst) __free_page(dst);
        if (table) __free_page(table);
        return -ENOMEM;
    }
    /* Identical two-page working set and function-call loop for both copies.
     * Interrupts excluded to isolate the page copy's attempt count. */
    local_irq_save(flags);
    START();
    for (i = 0; i < REPS; ++i)
        copy_page(page_address(dst), page_address(src));
    STOP();
    START();
    for (i = 0; i < REPS; ++i)
        copy_highpage(dst, src);
    STOP();
    local_irq_restore(flags);
    pr_info("RAW_PAGE copies=%d plain_then_arch windows=2\n", REPS);

    /* A flat private table only for minting a storage tag. Restore the
     * previous root before anything can schedule; raw copies ignore liveness. */
    header = page_address(table);
    header[0] = 255;
    header[1] = 1;
    desc[0] = (unsigned long)page_address(src);
    desc[1] = desc[0] + PAGE_SIZE;
    desc[2] = 0x107; /* NONLIN RWX */
    desc[3] = desc[0];
    desc[4] = desc[5] = 0;
    local_irq_save(flags);
    oldroot = csr_read(0x5c1);
    csr_write(0x5c1, page_to_phys(table));
    asm volatile(".insn r 0x5b, 1, 0x50, %0, %1, %2"
                 : "=r" (id) : "r" (page_address(src)), "r" (desc) : "memory");
    csr_write(0x5c1, oldroot);
    local_irq_restore(flags);
    if (!id || !any_tag(page_to_phys(src))) {
        pr_info("RAW_PAGE FAIL mint\n");
        bad = 1;
    } else {
        copy_highpage(dst, src);
        if (!any_tag(page_to_phys(dst))) {
            pr_info("RAW_PAGE FAIL copy_tag\n");
            bad = 1;
        }
        pa = page_to_phys(src);
        /* PCP normally returns the just-freed frame. Report the actual
         * equality, never infer physical reuse from a virtual address. */
        __free_page(src);
        src = NULL;
        reuse = alloc_page(GFP_KERNEL);
        if (!reuse) {
            bad = 1;
        } else {
            bool tagged = any_tag(page_to_phys(reuse));
            pr_info("RAW_PAGE reused_same_frame=%d tag=%d\n",
                    page_to_phys(reuse) == pa, tagged);
            bad |= tagged || page_to_phys(reuse) != pa;
            __free_page(reuse);
        }
    }
    if (src) __free_page(src);
    __free_page(dst);
    __free_page(table);
    pr_info("RAW_PAGE %s\n", bad ? "FAIL" : "PASS");
    return 0;
}

static void __exit raw_page_probe_exit(void) {}
module_init(raw_page_probe_init);
module_exit(raw_page_probe_exit);
MODULE_LICENSE("GPL");
