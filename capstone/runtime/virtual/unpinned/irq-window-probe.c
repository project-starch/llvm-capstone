// SPDX-License-Identifier: GPL-2.0-only
/* Trusted private-guest fixture: does a firmware-handled timer interrupt that
 * lands between a raw granule load and its store lose the cargo?
 *
 * Each iteration arms the platform timer a few hundred instructions ahead with
 * the S-level interrupts masked, runs the kernel's own copy_highpage over a page
 * whose every granule holds a tag, waits until the timer has fired and was
 * handled, and counts granules of the copy that lost their tag. Deadlines are
 * spread over the whole copy window by a varying delay and a varying distance.
 * Module parameters select the distance range, so a control with the deadline
 * far outside the copy window uses the same code. */
#include <linux/module.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <linux/timex.h>
#include <linux/string.h>
#include <asm/csr.h>
#include <asm/sbi.h>

/* sbi_set_timer is not exported to modules; the TIME extension is. */
static void arm_timer(u64 deadline)
{
    sbi_ecall(SBI_EXT_TIME, SBI_EXT_TIME_SET_TIMER, deadline, 0, 0, 0, 0, 0);
}

static unsigned int iters = 4000, dmin = 1, dmax = 24;
module_param(iters, uint, 0444);
module_param(dmin, uint, 0444);
module_param(dmax, uint, 0444);

static unsigned long query(struct page *page, bool ack)
{
    unsigned long result, pa = page_to_phys(page);
    if (ack)
        asm volatile(".insn i 0x0b, 6, %0, %1, 1" : "=r"(result) : "r"(pa) : "memory");
    else
        asm volatile(".insn i 0x0b, 6, %0, %1, 0" : "=r"(result) : "r"(pa) : "memory");
    return result;
}

static void raw16(void *dst, const void *src)
{
    asm volatile(".insn i 0x0b, 4, t0, %1, 0\n\t"
                 ".insn s 0x0b, 5, t0, 0(%0)\n\tli t0, 0"
                 : : "r"(dst), "r"(src) : "t0", "memory");
}

static void scrub(struct page *page)
{
    unsigned long *p = page_address(page);
    unsigned int j;
    for (j = 0; j < PAGE_SIZE / sizeof(*p); j += 2)
        WRITE_ONCE(p[j], 0);
    WARN_ON(query(page, true));
}

/* Granules of page copy that do not hold a tag. */
static unsigned int missing(struct page *copy, struct page *check)
{
    unsigned int i, lost = 0;
    unsigned long *c = page_address(check);
    for (i = 0; i < PAGE_SIZE; i += 16) {
        raw16(c, page_address(copy) + i);
        lost += query(check, true) != 1;
        WRITE_ONCE(c[0], 0);
    }
    return lost;
}

static int __init probe_init(void)
{
    struct page *src = alloc_page(GFP_KERNEL | __GFP_ZERO);
    struct page *dst = alloc_page(GFP_KERNEL | __GFP_ZERO);
    struct page *check = alloc_page(GFP_KERNEL | __GFP_ZERO);
    struct page *table = alloc_pages(GFP_KERNEL | __GFP_ZERO, 2);
    unsigned long desc[6] __aligned(16), flags, oldroot, id, deadline, t0, t1, calib = 0;
    unsigned int i, k, n, hit = 0, lost_iter = 0, lost_hit = 0, lost_total = 0,
                 first_k = 0, first_d = 0, first_j = 0, setup_bad;
    unsigned long *header;
    int error = -ENOMEM;

    if (!src || !dst || !check || !table)
        goto out;
    header = page_address(table);
    header[0] = 1023;
    header[1] = 1;
    desc[0] = (unsigned long)page_address(src);
    desc[1] = desc[0] + PAGE_SIZE;
    desc[2] = 0x107;
    desc[3] = desc[0];
    desc[4] = desc[5] = 0;
    local_irq_save(flags);
    oldroot = csr_read(0x5c1);
    csr_write(0x5c1, page_to_phys(table));
    asm volatile(".insn r 0x5b, 1, 0x50, %0, %1, %2"
                 : "=r"(id) : "r"(page_address(src)), "r"(desc) : "memory");
    csr_write(0x5c1, oldroot);
    error = -EIO;
    if (!id)
        goto out_irq;
    for (i = 16; i < PAGE_SIZE; i += 16)
        raw16(page_address(src) + i, page_address(src));
    setup_bad = missing(src, check);
    pr_info("IRQ_WINDOW setup_missing=%u\n", setup_bad);
    if (setup_bad)
        goto out_irq;
    /* Timer ticks the arming call itself takes; deadlines are placed after it. */
    for (k = 0; k < 16; k++) {
        t0 = get_cycles64();
        arm_timer(-1ULL);
        calib = max_t(unsigned long, calib, get_cycles64() - t0);
    }
    pr_info("IRQ_WINDOW arm_call_ticks=%lu\n", calib);
    for (k = 0; k < iters; k++) {
        unsigned int d = dmin + k % (dmax - dmin + 1), j = (k * 37) % 101, lost;
        for (n = 0; n < j; n++)
            asm volatile("nop");
        deadline = get_cycles64() + calib + d;
        arm_timer(deadline);
        t0 = get_cycles64();
        copy_highpage(dst, src);
        t1 = get_cycles64();
        while (get_cycles64() < deadline + 3)
            cpu_relax();
        lost = missing(dst, check);
        if (t0 <= deadline && deadline <= t1)
            hit++;
        if (lost) {
            if (!lost_iter) { first_k = k; first_d = d; first_j = j; }
            lost_iter++;
            lost_total += lost;
            lost_hit += t0 <= deadline && deadline <= t1;
        }
        scrub(dst);
    }
    arm_timer(-1ULL);
    error = 0;
out_irq:
    local_irq_restore(flags);
    pr_info("IRQ_WINDOW iterations=%u deadline_in_copy=%u lost_iterations=%u lost_in_copy=%u lost_granules=%u first_k=%u first_d=%u first_j=%u\n",
            iters, hit, lost_iter, lost_hit, lost_total, first_k, first_d, first_j);
out:
    if (src) { scrub(src); __free_page(src); }
    if (dst) { scrub(dst); __free_page(dst); }
    if (check) { scrub(check); __free_page(check); }
    if (table) __free_pages(table, 2);
    pr_info("IRQ_WINDOW %s\n", error ? "FAIL" : "DONE");
    return error;
}

static void __exit probe_exit(void) {}
module_init(probe_init);
module_exit(probe_exit);
MODULE_LICENSE("GPL");
