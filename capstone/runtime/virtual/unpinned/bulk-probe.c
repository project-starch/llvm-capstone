// SPDX-License-Identifier: GPL-2.0-only
/* Trusted private-guest fixture: dense live/stale raw transport and order-1
 * free-hook clearing, including kernel reuse without __GFP_ZERO. */
#include <linux/module.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <linux/interrupt.h>
#include <linux/string.h>
#include <asm/csr.h>

static unsigned long query(struct page *page, bool ack)
{
    unsigned long result, pa = page_to_phys(page);
    if (ack)
        asm volatile(".insn i 0x0b, 6, %0, %1, 1"
                     : "=r"(result) : "r"(pa) : "memory");
    else
        asm volatile(".insn i 0x0b, 6, %0, %1, 0"
                     : "=r"(result) : "r"(pa) : "memory");
    return result;
}

static void raw16(void *dst, const void *src)
{
    unsigned long flags;
    local_irq_save(flags);
    asm volatile(".insn i 0x0b, 4, t0, %1, 0\n\t"
                 ".insn s 0x0b, 5, t0, 0(%0)\n\tli t0, 0"
                 : : "r"(dst), "r"(src) : "t0", "memory");
    local_irq_restore(flags);
}

/* Independent scalar cleanup also works on deliberately broken kernels. */
static void scrub(struct page *page, int order)
{
    unsigned int i, j;
    for (i = 0; i < (1U << order); i++) {
        unsigned long *p = page_address(page + i);
        for (j = 0; j < PAGE_SIZE / sizeof(*p); j += 2)
            WRITE_ONCE(p[j], 0);
        WARN_ON(query(page + i, true));
    }
}

static int check_copy(struct page *src, struct page *dst, struct page *check)
{
    unsigned int i, missing = 0, bad_bytes = 0;
    unsigned char *a = page_address(src), *b = page_address(dst);
    for (i = 0; i < PAGE_SIZE; i++)
        bad_bytes += READ_ONCE(a[i]) != READ_ONCE(b[i]);
    for (i = 0; i < PAGE_SIZE; i += 16) {
        raw16(page_address(check), b + i);
        missing += query(check, true) != 1;
        scrub(check, 0);
    }
    pr_info("BULK_MM copy missing_tags=%u bad_bytes=%u granules=%lu\n",
            missing, bad_bytes, PAGE_SIZE / 16);
    return missing || bad_bytes ? -EIO : 0;
}

static int __init probe_init(void)
{
    struct page *src = alloc_pages(GFP_KERNEL | __GFP_ZERO, 1);
    struct page *dst = alloc_pages(GFP_KERNEL | __GFP_ZERO, 1);
    struct page *table = alloc_page(GFP_KERNEL | __GFP_ZERO);
    struct page *check = alloc_page(GFP_KERNEL | __GFP_ZERO);
    unsigned long desc[6] __aligned(16), flags, oldroot, id, stale_id, retired;
    unsigned long oldpa, *header, sticky, actual;
    unsigned int i;
    int error = -ENOMEM;

    if (!src || !dst || !table || !check)
        goto out;
    header = page_address(table);
    header[0] = 255;
    header[1] = 1;
    desc[0] = (unsigned long)page_address(src);
    desc[1] = desc[0] + 2 * PAGE_SIZE;
    desc[2] = 0x107;
    desc[3] = desc[0];
    desc[4] = desc[5] = 0;
    local_irq_save(flags);
    oldroot = csr_read(0x5c1);
    csr_write(0x5c1, page_to_phys(table));
    asm volatile(".insn r 0x5b, 1, 0x50, %0, %1, %2"
                 : "=r"(id) : "r"(page_address(src)), "r"(desc) : "memory");
    asm volatile(".insn r 0x5b, 1, 0x50, %0, %1, %2"
                 : "=r"(stale_id) : "r"(page_address(src) + 16), "r"(desc) : "memory");
    asm volatile(".insn r 0x5b, 1, 0x53, %0, %1, zero"
                 : "=r"(retired) : "r"(stale_id) : "memory");
    csr_write(0x5c1, oldroot);
    local_irq_restore(flags);
    error = -EIO;
    if (!id || !stale_id || id == stale_id || retired || !query(src, true))
        goto out;
    for (i = 32; i < 2 * PAGE_SIZE; i += 16)
        raw16(page_address(src) + i, page_address(src) + (i & 16));
    copy_highpage(dst, src);
    copy_user_highpage(dst + 1, src + 1, 0, NULL);
    if (check_copy(src, dst, check) || check_copy(src + 1, dst + 1, check))
        goto out;
    for (i = 0; i < 2; i++) {
        clear_page(page_address(dst + i));
        sticky = query(dst + i, false);
        actual = query(dst + i, true);
        pr_info("BULK_MM zero page=%u sticky=%lu actual_tags=%lu\n", i, sticky, actual);
        if (sticky != 1 || actual ||
            memchr_inv(page_address(dst + i), 0, PAGE_SIZE) || !query(src + i, true))
            goto out;
    }
    oldpa = page_to_phys(src);
    __free_pages(src, 1);
    src = alloc_pages(GFP_KERNEL, 1);
    if (!src) {
        error = -ENOMEM;
        goto out;
    }
    actual = query(src, false) | query(src, true) |
             query(src + 1, false) | query(src + 1, true);
    pr_info("BULK_MM kernel_reuse same_frame=%u actual_tags=%lu pages=2\n",
            page_to_phys(src) == oldpa, actual);
    if (actual || page_to_phys(src) != oldpa)
        goto out;
    error = 0;
out:
    if (src) { scrub(src, 1); __free_pages(src, 1); }
    if (dst) { scrub(dst, 1); __free_pages(dst, 1); }
    if (check) { scrub(check, 0); __free_page(check); }
    if (table) __free_page(table);
    pr_info("BULK_MM %s\n", error ? "FAIL" : "PASS");
    return error;
}

static void __exit probe_exit(void) {}
module_init(probe_init);
module_exit(probe_exit);
MODULE_LICENSE("GPL");
