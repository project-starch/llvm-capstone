// SPDX-License-Identifier: GPL-2.0-only
/* Standalone trusted-mm qualification. No protected application may run in
 * this guest. CPTAG acknowledgement inspects actual tag words before clearing
 * the sticky summary. Query alone can retain conservative false positives. */
#include <linux/module.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <linux/interrupt.h>
#include <asm/csr.h>

static bool expect_copy_tags = true;
module_param(expect_copy_tags, bool, 0444);

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

static void scrub(struct page *page)
{
    unsigned long *p = page_address(page);
    unsigned int i;
    for (i = 0; i < PAGE_SIZE / sizeof(*p); i += 2)
        WRITE_ONCE(p[i], 0);
    WARN_ON(query(page, true));
}

static int __init probe_init(void)
{
    struct page *src = alloc_page(GFP_KERNEL | __GFP_ZERO);
    struct page *dst = alloc_page(GFP_KERNEL | __GFP_ZERO);
    struct page *table = alloc_page(GFP_KERNEL | __GFP_ZERO);
    unsigned long flags, oldroot, id, oldpa, tagged;
    unsigned long desc[6] __aligned(16);
    unsigned long *header;
    int error = -ENOMEM;

    if (!src || !dst || !table) goto out;
    header = page_address(table);
    header[0] = 255;
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
    local_irq_restore(flags);
    error = -EIO;
    if (!id || !query(src, false) || !query(src, true) || !query(src, false))
        goto clean;
    pr_info("RAW_MM acknowledge_refuses_live_tag=1\n");
    copy_highpage(dst, src);
    tagged = query(dst, true);
    pr_info("RAW_MM copied_tag=%lu expected=%u\n", tagged, expect_copy_tags);
    if (tagged != expect_copy_tags) goto clean;
    if (expect_copy_tags) {
        oldpa = page_to_phys(src);
        __free_page(src);
        src = alloc_page(GFP_KERNEL);
        if (!src) { error = -ENOMEM; goto clean; }
        tagged = query(src, false) | query(src, true);
        pr_info("RAW_MM reused_same_frame=%u tag=%lu\n",
                page_to_phys(src) == oldpa, tagged);
        if (tagged || page_to_phys(src) != oldpa) goto clean;
    }
    error = 0;
clean:
    /* The captag=off negative control must not return tagged frames to Linux. */
    if (src) scrub(src);
    scrub(dst);
out:
    if (src) __free_page(src);
    if (dst) __free_page(dst);
    if (table) __free_page(table);
    pr_info("RAW_MM %s\n", error ? "FAIL" : "PASS");
    return error;
}
static void __exit probe_exit(void) {}
module_init(probe_init);
module_exit(probe_exit);
MODULE_LICENSE("GPL");
