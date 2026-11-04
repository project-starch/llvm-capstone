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

/* Invalidate a byte outside the word being sampled. Integer reads now see
 * stored bits, including both halves of a formerly tagged representation. */
static void stored_words(struct page *check, const void *src, unsigned long words[2])
{
    unsigned char *p = page_address(check);
    raw16(p, src);
    WRITE_ONCE(p[15], 0);
    words[0] = READ_ONCE(((unsigned long *)p)[0]);
    raw16(p, src);
    WRITE_ONCE(p[0], 0);
    words[1] = READ_ONCE(((unsigned long *)p)[1]);
    scrub(check, 0);
}

enum { ALL_TAGGED, MIXED, NONE_TAGGED };

static int check_copy(struct page *src, struct page *dst, struct page *check, int mode)
{
    unsigned int i, missing = 0, extra = 0, bad_words = 0;
    unsigned long a_words[2], b_words[2], tagged;
    unsigned char *a = page_address(src), *b = page_address(dst);
    for (i = 0; i < PAGE_SIZE; i += 16) {
        bool expected = mode == ALL_TAGGED || (mode == MIXED && (i / 16) % 4 < 2);
        raw16(page_address(check), b + i);
        tagged = query(check, true);
        missing += expected && !tagged;
        extra += !expected && tagged;
        scrub(check, 0);
        stored_words(check, a + i, a_words);
        stored_words(check, b + i, b_words);
        bad_words += a_words[0] != b_words[0];
        bad_words += a_words[1] != b_words[1];
    }
    pr_info("BULK_MM %s missing_tags=%u extra_tags=%u bad_words=%u granules=%lu\n",
            mode == MIXED ? "mixed_copy" : mode == NONE_TAGGED ? "plain_copy" : "copy",
            missing, extra, bad_words, PAGE_SIZE / 16);
    return missing || extra || bad_words ? -EIO : 0;
}

static int __init probe_init(void)
{
    struct page *src = alloc_pages(GFP_KERNEL | __GFP_ZERO, 1);
    struct page *dst = alloc_pages(GFP_KERNEL | __GFP_ZERO, 1);
    /* Each CSMINT allocates a child and its trusted ancestor. */
    struct page *table = alloc_pages(GFP_KERNEL | __GFP_ZERO, 2);
    struct page *check = alloc_page(GFP_KERNEL | __GFP_ZERO);
    struct page *plain = alloc_pages(GFP_KERNEL | __GFP_ZERO, 1);
    unsigned long desc[6] __aligned(16), flags, oldroot, id, stale_id, retired;
    unsigned long oldpa, *header, sticky, actual;
    unsigned int i;
    int error = -ENOMEM;

    if (!src || !dst || !table || !check || !plain)
        goto out;
    header = page_address(table);
    header[0] = 1023;
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
    if (check_copy(src, dst, check, ALL_TAGGED) || check_copy(src + 1, dst + 1, check, ALL_TAGGED))
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
    /* Unique cursors distinguish reordered granules. Mixed live/stale,
     * scalar and zero cargo must replace a destination tagged everywhere. */
    for (i = 0; i < 2 * PAGE_SIZE; i += 16)
        raw16(page_address(dst) + i, page_address(src));
    scrub(src, 1);
    clear_page(page_address(src));
    clear_page(page_address(src + 1));
    local_irq_save(flags);
    oldroot = csr_read(0x5c1);
    csr_write(0x5c1, page_to_phys(table));
    retired = 0;
    for (i = 0; i < 2 * PAGE_SIZE; i += 16) {
        if ((i / 16) % 4 < 2) {
            desc[3] = desc[0] + i;
            asm volatile(".insn r 0x5b, 1, 0x50, %0, %1, %2"
                         : "=r"(id) : "r"(page_address(src) + i), "r"(desc) : "memory");
            if (!id) { retired = 1; break; }
            if ((i / 16) % 4 == 1) {
                asm volatile(".insn r 0x5b, 1, 0x53, %0, %1, zero"
                             : "=r"(retired) : "r"(id) : "memory");
                if (retired) break;
            }
        } else if ((i / 16) % 4 == 2) {
            unsigned long *p = page_address(src) + i;
            WRITE_ONCE(p[0], 0xa123000000000000UL + i);
            WRITE_ONCE(p[1], 0xb456000000000000UL + i);
        }
    }
    csr_write(0x5c1, oldroot);
    local_irq_restore(flags);
    if (retired)
        goto out;
    copy_highpage(dst, src);
    copy_user_highpage(dst + 1, src + 1, 0, NULL);
    if (check_copy(src, dst, check, MIXED) || check_copy(src + 1, dst + 1, check, MIXED))
        goto out;
    /* A source whose flag is clear is copied by ordinary stores. They must
     * still leave a destination that held tags on every granule tag-free. */
    scrub(plain, 1);
    if (query(plain, false) || query(plain + 1, false)) {
        pr_info("BULK_MM plain source is flagged\n");
        goto out;
    }
    for (i = 0; i < 2 * PAGE_SIZE; i += 16)
        raw16(page_address(dst) + i, page_address(src));
    copy_highpage(dst, plain);
    copy_user_highpage(dst + 1, plain + 1, 0, NULL);
    if (check_copy(plain, dst, check, NONE_TAGGED) ||
        check_copy(plain + 1, dst + 1, check, NONE_TAGGED))
        goto out;
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
    if (plain) { scrub(plain, 1); __free_pages(plain, 1); }
    if (table) __free_pages(table, 2);
    pr_info("BULK_MM %s\n", error ? "FAIL" : "PASS");
    return error;
}

static void __exit probe_exit(void) {}
module_init(probe_init);
module_exit(probe_exit);
MODULE_LICENSE("GPL");
