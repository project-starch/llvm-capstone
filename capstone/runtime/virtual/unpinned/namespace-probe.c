// SPDX-License-Identifier: GPL-2.0-only
/* Trusted qualification fixture. Load only in a private test guest without
 * protected applications. Seeds an old namespace tag and reads raw bytes. */
#include <linux/module.h>
#include <linux/mm.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/interrupt.h>
#include <asm/csr.h>
#include "namespace-probe.h"

static long probe_ioctl(struct file *f, unsigned int op, unsigned long arg)
{
    struct onq_namespace_sample s;
    struct page *page = NULL, *table = NULL, *copy = NULL;
    unsigned long flags, root, id, value, desc[6] __aligned(16);
    unsigned long *data;
    int error = -ENOMEM;
    if ((op != ONQ_NS_SEED && op != ONQ_NS_SAMPLE && op != ONQ_NS_CLEAR) ||
        copy_from_user(&s, (void __user *)arg, sizeof(s)) ||
        (s.address & (PAGE_SIZE - 1))) return -EINVAL;
    if (pin_user_pages_fast(s.address, 1, FOLL_WRITE, &page) != 1) return -EFAULT;
    table = alloc_page(GFP_KERNEL | __GFP_ZERO);
    copy = alloc_page(GFP_KERNEL | __GFP_ZERO);
    if (!table || !copy) goto out;
    data = page_address(page);
    s.frame = page_to_phys(page);
    if (op == ONQ_NS_SEED) {
        unsigned long *header = page_address(table);
        header[0] = 255; header[1] = 1;
        desc[0] = s.address; desc[1] = s.address + PAGE_SIZE;
        desc[2] = 0x107; desc[3] = s.address; desc[4] = desc[5] = 0;
        local_irq_save(flags);
        root = csr_read(0x5c1);
        csr_write(0x5c1, page_to_phys(table));
        asm volatile(".insn r 0x5b, 1, 0x50, %0, %1, %2"
                     : "=r"(id) : "r"(data), "r"(desc) : "memory");
        csr_write(0x5c1, root);
        local_irq_restore(flags);
        if (!id) { error = -EIO; goto out; }
    }
    local_irq_save(flags);
    if (op == ONQ_NS_CLEAR)
        asm volatile(".insn i 0x0b, 4, %0, %1, 0\n sd %0, 0(%1)\n li %0, 0"
                     : "=&r"(value) : "r"(data) : "memory");
    asm volatile(".insn i 0x0b, 6, %0, %1, 1"
                 : "=r"(s.tag) : "r"(page_to_phys(page)) : "memory");
    /* Copy the exact representation, then clear the scratch tag by writing
     * its unchanged low word. Both words can now be read without projection. */
    asm volatile(".insn i 0x0b, 4, %0, %1, 0\n"
                 ".insn s 0x0b, 5, %0, 0(%2)\n"
                 "sd %0, 0(%2)\n li %0, 0"
                 : "=&r"(value) : "r"(data), "r"(page_address(copy)) : "memory");
    local_irq_restore(flags);
    s.low = ((unsigned long *)page_address(copy))[0];
    s.high = ((unsigned long *)page_address(copy))[1];
    error = copy_to_user((void __user *)arg, &s, sizeof(s)) ? -EFAULT : 0;
out:
    if (copy) __free_page(copy);
    if (table) __free_page(table);
    unpin_user_page(page);
    return error;
}
static const struct file_operations probe_ops = {
    .owner = THIS_MODULE, .unlocked_ioctl = probe_ioctl,
};
static struct miscdevice probe_device = {
    .minor = MISC_DYNAMIC_MINOR, .name = "onq-namespace-probe",
    .fops = &probe_ops, .mode = 0600,
};
static int __init probe_init(void) { return misc_register(&probe_device); }
static void __exit probe_exit(void) { misc_deregister(&probe_device); }
module_init(probe_init);
module_exit(probe_exit);
MODULE_LICENSE("GPL");
