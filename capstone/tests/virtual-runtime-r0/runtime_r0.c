// SPDX-License-Identifier: GPL-2.0-only
/* R0 adapter: ordinary Linux mappings, pinned backing, explicit C context. */
#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/sched/signal.h>
#include <linux/mutex.h>
#include <asm/csr.h>
#include <asm/io.h>
#include "wire.h"

static DEFINE_MUTEX(r0_lock);

static void mint(unsigned long *frame, unsigned reg, unsigned long base,
                 unsigned long end, unsigned long perms)
{
    unsigned long desc[3] = {base, end, perms};
    void *slot = (char *)frame + 64 + reg * 16;
    asm volatile(".insn r 0x5b, 1, 0x50, zero, %0, %1"
                 : : "r"(slot), "r"(desc) : "memory");
}

static unsigned long run_context(unsigned long frame, unsigned long action)
{
    unsigned long result;
    asm volatile(".insn r 0x5b, 1, 0x24, %0, %1, %2"
                 : "=r"(result) : "r"(frame), "r"(action) : "memory");
    return result;
}

static long r0_ioctl(struct file *file, unsigned int op, unsigned long arg)
{
    struct r0_request r;
    struct page *pages[5] = {};
    unsigned long *frame = NULL, *table = NULL;
    unsigned long flags, old_root, status;
    long n, rc = -EINVAL;
    unsigned pinned = 0, i, j;
    unsigned long starts[4], lengths[4] = {PAGE_SIZE, 2*PAGE_SIZE, PAGE_SIZE, PAGE_SIZE};
    bool started = false;

    if (op != R0_RUN)
        return -ENOTTY;
    if (copy_from_user(&r, (void __user *)arg, sizeof(r)))
        return -EFAULT;
    if (!current->mm || atomic_read(&current->mm->mm_users) != 1 ||
        ((r.code | r.data | r.stack | r.tls) & (PAGE_SIZE - 1)) ||
        r.code_bytes < 4 || r.code_bytes > PAGE_SIZE ||
        r.data_bytes < PAGE_SIZE || r.data_bytes > 2 * PAGE_SIZE ||
        r.data_perms > 7 || r.code_perms > 7 || r.c_entry > 1 ||
        !access_ok((void __user *)r.code, PAGE_SIZE) ||
        !access_ok((void __user *)r.data, 2 * PAGE_SIZE) ||
        !access_ok((void __user *)r.stack, PAGE_SIZE) ||
        !access_ok((void __user *)r.tls, PAGE_SIZE))
        return -EINVAL;
    starts[0] = r.code; starts[1] = r.data;
    starts[2] = r.stack; starts[3] = r.tls;
    for (i = 0; i < 4; ++i)
        for (j = i + 1; j < 4; ++j)
            if (starts[i] < starts[j] + lengths[j] &&
                starts[j] < starts[i] + lengths[i])
                return -EINVAL;
    mmap_read_lock(current->mm);
    for (i = 0; i < 4; ++i) {
        unsigned long addr = starts[i], end = addr + lengths[i];
        while (addr < end) {
            struct vm_area_struct *vma = find_vma(current->mm, addr);
            if (!vma || vma->vm_start > addr || vma->vm_file ||
                (vma->vm_flags & VM_SHARED) || !(vma->vm_flags & VM_READ)) {
                mmap_read_unlock(current->mm);
                return -EOPNOTSUPP;
            }
            addr = min(end, vma->vm_end);
        }
    }
    mmap_read_unlock(current->mm);
    if (mutex_lock_interruptible(&r0_lock))
        return -EINTR;
    /* Read pins also permit the deliberate read-only-PTE negative control.
     * The test accepts only already resident private mappings. No asynchronous
     * I/O or kernel access to an application's interior pointers is exposed. */
    n = pin_user_pages_fast(r.data, 2, 0, pages);
    if (n > 0) pinned = n;
    if (n != 2) { rc = -EFAULT; goto out; }
    n = pin_user_pages_fast(r.code, 1, 0, pages + 2);
    if (n == 1) ++pinned;
    else { rc = -EFAULT; goto out; }
    n = pin_user_pages_fast(r.stack, 1, FOLL_WRITE, pages + 3);
    if (n == 1) ++pinned;
    else { rc = -EFAULT; goto out; }
    n = pin_user_pages_fast(r.tls, 1, FOLL_WRITE, pages + 4);
    if (n == 1) ++pinned;
    else { rc = -EFAULT; goto out; }
    for (i = 0; i < pinned; ++i) {
        if (!PageAnon(pages[i]) || page_mapcount(pages[i]) != 1) {
            rc = -EOPNOTSUPP; goto out;
        }
        for (j = i + 1; j < pinned; ++j)
            if (pages[i] == pages[j]) { rc = -EINVAL; goto out; }
    }
    r.scattered = page_to_pfn(pages[1]) != page_to_pfn(pages[0]) + 1 &&
                  page_to_pfn(pages[0]) != page_to_pfn(pages[1]) + 1;
    frame = (void *)get_zeroed_page(GFP_KERNEL);
    table = (void *)get_zeroed_page(GFP_KERNEL);
    if (!frame || !table) { rc = -ENOMEM; goto out; }
    table[0] = PAGE_SIZE / 16;
    table[1] = 1;
    frame[1] = virt_to_phys(table);
    preempt_disable();
    local_irq_save(flags);
    frame[0] = csr_read(CSR_SATP);
    asm volatile("csrr %0, 0x5c1\ncsrw 0x5c1, %1"
                 : "=&r"(old_root) : "r"(frame[1]) : "memory");
    mint(frame, 0, r.code, r.code + r.code_bytes, r.code_perms);
    mint(frame, r.c_entry ? 10 : 18, r.data, r.data + r.data_bytes, r.data_perms);
    mint(frame, 2, r.stack, r.stack + PAGE_SIZE, 6);
    mint(frame, 4, r.tls, r.tls + PAGE_SIZE, 6);
    /* Prepare a downward-growing stack through the existing S context path.
     * t0 is consumed by STC before interrupts or C code can observe it. */
    asm volatile(".insn i 0x5b, 3, t0, %0, 0\n"
                 "li t1, 4096\n"
                 ".insn r 0x5b, 1, 0x0c, t0, t0, t1\n"
                 ".insn s 0x5b, 4, t0, 0(%0)"
                 : : "r"((char *)frame + 96) : "t0", "t1", "memory");
    asm volatile("csrw 0x5c1, %0" : : "r"(old_root) : "memory");
    local_irq_restore(flags);
    preempt_enable();
    r.preemptions = 0;
    do {
        preempt_disable();
        local_irq_save(flags);
        status = run_context(virt_to_phys(frame), 0);
        started = true;
        local_irq_restore(flags);
        preempt_enable();
        if (status != 1) break;
        ++r.preemptions;
        if (signal_pending(current)) { rc = -EINTR; goto out; }
        cond_resched();
    } while (r.preemptions < 1000);
    r.kind = frame[2]; r.cause = frame[3]; r.pc = frame[4];
    r.address = frame[5]; r.result = frame[6];
    rc = status == 2 ? 0 : -ETIMEDOUT;
    if (!rc && copy_to_user((void __user *)arg, &r, sizeof(r)))
        rc = -EFAULT;
out:
    if (started) {
        preempt_disable();
        local_irq_save(flags);
        run_context(virt_to_phys(frame), 1);
        local_irq_restore(flags);
        preempt_enable();
    }
    /* No saved authority or table survives reuse of kernel storage. */
    if (frame) { memset(frame, 0, PAGE_SIZE); free_page((unsigned long)frame); }
    if (table) { memset(table, 0, PAGE_SIZE); free_page((unsigned long)table); }
    for (i = 0; i < pinned; ++i) {
        unsigned long *words = page_address(pages[i]);
        /* This bounded adapter destroys the whole execution namespace on
         * every ioctl. Preserve output bytes but clear ALL user-memory tags
         * before a later invocation can reuse the table's local identities.
         * A volatile scalar store invokes the physical tag-clear rule. */
        if (started) {
            for (j = 0; j < PAGE_SIZE / sizeof(*words); ++j)
                WRITE_ONCE(words[j], READ_ONCE(words[j]));
            set_page_dirty_lock(pages[i]);
        }
        unpin_user_page(pages[i]);
    }
    mutex_unlock(&r0_lock);
    return rc;
}
static const struct file_operations r0_ops = {
    .owner = THIS_MODULE, .unlocked_ioctl = r0_ioctl,
};
static struct miscdevice r0_device = {
    .minor = MISC_DYNAMIC_MINOR, .name = "capstone-r0", .fops = &r0_ops,
    .mode = 0600,
};
static int __init r0_init(void) { return misc_register(&r0_device); }
static void __exit r0_exit(void) { misc_deregister(&r0_device); }
module_init(r0_init);
module_exit(r0_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Resident virtual C execution experiment");
