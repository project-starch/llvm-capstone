// SPDX-License-Identifier: GPL-2.0-only
/* Trusted OS adapter: registered private pages and one explicit C context. */
#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/sched/signal.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/list.h>
#include <asm/csr.h>
#include <asm/io.h>
#include "wire.h"

struct arena {
    unsigned long address, bytes, ancestor;
    struct page **pages;
    unsigned count;
    bool lazy;
};
struct context {
    struct list_head link;
    struct mm_struct *mm;
    unsigned long *frame, *table;
    struct arena arenas[CV_MAX_ARENAS];
    struct cv_stats stats;
    bool started, reply_cap, terminal;
    unsigned long event;
    u64 next_id, ids[CV_MAX_ARENAS];
};
static DEFINE_MUTEX(vm_lock);
static LIST_HEAD(contexts);
static u64 allocated_total, high_water, collections_total, reclaimed_total;

static unsigned long run(unsigned long frame, unsigned long action);

static int pin_remaining(struct context *c, struct arena *a)
{
    for (unsigned j = 0; j < a->count; ++j) {
        struct page *page;
        if (a->pages[j]) continue;
        if (pin_user_pages_fast(a->address + j * PAGE_SIZE, 1, FOLL_WRITE, &page) != 1)
            return -EFAULT;
        if (!PageAnon(page) || page_mapcount(page) != 1) {
            unpin_user_page(page); return -EACCES;
        }
        a->pages[j] = page;
        ++c->stats.pinned_pages;
        c->stats.peak_pages = max(c->stats.peak_pages, c->stats.pinned_pages);
    }
    return 0;
}

static int collect(struct context *c)
{
    unsigned long count, at = 0, flags, result, *pages;
    unsigned order;
    /* Include pages even if Linux or the trusted launcher populated a lazy
     * arena without a C fault. This conservative sweep materializes remaining
     * holes at the first collection; a failed GUP never releases any IDs. */
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i) {
        int error = pin_remaining(c, &c->arenas[i]);
        if (error) return error;
    }
    count = c->stats.pinned_pages + 1;
    order = get_order(count * sizeof(unsigned long));
    pages = (void *)__get_free_pages(GFP_KERNEL, order);
    if (!pages) return -ENOMEM;
    pages[at++] = virt_to_phys(c->frame);
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
        for (unsigned j = 0; j < c->arenas[i].count; ++j)
            if (c->arenas[i].pages[j]) pages[at++] = page_to_phys(c->arenas[i].pages[j]);
    c->frame[80] = at;
    c->frame[81] = virt_to_phys(pages);
    preempt_disable(); local_irq_save(flags);
    result = run(virt_to_phys(c->frame), 3);
    local_irq_restore(flags); preempt_enable();
    c->frame[80] = c->frame[81] = 0;
    free_pages((unsigned long)pages, order);
    if (result == ULONG_MAX) return -EIO;
    ++c->stats.collections; c->stats.reclaimed += result;
    return 0;
}

static unsigned long run(unsigned long frame, unsigned long action)
{
    unsigned long result;
    asm volatile(".insn r 0x5b, 1, 0x24, %0, %1, %2"
                 : "=r"(result) : "r"(frame), "r"(action) : "memory");
    return result;
}
static unsigned long select_root(struct context *c)
{
    unsigned long old;
    asm volatile("csrr %0, 0x5c1\ncsrw 0x5c1, %1"
                 : "=&r"(old) : "r"(virt_to_phys(c->table)) : "memory");
    return old;
}
static void restore_root(unsigned long root)
{
    asm volatile("csrw 0x5c1, %0" : : "r"(root) : "memory");
}
static int retire(struct context *c, struct arena *a, bool destroying)
{
    unsigned long flags, old, status;
    unsigned i, j;
    if (!destroying) {
        int error = pin_remaining(c, a);
        if (error) return error;
    }
    preempt_disable(); local_irq_save(flags);
    old = select_root(c);
    asm volatile(".insn r 0x5b, 1, 0x53, %0, %1, zero"
                 : "=r"(status) : "r"(a->ancestor) : "memory");
    restore_root(old);
    local_irq_restore(flags); preempt_enable();
    if (status && !destroying) return -EIO;
    /* Clear every tag before returning frames to Linux. Later collection can
     * reuse dead IDs only after sweeping all remaining namespace storage. */
    for (i = 0; i < a->count; ++i) {
        unsigned long *p;
        if (!a->pages[i]) continue;
        p = page_address(a->pages[i]);
        for (j = 0; j < PAGE_SIZE / sizeof(*p); ++j)
            WRITE_ONCE(p[j], READ_ONCE(p[j]));
        set_page_dirty_lock(a->pages[i]);
        unpin_user_page(a->pages[i]);
        --c->stats.pinned_pages;
    }
    --c->stats.arenas;
    kvfree(a->pages);
    memset(a, 0, sizeof(*a));
    return 0;
}
static int add(struct context *c, struct cv_map *r)
{
    unsigned i, j, index = 0;
    struct arena *a = NULL;
    unsigned long flags, old, desc[3], ancestor, *slot, mapped = 0;
    struct page **pages;
    long n;
    if (!r->bytes || r->bytes > CV_MAX_REGION_BYTES ||
        ((r->address | r->bytes) & (PAGE_SIZE - 1)) ||
        !is_power_of_2(r->bytes) || (r->address & (r->bytes - 1)) ||
        r->address > ULONG_MAX - r->bytes ||
        !access_ok((void __user *)r->address, r->bytes) ||
        r->permissions > 7 || r->linear > 1 || r->reg > 31 ||
        r->cursor < r->address || r->cursor > r->address + r->bytes ||
        (r->linear && r->cursor != r->address) ||
        (c->started && (c->event != 3 || r->reg != 10 || c->reply_cap)) ||
        (!c->started && r->reg != 2 && r->reg != 3 &&
         r->reg != 10 && r->reg != 11 && r->reg != 12 && r->reg != 13) ||
        (r->reg == 3 && (!(r->permissions & 1) || r->linear)))
        return -EINVAL;
    for (i = 0; i < CV_MAX_ARENAS; ++i) {
        struct arena *b = &c->arenas[i];
        if (!b->pages) { if (!a) { a = b; index = i; } continue; }
        mapped += b->bytes;
        if (r->address < b->address + b->bytes && b->address < r->address + r->bytes)
            return -EINVAL;
    }
    if (!a) return -ENOSPC;
    if (mapped + r->bytes > CV_MAX_BYTES) return -ENOMEM;
    if (((c->table[0] & ~(1UL << 63)) - c->table[1] + (c->table[2] >> 32)) < 2) return -ENOSPC;
    mmap_read_lock(c->mm);
    for (unsigned long at = r->address; at < r->address + r->bytes;) {
        struct vm_area_struct *v = find_vma(c->mm, at);
        if (!v || v->vm_start > at || v->vm_file || (v->vm_flags & VM_SHARED) ||
            !(v->vm_flags & VM_READ) ||
            ((r->permissions & 2) && !(v->vm_flags & VM_WRITE)) ||
            ((r->permissions & 1) && !(v->vm_flags & VM_EXEC))) {
            mmap_read_unlock(c->mm); return -EACCES;
        }
#ifdef CONFIG_TRANSPARENT_HUGEPAGE
        if (!(v->vm_flags & VM_NOHUGEPAGE)) {
            mmap_read_unlock(c->mm); return -EOPNOTSUPP;
        }
#endif
        at = min((unsigned long)(r->address + r->bytes), v->vm_end);
    }
    mmap_read_unlock(c->mm);
    pages = kvcalloc(r->bytes / PAGE_SIZE, sizeof(*pages), GFP_KERNEL);
    if (!pages) return -ENOMEM;
    n = 0;
    /* Growing RW arenas start with absent PTEs. Linux resolves each first
     * touch through GUP, then the adapter pins that backing until retirement. */
    if (!r->linear) {
        n = pin_user_pages_fast(r->address, r->bytes / PAGE_SIZE, FOLL_WRITE, pages);
        if (n != r->bytes / PAGE_SIZE) goto unpin;
    }
    for (i = 0; i < n; ++i) {
        if (!PageAnon(pages[i]) || page_mapcount(pages[i]) != 1) goto unpin;
        for (j = 0; j < i; ++j) if (pages[i] == pages[j]) goto unpin;
    }
    slot = c->frame + 8 + r->reg * 2;
    if (slot[0] || slot[1]) goto unpin;
    desc[0] = r->address; desc[1] = r->address + r->bytes; desc[2] = r->permissions;
    preempt_disable(); local_irq_save(flags);
    old = select_root(c);
    asm volatile(".insn r 0x5b, 1, 0x50, %0, %1, %2"
                 : "=r"(ancestor) : "r"(slot), "r"(desc) : "memory");
    /* C pointers are copyable. Linear mapping grants remain in move-only
     * slots until the allocator derives a non-linear object from them. */
    if (!r->linear) {
        asm volatile(".insn i 0x5b, 3, t0, %0, 0\n"
                     ".insn r 0x5b, 1, 3, t0, zero, zero\n"
                     ".insn r 0x5b, 1, 5, t0, t0, %1\n"
                     ".insn s 0x5b, 4, t0, 0(%0)\nli t0, 0"
                     : : "r"(slot), "r"(r->cursor) : "t0", "memory");
    }
    if (r->reg == 3) {
        asm volatile(".insn i 0x5b, 3, t0, %0, 0\n"
                     ".insn s 0x5b, 4, t0, 0(%1)\nli t0, 0"
                     : : "r"(slot), "r"(c->frame + 8) : "t0", "memory");
    }
    restore_root(old);
    local_irq_restore(flags); preempt_enable();
    *a = (struct arena){r->address, r->bytes, ancestor, pages,
                       r->bytes / PAGE_SIZE, r->linear};
    r->id = c->ids[index] = ++c->next_id;
    ++c->stats.arenas; c->stats.pinned_pages += n;
    c->stats.peak_pages = max(c->stats.peak_pages, c->stats.pinned_pages);
    if (c->started) c->reply_cap = true;
    return 0;
unpin:
    for (i = 0; i < max(n, 0L); ++i) unpin_user_page(pages[i]);
    kvfree(pages);
    return -EFAULT;
}
static int resolve(struct context *c)
{
    unsigned long address = c->frame[5], cause = c->frame[3];
    struct page *page;
    if (!c->started || c->event != 4 || (cause != 13 && cause != 15)) return -EINVAL;
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i) {
        struct arena *a = &c->arenas[i];
        unsigned index;
        if (!a->pages || !a->lazy || address < a->address ||
            address - a->address >= a->bytes) continue;
        index = (address - a->address) / PAGE_SIZE;
        /* A fault on an already pinned page is a permission failure. Do not
         * alter the VMA or grant permissions in response to such a fault. */
        if (a->pages[index]) return -EACCES;
        if (pin_user_pages_fast(address & PAGE_MASK, 1, FOLL_WRITE, &page) != 1)
            return -EFAULT;
        if (!PageAnon(page) || page_mapcount(page) != 1) {
            unpin_user_page(page); return -EACCES;
        }
        a->pages[index] = page;
        ++c->stats.pinned_pages;
        c->stats.peak_pages = max(c->stats.peak_pages, c->stats.pinned_pages);
        return 0;
    }
    return -EFAULT;
}
static void node_stats(struct context *c)
{
    c->stats.nodes = c->table[3];
    c->stats.nodes_high_water = c->table[1] - 2;
    c->stats.nodes_live = c->stats.nodes_retired = 0;
    for (unsigned long id = 2; id < c->table[1]; ++id) {
        unsigned long flags = c->table[2 * id + 1] >> 32;
        if (flags & 1) ++c->stats.nodes_live;
        else if (!(flags & 4)) ++c->stats.nodes_retired;
    }
    high_water = max(high_water, c->stats.nodes_high_water);
}

static long vm_ioctl(struct file *f, unsigned int op, unsigned long arg)
{
    struct context *c = f->private_data;
    long rc = -EINVAL;
    unsigned long flags;
    if (current->mm != c->mm || atomic_read(&c->mm->mm_users) != 1)
        return -EPERM;
    if (mutex_lock_interruptible(&vm_lock)) return -EINTR;
    if (op == CV_GLOBAL) {
        struct context *other;
        struct cv_global stats = {.nodes_allocated = allocated_total,
            .collections = collections_total, .reclaimed = reclaimed_total};
        list_for_each_entry(other, &contexts, link) {
            node_stats(other);
            stats.arenas += other->stats.arenas;
            stats.pinned_pages += other->stats.pinned_pages;
            stats.nodes_allocated += other->stats.nodes;
            stats.nodes_live += other->stats.nodes_live;
            stats.nodes_retired += other->stats.nodes_retired;
            stats.collections += other->stats.collections;
            stats.reclaimed += other->stats.reclaimed;
            if (other->stats.arenas) ++stats.contexts;
        }
        stats.nodes_high_water = high_water;
        rc = copy_to_user((void __user *)arg, &stats, sizeof(stats)) ? -EFAULT : 0;
    } else if (op == CV_ADD) {
        struct cv_map r;
        if (c->terminal) goto out;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        rc = add(c, &r);
        if (!rc && copy_to_user((void __user *)arg, &r, sizeof(r))) {
            /* The caller did not receive the handle. Reclaim the entire
             * context on close; never silently remove a delivered grant. */
            c->terminal = true; rc = -EFAULT;
        }
    } else if (op == CV_STEP) {
        struct cv_step r;
        if (c->terminal) goto out;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        if (r.reply > 1 || (r.reply != (c->started && c->event == 3))) goto out;
        if (!c->started && (!c->frame[8] || !c->frame[12])) goto out;
        if (r.reply && !c->reply_cap) {
            c->frame[28] = r.result; c->frame[29] = 0;
        }
        preempt_disable(); local_irq_save(flags);
        if (!c->started) { c->frame[0] = csr_read(CSR_SATP); c->frame[7] = 3; }
        c->event = run(virt_to_phys(c->frame), r.reply ? 2 : 0);
        c->started = true; c->reply_cap = false;
        local_irq_restore(flags); preempt_enable();
        ++c->stats.steps;
        if (c->event == 5) {
            int error = collect(c);
            unsigned long available = (c->table[0] & ~(1UL << 63)) - c->table[1] + (c->table[2] >> 32);
            /* Return through Linux between retries. Refuse endless retries
             * when the namespace consists of live or pinned identities. */
            c->frame[2] = error || available <= 256 ? 2 : 1;
        }
        r.kind = c->frame[2]; r.cause = c->frame[3];
        r.pc = c->frame[4]; r.address = c->frame[5];
        memcpy(r.args, c->frame + 72, sizeof(r.args));
        c->terminal = r.kind == 2;
        if (r.kind == 4) ++c->stats.faults;
        rc = copy_to_user((void __user *)arg, &r, sizeof(r)) ? -EFAULT : 0;
        if (rc) c->terminal = true;
    } else if (op == CV_RESOLVE) {
        rc = resolve(c);
    } else if (op == CV_RETIRE) {
        u64 id;
        if (copy_from_user(&id, (void __user *)arg, sizeof(id))) { rc = -EFAULT; goto out; }
        for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
            if (c->ids[i] == id && c->arenas[i].pages) { rc = retire(c, &c->arenas[i], false); break; }
    } else if (op == CV_STATS) {
        node_stats(c);
        rc = copy_to_user((void __user *)arg, &c->stats, sizeof(c->stats)) ? -EFAULT : 0;
    } else rc = -ENOTTY;
out:
    mutex_unlock(&vm_lock);
    return rc;
}
static int vm_open(struct inode *inode, struct file *f)
{
    struct context *c;
    if (num_online_cpus() != 1 || !current->mm) return -EOPNOTSUPP;
    c = kvzalloc(sizeof(*c), GFP_KERNEL);
    if (!c) return -ENOMEM;
    c->frame = (void *)get_zeroed_page(GFP_KERNEL);
    c->table = (void *)__get_free_pages(GFP_KERNEL | __GFP_ZERO, CV_NODE_ORDER);
    if (!c->frame || !c->table) {
        if (c->frame) free_page((unsigned long)c->frame);
        if (c->table) free_pages((unsigned long)c->table, CV_NODE_ORDER);
        kvfree(c); return -ENOMEM;
    }
    c->table[0] = (1UL << 63) | (CV_NODE_BYTES / 16); c->table[1] = 2;
    c->frame[1] = virt_to_phys(c->table);
    c->mm = current->mm; mmgrab(c->mm);
    f->private_data = c;
    mutex_lock(&vm_lock);
    list_add(&c->link, &contexts);
    mutex_unlock(&vm_lock);
    return 0;
}
static int vm_release(struct inode *inode, struct file *f)
{
    struct context *c = f->private_data;
    unsigned long flags;
    mutex_lock(&vm_lock);
    list_del(&c->link);
    node_stats(c);
    allocated_total += c->stats.nodes;
    collections_total += c->stats.collections;
    reclaimed_total += c->stats.reclaimed;
    if (c->started) {
        preempt_disable(); local_irq_save(flags);
        run(virt_to_phys(c->frame), 1);
        local_irq_restore(flags); preempt_enable();
    }
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
        if (c->arenas[i].pages) retire(c, &c->arenas[i], true);
    memset(c->frame, 0, PAGE_SIZE); free_page((unsigned long)c->frame);
    memset(c->table, 0, CV_NODE_BYTES); free_pages((unsigned long)c->table, CV_NODE_ORDER);
    mmdrop(c->mm); kvfree(c);
    mutex_unlock(&vm_lock);
    return 0;
}
static const struct file_operations ops = {
    .owner = THIS_MODULE, .open = vm_open, .release = vm_release, .unlocked_ioctl = vm_ioctl,
};
static struct miscdevice device = {
    .minor = MISC_DYNAMIC_MINOR, .name = "capstone-vm", .fops = &ops, .mode = 0600,
};
static int __init init(void) { return misc_register(&device); }
static void __exit done(void) { misc_deregister(&device); }
module_init(init); module_exit(done);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Private virtual Capstone execution adapter");
