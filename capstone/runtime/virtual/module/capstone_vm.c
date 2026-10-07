// SPDX-License-Identifier: GPL-2.0-only
/* Trusted OS adapter: one shared-mm namespace and explicit C contexts. */
#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/sched/signal.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/list.h>
#include <linux/rbtree.h>
#include <asm/csr.h>
#include <asm/io.h>
#include <asm/pgtable.h>
#include "wire.h"
#include "cap_rev_table_abi.h"
_Static_assert(CV_NODE_RESERVE == CAP_REV_TABLE_RESERVE, "processor reserve ABI");

struct arena {
    unsigned long address, bytes, ancestor;
    struct page **pages;
    unsigned count;
    bool lazy;
};
struct heap_object {
    struct rb_node link;
    unsigned long address, bytes, ancestor, node;
    bool linear;
};
struct heap_page { struct page *page; unsigned long address; struct rb_node link; };
struct context;
struct cv_thread {
    struct list_head link;
    struct context *context;
    u64 id;
    unsigned long *frame;
    unsigned long frame_pa;
    unsigned long frame_user;
    struct page *frame_page;
    bool user_frame;
    bool started, reply_cap, terminal;
    unsigned long event;
};
struct context {
    struct list_head link;
    struct mm_struct *mm;
    unsigned long *frame, *table;
    struct cv_thread *main_thread;
    struct list_head threads;
    u64 next_thread;
    struct arena arenas[CV_MAX_ARENAS];
    struct cv_stats stats;
    u64 next_id, ids[CV_MAX_ARENAS];
    struct rb_root heap_objects, heap_pages;
    bool heap_busy, heap_enabled;
    u64 heap_owner, heap_op;
    struct heap_object *heap_old, *heap_pending;
    unsigned long heap_stats[9], heap_live;
};
static int heap_pin(struct context *c, unsigned long address, struct page **result);
static int heap_resident(struct context *c, struct heap_object *o);
static DEFINE_MUTEX(vm_lock);
/* Quotas are in base pages; zero means the 31-bit ISA limit. Parameters are
 * read-only after module load so concurrent contexts see a stable policy. */
static unsigned long node_initial_pages = 4, node_batch_pages = 16;
static unsigned long node_max_pages;
module_param(node_initial_pages, ulong, 0444);
module_param(node_batch_pages, ulong, 0444);
module_param(node_max_pages, ulong, 0444);

static unsigned long node_capacity(const struct context *c)
{ return c->table[0] & ~CAP_REV_TABLE_FLAGS; }
static unsigned long node_available(const struct context *c)
{ return node_capacity(c) - c->table[1] + (c->table[2] >> 32); }

static unsigned long *node_new_page(struct context *c)
{
    unsigned long *p = (void *)get_zeroed_page(GFP_KERNEL);
    if (p) c->stats.node_bytes += PAGE_SIZE;
    return p;
}

/* These are metadata pages owned by the adapter, not application arenas.
 * vm_lock excludes every C execution while a published root grows. */
static int node_add_page(struct context *c, unsigned long id)
{
    unsigned long *entry = c->table + CAP_REV_TABLE_DIRECTORY / 8 + (id >> 26);
    for (unsigned level = 0; level < 3; ++level) {
        unsigned long *page;
        if (!*entry) {
            page = node_new_page(c);
            if (!page) return -ENOMEM;
            WRITE_ONCE(*entry, virt_to_phys(page));
        } else page = phys_to_virt(*entry);
        if (level != 2)
            entry = page + ((id >> (level == 0 ? 17 : 8)) & 511);
    }
    smp_wmb();
    WRITE_ONCE(c->table[0], CAP_REV_TABLE_FLAGS | (id + 256));
    return 0;
}

static int node_grow(struct context *c, unsigned long required)
{
    unsigned long limit = node_max_pages ?: (1UL << 23);
    unsigned long pages = node_capacity(c) / 256, before = pages;
    unsigned long need, end;
    if (node_available(c) >= required) return 0;
    need = DIV_ROUND_UP(required - node_available(c), 256UL);
    end = min(limit, pages + max(node_batch_pages, need));
    for (; pages < end; ++pages)
        if (node_add_page(c, pages * 256)) break;
    if (pages != before) ++c->stats.node_growths;
    return node_available(c) >= required ? 0 : -ENOMEM;
}

static void node_free_table(struct context *c)
{
    if (!c->table) return;
    for (unsigned i = 0; i < CAP_REV_TABLE_ROOT_ENTRIES; ++i) {
        unsigned long pa = c->table[CAP_REV_TABLE_DIRECTORY / 8 + i];
        unsigned long *middle;
        if (!pa) continue;
        middle = phys_to_virt(pa);
        for (unsigned j = 0; j < 512; ++j) {
            unsigned long *leaf;
            if (!middle[j]) continue;
            leaf = phys_to_virt(middle[j]);
            for (unsigned k = 0; k < 512; ++k)
                if (leaf[k]) free_page((unsigned long)phys_to_virt(leaf[k]));
            free_page((unsigned long)leaf);
        }
        free_page((unsigned long)middle);
    }
    free_page((unsigned long)c->table);
    c->table = NULL;
}
static LIST_HEAD(contexts);
static u64 allocated_total, high_water, collections_total, reclaimed_total;

static unsigned long run(unsigned long frame, unsigned long action);

static struct cv_thread *find_thread(struct context *c, u64 id)
{
    struct cv_thread *t;
    list_for_each_entry(t, &c->threads, link)
        if (t->id == id) return t;
    return NULL;
}

static unsigned thread_count(struct context *c)
{
    unsigned count = 0;
    struct cv_thread *t;
    list_for_each_entry(t, &c->threads, link) ++count;
    return count;
}

static struct arena *find_arena(struct context *c, unsigned long address,
                                unsigned long bytes)
{
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
        if (c->arenas[i].pages && c->arenas[i].address == address &&
            c->arenas[i].bytes == bytes)
            return &c->arenas[i];
    return NULL;
}

static bool arena_has_thread_frame(struct context *c, struct arena *a)
{
    struct cv_thread *t;
    list_for_each_entry(t, &c->threads, link)
        if (t->user_frame && t->frame_user == a->address)
            return true;
    return false;
}

/* A child start frame stays in the owning mm. Pin exactly one page so the
 * QEMU supervisor sees a stable physical frame and collection can include it.
 * Requiring page alignment also prevents a frame from straddling an unpinned
 * page and keeps the frame ABI independently inspectable by the trusted
 * kernel. */
static int pin_thread_frame(struct context *c, unsigned long address,
                            struct cv_thread *t)
{
    struct page *page;
    struct vm_area_struct *v;
    if (!address || (address & (PAGE_SIZE - 1)) ||
        !access_ok((void __user *)address, PAGE_SIZE))
        return -EINVAL;
    mmap_read_lock(c->mm);
    v = find_vma(c->mm, address);
    if (!v || v->vm_start > address || address > v->vm_end - PAGE_SIZE ||
        v->vm_file || (v->vm_flags & VM_SHARED) ||
        !(v->vm_flags & VM_READ) || !(v->vm_flags & VM_WRITE)) {
        mmap_read_unlock(c->mm);
        return -EACCES;
    }
    mmap_read_unlock(c->mm);
    if (pin_user_pages_fast(address, 1, FOLL_WRITE, &page) != 1)
        return -EFAULT;
    if (!page_address(page)) {
        unpin_user_page(page);
        return -EOPNOTSUPP;
    }
    t->frame_page = page;
    t->frame = page_address(page);
    t->frame_pa = page_to_phys(page);
    t->frame_user = address;
    t->user_frame = true;
    return 0;
}

static void drop_thread_frame(struct cv_thread *t)
{
    if (t->user_frame && t->frame_page)
        unpin_user_page(t->frame_page);
    t->frame_page = NULL;
    t->frame = NULL;
    t->frame_pa = 0;
}

static void clear_thread_frame(struct cv_thread *t)
{
    if (!t->frame) return;
    /* The frame contains tagged PCC/GPR slots. Scalar clearing removes the
     * tags before the page is returned to the application's allocator. */
    memset(t->frame, 0, 656);
}

/* Called with mmap_read_lock held. Missing PTEs contain no application tags;
 * swapped/huge mappings are outside this profile and fail closed. */
static int resident_page(struct mm_struct *mm, unsigned long address)
{
    pgd_t *pgd = pgd_offset(mm, address);
    p4d_t *p4d;
    pud_t *pud;
    pmd_t *pmd;
    pte_t *pte, value;
    spinlock_t *lock;
    if (pgd_none(*pgd)) return 0;
    if (pgd_bad(*pgd)) return -EFAULT;
    p4d = p4d_offset(pgd, address);
    if (p4d_none(*p4d)) return 0;
    if (p4d_bad(*p4d)) return -EFAULT;
    pud = pud_offset(p4d, address);
    if (pud_none(*pud)) return 0;
    if (pud_leaf(*pud)) return -EOPNOTSUPP;
    if (pud_bad(*pud)) return -EFAULT;
    pmd = pmd_offset(pud, address);
    if (pmd_none(*pmd)) return 0;
    if (pmd_leaf(*pmd)) return -EOPNOTSUPP;
    if (pmd_bad(*pmd)) return -EFAULT;
    pte = pte_offset_map_lock(mm, pmd, address, &lock);
    if (!pte) return -EFAULT;
    value = READ_ONCE(*pte);
    pte_unmap_unlock(pte, lock);
    if (pte_none(value)) return 0;
    return pte_present(value) ? 1 : -EOPNOTSUPP;
}
static int pin_remaining(struct context *c, struct arena *a)
{
    int error = 0;
    mmap_read_lock(c->mm);
    for (unsigned j = 0; j < a->count; ++j) {
        struct page *page;
        unsigned long address = a->address + j * PAGE_SIZE;
        int resident;
        if (a->pages[j]) continue;
        resident = resident_page(c->mm, address);
        if (!resident) continue;
        if (resident < 0) { error = resident; break; }
        /* Trusted inspection may read PROT_NONE/RO pages, but it never faults
         * in unused pages, changes PTE rights or manufactures fresh backing. */
        if (pin_user_pages_remote(c->mm, address, 1, FOLL_FORCE | FOLL_NOFAULT,
                                  &page, NULL, NULL) != 1) {
            error = -EFAULT; break;
        }
        if (is_zero_pfn(page_to_pfn(page))) {
            unpin_user_page(page); continue;
        }
        if (!PageAnon(page) || page_mapcount(page) != 1) {
            unpin_user_page(page); error = -EACCES; break;
        }
        a->pages[j] = page;
        ++c->stats.pinned_pages;
        c->stats.peak_pages = max(c->stats.peak_pages, c->stats.pinned_pages);
    }
    mmap_read_unlock(c->mm);
    return error;
}

struct collect_list {
    unsigned long *head, *tail, count;
};

static int collect_page(struct collect_list *list, unsigned long pa)
{
    if (!list->tail || list->tail[1] == CAP_REV_COLLECT_ENTRIES) {
        unsigned long *page = (void *)get_zeroed_page(GFP_KERNEL);
        if (!page) return -ENOMEM;
        if (list->tail) list->tail[0] = virt_to_phys(page);
        else list->head = page;
        list->tail = page;
    }
    list->tail[2 + list->tail[1]++] = pa;
    ++list->count;
    return 0;
}

static void collect_free(struct collect_list *list)
{
    unsigned long *page = list->head;
    while (page) {
        unsigned long next = page[0];
        free_page((unsigned long)page);
        page = next ? phys_to_virt(next) : NULL;
    }
}

static int collect(struct context *c, struct cv_thread *owner)
{
    unsigned long flags, result;
    struct collect_list list = {0};
    int error = -ENOMEM;
    struct cv_thread *t;
    /* Include resident pages populated by Linux/the trusted launcher without
     * a C fault. Missing pages are not materialized; any inspection failure
     * prevents identity reuse. Saved contexts share this mm's namespace. */
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i) {
        int error = pin_remaining(c, &c->arenas[i]);
        if (error) return error;
    }
    for (struct rb_node *n = rb_first(&c->heap_objects); n; n = rb_next(n)) {
        int error = heap_resident(c, rb_entry(n, struct heap_object, link));
        if (error) return error;
    }
    list_for_each_entry(t, &c->threads, link)
        if (collect_page(&list, t->frame_pa)) goto out;
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
        for (unsigned j = 0; j < c->arenas[i].count; ++j)
            if (c->arenas[i].pages[j] &&
                collect_page(&list, page_to_phys(c->arenas[i].pages[j]))) goto out;
    for (struct rb_node *n = rb_first(&c->heap_pages); n; n = rb_next(n))
        if (collect_page(&list, page_to_phys(rb_entry(n, struct heap_page, link)->page))) goto out;
    owner->frame[80] = list.count | CAP_REV_COLLECT_CHAIN;
    owner->frame[81] = virt_to_phys(list.head);
    preempt_disable(); local_irq_save(flags);
    result = run(owner->frame_pa, 3);
    local_irq_restore(flags); preempt_enable();
    owner->frame[80] = owner->frame[81] = 0;
    if (result == ULONG_MAX) { error = -EIO; goto out; }
    ++c->stats.collections; c->stats.reclaimed += result;
    error = 0;
out:
    collect_free(&list);
    return error;
}

/* Caller holds vm_lock, with interrupts/preemption enabled. The same path
 * supplies user instructions, privileged mint and the allocator's preflight.
 * A count is available capacity, not a reservation against other C threads. */
static int ensure_nodes(struct context *c, unsigned long required)
{
    struct cv_thread *owner = NULL, *t;
    unsigned long available = node_available(c);
    unsigned long sweep_pages = thread_count(c), collect_min;
    bool collected = false;
    int error;
    if (required > (1UL << 31)) return -ENOMEM;
    if (available >= required) return 0;
    list_for_each_entry(t, &c->threads, link)
        if (t->started && !t->terminal) { owner = t; break; }
    /* pin_remaining inspects every registered virtual page, including holes.
     * Amortize that scan over at least as many retired identities and one
     * growth batch; a tiny table beside a large sparse VMA must not cause a
     * full scan every few hundred allocations. Existing free IDs win above. */
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
        sweep_pages += c->arenas[i].bytes / PAGE_SIZE;
    collect_min = max(sweep_pages, node_batch_pages * 256);
    collect_min = max(collect_min, node_capacity(c) / 4);
    if (owner && c->table[5] >= max(required - available, collect_min)) {
        error = collect(c, owner);
        if (error) return error;
        collected = true;
        if (node_available(c) >= required) return 0;
    }
    error = node_grow(c, required);
    if (!error) return 0;
    /* At a quota or memory limit, even a small successful sweep can help. */
    if (!collected && owner && c->table[5]) {
        error = collect(c, owner);
        if (error) return error;
    }
    return node_available(c) >= required ? 0 : -ENOMEM;
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
#include "heap-native.inc"
static int retire(struct context *c, struct arena *a, bool destroying)
{
    unsigned long flags, old, status;
    unsigned i, j;
    if (!destroying && arena_has_thread_frame(c, a))
        return -EBUSY;
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
static int add(struct context *c, struct cv_thread *t, struct cv_map *r)
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
        (t->started && (t->event != 3 || r->reg != 10 || t->reply_cap)) ||
        (!t->started && r->reg != 2 && r->reg != 3 &&
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
    if (ensure_nodes(c, 2 + CAP_REV_TABLE_RESERVE)) return -ENOMEM;
    mmap_read_lock(c->mm);
    for (unsigned long at = r->address; at < r->address + r->bytes;) {
        struct vm_area_struct *v = find_vma(c->mm, at);
        if (!v || v->vm_start > at || v->vm_file || (v->vm_flags & VM_SHARED) ||
            ((r->permissions & 4) && !(v->vm_flags & (r->linear ? VM_MAYREAD : VM_READ))) ||
            ((r->permissions & 2) && !(v->vm_flags & (r->linear ? VM_MAYWRITE : VM_WRITE))) ||
            ((r->permissions & 1) && !(v->vm_flags & (r->linear ? VM_MAYEXEC : VM_EXEC)))) {
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
    slot = t->frame + 8 + r->reg * 2;
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
                     : : "r"(slot), "r"(t->frame + 8) : "t0", "memory");
    }
    restore_root(old);
    local_irq_restore(flags); preempt_enable();
    *a = (struct arena){r->address, r->bytes, ancestor, pages,
                       r->bytes / PAGE_SIZE, r->linear};
    r->id = c->ids[index] = ++c->next_id;
    ++c->stats.arenas; c->stats.pinned_pages += n;
    c->stats.peak_pages = max(c->stats.peak_pages, c->stats.pinned_pages);
    if (t->started) t->reply_cap = true;
    return 0;
unpin:
    for (i = 0; i < max(n, 0L); ++i) unpin_user_page(pages[i]);
    kvfree(pages);
    return -EFAULT;
}
static int resolve(struct context *c, struct cv_thread *t)
{
    unsigned long address = t->frame[5], cause = t->frame[3];
    struct page *page;
    struct vm_area_struct *v;
    unsigned long required = cause == 15 ? VM_WRITE : cause == 12 ? VM_EXEC : VM_READ;
    if (!t->started || t->event != 4 || (cause != 12 && cause != 13 && cause != 15))
        return -EINVAL;
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i) {
        struct arena *a = &c->arenas[i];
        unsigned index;
        long n;
        if (!a->pages || !a->lazy || address < a->address ||
            address - a->address >= a->bytes) continue;
        index = (address - a->address) / PAGE_SIZE;
        mmap_read_lock(c->mm);
        v = find_vma(c->mm, address);
        if (!v || v->vm_start > address || !(v->vm_flags & required) ||
            v->vm_file || (v->vm_flags & VM_SHARED)) {
            mmap_read_unlock(c->mm); return -EACCES;
        }
        /* Check actual PTE policy first. FORCE then materializes private
         * anonymous backing even for a read-only zero-page first touch. The
         * application still sees its original VMA/PTE protection. */
        n = pin_user_pages_remote(c->mm, address & PAGE_MASK, 1,
                                  FOLL_WRITE | ((v->vm_flags & VM_WRITE) ? 0 : FOLL_FORCE),
                                  &page, NULL, NULL);
        mmap_read_unlock(c->mm);
        if (n != 1) return -EFAULT;
        if (!PageAnon(page) || page_mapcount(page) != 1) {
            unpin_user_page(page); return -EACCES;
        }
        if (a->pages[index]) {
            /* mprotect may leave a write fault for Linux to resolve. It must
             * resolve to the same backing: tag-preserving moves are not part
             * of this pinned profile. Drop only the additional pin. */
            int same = a->pages[index] == page;
            unpin_user_page(page);
            return same ? 0 : -EACCES;
        }
        a->pages[index] = page;
        ++c->stats.pinned_pages;
        c->stats.peak_pages = max(c->stats.peak_pages, c->stats.pinned_pages);
        return 0;
    }
    return heap_resolve(c, address);
}

static void node_stats(struct context *c)
{
    c->stats.nodes = c->table[3];
    c->stats.nodes_high_water = c->table[1] - 2;
    c->stats.nodes_retired = c->table[5];
    c->stats.nodes_live = c->stats.nodes_high_water - c->table[5] - (c->table[2] >> 32);
    c->stats.node_capacity = node_capacity(c);
    high_water = max(high_water, c->stats.nodes_high_water);
}

static long vm_ioctl(struct file *f, unsigned int op, unsigned long arg)
{
    struct context *c = f->private_data;
    long rc = -EINVAL;
    unsigned long flags;
    if (current->mm != c->mm)
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
            stats.node_capacity += other->stats.node_capacity;
            stats.node_bytes += other->stats.node_bytes;
            if (other->stats.arenas) ++stats.contexts;
        }
        stats.nodes_high_water = high_water;
        rc = copy_to_user((void __user *)arg, &stats, sizeof(stats)) ? -EFAULT : 0;
    } else if (op == CV_HEAP_BEGIN || op == CV_HEAP_COMMIT) {
        struct cv_heap r;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        rc = op == CV_HEAP_BEGIN ? heap_begin(c, &r) : heap_commit(c, &r);
    } else if (op == CV_HEAP_RELEASE) {
        u64 address;
        if (copy_from_user(&address, (void __user *)arg, sizeof(address))) { rc = -EFAULT; goto out; }
        rc = heap_release(c, address);
    } else if (op == CV_HEAP_COPY) {
        struct cv_heap_copy r;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        rc = heap_copy(c, &r);
    } else if (op == CV_HEAP_RANGE) {
        struct cv_heap_range r;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        rc = heap_range(c, &r);
    } else if (op == CV_HEAP_SPAN) {
        struct cv_heap_range r;
        struct rb_node *n = c->heap_objects.rb_node;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        rc = -EINVAL;
        while (n) {
            struct heap_object *o = rb_entry(n, struct heap_object, link);
            if (r.address >= o->address && r.address - o->address <= o->bytes &&
                r.bytes <= o->bytes - (r.address - o->address)) { rc = 0; break; }
            n = r.address < o->address ? n->rb_left : n->rb_right;
        }
    } else if (op == CV_HEAP_STATS) {
        struct cv_heap_stats r;
        struct cv_thread *t;
        unsigned long info[6];
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        t = find_thread(c, r.thread);
        if (!t || t->event != 3 || !heap_info(c, t->frame + 28, info) ||
            info[2] != r.address || !(info[5] & 2) || r.address < info[0] ||
            r.address > info[1] ||
            info[1] - r.address < sizeof(c->heap_stats)) { rc = -EFAULT; goto out; }
        rc = copy_to_user((void __user *)r.address, c->heap_stats, sizeof(c->heap_stats)) ? -EFAULT : 0;
    } else if (op == CV_NODES) {
        struct cv_nodes r;
        struct cv_thread *t;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        t = find_thread(c, r.thread);
        if (!t || t->terminal) { rc = -ESRCH; goto out; }
        if (r.available > (1UL << 31) - CAP_REV_TABLE_RESERVE) { rc = -ENOMEM; goto out; }
        rc = ensure_nodes(c, r.available + CAP_REV_TABLE_RESERVE);
    } else if (op == CV_ADD) {
        struct cv_map r;
        struct cv_thread *t;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        t = find_thread(c, r.thread);
        if (!t || t->terminal) { rc = -ESRCH; goto out; }
        rc = add(c, t, &r);
        if (!rc && copy_to_user((void __user *)arg, &r, sizeof(r))) {
            /* The caller did not receive the handle. Reclaim the entire
             * context on close; never silently remove a delivered grant. */
            t->terminal = true; rc = -EFAULT;
        }
    } else if (op == CV_THREAD_CREATE) {
        struct cv_thread_create r;
        struct cv_thread *t;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        if (!r.frame || !find_arena(c, r.frame, PAGE_SIZE)) { rc = -EINVAL; goto out; }
        if (thread_count(c) >= 32) { rc = -ENOSPC; goto out; }
        t = kvzalloc(sizeof(*t), GFP_KERNEL);
        if (!t) { rc = -ENOMEM; goto out; }
        t->context = c; t->id = ++c->next_thread;
        rc = pin_thread_frame(c, r.frame, t);
        if (rc) { kvfree(t); goto out; }
        /* The kernel owns the process address space and lifetime root. The
         * application supplies only the entry/register capabilities. */
        t->frame[0] = csr_read(CSR_SATP);
        t->frame[1] = virt_to_phys(c->table);
        t->frame[2] = t->frame[3] = t->frame[4] = t->frame[5] = t->frame[6] = 0;
        t->frame[7] = 3;
        t->frame[80] = t->frame[81] = 0;
        INIT_LIST_HEAD(&t->link);
        list_add_tail(&t->link, &c->threads);
        r.thread = t->id;
        rc = copy_to_user((void __user *)arg, &r, sizeof(r)) ? -EFAULT : 0;
        if (rc) {
            list_del(&t->link); clear_thread_frame(t); drop_thread_frame(t); kvfree(t);
        }
    } else if (op == CV_THREAD_EXIT) {
        struct cv_thread_control r;
        struct cv_thread *t;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        t = find_thread(c, r.thread);
        if (!t || t == c->main_thread) { rc = -EINVAL; goto out; }
        r.frame = t->frame_user;
        if (t->started) {
            preempt_disable(); local_irq_save(flags);
            run(t->frame_pa, 1);
            local_irq_restore(flags); preempt_enable();
        }
        list_del(&t->link); clear_thread_frame(t); drop_thread_frame(t);
        kvfree(t);
        rc = copy_to_user((void __user *)arg, &r, sizeof(r)) ? -EFAULT : 0;
    } else if (op == CV_STEP) {
        struct cv_step r;
        struct cv_thread *t;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        t = find_thread(c, r.thread);
        if (c->heap_busy) { rc = -EAGAIN; goto out; }
        if (!t || t->terminal) { rc = -ESRCH; goto out; }
        if (r.reply > 1 || (r.reply != (t->started && t->event == 3))) goto out;
        if (!t->started && (!t->frame[8] || !t->frame[12])) goto out;
        if (r.reply && !t->reply_cap) {
            t->frame[28] = r.result; t->frame[29] = 0;
        }
        preempt_disable(); local_irq_save(flags);
        if (!t->started) {
            t->frame[0] = csr_read(CSR_SATP);
            t->frame[1] = virt_to_phys(c->table);
            t->frame[7] = 3;
        }
        t->event = run(t->frame_pa, r.reply ? 2 : 0);
        t->started = true; t->reply_cap = false;
        local_irq_restore(flags); preempt_enable();
        ++c->stats.steps;
        if (t->event == 5) {
            int error = ensure_nodes(c, CAP_REV_TABLE_RESERVE + 1);
            /* Return through Linux between retries. Refuse endless retries
             * when the namespace consists of live or pinned identities. */
            t->frame[2] = error ? 2 : 1;
        }
        r.kind = t->frame[2]; r.cause = t->frame[3];
        r.pc = t->frame[4]; r.address = t->frame[5];
        memcpy(r.args, t->frame + 72, sizeof(r.args));
        r.thread = t->id;
        t->terminal = r.kind == 2;
        if (r.kind == 4) ++c->stats.faults;
        rc = copy_to_user((void __user *)arg, &r, sizeof(r)) ? -EFAULT : 0;
        if (rc) t->terminal = true;
    } else if (op == CV_RESOLVE) {
        struct cv_thread_control r;
        struct cv_thread *t;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        t = find_thread(c, r.thread);
        if (c->heap_busy) { rc = -EAGAIN; goto out; }
        rc = t ? resolve(c, t) : -ESRCH;
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
    struct context *existing;
    struct cv_thread *t;
    unsigned long flags, previous_root, available;
    if (num_online_cpus() != 1 || !current->mm || PAGE_SIZE != 4096)
        return -EOPNOTSUPP;
    if (node_initial_pages < 2 || node_initial_pages > (1UL << 23) ||
        !node_batch_pages || node_batch_pages > (1UL << 23) ||
        node_max_pages > (1UL << 23) ||
        (node_max_pages && node_max_pages < node_initial_pages)) return -EINVAL;
    c = kvzalloc(sizeof(*c), GFP_KERNEL);
    if (!c) return -ENOMEM;
    asm volatile("csrr %0, 0x5c0" : "=r"(available));
    c->heap_enabled = !!(available & 0x100);
    INIT_LIST_HEAD(&c->threads);
    c->next_thread = 0;
    t = kvzalloc(sizeof(*t), GFP_KERNEL);
    if (!t) { kvfree(c); return -ENOMEM; }
    c->frame = (void *)get_zeroed_page(GFP_KERNEL);
    c->table = node_new_page(c);
    if (!c->frame || !c->table) {
        if (c->frame) free_page((unsigned long)c->frame);
        node_free_table(c);
        kvfree(t); kvfree(c); return -ENOMEM;
    }
    c->table[0] = CAP_REV_TABLE_FLAGS;
    c->table[1] = 2;
    c->table[4] = CAP_REV_TABLE_MAGIC;
    for (unsigned long i = 0; i < node_initial_pages; ++i) {
        if (node_add_page(c, i * 256)) {
            node_free_table(c); free_page((unsigned long)c->frame);
            kvfree(t); kvfree(c); return -ENOMEM;
        }
    }
    /* An older processor rejects the versioned format through urevavail.
     * Detect that before CSMINT could fault in the kernel context. */
    preempt_disable(); local_irq_save(flags);
    previous_root = select_root(c);
    asm volatile("csrr %0, 0xcc0" : "=r"(available));
    restore_root(previous_root);
    local_irq_restore(flags); preempt_enable();
    if (available != node_capacity(c) - 2) {
        node_free_table(c); free_page((unsigned long)c->frame);
        kvfree(t); kvfree(c); return -EOPNOTSUPP;
    }
    c->frame[1] = virt_to_phys(c->table);
    t->context = c; t->id = 0; t->frame = c->frame;
    t->frame_pa = virt_to_phys(c->frame); t->user_frame = false;
    c->main_thread = t;
    list_add(&t->link, &c->threads);
    c->mm = current->mm; mmgrab(c->mm);
    mutex_lock(&vm_lock);
    list_for_each_entry(existing, &contexts, link)
        if (existing->mm == c->mm) {
            mutex_unlock(&vm_lock);
            mmdrop(c->mm);
            node_free_table(c);
            memset(c->frame, 0, PAGE_SIZE);
            free_page((unsigned long)c->frame);
            kvfree(t); kvfree(c);
            return -EBUSY;
        }
    f->private_data = c;
    list_add(&c->link, &contexts);
    mutex_unlock(&vm_lock);
    return 0;
}
static int vm_release(struct inode *inode, struct file *f)
{
    struct context *c = f->private_data;
    unsigned long flags;
    struct cv_thread *t, *tmp;
    mutex_lock(&vm_lock);
    list_del(&c->link);
    node_stats(c);
    allocated_total += c->stats.nodes;
    collections_total += c->stats.collections;
    reclaimed_total += c->stats.reclaimed;
    /* Stop every supervisor slot while its frame pages are still pinned. */
    list_for_each_entry(t, &c->threads, link) {
        if (t->started) {
            preempt_disable(); local_irq_save(flags);
            run(t->frame_pa, 1);
            local_irq_restore(flags); preempt_enable();
        }
    }
    /* Teardown still clears tags in user frames (including thread start
     * frames) before the final pins are dropped. */
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
        if (c->arenas[i].pages || c->arenas[i].address)
            /* The task may be in exit_mm() here.  A teardown must never
             * call pin_user_pages_fast() to materialize lazy holes; only
             * already pinned pages can safely be revoked and released. */
            retire(c, &c->arenas[i], true);
    while (c->heap_objects.rb_node)
        heap_retire(c, rb_entry(rb_first(&c->heap_objects), struct heap_object, link));
    while (c->heap_pages.rb_node)
        heap_drop_page(c, rb_entry(rb_first(&c->heap_pages), struct heap_page, link), true);
    kfree(c->heap_pending);
    list_for_each_entry_safe(t, tmp, &c->threads, link) {
        list_del(&t->link);
        if (t->user_frame) { clear_thread_frame(t); drop_thread_frame(t); }
        else if (t->frame) { memset(t->frame, 0, PAGE_SIZE); free_page((unsigned long)t->frame); }
        kvfree(t);
    }
    node_free_table(c);
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
